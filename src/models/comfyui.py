"""ComfyUI provider adapter — local/self-hosted image and video generation.

Targets any ComfyUI instance reachable over HTTP (LAN, Tailscale, localhost).
Two model classes share one client:

- ``ComfyUIImageModel``  — Qwen-Image 2.1 (T2I + reference-image I2I)
- ``ComfyUIVideoModel``  — MiniMax-H3 Ref2VA (R2V + I2V, joint audio/video)

Workflows are NOT built in code. Each backing model has an *API-format*
workflow template under ``config/comfyui_workflows/`` plus a ``bindings``
map saying which node/field each LumenX parameter writes to. Adding a new
ComfyUI-hosted model is therefore a JSON drop-in, not a code change.

To produce a template: run the graph once in the ComfyUI web UI, then pull
the flattened API-format prompt back out of ``GET /history`` (the web UI's
"save workflow" writes canvas format, which ``/prompt`` rejects).

Environment:
    COMFYUI_IMAGE_HOST      host:port for the image instance
    COMFYUI_VIDEO_HOST      host:port for the video instance
    COMFYUI_IMAGE_WORKFLOW  template name (default: qwen_image_2.1)
    COMFYUI_VIDEO_WORKFLOW  template name (default: minimax_h3_video)
    COMFYUI_HOST_HEADER     override Host header (Tailscale serve routes by it)
    COMFYUI_TIMEOUT         per-job ceiling in seconds (default 3600)
"""

import json
import logging
import mimetypes
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

from .base import ImageGenModel, VideoGenModel

logger = logging.getLogger(__name__)

WORKFLOW_DIR = os.path.join("config", "comfyui_workflows")

DEFAULT_IMAGE_HOST = "127.0.0.1:8188"
DEFAULT_VIDEO_HOST = "127.0.0.1:8188"
DEFAULT_TIMEOUT = 3600
POLL_INTERVAL = 5

# ResolutionSelector's aspect_ratio is a fixed COMBO — these are its exact
# option strings, so we map LumenX's plain "9:16" onto them.
ASPECT_OPTIONS = {
    "1:1": "1:1 (Square)",
    "2:3": "2:3 (Portrait Photo)",
    "3:2": "3:2 (Photo)",
    "3:4": "3:4 (Portrait Standard)",
    "4:3": "4:3 (Standard)",
    "9:16": "9:16 (Portrait Widescreen)",
    "16:9": "16:9 (Widescreen)",
    "21:9": "21:9 (Ultrawide)",
}

# H3 sizes by *pixel budget*, not by short edge, so LumenX's resolution
# labels map onto a megapixel figure. Measured on Apple Silicon (M-series,
# 8-step Parasyte Turbo, per 5s of output):
#   0.1MP ≈ 256x416  ~3.5min     0.2MP ≈ 352x608  ~6.5min
#   0.4MP ≈ 480x864  ~13min
# 720p and above are extrapolated, not measured — expect them to be slow.
RESOLUTION_MEGAPIXELS = {
    "360p": 0.2,
    "480p": 0.4,
    "720p": 0.92,
    "1080p": 2.07,
}

# H3 was trained on ~124-362 frames at 24fps (roughly 5-15s). Outside that
# range quality is untested, so we clamp and say so rather than silently
# handing the model something it has never seen.
H3_MIN_SECONDS = 1.0
H3_MAX_SECONDS = 15.0


class ComfyUIError(RuntimeError):
    """A ComfyUI submission or execution failed."""


def rewrite_reference_tags(text: str, style: str) -> str:
    """Rewrite LumenX's ``[characterN:name]`` slots into a model's own syntax.

    ``prompt_assembly.inject_reference_tags`` emits ``名字[character1:名字]``
    where N is the 1-based reference-image slot. Both ComfyUI models bind
    references positionally from the prompt text, just with different
    spellings:

        MiniMax-H3      "The woman from <Picture 1> and <Picture 2> ..."
        Qwen-Image 2.1  "Keep the character in <image1> unchanged ..."

    Without the tag the reference image is silently ignored and the model
    behaves as plain T2I — this is the single most common way to get a
    "why did it change the face" result.

    ``style`` is ``"picture"`` (H3) or ``"image"`` (Qwen).
    """
    if not text:
        return text
    template = "<Picture {n}>" if style == "picture" else "<image{n}>"
    return re.sub(
        r"\[character(\d+):[^\]]*\]",
        lambda m: template.format(n=m.group(1)),
        text,
    )


def _has_reference_tag(text: str, style: str) -> bool:
    pattern = r"<Picture\s*\d+>" if style == "picture" else r"<image\s*\d+>"
    return bool(re.search(pattern, text or "", re.IGNORECASE))


def ensure_reference_tags(text: str, count: int, style: str) -> str:
    """Guarantee the prompt actually addresses its reference images.

    Appends an explicit identity-preservation clause when the caller's
    prompt carries no tags. Doing nothing here would mean paying for a
    generation that ignores the references entirely.
    """
    if count <= 0:
        return text
    text = rewrite_reference_tags(text, style)
    if _has_reference_tag(text, style):
        return text
    # Prompt-side tags are 1-based for BOTH models, even though the graph's
    # slot names differ (H3: <Picture 1> -> ref_image_0; Qwen: <image1> ->
    # image_1). Only the slot naming is template-specific.
    template = "<Picture {n}>" if style == "picture" else "<image{n}>"
    refs = " and ".join(template.format(n=i) for i in range(1, count + 1))
    logger.info("[ComfyUI] Prompt had no reference tags; appending %s", refs)
    return (
        f"{text}\n\nKeep the subject from {refs} exactly the same — "
        f"same face, same hair, same clothing."
    ).strip()


def _resolve_local_media(ref: str) -> Optional[str]:
    """Best-effort resolve a LumenX media ref to a readable local file.

    LumenX stores media local-first: paths are normally relative to
    ``output/``. Absolute paths and already-resolved paths pass through.
    Remote refs (http/OSS keys) are not handled here — the caller is
    expected to have materialized them first.
    """
    if not ref:
        return None
    if os.path.isabs(ref) and os.path.exists(ref):
        return ref
    candidate = os.path.join("output", ref)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    if os.path.exists(ref):
        return os.path.abspath(ref)
    return None


class ComfyUIClient:
    """Minimal ComfyUI HTTP client: upload, submit, poll, download.

    Two non-obvious requirements are handled here because getting either
    wrong produces a confusing failure:

    - ``Origin`` must match the host. ComfyUI's anti-DNS-rebinding
      middleware returns a bare 403 with no explanation otherwise.
    - ``Host`` may need overriding. A ComfyUI behind ``tailscale serve``
      is routed by Host header, so requests to the raw 100.x IP get a
      plain-text 404 even though the service is healthy.
    """

    def __init__(self, host: str, host_header: Optional[str] = None,
                 timeout: int = DEFAULT_TIMEOUT):
        self.host = host.rstrip("/")
        self.base = f"http://{self.host}"
        self.timeout = timeout
        self.headers = {"Origin": self.base}
        if host_header:
            self.headers["Host"] = host_header
        # trust_env=False keeps a machine-level HTTP_PROXY from swallowing
        # requests to private/Tailscale addresses.
        self.session = requests.Session()
        self.session.trust_env = False

    def _url(self, path: str) -> str:
        return f"{self.base}{path}"

    def health(self) -> Dict[str, Any]:
        r = self.session.get(self._url("/system_stats"), headers=self.headers, timeout=15)
        r.raise_for_status()
        return r.json()

    def upload_image(self, path: str, overwrite: bool = True) -> str:
        """Upload a local image/video into ComfyUI's input dir.

        Returns the name to hand to a ``LoadImage`` / ``LoadVideo`` node.
        """
        name = os.path.basename(path)
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        with open(path, "rb") as fh:
            files = {"image": (name, fh, ctype)}
            data = {"overwrite": "true" if overwrite else "false"}
            r = self.session.post(self._url("/upload/image"), files=files, data=data,
                                  headers=self.headers, timeout=300)
        r.raise_for_status()
        payload = r.json()
        subfolder = payload.get("subfolder") or ""
        return f"{subfolder}/{payload['name']}" if subfolder else payload["name"]

    def submit(self, graph: Dict[str, Any], client_id: Optional[str] = None) -> str:
        body = {"prompt": graph, "client_id": client_id or str(uuid.uuid4())}
        r = self.session.post(self._url("/prompt"), json=body,
                              headers=self.headers, timeout=120)
        if r.status_code >= 400:
            # node_errors carries the actual reason (bad node id, wrong
            # widget type, missing model file) — surface it verbatim.
            raise ComfyUIError(f"submit failed HTTP {r.status_code}: {r.text[:1200]}")
        payload = r.json()
        if "prompt_id" not in payload:
            raise ComfyUIError(f"submit rejected: {json.dumps(payload, ensure_ascii=False)[:1200]}")
        return payload["prompt_id"]

    def wait(self, prompt_id: str, timeout: Optional[int] = None,
             poll_interval: int = POLL_INTERVAL) -> Tuple[Dict[str, Any], float]:
        """Poll until the job finishes, tolerating transient poll failures.

        A single failed poll must not abort the job. Video sampling keeps the
        host busy for minutes at a time and these requests often cross a VPN
        or LAN, so an occasional read timeout is expected — dropping the job
        over one would throw away work that is still running server-side.
        Only a real execution failure, or the overall deadline, ends the wait.
        """
        deadline = time.time() + (timeout or self.timeout)
        started = time.time()
        consecutive_errors = 0
        while time.time() < deadline:
            try:
                r = self.session.get(self._url(f"/history/{prompt_id}"),
                                     headers=self.headers, timeout=60)
                r.raise_for_status()
                history = r.json()
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                # Noisy at first, then quiet — a long outage should leave a
                # trail without flooding the log for the whole deadline.
                if consecutive_errors <= 3 or consecutive_errors % 20 == 0:
                    logger.warning("[ComfyUI] poll %s failed (%d in a row): %s",
                                   prompt_id, consecutive_errors, exc)
                time.sleep(poll_interval)
                continue

            if history:
                record = history[prompt_id]
                status = record.get("status", {})
                if status.get("status_str") != "success":
                    raise ComfyUIError(
                        f"execution failed: {json.dumps(status, ensure_ascii=False)[:1200]}"
                    )
                return record, time.time() - started
            time.sleep(poll_interval)
        raise ComfyUIError(f"timed out after {timeout or self.timeout}s (prompt_id={prompt_id})")

    def collect_outputs(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten a history record's outputs into file descriptors.

        H3 emits the video under ``images`` (with ``animated: true``) and a
        separate audio-only FLAC from its PreviewAudio node, so we sweep
        every known bucket rather than assuming one.
        """
        found: List[Dict[str, Any]] = []
        for out in record.get("outputs", {}).values():
            for bucket in ("images", "gifs", "videos", "audio"):
                for item in out.get(bucket) or []:
                    if isinstance(item, dict) and "filename" in item:
                        found.append(item)
        return found

    def download(self, item: Dict[str, Any], dest_path: str, attempts: int = 3) -> str:
        params = {
            "filename": item["filename"],
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        }
        # The artifact already exists server-side; a dropped transfer is worth
        # retrying rather than discarding a completed generation.
        for attempt in range(1, attempts + 1):
            try:
                r = self.session.get(self._url("/view"), params=params,
                                     headers=self.headers, timeout=600)
                r.raise_for_status()
                break
            except Exception as exc:
                if attempt == attempts:
                    raise
                logger.warning("[ComfyUI] download %s failed (attempt %d/%d): %s",
                               item["filename"], attempt, attempts, exc)
                time.sleep(5 * attempt)
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        with open(dest_path, "wb") as fh:
            fh.write(r.content)
        return dest_path


class WorkflowTemplate:
    """An API-format ComfyUI graph plus its LumenX parameter bindings."""

    def __init__(self, payload: Dict[str, Any]):
        self.bindings: Dict[str, Dict[str, str]] = payload.get("bindings", {})
        self.ref_spec: Optional[Dict[str, Any]] = payload.get("ref_images")
        self.first_frame: Optional[Dict[str, str]] = payload.get("first_frame")
        self.last_frame: Optional[Dict[str, str]] = payload.get("last_frame")
        self.reserved_ids: List[str] = [str(i) for i in payload.get("reserved_node_ids", [])]
        self._graph: Dict[str, Any] = payload["graph"]

    @classmethod
    def load(cls, name: str, directory: str = WORKFLOW_DIR) -> "WorkflowTemplate":
        path = name if name.endswith(".json") else os.path.join(directory, f"{name}.json")
        if not os.path.exists(path):
            raise ComfyUIError(
                f"workflow template not found: {path}. "
                f"Export one with 'Workflow -> Export (API)' or pull it from GET /history."
            )
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def instantiate(self) -> Dict[str, Any]:
        """A deep copy, so concurrent jobs never share mutable node state."""
        return json.loads(json.dumps(self._graph))

    def set(self, graph: Dict[str, Any], key: str, value: Any) -> bool:
        """Write one bound parameter. Unbound keys are skipped, not fatal —
        a template legitimately may not expose every knob."""
        binding = self.bindings.get(key)
        if not binding:
            return False
        node, field = str(binding["node"]), binding["field"]
        if node not in graph:
            logger.warning("[ComfyUI] binding '%s' points at missing node %s", key, node)
            return False
        graph[node]["inputs"][field] = value
        return True

    def attach_references(self, graph: Dict[str, Any], names: List[str]) -> int:
        """Inject LoadImage nodes for uploaded refs and wire the AUTOGROW input.

        ComfyUI serializes a dynamic (AUTOGROW) input as a plain dict of
        ``{slot_name: [node_id, output_index]}`` — H3 uses ``ref_image_0..8``
        and Qwen uses ``image_1..16``, which is why start/prefix live in the
        template rather than in code.
        """
        if not names:
            return 0
        if not self.ref_spec:
            raise ComfyUIError("this workflow template declares no 'ref_images' slot")

        spec = self.ref_spec
        limit = int(spec.get("max", len(names)))
        if len(names) > limit:
            logger.warning("[ComfyUI] %d refs supplied, template accepts %d — truncating",
                           len(names), limit)
            names = names[:limit]

        node_ids = list(self.reserved_ids)
        if len(node_ids) < len(names):
            highest = max((int(k) for k in graph if k.isdigit()), default=0)
            node_ids += [str(highest + 1 + i) for i in range(len(names) - len(node_ids))]

        wired: Dict[str, Any] = {}
        start, prefix = int(spec.get("start", 0)), spec.get("prefix", "ref_image_")
        for i, name in enumerate(names):
            node_id = node_ids[i]
            graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
            wired[f"{prefix}{start + i}"] = [node_id, 0]

        graph[str(spec["node"])]["inputs"][spec["field"]] = wired
        return len(names)

    def attach_frame(self, graph: Dict[str, Any], slot: str, name: str) -> None:
        """Wire a first/last frame image (I2V mode)."""
        binding = self.first_frame if slot == "first_frame" else self.last_frame
        if not binding:
            raise ComfyUIError(f"template has no '{slot}' binding")
        highest = max((int(k) for k in graph if k.isdigit()), default=0)
        node_id = str(highest + 1)
        graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        graph[str(binding["node"])]["inputs"][binding["field"]] = [node_id, 0]


class _ComfyUIBase:
    """Shared config/plumbing for the image and video adapters."""

    host_env = ""
    workflow_env = ""
    default_host = DEFAULT_IMAGE_HOST
    default_workflow = ""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.params = self.config.get("params", {})
        self._client: Optional[ComfyUIClient] = None
        self._template: Optional[WorkflowTemplate] = None

    @property
    def host(self) -> str:
        return (self.config.get("host")
                or os.getenv(self.host_env, "").strip()
                or self.default_host)

    @property
    def client(self) -> ComfyUIClient:
        if self._client is None:
            self._client = ComfyUIClient(
                self.host,
                host_header=os.getenv("COMFYUI_HOST_HEADER", "").strip() or None,
                timeout=int(os.getenv("COMFYUI_TIMEOUT", str(DEFAULT_TIMEOUT))),
            )
        return self._client

    def template(self, name: Optional[str] = None) -> WorkflowTemplate:
        wanted = (name
                  or self.config.get("workflow")
                  or os.getenv(self.workflow_env, "").strip()
                  or self.default_workflow)
        if self._template is None or getattr(self._template, "_name", None) != wanted:
            self._template = WorkflowTemplate.load(wanted)
            self._template._name = wanted  # type: ignore[attr-defined]
        return self._template

    def _upload_refs(self, paths: List[str]) -> List[str]:
        names = []
        for path in paths:
            local = _resolve_local_media(path)
            if not local:
                logger.warning("[ComfyUI] reference not found locally, skipping: %s", path)
                continue
            names.append(self.client.upload_image(local))
        return names

    def _run(self, graph: Dict[str, Any], output_path: str,
             want: str = "image") -> Tuple[str, float, List[str]]:
        """Submit, wait, and save outputs.

        Returns ``(primary_path, elapsed_seconds, extra_paths)``. ``extra``
        matters for H3, which hands back a standalone audio track alongside
        the muxed video — LumenX's dub stage can use it directly instead of
        shelling out to ffmpeg to demux.
        """
        prompt_id = self.client.submit(graph)
        logger.info("[ComfyUI] submitted %s to %s", prompt_id, self.host)
        record, elapsed = self.client.wait(prompt_id)
        items = self.client.collect_outputs(record)
        if not items:
            raise ComfyUIError(f"job {prompt_id} produced no output files")

        video_exts = (".mp4", ".webm", ".mov", ".mkv", ".gif")
        audio_exts = (".flac", ".wav", ".mp3", ".ogg")

        def is_video(it):
            return it["filename"].lower().endswith(video_exts)

        def is_audio(it):
            return it["filename"].lower().endswith(audio_exts)

        if want == "video":
            primary = next((i for i in items if is_video(i)), None)
        else:
            primary = next((i for i in items if not is_video(i) and not is_audio(i)), None)
        if primary is None:
            raise ComfyUIError(
                f"job {prompt_id} returned no {want}; got "
                f"{[i['filename'] for i in items]}"
            )

        self.client.download(primary, output_path)
        extras = []
        base, _ = os.path.splitext(output_path)
        for item in items:
            if item is primary:
                continue
            ext = os.path.splitext(item["filename"])[1]
            side = f"{base}{ext}"
            if side == output_path:
                continue
            try:
                extras.append(self.client.download(item, side))
            except Exception as exc:  # side-car outputs are best-effort
                logger.warning("[ComfyUI] could not fetch side output %s: %s",
                               item["filename"], exc)
        logger.info("[ComfyUI] %s done in %.0fs -> %s (extras: %s)",
                    prompt_id, elapsed, output_path, extras)
        return output_path, elapsed, extras


class ComfyUIImageModel(_ComfyUIBase, ImageGenModel):
    """Qwen-Image 2.1 (or any ComfyUI image graph) behind LumenX's ImageGenModel."""

    host_env = "COMFYUI_IMAGE_HOST"
    workflow_env = "COMFYUI_IMAGE_WORKFLOW"
    default_host = DEFAULT_IMAGE_HOST
    default_workflow = "qwen_image_2.1"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        _ComfyUIBase.__init__(self, config)
        ImageGenModel.__init__(self, config or {})

    @staticmethod
    def _parse_size(size: Optional[str]) -> Tuple[int, int]:
        """Accept DashScope's ``1024*576`` and OpenAI's ``1024x576`` alike."""
        if not size:
            return 1024, 1024
        try:
            w, h = re.split(r"[*x×]", str(size).strip().lower())[:2]
            return int(w), int(h)
        except (ValueError, IndexError):
            logger.warning("[ComfyUI] unparseable size %r, falling back to 1024x1024", size)
            return 1024, 1024

    def generate(self, prompt: str, output_path: str,
                 ref_image_path: Optional[str] = None,
                 ref_image_paths: Optional[List[str]] = None,
                 model_name: Optional[str] = None,
                 **kwargs) -> Tuple[str, float]:
        template = self.template(kwargs.get("workflow"))
        graph = template.instantiate()

        refs = list(ref_image_paths or [])
        if ref_image_path:
            refs.append(ref_image_path)
        names = self._upload_refs(refs) if refs else []
        attached = template.attach_references(graph, names) if names else 0

        # Reference images bind positionally through <imageN> in the prompt.
        # No tag means the model quietly runs plain T2I.
        text = ensure_reference_tags(prompt, attached, style="image")

        width, height = self._parse_size(kwargs.get("size") or self.params.get("size"))
        template.set(graph, "prompt", text)
        template.set(graph, "negative_prompt", kwargs.get("negative_prompt") or "")
        template.set(graph, "width", width)
        template.set(graph, "height", height)
        template.set(graph, "batch_size", int(kwargs.get("n", 1)))
        template.set(graph, "steps", int(kwargs.get("steps", self.params.get("steps", 25))))
        template.set(graph, "cfg", float(kwargs.get("cfg", self.params.get("cfg", 1.0))))
        template.set(graph, "filename_prefix", kwargs.get("filename_prefix", "lumenx"))

        seed = kwargs.get("seed")
        template.set(graph, "seed", int(seed) if seed is not None else uuid.uuid4().int % (2 ** 31))

        # resolution=0 keeps each reference at native size. Downscaling refs
        # measurably degrades identity retention, so only honour an explicit
        # override.
        template.set(graph, "reference_resolution",
                     int(kwargs.get("reference_resolution", 0)))

        path, elapsed, _ = self._run(graph, output_path, want="image")
        return path, elapsed


class ComfyUIVideoModel(_ComfyUIBase, VideoGenModel):
    """MiniMax-H3 (or any ComfyUI video graph) behind LumenX's VideoGenModel.

    Supports ``generation_mode`` of ``r2v`` (reference images) and ``i2v``
    (first frame). H3 generates audio jointly with video; the standalone
    audio track is saved next to the mp4 and returned via
    :attr:`last_extra_outputs` for the dubbing stage.
    """

    host_env = "COMFYUI_VIDEO_HOST"
    workflow_env = "COMFYUI_VIDEO_WORKFLOW"
    default_host = DEFAULT_VIDEO_HOST
    default_workflow = "minimax_h3_video"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        _ComfyUIBase.__init__(self, config)
        VideoGenModel.__init__(self, config or {})
        self.last_extra_outputs: List[str] = []

    @staticmethod
    def _megapixels(resolution: Optional[str], override: Optional[float]) -> float:
        if override:
            return float(override)
        key = str(resolution or "480p").lower()
        if key not in RESOLUTION_MEGAPIXELS:
            logger.warning("[ComfyUI] unknown resolution %r, using 480p budget", resolution)
        return RESOLUTION_MEGAPIXELS.get(key, RESOLUTION_MEGAPIXELS["480p"])

    @staticmethod
    def _aspect(ratio: Optional[str]) -> str:
        key = (ratio or "9:16").strip()
        if key in ASPECT_OPTIONS:
            return ASPECT_OPTIONS[key]
        if key in ASPECT_OPTIONS.values():
            return key
        logger.warning("[ComfyUI] unknown aspect_ratio %r, using 9:16", ratio)
        return ASPECT_OPTIONS["9:16"]

    def generate(self, prompt: str, output_path: str,
                 img_path: Optional[str] = None,
                 img_url: Optional[str] = None,
                 model_name: Optional[str] = None,
                 **kwargs) -> Tuple[str, float]:
        template = self.template(kwargs.get("workflow"))
        graph = template.instantiate()

        mode = (kwargs.get("generation_mode") or "r2v").lower()
        ref_urls = kwargs.get("ref_image_urls") or kwargs.get("reference_image_urls") or []

        attached = 0
        if mode == "r2v" and ref_urls:
            names = self._upload_refs(list(ref_urls))
            attached = template.attach_references(graph, names)
            template.set(graph, "mode_switch", True)
        else:
            # I2V: drive from a first frame instead of reference images.
            template.set(graph, "mode_switch", False)
            first = img_path or _resolve_local_media(img_url or "")
            if not first:
                raise ComfyUIError(
                    "i2v mode needs a first frame (img_path/img_url), and r2v mode "
                    "needs ref_image_urls — neither was usable"
                )
            template.attach_frame(graph, "first_frame", self.client.upload_image(first))

        text = ensure_reference_tags(prompt, attached, style="picture")
        template.set(graph, "prompt", text)

        seconds = float(kwargs.get("duration") or self.params.get("duration", 5))
        clamped = max(H3_MIN_SECONDS, min(H3_MAX_SECONDS, seconds))
        if clamped != seconds:
            logger.warning("[ComfyUI] duration %.1fs outside trained range, clamped to %.1fs",
                           seconds, clamped)
        template.set(graph, "seconds", clamped)

        template.set(graph, "megapixels",
                     self._megapixels(kwargs.get("resolution"), kwargs.get("megapixels")))
        template.set(graph, "aspect_ratio", self._aspect(kwargs.get("aspect_ratio")))
        template.set(graph, "filename_prefix", kwargs.get("filename_prefix", "video/lumenx"))

        seed = kwargs.get("seed")
        template.set(graph, "seed", int(seed) if seed else uuid.uuid4().int % (2 ** 31))

        # "max" reads refs at 2048px short edge for best identity retention;
        # "match" scales them to the output budget and samples faster.
        template.set(graph, "ref_image_size", kwargs.get("ref_image_size", "max"))

        path, elapsed, extras = self._run(graph, output_path, want="video")
        self.last_extra_outputs = extras
        return path, elapsed
