"""IndexTTS 2.5 provider adapter — self-hosted zero-shot voice cloning TTS.

Drop-in alternative to the DashScope CosyVoice/Qwen3-TTS engine in
``src/audio/tts.py``. ``synthesize()`` keeps that signature so callers in
``apps/comic_gen/audio.py`` need no changes.

Model differences worth knowing before wiring this up:

- **Voice identity is an audio file, not an ID.** CosyVoice picks a voice
  from a catalogue of string IDs; IndexTTS clones whatever reference clip
  you hand it. So ``voice`` here is either a path to a reference clip or a
  bare id resolved against ``voice_dir``.
  IndexTTS has its own server-side *preset* system, but it is write-only
  over the API: ``on_preset_save`` and ``refresh_preset_choices`` work
  while ``on_preset_load`` is a UI-only handler that errors for every
  input. LumenX therefore keeps its own copy of each reference clip and
  treats the preset purely as something that shows up in the WebUI.
- **Emotion is an 8-dim vector, not free text.** LumenX's natural-language
  ``instructions`` are mapped onto those dims by :func:`instructions_to_vector`.
- **No pitch or volume control.** Those arguments are accepted for
  signature compatibility and ignored with a log line, rather than being
  silently dropped.

The transport is plain HTTP against Gradio's REST API — deliberately not
``gradio_client``, to avoid adding a heavyweight dependency for three
endpoints.

Environment:
    INDEXTTS_HOST         host:port, e.g. 100.72.128.46:7860
    INDEXTTS_HOST_HEADER  Host header override; REQUIRED behind `tailscale
                          serve`, which routes by hostname — requests to the
                          raw IP get a plain-text 404 even when healthy.
    INDEXTTS_LANG         default language (ZH/EN/JA/AR/ES), default ZH
"""

import json
import logging
import os
import shutil
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1:7860"
SUPPORTED_LANGS = ("ZH", "EN", "JA", "AR", "ES")

EMO_SAME_AS_VOICE = "与音色参考音频相同"
EMO_FROM_AUDIO = "使用情感参考音频"
EMO_FROM_VECTOR = "使用情感向量控制"

# /gen_single's 8 emotion dims, in positional order.
EMOTION_DIMS = ("喜", "怒", "哀", "惧", "厌恶", "低落", "惊喜", "平静")

# Maps LumenX's free-text `instructions` onto those dims. Deliberately
# coarse: IndexTTS normalizes the vector server-side and the docs advise
# emphasizing only one or two dims, so a keyword hit sets one strong value
# rather than painting across all eight.
EMOTION_KEYWORDS: Dict[str, Tuple[int, float]] = {
    "开心": (0, 0.8), "高兴": (0, 0.8), "喜悦": (0, 0.8), "愉快": (0, 0.6),
    "happy": (0, 0.8), "cheerful": (0, 0.7), "joyful": (0, 0.8),
    "生气": (1, 0.8), "愤怒": (1, 0.9), "恼火": (1, 0.6), "暴躁": (1, 0.7),
    "angry": (1, 0.8), "furious": (1, 0.9), "irritated": (1, 0.6),
    "悲伤": (2, 0.8), "难过": (2, 0.8), "哀伤": (2, 0.8), "哭腔": (2, 0.9),
    "sad": (2, 0.8), "sorrowful": (2, 0.8), "tearful": (2, 0.9),
    "害怕": (3, 0.8), "恐惧": (3, 0.9), "惊恐": (3, 0.8), "紧张": (3, 0.6),
    "afraid": (3, 0.8), "scared": (3, 0.8), "fearful": (3, 0.8),
    "厌恶": (4, 0.8), "嫌弃": (4, 0.7), "反感": (4, 0.7),
    "disgusted": (4, 0.8),
    "低落": (5, 0.8), "沮丧": (5, 0.8), "失落": (5, 0.7), "疲惫": (5, 0.6),
    "depressed": (5, 0.8), "weary": (5, 0.6), "dejected": (5, 0.8),
    "惊讶": (6, 0.8), "惊喜": (6, 0.8), "震惊": (6, 0.9),
    "surprised": (6, 0.8), "shocked": (6, 0.9), "astonished": (6, 0.8),
    "平静": (7, 0.8), "冷静": (7, 0.8), "淡然": (7, 0.7), "温柔": (7, 0.6),
    "calm": (7, 0.8), "neutral": (7, 0.7), "gentle": (7, 0.6),
}

# duration_factor is a *slowness* multiplier, the inverse of LumenX's
# speech_rate, and IndexTTS only accepts 0.5-2.0.
DURATION_FACTOR_MIN = 0.5
DURATION_FACTOR_MAX = 2.0


class IndexTTSError(RuntimeError):
    """An IndexTTS call failed."""


def instructions_to_vector(instructions: Optional[str]) -> Optional[List[float]]:
    """Map a natural-language emotion instruction onto IndexTTS's 8 dims.

    Returns ``None`` when nothing matches, which tells the caller to stay
    in "same as voice reference" mode — the most stable and cheapest path,
    and a better default than forcing a half-guessed vector.

    For higher-fidelity mapping, have the LLM emit the vector directly;
    ``ScriptProcessor`` already does this kind of parameter translation in
    ``translate_character_to_voice_prompt``.
    """
    if not instructions or not instructions.strip():
        return None
    text = instructions.lower()
    vector = [0.0] * 8
    for keyword, (index, weight) in EMOTION_KEYWORDS.items():
        if keyword.lower() in text:
            vector[index] = max(vector[index], weight)
    if not any(vector):
        logger.info("[IndexTTS] no emotion keyword matched %r; using voice-reference emotion",
                    instructions[:60])
        return None
    hits = [f"{EMOTION_DIMS[i]}={v:.2f}" for i, v in enumerate(vector) if v]
    logger.info("[IndexTTS] instructions -> emotion vector (%s)", ", ".join(hits))
    return vector


class IndexTTSClient:
    """Gradio REST client for IndexTTS (upload -> call -> poll SSE -> fetch)."""

    def __init__(self, host: Optional[str] = None, host_header: Optional[str] = None,
                 timeout: int = 600):
        self.host = (host or os.getenv("INDEXTTS_HOST", "").strip() or DEFAULT_HOST).rstrip("/")
        self.base = f"http://{self.host}"
        self.timeout = timeout
        self.headers: Dict[str, str] = {}
        header = host_header or os.getenv("INDEXTTS_HOST_HEADER", "").strip()
        if header:
            self.headers["Host"] = header
        self.session = requests.Session()
        self.session.trust_env = False

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base}/gradio_api/info",
                                 headers=self.headers, timeout=15)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def upload(self, path: str) -> str:
        """Upload a local file to Gradio's temp store; returns the server path."""
        if not os.path.exists(path):
            raise IndexTTSError(f"reference audio not found: {path}")
        with open(path, "rb") as fh:
            r = self.session.post(
                f"{self.base}/gradio_api/upload",
                params={"upload_id": uuid.uuid4().hex},
                files={"files": (os.path.basename(path), fh, "application/octet-stream")},
                headers=self.headers, timeout=self.timeout,
            )
        r.raise_for_status()
        return r.json()[0]

    @staticmethod
    def file_ref(server_path: Optional[str]) -> Optional[Dict[str, Any]]:
        if not server_path:
            return None
        return {"path": server_path, "meta": {"_type": "gradio.FileData"}}

    def call(self, endpoint: str, data: Sequence[Any]) -> Any:
        """POST to a named endpoint, then read its SSE result stream."""
        r = self.session.post(f"{self.base}/gradio_api/call/{endpoint}",
                              json={"data": list(data)},
                              headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        event_id = r.json().get("event_id")
        if not event_id:
            raise IndexTTSError(f"{endpoint}: no event_id in {r.text[:400]}")

        stream = self.session.get(f"{self.base}/gradio_api/call/{endpoint}/{event_id}",
                                  headers=self.headers, timeout=self.timeout)
        stream.raise_for_status()
        result, error = None, None
        for block in stream.text.split("\n\n"):
            event = payload = None
            for line in block.splitlines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    payload = line[5:].strip()
            if event == "complete" and payload:
                result = json.loads(payload)
            elif event == "error":
                error = payload
        if result is None:
            raise IndexTTSError(f"{endpoint} returned no result (error={error})")
        return result

    def fetch(self, server_path: str, dest_path: str) -> str:
        """Download a server-side file. The path is sent raw — it is a
        Windows path in the reference deployment and must not be escaped."""
        r = self.session.get(f"{self.base}/gradio_api/file={server_path}",
                             headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        with open(dest_path, "wb") as fh:
            fh.write(r.content)
        return dest_path

    # --- presets: IndexTTS's stand-in for a voice ID -------------------

    def list_presets(self) -> List[str]:
        result = self.call("refresh_preset_choices", [])
        names: List[str] = []
        for update in result if isinstance(result, list) else []:
            for choice in (update or {}).get("choices", []) or []:
                label = choice[0] if isinstance(choice, (list, tuple)) else choice
                if label:
                    names.append(label)
        return sorted(set(names))

    def save_preset(self, name: str, reference_audio_path: str,
                    emotion_vector: Optional[Sequence[float]] = None) -> List[str]:
        """Persist a voice preset — the server-side anchor for a LumenX voice_id."""
        server_ref = self.upload(reference_audio_path)
        vector = list(emotion_vector) if emotion_vector else [0.0] * 8
        method = EMO_FROM_VECTOR if emotion_vector else EMO_SAME_AS_VOICE
        self.call("on_preset_save", [
            name, self.file_ref(server_ref), method, None, 0.65, *vector,
            "", False, True, 0.8, 30, 0.8, 0.0, 3, 10.0, 1500, 120,
        ])
        return self.list_presets()

    # NOTE: the sibling `on_preset_load` endpoint is deliberately not wrapped.
    # It is a UI-only Gradio event handler: `/gradio_api/info` declares its
    # argument as `Literal['']`, and calling it over REST returns
    # `event: error` for every value, including a valid preset name. So a
    # preset cannot be read back through the API — only written and listed.
    # LumenX therefore keeps its own copy of each voice's reference audio
    # (see IndexTTSEngine._server_reference) rather than treating the
    # server-side preset as the source of truth.


class IndexTTSEngine:
    """LumenX-facing TTS engine backed by IndexTTS 2.5.

    Mirrors ``src.audio.tts.TTSEngine.synthesize`` so ``AudioGenerator`` can
    swap engines without touching call sites.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.client = IndexTTSClient(self.config.get("host"), self.config.get("host_header"))
        self.default_lang = (os.getenv("INDEXTTS_LANG", "ZH").strip().upper() or "ZH")
        self.voice = self.config.get("voice")
        # Where LumenX keeps each voice's reference clip. This is the source
        # of truth for voice identity, because IndexTTS presets can be
        # written over the API but not read back.
        self.voice_dir = self.config.get("voice_dir") or os.path.join("output", "audio", "voices")
        # Reference clips are uploaded once per path and reused — re-uploading
        # the same character voice for every line is pure latency.
        self._upload_cache: Dict[str, str] = {}

    @property
    def is_configured(self) -> bool:
        return bool(os.getenv("INDEXTTS_HOST", "").strip() or self.config.get("host"))

    def _candidate_paths(self, voice: str) -> List[str]:
        """Local files a ``voice_id`` may refer to, in resolution order."""
        candidates = [voice, os.path.join("output", voice)]
        # A bare voice_id (no extension) means "the clip LumenX stored for
        # this voice" — the server-side preset cannot be read back, so our
        # own copy is the source of truth.
        if not os.path.splitext(voice)[1]:
            for ext in (".wav", ".flac", ".mp3", ".m4a"):
                candidates.append(os.path.join(self.voice_dir, f"{voice}{ext}"))
        return candidates

    def _server_reference(self, voice: str) -> str:
        """Resolve a LumenX ``voice_id`` to a server-side reference audio path.

        Accepts an absolute path, a path relative to ``output/``, or a bare
        voice id stored under ``voice_dir``. Uploads are cached per id —
        re-sending the same clip for every line of dialogue is pure latency.
        """
        if voice in self._upload_cache:
            return self._upload_cache[voice]

        for candidate in self._candidate_paths(voice):
            if candidate and os.path.exists(candidate):
                server_path = self.client.upload(candidate)
                self._upload_cache[voice] = server_path
                return server_path

        raise IndexTTSError(
            f"voice {voice!r} did not resolve to a reference audio file. "
            f"IndexTTS clones a voice from a clip, so voice_id must point at one "
            f"(looked in: {', '.join(self._candidate_paths(voice))}). "
            f"Register one with clone_voice()."
        )

    def list_voices(self) -> List[Dict[str, str]]:
        """Presets, shaped like the DashScope voice registry entries."""
        return [{"id": name, "name": name, "family": "indextts"}
                for name in self.client.list_presets()]

    def clone_voice(self, name: str, reference_audio_path: str) -> str:
        """Register a new voice from a reference clip (LumenX 声音复刻).

        Keeps a local copy under ``voice_dir`` — that copy is what later
        ``synthesize(voice=name)`` calls resolve against. The server-side
        preset is written too, but only so the clip shows up in the
        IndexTTS WebUI; it cannot be read back over the API.
        """
        if not os.path.exists(reference_audio_path):
            raise IndexTTSError(f"reference audio not found: {reference_audio_path}")

        ext = os.path.splitext(reference_audio_path)[1] or ".wav"
        os.makedirs(self.voice_dir, exist_ok=True)
        local_copy = os.path.join(self.voice_dir, f"{name}{ext}")
        if os.path.abspath(local_copy) != os.path.abspath(reference_audio_path):
            shutil.copyfile(reference_audio_path, local_copy)

        try:
            self.client.save_preset(name, local_copy)
        except Exception as exc:  # preset is a convenience, not a requirement
            logger.warning("[IndexTTS] preset %r not saved server-side (%s); "
                           "the local reference clip still works", name, exc)
        self._upload_cache.pop(name, None)
        logger.info("[IndexTTS] voice %r registered -> %s", name, local_copy)
        return name

    def synthesize(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        speech_rate: float = 1.0,
        pitch_rate: float = 1.0,
        volume: int = 50,
        instructions: Optional[str] = None,
        model_override: Optional[str] = None,
        family_override: Optional[str] = None,
        lang: Optional[str] = None,
        emotion_vector: Optional[Sequence[float]] = None,
        max_text_tokens_per_segment: int = 120,
    ) -> Tuple[str, float, str]:
        """Synthesize ``text`` to ``output_path``.

        Returns ``(output_path, elapsed_ms, request_id)`` to match the
        DashScope engine's contract. ``elapsed_ms`` is wall-clock rather
        than first-package latency — IndexTTS is non-streaming.
        """
        if not text or not text.strip():
            raise IndexTTSError("empty text")

        voice = voice or self.voice
        if not voice:
            raise IndexTTSError("no voice: pass a preset name or a reference audio path")

        if pitch_rate != 1.0 or volume != 50:
            logger.info("[IndexTTS] pitch_rate/volume are not supported by this engine; "
                        "ignoring (pitch=%.2f volume=%d)", pitch_rate, volume)
        if model_override or family_override:
            logger.debug("[IndexTTS] model/family overrides do not apply; ignoring")

        language = (lang or self.default_lang).upper()
        if language not in SUPPORTED_LANGS:
            logger.warning("[IndexTTS] unsupported lang %r, falling back to ZH", language)
            language = "ZH"

        vector = list(emotion_vector) if emotion_vector else instructions_to_vector(instructions)
        method = EMO_FROM_VECTOR if vector else EMO_SAME_AS_VOICE
        vector = vector or [0.0] * 8

        # speech_rate is a speed multiplier; duration_factor is its inverse.
        duration_factor = 1.0 / speech_rate if speech_rate else 1.0
        duration_factor = max(DURATION_FACTOR_MIN, min(DURATION_FACTOR_MAX, duration_factor))

        server_ref = self._server_reference(voice)
        started = time.time()
        result = self.client.call("gen_single", [
            method,
            self.client.file_ref(server_ref),
            text,
            language,
            None,                      # emo_ref_path (mode 1 only)
            0.65,                      # emo_weight
            *[float(v) for v in vector],
            "",                        # emo_text — needs QwenEmotion, often unloaded
            False,                     # emo_random
            float(max_text_tokens_per_segment),
            float(duration_factor),
            True, 0.8, 30, 0.8, 0.0, 3, 10.0, 1500,   # do_sample..max_mel_tokens
        ])
        elapsed_ms = (time.time() - started) * 1000.0

        entry = result[0] if isinstance(result, list) and result else result
        if isinstance(entry, dict) and entry.get("__type__") == "update":
            entry = entry.get("value")
        server_path = entry.get("path") if isinstance(entry, dict) else entry
        if not server_path:
            raise IndexTTSError(f"no audio in response: {json.dumps(result, ensure_ascii=False)[:400]}")

        self.client.fetch(server_path, output_path)
        # The server path is a Windows path in the reference deployment, so
        # os.path.basename() on POSIX would not split it. Normalize first.
        leaf = str(server_path).replace("\\", "/").rsplit("/", 1)[-1]
        request_id = os.path.splitext(leaf)[0]
        logger.info("[IndexTTS] synthesized %d chars in %.0fms -> %s",
                    len(text), elapsed_ms, output_path)
        return output_path, elapsed_ms, request_id
