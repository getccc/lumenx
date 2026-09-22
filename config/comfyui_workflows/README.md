# ComfyUI Workflow 模板

`src/models/comfyui.py` 不在代码里拼工作流，而是从这个目录读取 **API 格式**的
ComfyUI 图 + 一份 `bindings` 映射表。接入一个新的自托管模型因此是「放一个 JSON」，
不是改代码。

## 文件格式

```jsonc
{
  "bindings": {                                   // LumenX 参数 -> 节点字段
    "prompt": {"node": "144", "field": "value"}
  },
  "ref_images": {                                 // 动态参考图槽位（可选）
    "node": "148", "field": "ref_images",
    "prefix": "ref_image_", "start": 0, "max": 9
  },
  "first_frame": {"node": "104", "field": "first_frame"},  // I2V 首帧（可选）
  "reserved_node_ids": ["201", "202"],            // 留给 LoadImage 注入的 id
  "graph": { /* ComfyUI API 格式的图 */ }
}
```

`bindings` 里没声明的键会被安静跳过——模板本来就不必暴露每一个旋钮。

图像模板认这些键：`prompt` `negative_prompt` `reference_resolution` `width`
`height` `batch_size` `seed` `steps` `cfg` `filename_prefix`。
视频模板认：`prompt` `seed` `seconds` `megapixels` `aspect_ratio`
`ref_image_size` `filename_prefix` `mode_switch`。

## 怎么拿到 API 格式的图

网页里「保存工作流」存的是**画布格式**（带 `nodes`/`links`/坐标），`/prompt`
不认。三种拿法，从稳到不稳：

1. **从执行历史里捞**（推荐）——先在网页里把图跑通一次，然后
   `GET /history`，取 `prompt[2]`。这拿到的是服务端真正接受过的图，
   参数组合也是验证过的。
2. 网页菜单 **工作流 → 导出 (API)**。
3. 浏览器控制台 `(await app.graphToPrompt()).output`。

用了 subgraph（节点 type 是一串 UUID）的工作流**不能**直接走 2/3，必须用 1。

## 两个模型都靠提示词里的位置标签绑定参考图

这是最容易踩的坑：参考图挂上了、服务端也收到了，但提示词里没有标签，
模型就当纯 T2I 处理，**静默忽略参考图**。

| 模型 | 提示词里的写法 | 图里的槽位名 |
|---|---|---|
| MiniMax-H3 | `<Picture 1>` `<Picture 2>` | `ref_image_0` `ref_image_1` |
| Qwen-Image 2.1 | `<image1>` `<image2>` | `image_1` `image_2` |

提示词标签**都是 1-based**，槽位起始编号不同（由模板的 `start` 声明）。
adapter 会把 LumenX 的 `[characterN:名字]` 自动改写成对应写法，提示词里一个
标签都没有时还会补一句身份保持的说明。

## 实测记录（2026-09-22）

**Qwen-Image 2.1** — RTX 4090 Laptop 16GB，int8：

| 任务 | 尺寸 | 耗时 |
|---|---|---|
| T2I | 768×1152 | ~16s |
| T2I 三视图设定图 | 1536×864 | ~18s |
| I2I（带参考图） | 832×1216 | ~17s |

`reference_resolution`（节点里的 `resolution`）传 **0** 表示参考图保留原始尺寸。
默认值 1024 会把参考图压缩，实测身份保真明显下降，所以 adapter 默认传 0。

**MiniMax-H3 Ref2VA** — Apple Silicon 48GB，Parasyte Turbo 8 步，3 秒片段：

| 画布 | 输出 | 耗时 | 身份保真 |
|---|---|---|---|
| 0.2 MP | 352×608 | 233s | 发型/服装色调跟得住，五官和服装款式会漂 |
| 0.4 MP | 480×864 | 432s | 明显更贴参考图（服装款式、脸型都对上了）|

参考图数量不是主要变量：同参数下把三视图设定图 + 面部特写换成单张面部特写，
输出几乎没有差别。**分辨率才是**。做正片建议至少 0.4MP。

`minimax_h3_video.json` 里节点 `141`/`119`/`9`/`17` 是 Parasyte Turbo 验证过的
配方组合（LoRA 1.5 / shift 12·3 / beta57 / er_sde / 8 步），动它会破坏出片质量。

H3 是音视频联合生成，除了 mp4 还会吐一条独立音轨，adapter 把它存在 mp4 旁边并
通过 `ComfyUIVideoModel.last_extra_outputs` 暴露出来——配音阶段可以直接用，
不需要再 ffmpeg 解复用。
