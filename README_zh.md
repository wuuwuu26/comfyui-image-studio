# ComfyUI Image Studio

[English](README.md) | 简体中文

一个把 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 生图能力包装成**极简中文 Web 界面**的本地部署方案，面向 RTX 4070 Laptop 8GB 这类消费级显卡。

手机连同一 Wi-Fi 即可访问；所有引导文字、按钮、滑杆刻度都与后端实际执行的参数一致——界面怎么写，后端就怎么跑，不偷改参数。

## 功能

- **三个生图引擎**，界面上一键切换：
  | 引擎 | 定位 | 实测速度 (8GB) |
  |---|---|---|
  | Krea 2 Turbo | 艺术旗舰，1920×1080 | ~50 s/张 |
  | Z-Image Turbo | 极速，1280×720，10 步 | 8~15 s/张 |
  | MOSS-Image | 文字排版/中文渲染最强，1280×720 | ~30 s/张 |

最高分辨率 3840×2160（Krea 2）。各引擎分辨率见下方表格。

- 文生图 / 图生图重塑（人物一致性 LoRA 仅 Krea 2 生效）
- 四档分辨率 + 异步轮询 + 画廊
- ESRGAN 4× 超分（4x-UltraSharp）
- 端口被占用自动顺延，页面显示的手机地址始终跟随真实端口

## 快速开始

### 前置条件

1. 一台带 NVIDIA 显卡的 Windows 机器（实测 RTX 4070 Laptop 8GB）
2. Python 3.12 + CUDA 版 PyTorch
3. 一个装好模型的 ComfyUI（见下）

### 安装

```bash
git clone https://github.com/wuuwuu26/comfyui-image-studio.git
cd comfyui-image-studio
pip install fastapi uvicorn python-multipart requests pydantic
```

### 配置

全部通过环境变量，不必改代码：

| 变量 | 说明 | 默认值 |
|---|---|---|
| `EASYAI_ROOT` | 项目根目录 | 脚本所在目录 |
| `COMFY_DIR` | ComfyUI 安装目录 | 项目上级目录的 `ComfyUI` |
| `EASYAI_PORT` | Web 服务端口 | `7860`（占用则顺延 7861…） |

示例：

```powershell
$env:COMFY_DIR = "D:\ComfyUI"
$env:EASYAI_PORT = 7860
python server.py
```

浏览器打开 `http://127.0.0.1:7860`。

### 模型布局

`server.py` 启动时会检查以下权重是否存在且体积达标（不达标界面上会标注引擎未就绪）：

```
ComfyUI/
├── models/
│   ├── diffusion_models/
│   │   ├── krea2_turbo_int8_convrot.safetensors   # Krea 2
│   │   ├── z_image_turbo_int8_convrot.safetensors # Z-Image
│   │   └── moss_image_sft_int8_convrot.safetensors# MOSS
│   ├── text_encoders/
│   │   ├── qwen3vl_4b_int8_convrot.safetensors
│   │   ├── qwen_3_4b_fp8_mixed.safetensors
│   │   └── qwen3.5_9b_w4a8_gptq.safetensors
│   ├── vae/
│   │   ├── qwen_image_vae.safetensors
│   │   ├── z_image_ae.safetensors
│   │   └── flux2-vae_small.safetensors
│   ├── loras/                                     # 人物一致性 LoRA 放这里
│   └── upscale_models/4x-UltraSharp.pth
```

## 调参记录（8GB 卡实测）

这些是踩坑后定下来的值，写死在后端以保证界面与执行一致：

- **Krea 2**：1920×1080，双阶段（潜空间 + 6 步精修），约 50 s/张，显存靠 `--lowvram` + 分块
- **Z-Image**：1280×720，10 步，CFG 约 2.0，8~15 s/张——**默认首选**
- **MOSS**：1280×720，中文文字渲染最强，约 30 s/张
- **人物一致性**：只在 Krea 2 图生图模式生效，识别 `loras/` 里文件名含 `face/id/identity/character/人物/人脸/一致性` 等的 LoRA
- **`ai_4k`（最高档）**：Krea 2 在该档输出 3840×2160，潜空间直接去噪、不接放大器，8GB 卡约 6~8 分钟/张；各引擎实际尺寸见下表
- **不要用 GGUF / NVMe 换页**：8GB 卡上重模型反复换页会让 GPU 占用 100% 但功耗掉到 40W，直接卡死

### 各引擎实际输出分辨率（16:9）

| 档位 | Krea 2 | Z-Image | MOSS |
|---|---|---|---|
| `fast_1k` | 1920×1080 | 1280×720 | 1280×720 |
| `native_2k` | 2560×1440 | 1920×1080 | 1920×1080 |
| `ai_4k` | 3840×2160 | 2560×1440 | 2560×1440 |

> 本机最高分辨率 3840×2160（Krea 2 引擎），8GB 卡约 6~8 分钟/张。
> Z-Image 8GB 显存上限约 3584×2016，默认限制在 2560×1440 作为速度与质量的平衡点。
> 9:16 与 1:1 按比例换算（如 2160×3840、3840×3840）。

ComfyUI 推荐启动参数（代码里已内置）：

```
--cuda-malloc --lowvram --reserve-vram 1.0
--disable-pinned-memory --disable-fast-disk --disable-auto-launch
```

## 技术栈

FastAPI + 原生 JS 前端 + ComfyUI HTTP API（`/prompt` 提交 + `/history` 轮询）。无前端框架、无构建步骤，单文件 HTML 直接静态托管。

## 许可

MIT
