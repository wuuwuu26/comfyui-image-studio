# ComfyUI Image Studio

English | [简体中文](README_zh.md)

A local deployment that wraps [ComfyUI](https://github.com/comfyanonymous/ComfyUI) image generation into a **minimal Chinese-language web UI**, tuned for consumer GPUs such as the RTX 4070 Laptop 8GB.

Open it from your phone on the same Wi-Fi. Every hint, button, and slider label matches what the backend actually executes — the UI never silently changes parameters.

## Features

- **Three generation engines**, switchable in one click:

  | Engine | Role | Measured speed (8GB) |
  |---|---|---|
  | Krea 2 Turbo | Artistic flagship, 1920×1080 | ~50 s/image |
  | Z-Image Turbo | Fast, 1280×720, 10 steps | 8–15 s/image |
  | MOSS-Image | Best for typography / Chinese text, 1280×720 | ~30 s/image |

  Maximum resolution: 3840×2160 (Krea 2). See the resolution table below for per-engine sizes.

- Text-to-image / image-to-image restyling (identity LoRA only with Krea 2)
- Four resolution tiers + async polling + gallery
- ESRGAN 4× upscaling (4x-UltraSharp)
- Automatic port fallback; the phone URL shown on the page always follows the real port

## Quick start

### Prerequisites

1. A Windows machine with an NVIDIA GPU (tested on RTX 4070 Laptop 8GB)
2. Python 3.12 with CUDA build of PyTorch
3. A ComfyUI installation with the models below

### Install

```bash
git clone https://github.com/wuuwuu26/comfyui-image-studio.git
cd comfyui-image-studio
pip install fastapi uvicorn python-multipart requests pydantic
```

### Configuration

Everything is configured via environment variables — no code edits:

| Variable | Meaning | Default |
|---|---|---|
| `EASYAI_ROOT` | Project root | directory of the script |
| `COMFY_DIR` | ComfyUI install directory | `ComfyUI` next to the project |
| `EASYAI_PORT` | Web server port | `7860` (falls back to 7861, 7862, …) |

Example:

```powershell
$env:COMFY_DIR = "D:\ComfyUI"
$env:EASYAI_PORT = 7860
python server.py
```

Open `http://127.0.0.1:7860` in your browser.

### Model layout

On startup, `server.py` checks that the following weights exist and meet a minimum size (otherwise the UI marks that engine as not ready):

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
│   ├── loras/                                     # identity LoRAs go here
│   └── upscale_models/4x-UltraSharp.pth
```

## Tuning notes (measured on an 8GB card)

These values were settled after trial and error and are fixed in the backend so the UI and execution always agree:

- **Krea 2**: 1920×1080, two-stage (latent + 6-step refinement), ~50 s/image, VRAM kept in check with `--lowvram` + tiling
- **Z-Image**: 1280×720, 10 steps, CFG ~2.0, 8–15 s/image — **default choice**
- **MOSS**: 1280×720, best Chinese text rendering, ~30 s/image
- **Identity consistency**: only active in Krea 2 image-edit mode; picks LoRAs from `loras/` whose filenames contain `face/id/identity/character`, etc.
- **`ai_4k` (top tier)**: Krea 2 produces 3840×2160 at this tier, direct latent denoising with no upscaler, ~6–8 min/image on 8GB; per-engine sizes below
- **Avoid GGUF / NVMe paging**: on an 8GB card, repeatedly paging a large model leaves the GPU at 100% but ~40 W power draw, which stalls completely

### Actual output resolution per engine (16:9)

| Tier | Krea 2 | Z-Image | MOSS |
|---|---|---|---|
| `fast_1k` | 1920×1080 | 1280×720 | 1280×720 |
| `native_2k` | 2560×1440 | 1920×1080 | 1920×1080 |
| `ai_4k` | 3840×2160 | 2560×1440 | 2560×1440 |

> Maximum on this machine: 3840×2160 (Krea 2), ~6–8 min/image on 8GB.
> Z-Image tops out around 3584×2016 on 8GB VRAM and is capped at 2560×1440 by default as a speed/quality balance.
> 9:16 and 1:1 scale accordingly (e.g. 2160×3840, 3840×3840).

Recommended ComfyUI flags (already built into the code):

```
--cuda-malloc --lowvram --reserve-vram 1.0
--disable-pinned-memory --disable-fast-disk --disable-auto-launch
```

## Stack

FastAPI + vanilla JS frontend + ComfyUI HTTP API (`/prompt` to submit, `/history` to poll). No frontend framework, no build step — a single HTML file served statically.

## License

The MIT license in [LICENSE](LICENSE) applies **only to the original code in this
repository** (server, frontend, scripts, docs) — not to the models.

This repository does not ship any weights; users supply their own. Refer to
[NOTICE](NOTICE) for third-party model licenses. In short: Z-Image and MOSS are
Apache-2.0, but **Krea 2 uses its own Community License** and **4x-UltraSharp is
CC BY-NC-SA 4.0 (non-commercial)**. Check the license of each checkpoint you use,
especially for commercial work.
