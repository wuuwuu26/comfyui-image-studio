import os
import sys
import time
import json
import uuid
import shutil
import subprocess
import threading
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="EasyAI Studio Pro Ultra")

# ===== 可配置路径：改环境变量即可换机器，不必动代码 =====
# EASYAI_ROOT  : 本项目根目录（默认与本文件同级）
# COMFY_DIR    : ComfyUI 安装目录
# EASYAI_PORT  : Web 服务端口（默认 7860）
BASE_DIR = os.environ.get("EASYAI_ROOT", os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")
COMFY_DIR = os.environ.get("COMFY_DIR", os.path.join(os.path.dirname(BASE_DIR), "ComfyUI"))
COMFY_INPUT = os.path.join(COMFY_DIR, "input")
COMFY_URL = "http://127.0.0.1:8188"

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(COMFY_INPUT, exist_ok=True)

# 神经超分组件
UPSCALE_MODEL = os.path.join(COMFY_DIR, "models", "upscale_models", "4x-UltraSharp.pth")

# Krea 2 Turbo 艺术生图组件 (生图唯一旗舰)
KREA2_UNET = os.path.join(COMFY_DIR, "models", "diffusion_models", "krea2_turbo_int8_convrot.safetensors")
KREA2_VAE = os.path.join(COMFY_DIR, "models", "vae", "qwen_image_vae.safetensors")
KREA2_QWEN = os.path.join(COMFY_DIR, "models", "text_encoders", "qwen3vl_4b_int8_convrot.safetensors")

# Z-Image-Turbo (通义造相 6B，8 步极速出图，中文文字渲染强)
ZIMG_UNET = os.path.join(COMFY_DIR, "models", "diffusion_models", "z_image_turbo_int8_convrot.safetensors")
ZIMG_TE = os.path.join(COMFY_DIR, "models", "text_encoders", "qwen_3_4b_fp8_mixed.safetensors")
ZIMG_VAE = os.path.join(COMFY_DIR, "models", "vae", "z_image_ae.safetensors")

# MOSS-Image (OpenMOSS 5B DiT，文字渲染/排版最强，走 ComfyUI 原生节点)
# 权重来自 Comfy-Org/MOSS-Image：SFT 版 + Qwen3.5-9B 文本编码器 + FLUX.2 小 VAE
MOSS_UNET = os.path.join(COMFY_DIR, "models", "diffusion_models", "moss_image_sft_int8_convrot.safetensors")
MOSS_TE = os.path.join(COMFY_DIR, "models", "text_encoders", "qwen3.5_9b_w4a8_gptq.safetensors")
MOSS_VAE = os.path.join(COMFY_DIR, "models", "vae", "flux2-vae_small.safetensors")

def is_krea2_ready():
    return (
        os.path.exists(KREA2_UNET) and os.path.getsize(KREA2_UNET) >= 13492686496 and
        os.path.exists(KREA2_VAE) and os.path.getsize(KREA2_VAE) > 200000000 and
        os.path.exists(KREA2_QWEN)
    )

def is_zimage_ready():
    return (
        os.path.exists(ZIMG_UNET) and os.path.getsize(ZIMG_UNET) > 6000000000 and
        os.path.exists(ZIMG_TE) and os.path.getsize(ZIMG_TE) > 5000000000 and
        os.path.exists(ZIMG_VAE) and os.path.getsize(ZIMG_VAE) > 300000000
    )

def is_moss_ready():
    return (
        os.path.exists(MOSS_UNET) and os.path.getsize(MOSS_UNET) > 5000000000 and
        os.path.exists(MOSS_TE) and os.path.getsize(MOSS_TE) > 6000000000 and
        os.path.exists(MOSS_VAE) and os.path.getsize(MOSS_VAE) > 200000000
    )

def is_upscale_ready():
    return os.path.exists(UPSCALE_MODEL) and os.path.getsize(UPSCALE_MODEL) > 50000000

# ===== 人物一致性 LoRA / 角色 LoRA =====
LORA_DIR = os.path.join(COMFY_DIR, "models", "loras")
# 文件名里出现这些关键词，视为「人物/人脸一致性」或「角色」LoRA
ID_LORA_KEYWORDS = (
    "face", "id", "identity", "character", "person", "portrait", "likeness",
    "人物", "人脸", "一致性", "角色", "肖像", "instantid", "influencer", "gal",
)

def detect_lora_engine(filename):
    """按文件名推测 LoRA 所属引擎：'krea2' / 'zimage' / None(通用或未知)。"""
    n = filename.lower()
    if "krea2" in n or "krea-2" in n or "krea_2" in n:
        return "krea2"
    if "zimage" in n or "z-image" in n or "z_image" in n or "-zit" in n or "_zit" in n or "zit" in n:
        return "zimage"
    if "moss" in n:
        return "moss"
    return None

def list_lora_files():
    """递归列出 loras 目录下所有可用 LoRA（相对路径，统一正斜杠）。"""
    items = []
    if not os.path.isdir(LORA_DIR):
        return items
    for root, _dirs, files in os.walk(LORA_DIR):
        for fn in files:
            if fn.lower().endswith((".safetensors", ".ckpt", ".pt")):
                if fn == "put_loras_here":
                    continue
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, LORA_DIR).replace("\\", "/")
                hay = rel.lower()  # 用整条相对路径判断，目录名里的关键词也算
                items.append({
                    "name": rel,
                    "size_mb": round(os.path.getsize(full) / (1024 * 1024), 1),
                    "is_id_lora": any(k in hay for k in ID_LORA_KEYWORDS),
                    "engine": detect_lora_engine(rel),
                })
    items.sort(key=lambda x: (not x["is_id_lora"], x["name"]))
    return items

def comfy_lora_name(rel_name: str) -> str:
    """ComfyUI 在 Windows 上按反斜杠匹配 lora_name，必须转换。"""
    return rel_name.replace("/", "\\")

def pick_id_lora(requested, engine=None):
    """选定要用的 LoRA：请求指定优先；否则在指定引擎的 LoRA 里挑第一个候选。找不到返回 None。"""
    files = list_lora_files()
    names = [f["name"] for f in files]
    if requested:
        norm = requested.replace("\\", "/")
        # 允许只传文件名
        for n in names:
            if n == norm or n.endswith("/" + norm) or os.path.basename(n) == norm:
                return n
        return None
    for f in files:
        if engine and f["engine"] not in (engine, None):
            continue
        if f["is_id_lora"]:
            return f["name"]
    return None

comfy_proc = None

def ensure_comfy_running():
    global comfy_proc
    try:
        r = requests.get(f"{COMFY_URL}/system_stats", timeout=1)
        if r.status_code == 200:
            return True
    except Exception:
        pass

    if comfy_proc is None or comfy_proc.poll() is not None:
        log_path = os.path.join(COMFY_DIR, "comfy_runtime.log")
        log_f = open(log_path, "a", encoding="utf-8")
        comfy_proc = subprocess.Popen(
            [
                os.path.join(COMFY_DIR, "venv", "Scripts", "python.exe"),
                os.path.join(COMFY_DIR, "main.py"),
                "--listen", "0.0.0.0",
                "--port", "8188",
                "--cuda-malloc",
                "--lowvram",
                "--reserve-vram", "1.0",
                "--disable-pinned-memory",
                "--disable-fast-disk",
                "--disable-auto-launch"
            ],
            cwd=COMFY_DIR,
            stdout=log_f,
            stderr=log_f
        )
    for _ in range(60):
        try:
            r = requests.get(f"{COMFY_URL}/system_stats", timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

class GenerateRequest(BaseModel):
    mode: str = "text2image"  # "text2image", "image_edit"
    engine: str = "krea2"  # "krea2"(艺术旗舰), "zimage"(造相极速), "moss"(文字排版旗舰)
    prompt: str
    negative_prompt: str = ""
    ratio: str = "16:9"
    resolution_mode: str = "fast_1k"  # "fast_1k"(原生1080p), "native_2k", "ai_4k"
    ref_image: Optional[str] = None
    denoise: float = 0.65
    custom_steps: Optional[int] = None

    # 二采精炼 (HiRes-Fix 二次采样)
    enable_second_pass: bool = True
    second_pass_steps: Optional[int] = None  # 二采步数：默认 6

    # 性能加速黑科技
    enable_teacache: bool = True

    # 人物一致性 LoRA（仅 Krea 2 图生图生效，需 comfyui-krea2edit 节点）
    enable_id_lora: bool = False
    id_lora_name: Optional[str] = None        # 空 = 自动挑选 loras 目录里的人物一致性 LoRA
    id_lora_strength: Optional[float] = None  # 0.0 ~ 1.5，空 = 1.0（官方推荐）
    id_lora_ref_boost: Optional[float] = None   # 0 ~ 10，越大越贴参考图，空 = 3.0
    id_lora_grounding_px: Optional[int] = None  # 接地分辨率，空 = 768

def align_prompt_semantics(text: str) -> str:
    t = text
    # 彻底杜绝名人歧义
    if any(k in t.lower() for k in ["科比", "kobe", "牢大"]):
        for k in ["科比·布莱恩特", "科比布莱恩特", "科比", "牢大", "kobe"]:
            t = t.replace(k, "African American NBA superstar Kobe Bryant, Black male with iconic shaved head, dark brown skin tone, recognizable Kobe Bryant face, wearing Lakers jersey")
    return t

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
        raise HTTPException(status_code=400, detail="只支持 PNG/JPG/WebP 图片")
    filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"
    dest = os.path.join(COMFY_INPUT, filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"success": True, "filename": filename, "preview_url": f"/input_files/{filename}"}

# ===== 提示词自动扩写（本地规则引擎，不联网、不占显存）=====
class ExpandRequest(BaseModel):
    prompt: str
    engine: str = "krea2"
    kind: str = ""   # 留空自动判断：product / portrait / landscape / anime / text / food / architecture

# 主体类型 → 补全字段。关键词命中越多越优先（关键词按“越长越具体”排前面）
EXPAND_SUBJECTS = [
    ("text", ["字", "海报", "标题", "logo", "文字", "字幕", "封面", "标语", "排版", "poster", "typography", "text"]),
    ("anime", ["动漫", "二次元", "漫画", "卡通", "赛璐璐", "anime", "manga", "cartoon", "chibi"]),
    ("portrait", ["人像", "肖像", "写真", "少女", "少年", "女孩", "男孩", "女生", "男生", "女人", "男人", "美女", "模特", "自拍", "人脸", "portrait", "selfie", "girl", "woman", "man", "face"]),
    ("food", ["美食", "料理", "蛋糕", "面包", "咖啡", "菜品", "水果", "冰淇淋", "汉堡", "拉面", "面条", "米饭", "早餐", "食物", "food", "coffee", "cake"]),
    ("architecture", ["建筑", "室内", "房间", "客厅", "卧室", "厨房", "装修", "楼宇", "house", "interior", "room", "architecture"]),
    ("landscape", ["风景", "山", "海", "森林", "湖", "日落", "日出", "天空", "城市", "夜景", "天际线", "草原", "沙漠", "雪山", "landscape", "mountain", "forest", "city", "sunset"]),
    ("animal", ["猫", "狗", "鸟", "动物", "兔子", "老虎", "狮子", "熊猫", "马", "鹿", "鱼", "cat", "dog", "animal", "bird"]),
    ("product", ["手机", "耳机", "相机", "手表", "汽车", "跑车", "产品", "香水", "瓶", "鞋子", "球鞋", "机械键盘", "鼠标", "笔记本", "product", "phone", "watch", "car"]),
]

# 每个主体类型的补全字段：构图 / 光线 / 材质细节 / 氛围 / 技术参数
EXPAND_FIELDS = {
    "product": {
        "compose": "居中三分法构图，主体占据画面黄金位置，留白干净",
        "light": "柔和的顶部主光配合侧向轮廓光，金属与玻璃表面呈现细腻高光反射",
        "detail": "表面纹理清晰，边缘倒角锐利，接缝工整，材质层次分明",
        "mood": "高级克制的商业氛围，纯净背景烘托主体",
        "tech": "超高清细节，专业产品摄影质感，锐利对焦，无噪点",
    },
    "portrait": {
        "compose": "人物半身构图，视线略微偏离镜头，背景适度虚化分离主体",
        "light": "大面积柔光箱主光，脸颊与发丝有温暖边缘光，眼神光自然",
        "detail": "皮肤保留真实纹理与毛孔，质感自然不磨皮，发丝根根分明",
        "mood": "安静而富有情绪的氛围，表情自然放松",
        "tech": "85mm 定焦镜头质感，浅景深，超高清细节，专业人像摄影",
    },
    "landscape": {
        "compose": "前景-中景-远景层次分明，视野开阔，地平线平稳",
        "light": "黄金时刻暖色阳光斜射，空气通透，光影反差自然",
        "detail": "植被、岩石、水面波纹细节丰富，远处雾气营造空气透视",
        "mood": "壮阔宁静的自然氛围，令人呼吸一滞的开阔感",
        "tech": "超广角镜头质感，超高清细节，专业风光摄影",
    },
    "anime": {
        "compose": "角色居中动态构图，姿态有张力，背景与角色呼应",
        "light": "明快通透的光影，高饱和但不刺眼的色彩",
        "detail": "干净的线稿，赛璐璐上色分层清晰，发丝与服饰细节精致",
        "mood": "清新治愈的氛围，画面富有故事感",
        "tech": "日式动画电影级质感，精致作画，超高清细节",
    },
    "text": {
        "compose": "版式居中对称，四周留白充足，主标题与副标题层级分明",
        "light": "均匀柔和的平光，文字与背景对比度高、边缘清晰",
        "detail": "字体笔画完整、字形准确、间距均匀，无多余字符",
        "mood": "极简高级的版式氛围，信息清晰易读",
        "tech": "印刷级排版质感，超高清细节，文字边缘锐利",
    },
    "food": {
        "compose": "45 度俯拍构图，主体居中，餐具与配菜环绕点缀",
        "light": "侧逆光勾勒食物轮廓，蒸汽与油脂高光生动",
        "detail": "食材纹理与色泽饱满，质感湿润真实，配菜新鲜",
        "mood": "温暖诱人的食欲氛围，热气腾腾的现场感",
        "tech": "微距镜头质感，超高清细节，专业美食摄影",
    },
    "architecture": {
        "compose": "广角透视构图，垂直线条笔直，视线引导至空间深处",
        "light": "自然天光与室内人工光源层次分明，窗边光柔和过渡",
        "detail": "材质接缝与纹理清晰，家具比例协调，空间尺度真实",
        "mood": "安静舒适的居住氛围，光影塑造空间层次",
        "tech": "超高清细节，建筑摄影质感，镜头无畸变",
    },
    "animal": {
        "compose": "动物占据画面中心，姿态自然生动，背景简洁衬托主体",
        "light": "柔和的自然光勾勒毛发轮廓，眼神光清澈明亮",
        "detail": "毛发根根分明，质地蓬松真实，眼睛湿润有神",
        "mood": "温暖治愈的动物氛围，神态灵动可爱",
        "tech": "长焦镜头质感，浅景深，超高清细节，专业动物摄影",
    },
    "poster": {
        "compose": "版式层次分明，主视觉占据画面重心，标题、副标题与正文形成清晰的视觉动线，四周留白得当",
        "light": "统一的主色调与高对比色彩搭配，主体元素与背景明暗区分明确，文字区域对比度高、便于阅读",
        "detail": "所有文字字形完整、笔画准确、无错字漏字，图标与装饰元素边缘干净，对齐严格",
        "mood": "整体调性统一、现代且高级，信息主次一目了然",
        "tech": "印刷级平面设计质感，文字边缘锐利清晰，超高清细节",
    },
    "general": {
        "compose": "主体居中且层次分明，画面有呼吸感的留白",
        "light": "自然柔和的光线塑造体积感，明暗过渡细腻",
        "detail": "主体细节清晰完整，材质与纹理真实可信",
        "mood": "氛围自然统一，画面富有感染力",
        "tech": "超高清细节，专业摄影质感，焦点锐利",
    },
}

EXPAND_SHORT_PROMPT_NOTE = "画面主体明确，主体细节完整，构图讲究，光影自然"

@app.post("/api/expand_prompt")
def expand_prompt(req: ExpandRequest):
    base = (req.prompt or "").strip()
    if not base:
        raise HTTPException(status_code=400, detail="提示词不能为空")

    # 已足够详细就不再堆砌，避免过度冗长反而干扰模型
    if len(base) >= 120:
        return {"success": True, "expanded": base, "skipped": True,
                "reason": "提示词已经足够详细（≥120 字），无需扩写"}

    kind = req.kind.strip().lower() if req.kind else ""
    if kind not in EXPAND_FIELDS:
        low = base.lower()
        kind = "general"
        best = 0
        for name, kws in EXPAND_SUBJECTS:
            hit = sum(1 for k in kws if k in low)
            if hit > best:
                best, kind = hit, name

    f = EXPAND_FIELDS[kind]
    zi = (req.engine == "zimage")
    # 中文逗号连接，Z-Image 中文渲染更稳；Krea 2 同样吃中文描述
    parts = []
    if len(base) < 30:
        parts.append(f"{base}，{EXPAND_SHORT_PROMPT_NOTE}")
    else:
        parts.append(base)
    parts.append(f["compose"])
    parts.append(f["light"])
    parts.append(f["detail"])
    parts.append(f["mood"])
    tech = f["tech"]
    if zi:
        tech = tech.replace("超高清细节", "高细节画面").replace("专业产品摄影质感", "真实产品摄影质感")
    parts.append(tech)

    # MOSS-Image 对「每个字块夹双引号」的训练数据最敏感，追加一句通用约束
    if req.engine == "moss":
        parts.append("画面中所有需要出现的文字都完整清晰、字形准确、可完整读出，无错字漏字")

    # 去重：避免用户已写的词重复
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return {"success": True, "expanded": "，".join(out), "kind": kind}

@app.get("/api/status")
def get_status():
    k2_ok = is_krea2_ready()
    zi_ok = is_zimage_ready()

    return {
        "engine_ready": True,
        "active_model": "Krea 2 Turbo (2026 艺术设计旗舰)",
        "is_krea2_ready": k2_ok,
        "is_zimage_ready": zi_ok,
        "is_moss_ready": is_moss_ready(),
        "upscale_ready": is_upscale_ready(),
        "id_lora_count": len(list_lora_files()),
    }

@app.get("/api/loras")
def get_loras():
    files = list_lora_files()
    return {
        "success": True,
        "lora_dir": LORA_DIR,
        "count": len(files),
        "id_lora_count": sum(1 for f in files if f["is_id_lora"]),
        "loras": files,
    }

last_generation_mode = None

@app.post("/api/generate")
def generate_image(req: GenerateRequest):
    global last_generation_mode
    if not ensure_comfy_running():
        raise HTTPException(status_code=500, detail="底层图像引擎未就绪")

    job_id = str(uuid.uuid4())[:8]

    # === Krea 2 Turbo 图像生成与图生图重塑 ===
    is_edit_mode = (req.mode == "image_edit") and (req.ref_image is not None) and os.path.exists(os.path.join(COMFY_INPUT, req.ref_image))

    # 准确计算潜空间尺寸
    # 1) fast_1k: 潜空间 1K，无放大器
    # 2) native_2k: 潜空间直接计算真 2K (2560 长边)，不接放大器！
    # 3) ai_4k: 潜空间直接计算真 4K (3840 长边, UHD 标准)，不接放大器！
    #    统一保证 16/32 整除：VAE 8x 下采样要求 8 的整数倍，二采/一采潜空间按 16 的倍数更稳
    if req.resolution_mode == "native_2k":
        if req.ratio == "16:9":
            base_w, base_h = 2560, 1440
        elif req.ratio == "9:16":
            base_w, base_h = 1440, 2560
        else:
            base_w, base_h = 2560, 2560
        target_w, target_h = base_w, base_h
    elif req.resolution_mode == "ai_4k":
        # 真 4K（UHD 3840×2160）：潜空间直接 3840 长边去噪，不再串接任何放大器
        if req.ratio == "16:9":
            base_w, base_h = 3840, 2160
        elif req.ratio == "9:16":
            base_w, base_h = 2160, 3840
        else:
            base_w, base_h = 3840, 3840
        target_w, target_h = base_w, base_h
    else: # fast_1k (标准原生 1080p 全高清)
        if req.ratio == "16:9":
            base_w, base_h = 1920, 1080
        elif req.ratio == "9:16":
            base_w, base_h = 1080, 1920
        else:
            base_w, base_h = 1080, 1080
        target_w, target_h = base_w, base_h

    # 风格预设模块已移除：提示词原样送入，不再自动追加任何风格词
    full_prompt = align_prompt_semantics(req.prompt)

    # ===== 引擎 A: Z-Image-Turbo（造相 6B · 8 步极速 · 仅文生图）=====
    if req.engine == "zimage":
        if not is_zimage_ready():
            raise HTTPException(status_code=500, detail="Z-Image-Turbo 引擎未就绪")
        if is_edit_mode:
            raise HTTPException(status_code=400, detail="Z-Image 引擎暂不支持图生图，请切换到 Krea 2")

        # Z-Image 三档分辨率（实测 8GB 显存上限 3584×2016，2560×1440 为性价比拐点）
        zi_sizes = {
            "fast_1k": {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (1024, 1024)},
            "native_2k": {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1536, 1536)},
            "ai_4k": {"16:9": (2560, 1440), "9:16": (1440, 2560), "1:1": (2048, 2048)},
        }
        zi_tier = zi_sizes.get(req.resolution_mode, zi_sizes["fast_1k"])
        zi_w, zi_h = zi_tier.get(req.ratio, zi_tier["16:9"])
        target_w, target_h = zi_w, zi_h

        zi_seed = int(time.time() * 1000) % (2**31)
        zi_steps = int(req.custom_steps) if req.custom_steps else 8
        # 蒸馏模型甜点区 8~12 步：低于 8 细节不足，高于 12 收益递减且变慢
        zi_steps = max(8, min(zi_steps, 12))

        workflow = {
            "1": {
                "inputs": {"unet_name": "z_image_turbo_int8_convrot.safetensors", "weight_dtype": "default"},
                "class_type": "UNETLoader"
            },
            "2": {
                "inputs": {"clip_name": "qwen_3_4b_fp8_mixed.safetensors", "type": "lumina2"},
                "class_type": "CLIPLoader"
            },
            "3": {
                "inputs": {"vae_name": "z_image_ae.safetensors"},
                "class_type": "VAELoader"
            },
        }
        zi_model = ["1", 0]
        zi_clip = ["2", 0]
        zi_lora_used = None

        # === 角色 LoRA（Z-Image 文生图）===
        if req.enable_id_lora:
            zi_lora_name = pick_id_lora(req.id_lora_name, engine="zimage")
            if not zi_lora_name:
                raise HTTPException(
                    status_code=400,
                    detail="启用了 Z-Image 角色 LoRA，但 loras 目录里没找到适用模型。请把 Z-Image 专用 LoRA 放进 <COMFY_DIR>/models/loras（文件名含 zit / zimage / character / 角色 等关键词即可自动识别）。",
                )
            try:
                zi_strength = float(req.id_lora_strength) if req.id_lora_strength is not None else 0.9
            except (TypeError, ValueError):
                zi_strength = 0.9
            zi_strength = max(0.0, min(zi_strength, 1.5))
            workflow["zi_lora"] = {
                "inputs": {
                    "lora_name": comfy_lora_name(zi_lora_name),
                    "strength_model": zi_strength,
                    "strength_clip": zi_strength,
                    "model": zi_model,
                    "clip": zi_clip,
                },
                "class_type": "LoraLoader",
            }
            zi_model = ["zi_lora", 0]
            zi_clip = ["zi_lora", 1]
            zi_lora_used = zi_lora_name

        workflow["4"] = {
            "inputs": {"text": full_prompt, "clip": zi_clip},
            "class_type": "CLIPTextEncode"
        }
        workflow["5"] = {
            "inputs": {"conditioning": ["4", 0]},
            "class_type": "ConditioningZeroOut"
        }
        workflow["5b"] = {
            "inputs": {"model": zi_model, "shift": 3.0},
            "class_type": "ModelSamplingAuraFlow"
        }
        workflow["6"] = {
            "inputs": {"width": zi_w, "height": zi_h, "batch_size": 1},
            "class_type": "EmptySD3LatentImage"
        }
        workflow["7"] = {
            "inputs": {
                "seed": zi_seed,
                "steps": zi_steps,
                "cfg": 1.0,
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["5b", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0]
            },
            "class_type": "KSampler"
        }
        workflow["8"] = {
            "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
            "class_type": "VAEDecode"
        }
        workflow["9"] = {
            "inputs": {"filename_prefix": f"ZIMG_T2I_{job_id}", "images": ["8", 0]},
            "class_type": "SaveImage"
        }
        save_node = "9"
        zi_lora_tag = f" · 角色LoRA" if zi_lora_used else ""
        model_display = f"Z-Image Turbo · 造相 ({zi_w}×{zi_h} · {zi_steps}步{zi_lora_tag})"
        prefix = f"ZIMG_T2I_{job_id}"

    # ===== 引擎 C: MOSS-Image（OpenMOSS 5B DiT · 文字渲染/排版旗舰 · 仅文生图）=====
    # 官方推荐总像素 1M / 1.5M / 2M；宽高必须为 16 的倍数；配合 Skip Layer Guidance(第22层) 与 Shift 5
    elif req.engine == "moss":
        if not is_moss_ready():
            raise HTTPException(status_code=500, detail="MOSS-Image 引擎未就绪（缺少权重文件，请先下载 moss_image_sft / qwen3.5_9b / flux2-vae_small）")
        if is_edit_mode:
            raise HTTPException(status_code=400, detail="MOSS-Image 引擎暂不支持图生图，请切换到 Krea 2")

        # MOSS-Image 三档分辨率（对齐 Z-Image 的 720p / 1080p / 2K，宽高均为 16 的倍数）
        moss_sizes = {
            "fast_1k": {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (1024, 1024)},
            "native_2k": {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1440, 1440)},
            "ai_4k": {"16:9": (2560, 1440), "9:16": (1440, 2560), "1:1": (2048, 2048)},
        }
        moss_tier = moss_sizes.get(req.resolution_mode, moss_sizes["fast_1k"])
        moss_w, moss_h = moss_tier.get(req.ratio, moss_tier["16:9"])
        target_w, target_h = moss_w, moss_h

        moss_seed = int(time.time() * 1000) % (2**31)
        # MOSS 不是蒸馏模型：默认 25 步 + CFG 2.5 是质量与速度的平衡点，30 步以上提升有限
        moss_steps = int(req.custom_steps) if req.custom_steps else 25
        moss_steps = max(10, min(moss_steps, 50))

        prefix = f"MOSS_T2I_{job_id}"
        moss_neg = (req.negative_prompt or "").strip() or "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"

        workflow = {
            "1": {
                "inputs": {"unet_name": "moss_image_sft_int8_convrot.safetensors", "weight_dtype": "default"},
                "class_type": "UNETLoader"
            },
            "2": {
                "inputs": {"clip_name": "qwen3.5_9b_w4a8_gptq.safetensors", "type": "moss_image"},
                "class_type": "CLIPLoader"
            },
            "3": {
                "inputs": {"vae_name": "flux2-vae_small.safetensors"},
                "class_type": "VAELoader"
            },
            "4": {
                "inputs": {"text": full_prompt, "clip": ["2", 0]},
                "class_type": "CLIPTextEncode"
            },
            "5": {
                "inputs": {"text": moss_neg, "clip": ["2", 0]},
                "class_type": "CLIPTextEncode"
            },
            # 负向分支跳过第 22 层（官方 SLG 设置）：更真实，代价是细节可能略收
            "15": {
                "inputs": {
                    "model": ["1", 0],
                    "double_layers": "",
                    "single_layers": "22",
                    "start_percent": 0.0,
                    "end_percent": 0.5
                },
                "class_type": "SkipLayerGuidanceDiTSimple"
            },
            "6": {
                "inputs": {"width": moss_w, "height": moss_h, "batch_size": 1},
                "class_type": "EmptyFlux2LatentImage"
            },
            "7": {
                "inputs": {
                    "seed": moss_seed,
                    "steps": moss_steps,
                    "cfg": 2.5,
                    "sampler_name": "er_sde",
                    "scheduler": "simple",
                    "denoise": 1.0,
                    "model": ["15", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["6", 0]
                },
                "class_type": "KSampler"
            },
            "8": {
                "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {"filename_prefix": prefix, "images": ["8", 0]},
                "class_type": "SaveImage"
            },
        }
        save_node = "9"
        model_display = f"MOSS-Image SFT · 文字排版 ({moss_w}×{moss_h} · {moss_steps}步)"

    else:
        # ===== 引擎 B: Krea 2 Turbo 艺术旗舰 =====
        if not is_krea2_ready():
            raise HTTPException(status_code=500, detail="Krea 2 图像引擎未就绪")

        prefix = f"KREA2_{'EDIT' if is_edit_mode else 'T2I'}_{job_id}"
        workflow = {}
        workflow["1"] = {
            "inputs": {
                "unet_name": "krea2_turbo_int8_convrot.safetensors",
                "weight_dtype": "default"
            },
            "class_type": "UNETLoader"
        }
        workflow["2"] = {
            "inputs": {
                "clip_name": "qwen3vl_4b_int8_convrot.safetensors",
                "type": "krea2"
            },
            "class_type": "CLIPLoader"
        }
        active_model = ["1", 0]
        active_clip = ["2", 0]

        workflow["3"] = {
            "inputs": {"vae_name": "qwen_image_vae.safetensors"},
            "class_type": "VAELoader"
        }

        # Krea 2 Turbo 步数真实与前端滑块 1:1 对齐
        v_steps = int(req.custom_steps) if req.custom_steps else 8
        seed_val = int(time.time() * 1000) % (2**31)
        load_ref_node = None
        id_lora_used = None
        if is_edit_mode:
            workflow["load_ref"] = {
                "inputs": {"image": req.ref_image},
                "class_type": "LoadImage"
            }
            load_ref_node = "load_ref"
            workflow["scale_ref"] = {
                "inputs": {
                    "upscale_method": "lanczos",
                    "width": base_w,
                    "height": base_h,
                    "crop": "disabled",
                    "image": ["load_ref", 0]
                },
                "class_type": "ImageScale"
            }
            workflow["encode_ref"] = {
                "inputs": {"pixels": ["scale_ref", 0], "vae": ["3", 0]},
                "class_type": "VAEEncode"
            }
            latent_input = ["encode_ref", 0]
            denoise_input = req.denoise

            # === 人物一致性 LoRA（Krea 2 Identity Edit，仅图生图）===
            # 该 LoRA 用「双条件」训练：VAE 潜空间 token + 图像接地 Qwen3-VL 编码，
            # 原生节点不提供，必须配 comfyui-krea2edit 的 Krea2EditModelPatch / Krea2EditGroundedEncode。
            if req.enable_id_lora:
                id_lora_name = pick_id_lora(req.id_lora_name, engine="krea2")
                if not id_lora_name:
                    raise HTTPException(
                        status_code=400,
                        detail="启用了人物一致性 LoRA，但 loras 目录里没有找到可用模型。请把一致性 LoRA 放进 <COMFY_DIR>/models/loras（文件名含 face/id/identity/character/人物/一致性 等关键词即可自动识别）。",
                    )
                if "krea2" not in id_lora_name.lower():
                    raise HTTPException(
                        status_code=400,
                        detail=f"人物一致性 LoRA 需使用 Krea 2 专用模型（如 krea2_identity_edit_v2_r128.safetensors），当前选中：{id_lora_name}。其他架构的 LoRA 无法用于 Krea 2。",
                    )
                try:
                    strength = float(req.id_lora_strength) if req.id_lora_strength is not None else 1.0
                except (TypeError, ValueError):
                    strength = 1.0
                strength = max(0.0, min(strength, 1.5))
                try:
                    ref_boost = float(req.id_lora_ref_boost) if req.id_lora_ref_boost is not None else 3.0
                except (TypeError, ValueError):
                    ref_boost = 3.0
                ref_boost = max(0.0, min(ref_boost, 10.0))
                try:
                    grounding_px = int(req.id_lora_grounding_px) if req.id_lora_grounding_px is not None else 768
                except (TypeError, ValueError):
                    grounding_px = 768
                grounding_px = max(0, min(grounding_px, 2048))

                # 1) LoRA 只作用于模型（该 LoRA 训练时不改 CLIP）
                workflow["id_lora"] = {
                    "inputs": {
                        "lora_name": comfy_lora_name(id_lora_name),
                        "strength_model": strength,
                        "model": active_model,
                    },
                    "class_type": "LoraLoaderModelOnly",
                }
                # 2) 模型补丁：把参考图作为 in-context 干净 token 注入
                workflow["id_patch"] = {
                    "inputs": {
                        "model": ["id_lora", 0],
                        "source_latent": ["encode_ref", 0],
                        "vae": ["3", 0],
                        "source_image": ["scale_ref", 0],
                        "target_latent": latent_input,
                        "fit_mode": "fit",
                        "ref_boost": ref_boost,
                    },
                    "class_type": "Krea2EditModelPatch",
                }
                # 3) 图像接地文本编码：CLIP 编码时「看着」参考图
                workflow["id_encode"] = {
                    "inputs": {
                        "clip": active_clip,
                        "prompt": full_prompt,
                        "image": ["scale_ref", 0],
                        "grounding_px": grounding_px,
                    },
                    "class_type": "Krea2EditGroundedEncode",
                }
                workflow["id_encode_neg"] = {
                    "inputs": {
                        "clip": active_clip,
                        "prompt": "",
                        "image": ["scale_ref", 0],
                        "grounding_px": grounding_px,
                    },
                    "class_type": "Krea2EditGroundedEncode",
                }
                active_model = ["id_patch", 0]
                id_lora_used = id_lora_name
        else:
            if req.enable_second_pass:
                # 二采 (HiRes-Fix)：一采在半分辨率潜空间快速起草构图
                p1_w = max(512, (base_w // 2) // 16 * 16)
                p1_h = max(512, (base_h // 2) // 16 * 16)
            else:
                p1_w, p1_h = base_w, base_h
            workflow["6"] = {
                "inputs": {"width": p1_w, "height": p1_h, "batch_size": 1},
                "class_type": "EmptyLatentImage"
            }
            latent_input = ["6", 0]
            denoise_input = 1.0

        # 文本编码放在 LoRA 挂载之后，确保 CLIP 侧权重生效
        if id_lora_used:
            # 一致性 LoRA 模式：正负条件都走图像接地编码
            cond_src = ["id_encode", 0]
            neg_src = ["id_encode_neg", 0]
        else:
            workflow["4"] = {
                "inputs": {"text": full_prompt, "clip": active_clip},
                "class_type": "CLIPTextEncode"
            }
            workflow["5"] = {
                "inputs": {"conditioning": ["4", 0]},
                "class_type": "ConditioningZeroOut"
            }
            cond_src = ["4", 0]
            neg_src = ["5", 0]

        workflow["7"] = {
            "inputs": {
                "seed": seed_val,
                "steps": v_steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": denoise_input,
                "model": active_model,
                "positive": cond_src,
                "negative": neg_src,
                "latent_image": latent_input
            },
            "class_type": "KSampler"
        }

        # === 二采精炼 (Second Sampling / HiRes-Fix) ===
        decode_src = ["7", 0]
        if req.enable_second_pass:
            # 二采步数：前端可选，默认 6 步；蒸馏模型 2~10 步都稳定，步数越多细节越扎实
            sp_steps = int(req.second_pass_steps) if req.second_pass_steps else 6
            sp_steps = max(2, min(sp_steps, 10))
            if is_edit_mode:
                # 修图模式：同分辨率低去噪二采，抛光细节、消除一采噪点
                sp_latent = ["7", 0]
                sp_denoise = 0.30
            else:
                # 文生图模式：潜空间双线性放大至全分辨率后二采重绘细节
                workflow["sp_up"] = {
                    "inputs": {
                        "upscale_method": "bislerp",
                        "width": base_w,
                        "height": base_h,
                        "crop": "disabled",
                        "samples": ["7", 0]
                    },
                    "class_type": "LatentUpscale"
                }
                sp_latent = ["sp_up", 0]
                sp_denoise = 0.50
            workflow["sp_sample"] = {
                "inputs": {
                    "seed": (seed_val + 1) % (2**31),
                    "steps": sp_steps,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": sp_denoise,
                    "model": active_model,
                    "positive": cond_src,
                    "negative": neg_src,
                    "latent_image": sp_latent
                },
                "class_type": "KSampler"
            }
            decode_src = ["sp_sample", 0]

        workflow["8"] = {
            "inputs": {"samples": decode_src, "vae": ["3", 0]},
            "class_type": "VAEDecode"
        }

        workflow["9"] = {
            "inputs": {"filename_prefix": prefix, "images": ["8", 0]},
            "class_type": "SaveImage"
        }
        save_node = "9"
        sp_tag = f" · 二采精炼 ({v_steps}步起草 + {sp_steps}步重绘)" if req.enable_second_pass else ""
        id_tag = f" · 一致性LoRA" if id_lora_used else ""
        if req.resolution_mode == "native_2k":
            model_display = f"Krea 2 Turbo · 2K 电影级 ({target_w}×{target_h}{sp_tag}{id_tag})"
        elif req.resolution_mode == "ai_4k":
            model_display = f"Krea 2 Turbo · 4K 原生采样 ({target_w}×{target_h}{sp_tag}{id_tag})"
        else:
            model_display = f"Krea 2 Turbo · 1080p 全高清 ({target_w}×{target_h}{sp_tag}{id_tag})"

    start_t = time.time()

    try:
        r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow}, timeout=5)
        res_data = r.json()
        if "prompt_id" not in res_data:
            err_msg = res_data.get("error", {}).get("message") or json.dumps(res_data.get("node_errors", {}))
            raise HTTPException(status_code=500, detail=f"工作流校验失败: {err_msg}")
        prompt_id = res_data["prompt_id"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提交任务失败: {str(e)}")

    out_file = None
    consecutive_conn_failures = 0
    # 4K 原生采样 (3840 长边去噪；开二采时 1920 起草 + 3840 重绘) 在 8G 笔记本上约 6~8 分钟
    if req.engine == "zimage":
        max_wait_secs = 600
    elif req.engine == "moss":
        # MOSS 常驻显存会挤掉 Krea/Z-Image，每次切换需重新加载 ~12GB，8G 显卡靠低显存模式回退
        max_wait_secs = 1500
    elif req.resolution_mode == "ai_4k":
        max_wait_secs = 1800
    else:
        max_wait_secs = 1200
    for _ in range(max_wait_secs):
        time.sleep(1)
        try:
            hist_resp = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=3)
            consecutive_conn_failures = 0
            hist = hist_resp.json()
            if prompt_id in hist:
                status_info = hist[prompt_id].get("status", {})
                if status_info.get("status_str") == "error":
                    msgs = status_info.get("messages", [])
                    err_detail = "底层推理出现错误"
                    for m in msgs:
                        if m[0] == "execution_error":
                            err_detail = m[1].get("exception_message", "")
                    raise HTTPException(status_code=500, detail=f"渲染失败: {err_detail.strip()}")

                outputs_node = hist[prompt_id].get("outputs", {}).get(save_node, {})
                imgs = outputs_node.get("images", [])
                if imgs:
                    out_file = imgs[0]["filename"]
                    break
        except HTTPException:
            raise
        except Exception:
            consecutive_conn_failures += 1
            if consecutive_conn_failures >= 8:
                ensure_comfy_running()
                raise HTTPException(status_code=500, detail="底层推理引擎通信中断，已自动重新唤醒，请重新发起一次生成！")

    if not out_file:
        raise HTTPException(status_code=504, detail="生成超时")

    src_path = os.path.join(COMFY_DIR, "output", out_file)
    dst_path = os.path.join(OUTPUTS_DIR, out_file)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)

    elapsed = round(time.time() - start_t, 2)

    return {
        "success": True,
        "media_url": f"/outputs/{out_file}",
        "elapsed": elapsed,
        "model_used": model_display,
        "prompt": req.prompt,
        "width": target_w,
        "height": target_h
    }

@app.get("/api/history")
def get_history():
    files = []
    if os.path.exists(OUTPUTS_DIR):
        for f in os.listdir(OUTPUTS_DIR):
            if f.lower().endswith((".png", ".jpg", ".webp")):
                files.append({
                    "filename": f,
                    "url": f"/outputs/{f}",
                    "mtime": os.path.getmtime(os.path.join(OUTPUTS_DIR, f))
                })
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return files[:30]

app.mount("/input_files", StaticFiles(directory=COMFY_INPUT), name="input_files")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

import threading
threading.Thread(target=ensure_comfy_running, daemon=True).start()

def _port_is_free(port: int) -> bool:
    # 注意：Windows 上 SO_REUSEADDR 会允许绑定已被占用的端口，
    # 这里必须用普通 bind 才能真正探测出占用。
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _pick_port(preferred: int, span: int = 20) -> int:
    # 首选端口可能只是被别的程序短暂占用（例如代理的临时连接），
    # 先原地等几秒重试，尽量让固定地址/手机书签继续有效。
    for _ in range(3):
        if _port_is_free(preferred):
            return preferred
        time.sleep(0.7)
    # 被占用就顺延到相邻端口（7861、7862…）
    for p in range(preferred + 1, preferred + span):
        if _port_is_free(p):
            return p
    return preferred


if __name__ == "__main__":
    import uvicorn
    BASE_PORT = int(os.environ.get("EASYAI_PORT", "7860"))
    PORT = _pick_port(BASE_PORT)
    if PORT != BASE_PORT:
        print(f">>> 端口 {BASE_PORT} 被长期占用，已自动顺延到 {PORT}")
    print(f">>> EasyAI Studio Pro Ultra 启动在: http://0.0.0.0:{PORT} (局域网手机可访问)")

    # 浏览器跟随真实端口打开（仅由启动脚本显式开启），避免端口顺延后仍打开旧地址
    if os.environ.get("EASYAI_OPEN_BROWSER", "0") == "1":
        import webbrowser

        def _open_browser():
            time.sleep(2.5)
            try:
                webbrowser.open(f"http://127.0.0.1:{PORT}")
            except Exception:
                pass

        threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=PORT)








