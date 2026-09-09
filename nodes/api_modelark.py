"""
JMCAI ModelArk (BytePlus / Volcengine Seedream 5.0) image generation node for ComfyUI.

Reference: https://docs.byteplus.com/en/docs/ModelArk/1541523
- POST /api/v3/images/generations
- Authentication: Authorization: Bearer ARK_API_KEY
- Response: data[].b64_json or data[].url
"""
import base64
import io
import logging
import re
import requests

from .base.api_image_config import BUILTIN_MODELS, get_api_config, get_model_list
from .base.api_image_utils import bytes_to_tensor, sanitize_url, tensor_to_pil

logger = logging.getLogger("JMCAI-APIImage")

DEFAULT_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
MAX_INPUT_PIXELS = 36_000_000
MAX_INPUT_BYTES = 30 * 1024 * 1024
MIN_OUTPUT_PIXELS = 921_600
MAX_OUTPUT_PIXELS = 4_624_220
MODEL_FAMILY_HINTS = {
    "auto": None,
    "seedream-5-0-pro": "pro",
    "seedream-5-0-lite": "lite",
    "seedream-4-5": "4.5",
    "seedream-4-0": "4.0",
    "seedream-3-0-t2i": "3.0-t2i",
}


def _model_family(model_name):
    normalized = (model_name or "").strip().lower()
    if "seedream-5-0-pro" in normalized:
        return "pro"
    if "seedream-5-0-lite" in normalized or normalized == "seedream-5-0-260128":
        return "lite"
    if "seedream-4-5" in normalized:
        return "4.5"
    if "seedream-4-0" in normalized:
        return "4.0"
    if "seedream-3-0-t2i" in normalized:
        return "3.0-t2i"
    return "unknown"


def _collect_reference_images(ref_images, image1, image2, image3):
    images = []
    for tensor in (ref_images, image1, image2, image3):
        if tensor is not None:
            images.extend(tensor_to_pil(tensor))
    return images


def _validate_reference_image(image, index):
    width, height = image.size
    if width <= 14 or height <= 14:
        raise ValueError(
            f"[JMCAI ModelArk] Reference image {index} must be larger than 14x14 px."
        )
    if max(width, height) / min(width, height) > 16:
        raise ValueError(
            f"[JMCAI ModelArk] Reference image {index} aspect ratio must be within 1:16-16:1."
        )
    if width * height > MAX_INPUT_PIXELS:
        raise ValueError(
            f"[JMCAI ModelArk] Reference image {index} exceeds 36000000 pixels."
        )


def _image_to_data_url(image, index):
    _validate_reference_image(image, index)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    image_bytes = buffer.getvalue()
    if len(image_bytes) > MAX_INPUT_BYTES:
        raise ValueError(
            f"[JMCAI ModelArk] Reference image {index} exceeds 30 MB limit."
        )
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _calibrate_and_validate_output_size(raw_size, family):
    if not raw_size:
        return "2K"
    clean_size = raw_size.split()[0].strip().replace("*", "x")
    if family not in {"pro", "lite", "4.5", "4.0"}:
        return clean_size
    if clean_size.upper() in {"1K", "2K"}:
        return clean_size.upper()

    match = re.fullmatch(r"(\d+)x(\d+)", clean_size.lower())
    if not match:
        logger.warning(f"[JMCAI ModelArk] 无法识别尺寸 '{raw_size}'，默认采用 2K")
        return "2K"

    width, height = int(match.group(1)), int(match.group(2))

    # 1. 宽和高对齐为 16 的倍数
    width = max(16, round(width / 16) * 16)
    height = max(16, round(height / 16) * 16)

    # 2. 长宽比限制在 1:16 到 16:1 之间
    if max(width, height) / min(width, height) > 16.0:
        if width > height:
            height = max(16, round((width / 16.0) / 16) * 16)
        else:
            width = max(16, round((height / 16.0) / 16) * 16)

    # 3. 像素范围限制：921,600 ~ 4,624,220
    pixels = width * height
    if pixels < MIN_OUTPUT_PIXELS:
        scale = (MIN_OUTPUT_PIXELS / pixels) ** 0.5
        width = max(16, int(round(width * scale / 16)) * 16)
        height = max(16, int(round(height * scale / 16)) * 16)
        while width * height < MIN_OUTPUT_PIXELS:
            width += 16
            height += 16
    elif pixels > MAX_OUTPUT_PIXELS:
        scale = (MAX_OUTPUT_PIXELS / pixels) ** 0.5
        width = max(16, int(round(width * scale / 16)) * 16)
        height = max(16, int(round(height * scale / 16)) * 16)
        while width * height > MAX_OUTPUT_PIXELS:
            if width >= height:
                width -= 16
            else:
                height -= 16

    calibrated = f"{width}x{height}"
    if calibrated != clean_size:
        logger.info(f"[JMCAI ModelArk] 尺寸规则校对: '{clean_size}' -> '{calibrated}'")
    return calibrated


def _generation_url(base_url):
    if base_url.endswith("/images/generations"):
        return base_url
    return f"{base_url}/images/generations"


def _error_text(error):
    if isinstance(error, dict):
        code = error.get("code", "UnknownError")
        message = error.get("message", "No error message")
        return f"{code}: {message}"
    return str(error or "Unknown error")


class JMCAI_ModelArkImageGenerate:
    """Generate or edit images through BytePlus / Volcengine ModelArk API (Seedream 5.0)."""

    CATEGORY = "JMCAI❤/API生图 (火山 · Seedream 5.0)"
    FUNCTION = "generate"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)

    @classmethod
    def INPUT_TYPES(cls):
        models = get_model_list("BytePlus ModelArk")
        if not models:
            models = BUILTIN_MODELS.get(
                "BytePlus ModelArk",
                ["dola-seedream-5-0-pro-260628"],
            )

        saved_config = get_api_config("BytePlus ModelArk")
        saved_key = saved_config.get("api_key", "")
        saved_url = saved_config.get("base_url", DEFAULT_BASE_URL)
        saved_model = saved_config.get("model_name", "")
        default_model = saved_model if saved_model in models else models[0]

        return {
            "required": {
                "提示词": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "请输入 Seedream 生图提示词（Prompt）...",
                }),
                "API密钥": ("STRING", {
                    "default": saved_key,
                    "placeholder": "填写火山方舟 ModelArk API Key",
                }),
                "接口地址": ("STRING", {
                    "default": saved_url,
                    "placeholder": DEFAULT_BASE_URL,
                }),
                "模型名称": (models, {
                    "default": default_model,
                }),
                "图像尺寸": ([
                    "2K (推荐默认 · 自动最佳超清)",
                    "1K (标准清晰度)",
                    "2048x2048 (2K · 1:1 方图)",
                    "1024x1024 (1K · 1:1 方图)",
                    "2816x1584 (2K · 16:9 宽屏壁纸)",
                    "1424x800 (1K · 16:9 宽屏)",
                    "1584x2816 (2K · 9:16 手机竖屏)",
                    "800x1424 (1K · 9:16 手机竖屏)",
                    "2368x1776 (2K · 4:3 经典横屏)",
                    "1152x864 (1K · 4:3 经典横屏)",
                    "1776x2368 (2K · 3:4 经典竖屏)",
                    "864x1152 (1K · 3:4 经典竖屏)",
                    "2496x1664 (2K · 3:2 摄影横屏)",
                    "1248x832 (1K · 3:2 摄影横屏)",
                    "1664x2496 (2K · 2:3 摄影竖屏)",
                    "832x1248 (1K · 2:3 摄影竖屏)",
                    "3136x1344 (2K · 21:9 电影全景)",
                    "1568x672 (1K · 21:9 电影全景)",
                ], {
                    "default": "2K (推荐默认 · 自动最佳超清)",
                }),
            },
            "optional": {
                "多参考图批次": ("IMAGE",),
                "参考图1": ("IMAGE",),
                "参考图2": ("IMAGE",),
                "参考图3": ("IMAGE",),
                "自定义模型": ("STRING", {
                    "default": "",
                    "placeholder": "留空默认使用下拉框，输入则优先指定模型名称",
                }),
                "模型架构系列": (list(MODEL_FAMILY_HINTS), {
                    "default": "auto",
                }),
                "自定义尺寸": ("STRING", {
                    "default": "",
                    "placeholder": "可选宽x高尺寸覆盖，如 2048x1152",
                }),
                "多图连续生成": (["disabled", "auto"], {
                    "default": "disabled",
                }),
                "最大生成张数": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 15,
                }),
                "输出格式": (["png", "jpeg"], {
                    "default": "png",
                }),
                "返回数据格式": (["b64_json", "url"], {
                    "default": "b64_json",
                }),
                "启用水印": ("BOOLEAN", {
                    "default": False,
                }),
                "提示词自动优化": (["standard", "fast", "disabled"], {
                    "default": "standard",
                }),
                "随机种子": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0x7FFFFFFF,
                }),
                "引导强度": ("FLOAT", {
                    "default": 2.5,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def generate(
        self,
        提示词=None,
        API密钥=None,
        接口地址=None,
        模型名称=None,
        图像尺寸=None,
        多参考图批次=None,
        参考图1=None,
        参考图2=None,
        参考图3=None,
        自定义模型="",
        自定义尺寸="",
        多图连续生成="disabled",
        最大生成张数=15,
        输出格式="png",
        返回数据格式="b64_json",
        启用水印=False,
        提示词自动优化="standard",
        随机种子=-1,
        引导强度=2.5,
        模型架构系列="auto",
        **kwargs,
    ):
        prompt = kwargs.get("prompt", 提示词)
        api_key = kwargs.get("api_key", API密钥)
        base_url = kwargs.get("base_url", 接口地址)
        model_name = kwargs.get("model_name", 模型名称)
        size = kwargs.get("size", 图像尺寸)
        ref_images = kwargs.get("ref_images", 多参考图批次)
        image1 = kwargs.get("image1", 参考图1)
        image2 = kwargs.get("image2", 参考图2)
        image3 = kwargs.get("image3", 参考图3)
        custom_model = kwargs.get("custom_model", 自定义模型)
        custom_size = kwargs.get("custom_size", 自定义尺寸)
        sequential_image_generation = kwargs.get("sequential_image_generation", 多图连续生成)
        max_images = kwargs.get("max_images", 最大生成张数)
        output_format = kwargs.get("output_format", 输出格式)
        response_format = kwargs.get("response_format", 返回数据格式)
        watermark = kwargs.get("watermark", 启用水印)
        optimize_prompt_mode = kwargs.get("optimize_prompt_mode", 提示词自动优化)
        seed = kwargs.get("seed", 随机种子)
        guidance_scale = kwargs.get("guidance_scale", 引导强度)
        model_family = kwargs.get("model_family", 模型架构系列)

        if not prompt or not prompt.strip():
            raise ValueError("[JMCAI ModelArk] Prompt is empty.")
        if not api_key or not api_key.strip():
            raise ValueError("[JMCAI ModelArk] API Key is not set.")
        if not base_url or not base_url.strip():
            raise ValueError("[JMCAI ModelArk] Base URL is not set.")

        effective_model = custom_model.strip() if custom_model and custom_model.strip() else model_name
        clean_url = sanitize_url(base_url) or DEFAULT_BASE_URL
        family = MODEL_FAMILY_HINTS[model_family] or _model_family(effective_model)
        
        raw_size = custom_size.strip() if custom_size and custom_size.strip() else size
        effective_size = _calibrate_and_validate_output_size(raw_size, family)

        reference_images = _collect_reference_images(ref_images, image1, image2, image3)

        payload = {
            "model": effective_model,
            "prompt": prompt,
            "size": effective_size,
            "response_format": response_format,
            "watermark": bool(watermark),
        }

        if reference_images:
            encoded_images = [
                _image_to_data_url(image, index)
                for index, image in enumerate(reference_images, start=1)
            ]
            payload["image"] = encoded_images[0] if len(encoded_images) == 1 else encoded_images

        if family in {"pro", "lite"}:
            payload["output_format"] = output_format

        if family in {"lite", "4.5", "4.0"}:
            payload["sequential_image_generation"] = sequential_image_generation
            if sequential_image_generation == "auto":
                payload["sequential_image_generation_options"] = {"max_images": max_images}

        if family in {"pro", "lite", "4.5", "4.0"} and optimize_prompt_mode != "disabled":
            payload["optimize_prompt_options"] = {"mode": optimize_prompt_mode}

        if seed is not None and seed >= 0:
            payload["seed"] = seed

        if family == "3.0-t2i" and guidance_scale is not None and guidance_scale > 0:
            payload["guidance_scale"] = guidance_scale

        url = _generation_url(clean_url)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300)
        except Exception as error:
            raise RuntimeError(f"[JMCAI ModelArk] Request failed: {error}")

        if response.status_code != 200:
            raise RuntimeError(f"[JMCAI ModelArk] API Error ({response.status_code}): {response.text[:500]}")

        try:
            json_response = response.json()
        except Exception:
            raise RuntimeError("[JMCAI ModelArk] Invalid JSON response.")

        image_bytes = []
        for item in json_response.get("data") or []:
            if item.get("b64_json"):
                image_bytes.append(base64.b64decode(item["b64_json"], validate=True))
            elif item.get("url"):
                download = requests.get(item["url"], timeout=60)
                download.raise_for_status()
                image_bytes.append(download.content)

        if not image_bytes:
            raise RuntimeError("[JMCAI ModelArk] No image data returned.")

        return (bytes_to_tensor(image_bytes),)


NODE_CLASS_MAPPINGS = {
    "JMCAI_ModelArkImageGenerate": JMCAI_ModelArkImageGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JMCAI_ModelArkImageGenerate": "JMCAI❤ 火山/Seedream 5.0 生图",
}
