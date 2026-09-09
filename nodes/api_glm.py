"""
JMCAI GLM Image Generation node for ComfyUI.

Uses the ZhipuAI REST API for image generation via GLM-Image / CogView models.
"""
import logging
import base64
import requests as http_requests

from .base.api_image_config import get_api_config, get_model_list, BUILTIN_MODELS
from .base.api_image_utils import bytes_to_tensor, sanitize_url, validate_ref_images

logger = logging.getLogger("JMCAI-APIImage")

MODEL_REF_IMAGE_LIMITS = {
    "glm-image": (0, 0),
    "cogview-4-250304": (0, 0),
}


def _calibrate_glm_size(model_name, raw_size):
    """
    Calibrate size for ZhipuAI GLM-Image / CogView:
    - GLM-Image requires width and height between 1024px and 2048px,
      max pixels <= 4,194,304, and both dimensions divisible by 32.
    """
    import re
    if not raw_size:
        return "1280x1280"

    clean_size = raw_size.split()[0].strip().replace("*", "x").replace("X", "x")
    match = re.fullmatch(r"(\d+)x(\d+)", clean_size.lower())
    if not match:
        logger.warning(f"[JMCAI GLM] 无法识别尺寸 '{raw_size}'，默认采用 1280x1280")
        return "1280x1280"

    w, h = int(match.group(1)), int(match.group(2))

    # 对齐为 32 的倍数
    new_w = max(32, round(w / 32) * 32)
    new_h = max(32, round(h / 32) * 32)

    # GLM-Image 限制: 宽与高均在 1024 ~ 2048 之间，总像素 <= 4,194,304
    if "glm-image" in (model_name or "").lower():
        new_w = max(1024, min(2048, new_w))
        new_h = max(1024, min(2048, new_h))
        # 限制总像素不超过 4,194,304 (2048x2048)
        if new_w * new_h > 4_194_304:
            scale = (4_194_304 / (new_w * new_h)) ** 0.5
            new_w = max(1024, int(round(new_w * scale / 32)) * 32)
            new_h = max(1024, int(round(new_h * scale / 32)) * 32)
        # 再次确认 32 倍数
        new_w = (new_w // 32) * 32
        new_h = (new_h // 32) * 32

    calibrated = f"{new_w}x{new_h}"
    if calibrated != clean_size:
        logger.info(f"[JMCAI GLM] 尺寸自动对齐 32 的倍数: '{clean_size}' -> '{calibrated}'")
    return calibrated


class JMCAI_GLMImageGenerate:
    """
    Generate images using ZhipuAI GLM-Image / CogView API.
    """

    CATEGORY = "JMCAI❤/API生图 (智谱 · GLM/CogView)"
    FUNCTION = "generate"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)

    @classmethod
    def INPUT_TYPES(cls):
        models = get_model_list("GLM Image")
        if not models:
            models = BUILTIN_MODELS.get("GLM Image", ["glm-image"])

        saved_config = get_api_config("GLM Image")
        saved_key = saved_config.get("api_key", "")
        saved_url = saved_config.get("base_url", "https://open.bigmodel.cn/api")

        return {
            "required": {
                "提示词": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "请输入图片生成提示词（Prompt）..."
                }),
                "API密钥": ("STRING", {
                    "default": saved_key,
                    "placeholder": "填写智谱 API Key"
                }),
                "模型名称": (models, {
                    "default": models[0] if models else "glm-image"
                }),
                "画质质量": (["hd", "standard"], {
                    "default": "hd"
                }),
                "图像尺寸": ([
                    "1280x1280 (1:1 · 正方默认)",
                    "1568x1056 (3:2 · 横屏摄影)",
                    "1056x1568 (2:3 · 竖屏人像)",
                    "1472x1088 (4:3 · 标准横屏)",
                    "1088x1472 (3:4 · 标准竖屏)",
                    "1728x960 (16:9 · 影视宽屏)",
                    "960x1728 (9:16 · 手机长屏)",
                    "1024x1024 (1:1 · 基础正方)",
                    "2048x1024 (2:1 · 宽景大片)",
                    "1024x2048 (1:2 · 竖版大片)",
                ], {
                    "default": "1280x1280 (1:1 · 正方默认)"
                }),
            },
            "optional": {
                "生成张数": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                }),
                "自定义尺寸": ("STRING", {
                    "default": "",
                    "placeholder": "自定义宽x高（需为32倍数，如 1280x1280）"
                }),
                "接口地址": ("STRING", {
                    "default": saved_url,
                    "placeholder": "https://open.bigmodel.cn/api"
                }),
                "多参考图批次": ("IMAGE",),
                "参考图1": ("IMAGE",),
                "参考图2": ("IMAGE",),
                "参考图3": ("IMAGE",),
                "自定义模型": ("STRING", {
                    "default": "",
                    "placeholder": "留空默认使用下拉框，输入则优先指定模型名称"
                }),
                "随机种子": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0x7FFFFFFF,
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
        模型名称=None,
        画质质量="hd",
        图像尺寸=None,
        生成张数=1,
        自定义尺寸="",
        接口地址="",
        多参考图批次=None,
        参考图1=None,
        参考图2=None,
        参考图3=None,
        自定义模型="",
        随机种子=0,
        **kwargs,
    ):
        prompt = kwargs.get("prompt", 提示词)
        api_key = kwargs.get("api_key", API密钥)
        model_name = kwargs.get("model_name", 模型名称)
        quality = kwargs.get("quality", 画质质量)
        size = kwargs.get("size", 图像尺寸)
        num_images = kwargs.get("num_images", 生成张数)
        custom_size = kwargs.get("custom_size", 自定义尺寸)
        base_url = kwargs.get("base_url", 接口地址)
        ref_images = kwargs.get("ref_images", 多参考图批次)
        image1 = kwargs.get("image1", 参考图1)
        image2 = kwargs.get("image2", 参考图2)
        image3 = kwargs.get("image3", 参考图3)
        custom_model = kwargs.get("custom_model", 自定义模型)
        seed = kwargs.get("seed", 随机种子)

        if not prompt or not prompt.strip():
            raise ValueError("[JMCAI GLM] Prompt is empty.")
        if not api_key or not api_key.strip():
            raise ValueError("[JMCAI GLM] API Key is not set.")

        effective_model = custom_model.strip() if custom_model and custom_model.strip() else model_name

        extra_img_count = sum(1 for s in [image1, image2, image3] if s is not None)
        validate_ref_images("GLM", effective_model, ref_images, MODEL_REF_IMAGE_LIMITS, extra_count=extra_img_count)

        effective_url = sanitize_url(base_url) or "https://open.bigmodel.cn/api"

        raw_size = custom_size.strip() if custom_size and custom_size.strip() else size
        calibrated_size = _calibrate_glm_size(effective_model, raw_size)

        all_images_data = []
        for gen_idx in range(num_images):
            images_data = self._call_api(
                effective_url, api_key, effective_model, prompt, quality, calibrated_size, seed=seed
            )
            all_images_data.extend(images_data)

        if not all_images_data:
            raise RuntimeError("[JMCAI GLM] No image data returned.")

        return (bytes_to_tensor(all_images_data),)

    def _call_api(self, base_url, api_key, model, prompt, quality, size, seed=0):
        url = f"{base_url}/paas/v4/images/generations"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}"
        }
        payload = {
            "model": model,
            "prompt": prompt,
            "quality": quality,
            "size": size,
        }
        if seed and seed > 0:
            payload["seed"] = seed

        try:
            response = http_requests.post(url, headers=headers, json=payload, timeout=120)
        except Exception as e:
            raise RuntimeError(f"[JMCAI GLM] Request failed: {e}")

        if response.status_code != 200:
            raise RuntimeError(f"[JMCAI GLM] API Error ({response.status_code}): {response.text[:500]}")

        try:
            json_response = response.json()
        except Exception:
            raise RuntimeError("[JMCAI GLM] Invalid JSON response.")

        images_data = []
        for item in json_response.get("data") or []:
            if item.get("url"):
                img_resp = http_requests.get(item["url"], timeout=60)
                img_resp.raise_for_status()
                images_data.append(img_resp.content)
            elif item.get("b64_json"):
                images_data.append(base64.b64decode(item["b64_json"]))

        return images_data


NODE_CLASS_MAPPINGS = {
    "JMCAI_GLMImageGenerate": JMCAI_GLMImageGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JMCAI_GLMImageGenerate": "JMCAI❤ 智谱/GLM-Image 生图",
}
