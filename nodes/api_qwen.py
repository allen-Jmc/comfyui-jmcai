"""
JMCAI Qwen Image Generation node for ComfyUI.

Uses the DashScope SDK for Alibaba Qwen / Tongyi Wanxiang image generation and editing.
"""
import logging
import io
import base64
import requests

from .base.api_image_config import get_api_config, get_model_list, BUILTIN_MODELS
from .base.api_image_utils import tensor_to_pil, mask_to_pil, bytes_to_tensor, sanitize_url, validate_ref_images

logger = logging.getLogger("JMCAI-APIImage")

MODEL_REF_IMAGE_LIMITS = {
    "qwen-image-plus": (0, 0),
    "qwen-image-edit": (1, 3),
}


def _calibrate_qwen_size(model_name, raw_size):
    """
    Calibrate size for DashScope Qwen Image:
    - Base 'qwen-image-edit' does NOT accept 'size' parameter at all (causes API error).
    - 'qwen-image-plus' / 'qwen-image' strictly requires 'width*height' format with specific ratios.
    - 'qwen-image-2.0' and 'qwen-image-edit-max' allow [512, 2048] dimensions.
    """
    import re
    normalized = (model_name or "").strip().lower()

    # Critical: Base 'qwen-image-edit' does not support size parameter
    if normalized == "qwen-image-edit":
        logger.info("[JMCAI Qwen] 基础版 qwen-image-edit 不支持传入 size 参数，已自动省略并按输入图自适应输出。")
        return None

    if not raw_size:
        return "1664*928"

    clean_size = raw_size.split()[0].strip().replace("x", "*").replace("X", "*")
    match = re.fullmatch(r"(\d+)\*(\d+)", clean_size)
    if match:
        w, h = int(match.group(1)), int(match.group(2))
        new_w = max(16, round(w / 16) * 16)
        new_h = max(16, round(h / 16) * 16)
        calibrated = f"{new_w}*{new_h}"
        if calibrated != clean_size:
            logger.info(f"[JMCAI Qwen] 尺寸自动对齐 16 的倍数: '{clean_size}' -> '{calibrated}'")
        return calibrated

    return clean_size


class JMCAI_QwenImageGenerate:
    """
    Generate or edit images using Alibaba Qwen (Tongyi Wanxiang) via DashScope SDK.
    """

    CATEGORY = "JMCAI❤/API生图 (阿里 · 通义万相)"
    FUNCTION = "generate"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("图像", "文本响应",)

    @classmethod
    def INPUT_TYPES(cls):
        models = get_model_list("Qwen Image")
        if not models:
            models = BUILTIN_MODELS.get("Qwen Image", ["qwen-image-plus"])

        saved_config = get_api_config("Qwen Image")
        saved_key = saved_config.get("api_key", "")
        saved_url = saved_config.get("base_url", "https://dashscope.aliyuncs.com/api/v1")

        return {
            "required": {
                "提示词": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "请输入图片生成提示词（Prompt）..."
                }),
                "API密钥": ("STRING", {
                    "default": saved_key,
                    "placeholder": "sk-... (填写 DashScope API Key)"
                }),
                "模型名称": (models, {
                    "default": models[0] if models else "qwen-image-plus"
                }),
                "图像尺寸": ([
                    "1664*928 (16:9 · 宽屏推荐)",
                    "1472*1104 (4:3 · 标准横屏)",
                    "1328*1328 (1:1 · 原生方图)",
                    "1104*1472 (3:4 · 标准竖屏)",
                    "928*1664 (9:16 · 手机长屏)",
                    "1024*1024 (1:1 · 编辑推荐)",
                    "1280*720 (16:9 · 编辑高清横屏)",
                    "720*1280 (9:16 · 编辑高清竖屏)",
                ], {
                    "default": "1664*928 (16:9 · 宽屏推荐)"
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
                    "placeholder": "自定义宽*高，如 1536*1024"
                }),
                "接口地址": ("STRING", {
                    "default": saved_url,
                    "placeholder": "https://dashscope.aliyuncs.com/api/v1"
                }),
                "多参考图批次": ("IMAGE",),
                "参考图1": ("IMAGE",),
                "参考图2": ("IMAGE",),
                "参考图3": ("IMAGE",),
                "局部重绘遮罩": ("MASK",),
                "负向提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "反向负面提示词（Negative Prompt）"
                }),
                "启用水印": ("BOOLEAN", {
                    "default": False,
                }),
                "提示词自动扩写": ("BOOLEAN", {
                    "default": True,
                }),
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

    def _upload_image_and_get_url(self, pil_img):
        import tempfile
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(buf.getvalue())
        tmp.close()
        return tmp.name

    def generate(
        self,
        提示词=None,
        API密钥=None,
        模型名称=None,
        图像尺寸=None,
        生成张数=1,
        自定义尺寸="",
        接口地址="",
        多参考图批次=None,
        参考图1=None,
        参考图2=None,
        参考图3=None,
        局部重绘遮罩=None,
        负向提示词="",
        启用水印=False,
        提示词自动扩写=True,
        自定义模型="",
        随机种子=0,
        **kwargs,
    ):
        import os

        prompt = kwargs.get("prompt", 提示词)
        api_key = kwargs.get("api_key", API密钥)
        model_name = kwargs.get("model_name", 模型名称)
        size = kwargs.get("size", 图像尺寸)
        num_images = kwargs.get("num_images", 生成张数)
        custom_size = kwargs.get("custom_size", 自定义尺寸)
        base_url = kwargs.get("base_url", 接口地址)
        ref_images = kwargs.get("ref_images", 多参考图批次)
        image1 = kwargs.get("image1", 参考图1)
        image2 = kwargs.get("image2", 参考图2)
        image3 = kwargs.get("image3", 参考图3)
        mask = kwargs.get("mask", 局部重绘遮罩)
        negative_prompt = kwargs.get("negative_prompt", 负向提示词)
        watermark = kwargs.get("watermark", 启用水印)
        prompt_extend = kwargs.get("prompt_extend", 提示词自动扩写)
        custom_model = kwargs.get("custom_model", 自定义模型)
        seed = kwargs.get("seed", 随机种子)

        if not prompt or not prompt.strip():
            raise ValueError("[JMCAI Qwen] Prompt is empty.")
        if not api_key or not api_key.strip():
            raise ValueError("[JMCAI Qwen] API Key is not set.")

        effective_model = custom_model.strip() if custom_model and custom_model.strip() else model_name
        raw_size = custom_size.strip() if custom_size and custom_size.strip() else size
        effective_size = _calibrate_qwen_size(effective_model, raw_size)

        try:
            import dashscope
            from dashscope import MultiModalConversation
        except ImportError as e:
            raise ImportError(
                f"[JMCAI Qwen] 缺少依赖包: {e}。请先运行: pip install dashscope"
            )

        effective_url = sanitize_url(base_url)
        if effective_url:
            dashscope.base_http_api_url = effective_url

        extra_img_count = sum(1 for s in [image1, image2, image3] if s is not None)
        validate_ref_images("Qwen", effective_model, ref_images, MODEL_REF_IMAGE_LIMITS, extra_count=extra_img_count)

        content = []
        temp_files = []
        ref_count = 0

        if ref_images is not None:
            try:
                batch_pils = tensor_to_pil(ref_images)
                for bimg in batch_pils:
                    tmp_path = self._upload_image_and_get_url(bimg)
                    temp_files.append(tmp_path)
                    content.append({"image": f"file://{tmp_path}"})
                    ref_count += 1
            except Exception as e:
                logger.warning(f"[JMCAI Qwen] Failed to process ref_images: {e}")

        for slot_name, slot_val in [("image1", image1), ("image2", image2), ("image3", image3)]:
            if slot_val is not None:
                try:
                    slot_pils = tensor_to_pil(slot_val)
                    for simg in slot_pils:
                        tmp_path = self._upload_image_and_get_url(simg)
                        temp_files.append(tmp_path)
                        content.append({"image": f"file://{tmp_path}"})
                        ref_count += 1
                except Exception as e:
                    logger.warning(f"[JMCAI Qwen] Failed to process {slot_name}: {e}")

        content.append({"text": prompt})

        if mask is not None:
            try:
                mask_pil = mask_to_pil(mask)
                tmp_path = self._upload_image_and_get_url(mask_pil)
                temp_files.append(tmp_path)
                content.append({"image": f"file://{tmp_path}"})
            except Exception as e:
                logger.warning(f"[JMCAI Qwen] Failed to process mask: {e}")

        messages = [{"role": "user", "content": content}]
        all_images_data = []
        text_messages = []

        try:
            for gen_idx in range(num_images):
                call_kwargs = {
                    "api_key": api_key.strip(),
                    "model": effective_model,
                    "messages": messages,
                    "result_format": "message",
                    "stream": False,
                    "watermark": watermark,
                }
                if negative_prompt and negative_prompt.strip():
                    call_kwargs["negative_prompt"] = negative_prompt.strip()
                if effective_size:
                    call_kwargs["size"] = effective_size
                if "edit" not in effective_model.lower():
                    call_kwargs["prompt_extend"] = prompt_extend
                if seed and seed > 0:
                    call_kwargs["seed"] = seed

                response = MultiModalConversation.call(**call_kwargs)

                if hasattr(response, 'status_code') and response.status_code != 200:
                    error_msg = getattr(response, 'message', str(response))
                    raise RuntimeError(f"[JMCAI Qwen] API Error ({response.status_code}): {error_msg}")

                choices = response.output.get("choices", []) if isinstance(response.output, dict) else []
                if not choices and isinstance(response.output, dict):
                    result_url = response.output.get("result_url", "")
                    if result_url:
                        img_resp = requests.get(result_url, timeout=60)
                        img_resp.raise_for_status()
                        all_images_data.append(img_resp.content)

                for choice in choices:
                    msg = choice.get("message", {})
                    msg_content = msg.get("content", [])
                    if isinstance(msg_content, list):
                        for item in msg_content:
                            if isinstance(item, dict):
                                if "image" in item:
                                    img_resp = requests.get(item["image"], timeout=60)
                                    img_resp.raise_for_status()
                                    all_images_data.append(img_resp.content)
                                elif "text" in item:
                                    text_messages.append(item["text"])
                            elif isinstance(item, str):
                                text_messages.append(item)
                    elif isinstance(msg_content, str):
                        text_messages.append(msg_content)

        finally:
            for tf in temp_files:
                try:
                    os.remove(tf)
                except Exception:
                    pass

        text_response = "\n".join(text_messages) if text_messages else ""
        if not all_images_data:
            raise RuntimeError(f"[JMCAI Qwen] No image data returned. Text: {text_response}")

        result_tensor = bytes_to_tensor(all_images_data)
        return (result_tensor, text_response,)


NODE_CLASS_MAPPINGS = {
    "JMCAI_QwenImageGenerate": JMCAI_QwenImageGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JMCAI_QwenImageGenerate": "JMCAI❤ 阿里/通义万相生图",
}
