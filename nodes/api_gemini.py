"""
JMCAI Gemini Image Generation node for ComfyUI.

Uses the official google-genai SDK for text-to-image and image editing.
Supports Nano Banana Pro (gemini-3-pro-image-preview) and Nano Banana 2 (gemini-2.5-flash-image).
Supports reference images (up to 14), masks, aspect ratio, and multi-turn conversations.
"""
import logging
import io
import base64
from typing import List, Tuple

from .base.api_image_config import get_api_config, get_model_list, BUILTIN_MODELS
from .base.api_image_utils import (
    tensor_to_pil,
    pil_to_tensor,
    bytes_to_tensor,
    mask_to_pil,
    sanitize_url,
    validate_ref_images,
)

logger = logging.getLogger("JMCAI-APIImage")

MODEL_REF_IMAGE_LIMITS = {
    "gemini-2.5-flash-image": (0, 3),
    "gemini-3-pro-image-preview": (0, 14),
}


class JMCAI_GeminiImageGenerate:
    """
    Generate or edit images using Google Gemini API (Nano Banana series).
    Supports text-to-image generation, image editing with up to 14 reference images,
    and mask-based inpainting.
    """

    CATEGORY = "JMCAI❤/API生图 (Gemini · Nano Banana)"
    FUNCTION = "generate"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("图像", "文本响应",)

    @classmethod
    def INPUT_TYPES(cls):
        models = get_model_list("Gemini Native")
        if not models:
            models = BUILTIN_MODELS.get("Gemini Native", ["gemini-3-pro-image-preview", "gemini-2.5-flash-image"])

        saved_config = get_api_config("Gemini Native")
        saved_key = saved_config.get("api_key", "")
        saved_url = saved_config.get("base_url", "")

        return {
            "required": {
                "提示词": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "请输入图片生成提示词（Prompt）..."
                }),
                "API密钥": ("STRING", {
                    "default": saved_key,
                    "placeholder": "AIzaSy... (填写 Google API Key)"
                }),
                "模型名称": (models, {
                    "default": models[0] if models else "gemini-3-pro-image-preview"
                }),
            },
            "optional": {
                "生成张数": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                }),
                "画面比例": ([
                    "Default (默认比例)",
                    "1:1 (方图)",
                    "16:9 (横屏宽屏)",
                    "9:16 (手机竖屏)",
                    "4:3 (标准横屏)",
                    "3:4 (标准竖屏)",
                    "3:2 (摄影横屏)",
                    "2:3 (摄影竖屏)",
                    "21:9 (电影全景)",
                ], {
                    "default": "Default (默认比例)"
                }),
                "分辨率": ([
                    "Default (默认清晰度)",
                    "1K (标准清晰度 1K)",
                    "2K (高清分辨率 2K)",
                    "4K (超高分辨率 4K)",
                ], {
                    "default": "Default (默认清晰度)"
                }),
                "接口地址": ("STRING", {
                    "default": saved_url,
                    "placeholder": "留空为官方默认，支持设置反代/中转代理 URL"
                }),
                "多参考图批次": ("IMAGE",),
                "参考图1": ("IMAGE",),
                "参考图2": ("IMAGE",),
                "参考图3": ("IMAGE",),
                "局部重绘遮罩": ("MASK",),
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
        生成张数=1,
        画面比例="Default (默认比例)",
        分辨率="Default (默认清晰度)",
        接口地址="",
        多参考图批次=None,
        参考图1=None,
        参考图2=None,
        参考图3=None,
        局部重绘遮罩=None,
        自定义模型="",
        随机种子=0,
        **kwargs,
    ):
        import torch

        prompt = kwargs.get("prompt", 提示词)
        api_key = kwargs.get("api_key", API密钥)
        model_name = kwargs.get("model_name", 模型名称)
        num_images = kwargs.get("num_images", 生成张数)
        aspect_ratio = kwargs.get("aspect_ratio", 画面比例)
        resolution = kwargs.get("resolution", 分辨率)
        base_url = kwargs.get("base_url", 接口地址)
        ref_images = kwargs.get("ref_images", 多参考图批次)
        image1 = kwargs.get("image1", 参考图1)
        image2 = kwargs.get("image2", 参考图2)
        image3 = kwargs.get("image3", 参考图3)
        mask = kwargs.get("mask", 局部重绘遮罩)
        custom_model = kwargs.get("custom_model", 自定义模型)
        seed = kwargs.get("seed", 随机种子)

        if not prompt or not prompt.strip():
            raise ValueError("[JMCAI Gemini] Prompt is empty. Please enter a prompt.")

        if not api_key or not api_key.strip():
            raise ValueError("[JMCAI Gemini] API Key is not set.")

        effective_model = custom_model.strip() if custom_model and custom_model.strip() else model_name
        logger.info(
            f"[JMCAI Gemini] Starting generation | Model: {effective_model} | "
            f"AspectRatio: {aspect_ratio} | HasMask: {mask is not None} | Seed: {seed}"
        )

        try:
            from google import genai
            from google.genai import types
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                f"[JMCAI Gemini] 缺少依赖包: {e}。请先运行: pip install google-genai Pillow"
            )

        effective_url = sanitize_url(base_url)

        image_config_kwargs = {}
        clean_res = resolution.split()[0].strip().upper() if resolution else "DEFAULT"
        if clean_res in {"1K", "2K", "4K"}:
            image_config_kwargs["image_size"] = clean_res

        clean_ar = aspect_ratio.split()[0].strip() if aspect_ratio else "Default"
        valid_ars = {"1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"}
        if clean_ar in valid_ars:
            image_config_kwargs["aspect_ratio"] = clean_ar

        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        ]

        gen_config_kwargs = {
            "response_modalities": ["TEXT", "IMAGE"],
            "safety_settings": safety_settings,
        }
        if image_config_kwargs:
            gen_config_kwargs["image_config"] = types.ImageConfig(**image_config_kwargs)

        gen_config_primary = types.GenerateContentConfig(**gen_config_kwargs)
        gen_config_fallback = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            safety_settings=safety_settings,
        )

        extra_img_count = sum(1 for s in [image1, image2, image3] if s is not None)
        validate_ref_images("Gemini", effective_model, ref_images, MODEL_REF_IMAGE_LIMITS, extra_count=extra_img_count)

        max_ref = None
        for known_model, (_, mx) in MODEL_REF_IMAGE_LIMITS.items():
            if known_model == effective_model or known_model in effective_model.lower():
                max_ref = mx
                break

        def _build_contents() -> List:
            contents_local = [prompt]
            all_ref_pils = []
            if ref_images is not None:
                try:
                    all_ref_pils.extend(tensor_to_pil(ref_images))
                except Exception as e:
                    logger.warning(f"[JMCAI Gemini] Failed to process ref_images: {e}")
            for slot_name, slot_val in [("image1", image1), ("image2", image2), ("image3", image3)]:
                if slot_val is not None:
                    try:
                        all_ref_pils.extend(tensor_to_pil(slot_val))
                    except Exception as e:
                        logger.warning(f"[JMCAI Gemini] Failed to process {slot_name}: {e}")

            if all_ref_pils:
                if max_ref is not None and max_ref > 0 and len(all_ref_pils) > max_ref:
                    all_ref_pils = all_ref_pils[:max_ref]

                for idx, pil_img in enumerate(all_ref_pils):
                    if pil_img.mode != "RGB":
                        pil_img = pil_img.convert("RGB")
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG", quality=95)
                    buf.seek(0)
                    final_img = Image.open(buf)
                    final_img.load()
                    contents_local.append(final_img)

            if mask is not None:
                try:
                    mask_pil = mask_to_pil(mask)
                    if mask_pil:
                        contents_local.append(mask_pil)
                except Exception as e:
                    logger.warning(f"[JMCAI Gemini] Failed to process mask: {e}")
            return contents_local

        contents = _build_contents()

        def _build_client(use_custom_endpoint: bool):
            try:
                client_kwargs = {"api_key": api_key.strip()}
                if use_custom_endpoint and effective_url:
                    client_kwargs["http_options"] = {"base_url": effective_url}
                    logger.info(f"[JMCAI Gemini] Using custom endpoint: {effective_url}")
                return genai.Client(**client_kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"[JMCAI Gemini] Failed to create Gemini client: {e}. Please check API key."
                )

        def _extract_parts(response_obj):
            parts_obj = None
            try:
                parts_obj = response_obj.parts
            except Exception:
                pass
            if not parts_obj:
                try:
                    if (
                        hasattr(response_obj, 'candidates')
                        and response_obj.candidates
                        and hasattr(response_obj.candidates[0], 'content')
                        and response_obj.candidates[0].content
                    ):
                        parts_obj = response_obj.candidates[0].content.parts
                except Exception:
                    pass
            return parts_obj

        def _extract_block_reason(response_obj) -> str:
            reasons = []
            if hasattr(response_obj, "prompt_feedback") and response_obj.prompt_feedback:
                pf = response_obj.prompt_feedback
                if hasattr(pf, "block_reason") and pf.block_reason:
                    reasons.append(str(pf.block_reason))
            if hasattr(response_obj, "candidates") and response_obj.candidates:
                for cand in response_obj.candidates:
                    if hasattr(cand, "finish_reason") and cand.finish_reason:
                        reasons.append(str(cand.finish_reason))
            return " | ".join(reasons)

        def _parse_response(response_obj) -> Tuple[List[bytes], List[str], str]:
            images_data_local = []
            text_messages_local = []
            parts_obj = _extract_parts(response_obj)
            if parts_obj:
                for part in parts_obj:
                    if hasattr(part, 'thought') and part.thought:
                        continue
                    if part.text is not None:
                        text_messages_local.append(part.text)
                    elif part.inline_data is not None:
                        try:
                            img = part.as_image()
                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            images_data_local.append(buf.getvalue())
                        except Exception as e1:
                            try:
                                raw = part.inline_data.data
                                if isinstance(raw, bytes):
                                    images_data_local.append(raw)
                                elif isinstance(raw, str):
                                    images_data_local.append(base64.b64decode(raw))
                            except Exception as e2:
                                logger.error(f"[JMCAI Gemini] Failed to extract image data: {e2}")
            return images_data_local, text_messages_local, _extract_block_reason(response_obj)

        all_images_data = []
        all_text_messages = []

        for gen_idx in range(num_images):
            use_custom_endpoint = bool(effective_url)
            has_any_ref = ref_images is not None or image1 is not None or image2 is not None or image3 is not None
            attempt_plan = [
                ("official-primary", use_custom_endpoint, contents, gen_config_primary),
            ]
            if has_any_ref:
                attempt_plan.append(
                    ("official-fallback", use_custom_endpoint, contents, gen_config_fallback)
                )

            success = False
            last_err = None
            for plan_name, use_custom, req_contents, req_config in attempt_plan:
                try:
                    client = _build_client(use_custom_endpoint=use_custom)
                    response = client.models.generate_content(
                        model=effective_model,
                        contents=req_contents,
                        config=req_config,
                    )
                    imgs, texts, block_reason = _parse_response(response)
                    if imgs:
                        all_images_data.extend(imgs)
                        all_text_messages.extend(texts)
                        success = True
                        break
                    else:
                        last_err = f"No images returned. Block reason: {block_reason or 'None'}"
                except Exception as e:
                    last_err = str(e)

            if not success and last_err:
                raise RuntimeError(f"[JMCAI Gemini] Generation failed: {last_err}")

        if not all_images_data:
            raise RuntimeError("[JMCAI Gemini] No images were generated.")

        result_tensor = bytes_to_tensor(all_images_data)
        out_text = "\n".join(all_text_messages) if all_text_messages else "Success"
        return (result_tensor, out_text,)


NODE_CLASS_MAPPINGS = {
    "JMCAI_GeminiImageGenerate": JMCAI_GeminiImageGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JMCAI_GeminiImageGenerate": "JMCAI❤ Gemini/Nano Banana 原生生图",
}
