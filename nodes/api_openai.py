"""
JMCAI OpenAI Compatible Image Generation node for ComfyUI.

Uses HTTP REST API to generate images via GPT Image 2, legacy DALL-E, or any
OpenAI-compatible endpoint (including Gemini Nano Banana proxied via OpenAI format).
Supports custom base_url for third-party providers with auto-sanitization.

Reference: https://developers.openai.com/api/reference/resources/images
- POST /v1/images/generations
- POST /v1/images/edits
- Response: data[].b64_json or data[].url
"""
import base64
import io
import logging
import re
import requests

from .base.api_image_config import get_api_config, get_model_list, BUILTIN_MODELS
from .base.api_image_utils import (
    bytes_to_tensor,
    mask_to_pil,
    sanitize_url,
    tensor_to_pil,
    validate_ref_images,
)

logger = logging.getLogger("JMCAI-APIImage")

MODEL_REF_IMAGE_LIMITS = {
    "dall-e-3": (0, 0),
    "gpt-image-2": (0, 16),
    "gpt-image-1.5": (0, 16),
    "gpt-image-1": (0, 16),
    "gpt-image-1-mini": (0, 16),
    "chatgpt-image-latest": (0, 16),
    "gemini-3-pro-image-preview": (0, 14),
    "gemini-2.5-flash-image": (0, 3),
    "dall-e-2": (0, 1),
}

GPT_IMAGE_2_MIN_PIXELS = 655_360
GPT_IMAGE_2_MAX_PIXELS = 8_294_400
GPT_IMAGE_2_MAX_EDGE = 3_840


def _is_gpt_image_model(model_name):
    """Return whether a model uses the GPT Image request contract."""
    normalized = (model_name or "").strip().lower()
    return normalized.startswith("gpt-image-") or normalized == "chatgpt-image-latest"


def _is_gpt_image_2(model_name):
    """Return whether a model is GPT Image 2 or one of its snapshots."""
    normalized = (model_name or "").strip().lower()
    return normalized == "gpt-image-2" or normalized.startswith("gpt-image-2-")


def _calibrate_and_validate_size(model_name, raw_size):
    """
    Intelligently validate and calibrate requested size according to the specific model's requirements:
    - GPT-Image-2: auto, or WIDTHxHEIGHT (edges div by 16, <= 3840, aspect <= 3:1, pixels in 655,360-8,294,400)
    - DALL-E 3: strictly 1024x1024, 1024x1792 (portrait), 1792x1024 (landscape)
    - DALL-E 2: strictly 1024x1024, 512x512, 256x256
    - Proxies / other: auto-aligned
    """
    if not raw_size:
        return "1024x1024"

    clean_size = raw_size.split()[0].strip().lower().replace("*", "x")
    normalized_model = (model_name or "").strip().lower()

    # Case 1: DALL-E 3
    if "dall-e-3" in normalized_model:
        allowed_dalle3 = {"1024x1024", "1024x1792", "1792x1024"}
        if clean_size in allowed_dalle3:
            return clean_size

        match = re.fullmatch(r"(\d+)x(\d+)", clean_size)
        if match:
            w, h = int(match.group(1)), int(match.group(2))
            if w > h * 1.2:
                calibrated = "1792x1024"
            elif h > w * 1.2:
                calibrated = "1024x1792"
            else:
                calibrated = "1024x1024"
        else:
            calibrated = "1024x1024"
        logger.info(f"[JMCAI OpenAI] DALL-E 3 仅支持 1024x1024, 1792x1024, 1024x1792，已自动校对为: {calibrated}")
        return calibrated

    # Case 2: DALL-E 2
    if "dall-e-2" in normalized_model:
        allowed_dalle2 = {"1024x1024", "512x512", "256x256"}
        if clean_size in allowed_dalle2:
            return clean_size
        logger.info(f"[JMCAI OpenAI] DALL-E 2 仅支持 1024x1024, 512x512, 256x256，已自动校对为: 1024x1024")
        return "1024x1024"

    # Case 3: GPT-Image-2 / GPT-Image-1
    if _is_gpt_image_model(normalized_model):
        if clean_size == "auto":
            return "auto"

        match = re.fullmatch(r"(\d+)x(\d+)", clean_size)
        if not match:
            logger.warning(f"[JMCAI OpenAI] 无法识别尺寸 '{clean_size}'，已安全回退为 1024x1024")
            return "1024x1024"

        w, h = int(match.group(1)), int(match.group(2))

        # 1. 宽和高对齐为 16 的倍数
        new_w = max(16, round(w / 16) * 16)
        new_h = max(16, round(h / 16) * 16)

        # 2. 长边不超过 3840
        if max(new_w, new_h) > GPT_IMAGE_2_MAX_EDGE:
            scale = GPT_IMAGE_2_MAX_EDGE / max(new_w, new_h)
            new_w = max(16, int(round(new_w * scale / 16)) * 16)
            new_h = max(16, int(round(new_h * scale / 16)) * 16)

        # 3. 长短边比例不超过 3:1
        if max(new_w, new_h) / min(new_w, new_h) > 3.0:
            if new_w > new_h:
                new_h = max(16, round((new_w / 3.0) / 16) * 16)
            else:
                new_w = max(16, round((new_h / 3.0) / 16) * 16)

        # 4. 像素范围：655,360 ~ 8,294,400
        pixels = new_w * new_h
        if pixels < GPT_IMAGE_2_MIN_PIXELS:
            scale = (GPT_IMAGE_2_MIN_PIXELS / pixels) ** 0.5
            new_w = max(16, int(round(new_w * scale / 16)) * 16)
            new_h = max(16, int(round(new_h * scale / 16)) * 16)
            while new_w * new_h < GPT_IMAGE_2_MIN_PIXELS:
                new_w += 16
                new_h += 16
        elif pixels > GPT_IMAGE_2_MAX_PIXELS:
            scale = (GPT_IMAGE_2_MAX_PIXELS / pixels) ** 0.5
            new_w = max(16, int(round(new_w * scale / 16)) * 16)
            new_h = max(16, int(round(new_h * scale / 16)) * 16)
            while new_w * new_h > GPT_IMAGE_2_MAX_PIXELS:
                if new_w >= new_h:
                    new_w -= 16
                else:
                    new_h -= 16

        calibrated = f"{new_w}x{new_h}"
        if calibrated != clean_size:
            logger.info(f"[JMCAI OpenAI] GPT-Image-2 尺寸规则校对: '{clean_size}' -> '{calibrated}'")
        return calibrated

    # Case 4: 中转站 / 第三方代理模型
    if clean_size == "auto":
        return "1024x1024"
    return clean_size


def _collect_reference_images(ref_images, image1, image2, image3):
    """Collect all ComfyUI image inputs while preserving their original sizes."""
    images = []
    for tensor in (ref_images, image1, image2, image3):
        if tensor is not None:
            images.extend(tensor_to_pil(tensor))
    return images


def _model_ref_limit(model_name):
    """Resolve a known reference-image limit using the shared matching rules."""
    normalized = (model_name or "").lower()
    for known_model, bounds in MODEL_REF_IMAGE_LIMITS.items():
        if known_model == normalized or known_model in normalized:
            return bounds
    return None


def _add_request_options(
    payload,
    model_name,
    quality,
    output_format,
    output_compression,
    background,
    moderation,
):
    """Add only parameters supported by the selected OpenAI model family."""
    norm = (model_name or "").lower()
    clean_quality = quality.split()[0].lower() if quality else "auto"

    if _is_gpt_image_model(model_name):
        payload["output_format"] = output_format
        if clean_quality not in {"auto", ""}:
            if clean_quality in {"hd", "high"}:
                payload["quality"] = "high"
            elif clean_quality in {"standard", "medium"}:
                payload["quality"] = "medium"
            elif clean_quality == "low":
                payload["quality"] = "low"
        if output_format in {"jpeg", "webp"} and output_compression != 100:
            payload["output_compression"] = output_compression
        if background != "auto":
            payload["background"] = background
        if moderation != "auto":
            payload["moderation"] = moderation
        return

    # DALL-E 3: standard or hd
    if "dall-e-3" in norm:
        payload["response_format"] = "b64_json"
        if clean_quality in {"high", "hd", "auto"}:
            payload["quality"] = "hd"
        else:
            payload["quality"] = "standard"
        return

    # Third-party proxy models
    payload["response_format"] = "b64_json"
    if clean_quality not in {"auto", ""}:
        if clean_quality in {"hd", "high"}:
            payload["quality"] = "hd"
        elif clean_quality in {"standard", "medium"}:
            payload["quality"] = "standard"


class JMCAI_OpenAIImageGenerate:
    """
    Generate or edit images using GPT Image 2, legacy DALL-E, Nano Banana, or compatible APIs.
    Supports custom base_url with auto /v1 trimming.
    """

    CATEGORY = "JMCAI❤/API生图 (OpenAI · GPT-Image-2)"
    FUNCTION = "generate"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        models = get_model_list("OpenAI Compatible")
        if not models:
            models = BUILTIN_MODELS.get("OpenAI Compatible", ["gpt-image-2"])

        saved_config = get_api_config("OpenAI Compatible")
        saved_key = saved_config.get("api_key", "")
        saved_url = saved_config.get("base_url", "https://api.openai.com")
        saved_model = saved_config.get("model_name", "")
        default_model = saved_model if saved_model in models else models[0]

        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your image generation prompt here..."
                }),
                "api_key": ("STRING", {
                    "default": saved_key,
                    "placeholder": "sk-... (OpenAI / Third-party API Key)"
                }),
                "base_url": ("STRING", {
                    "default": saved_url,
                    "placeholder": "https://api.openai.com (Auto trims trailing /v1)"
                }),
                "model_name": (models, {
                    "default": default_model if models else "gpt-image-2"
                }),
                "size": ([
                    "auto (GPT-Image 自动最佳)",
                    "1024x1024 (通用 · 1:1 方图)",
                    "1536x1024 (GPT-Image · 3:2 横屏)",
                    "1024x1536 (GPT-Image · 2:3 竖屏)",
                    "2048x1152 (GPT-Image 2K · 16:9 横屏)",
                    "1152x2048 (GPT-Image 2K · 9:16 竖屏)",
                    "2048x2048 (GPT-Image 2K · 1:1 高清)",
                    "3840x2160 (GPT-Image 4K · 16:9 超清横屏)",
                    "2160x3840 (GPT-Image 4K · 9:16 超清长图)",
                    "1792x1024 (DALL-E 3 · 16:9 横屏)",
                    "1024x1792 (DALL-E 3 · 9:16 竖屏)",
                    "512x512 (DALL-E 2 / 兼容)",
                    "256x256 (DALL-E 2 / 兼容)",
                ], {
                    "default": "1024x1024 (通用 · 1:1 方图)"
                }),
            },
            "optional": {
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                }),
                "quality": (["auto", "high (高清/HD)", "medium (标准/Standard)", "low (极速/Low)"], {
                    "default": "auto"
                }),
                "ref_images": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "mask": ("MASK",),
                "custom_model": ("STRING", {
                    "default": "",
                    "placeholder": "Leave empty to use dropdown; fill to override"
                }),
                "custom_size": ("STRING", {
                    "default": "",
                    "placeholder": "Size override, e.g. 1536x864"
                }),
                "output_format": (["png", "jpeg", "webp"], {
                    "default": "png"
                }),
                "output_compression": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 100,
                }),
                "background": (["auto", "opaque", "transparent"], {
                    "default": "auto"
                }),
                "moderation": (["auto", "low"], {
                    "default": "auto"
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def generate(
        self,
        prompt,
        api_key,
        base_url,
        model_name,
        size,
        num_images=1,
        quality="auto",
        ref_images=None,
        image1=None,
        image2=None,
        image3=None,
        mask=None,
        custom_model="",
        custom_size="",
        output_format="png",
        output_compression=100,
        background="auto",
        moderation="auto",
    ):
        if not prompt or not prompt.strip():
            raise ValueError("[JMCAI OpenAI] Prompt is empty.")
        if not api_key or not api_key.strip():
            raise ValueError("[JMCAI OpenAI] API Key is not set.")

        effective_model = custom_model.strip() if custom_model and custom_model.strip() else model_name
        raw_size = custom_size.strip() if custom_size and custom_size.strip() else size
        effective_size = _calibrate_and_validate_size(effective_model, raw_size)

        # Strip trailing /v1 automatically to prevent 404
        clean_url = sanitize_url(base_url, strip_v1=True) or "https://api.openai.com"

        if not effective_model or not effective_model.strip():
            raise ValueError("[JMCAI OpenAI] Model name is empty.")
        if num_images < 1 or num_images > 10:
            raise ValueError("[JMCAI OpenAI] num_images must be between 1 and 10.")
        if effective_model.lower() == "dall-e-3" and num_images != 1:
            raise ValueError("[JMCAI OpenAI] DALL-E 3 supports only num_images=1.")

        if _is_gpt_image_2(effective_model):
            if background == "transparent":
                raise ValueError(
                    "[JMCAI OpenAI] GPT-image-2 does not support transparent backgrounds."
                )

        extra_img_count = sum(1 for s in [image1, image2, image3] if s is not None)
        validate_ref_images(
            "OpenAI",
            effective_model,
            ref_images,
            MODEL_REF_IMAGE_LIMITS,
            extra_count=extra_img_count,
        )
        reference_images = _collect_reference_images(
            ref_images,
            image1,
            image2,
            image3,
        )
        ref_limit = _model_ref_limit(effective_model)
        if ref_limit and ref_limit[1] > 0 and len(reference_images) > ref_limit[1]:
            logger.warning(
                f"[JMCAI OpenAI] Model '{effective_model}' supports up to {ref_limit[1]} "
                f"reference images; {len(reference_images)} provided."
            )

        logger.info(
            f"[JMCAI OpenAI] Starting generation | URL: {clean_url} | "
            f"Model: {effective_model} | Size: {effective_size} | NumImages: {num_images}"
        )

        if reference_images:
            # === EDITING MODE: /v1/images/edits with multipart form data ===
            url = f"{clean_url}/v1/images/edits"
            headers = {"Authorization": f"Bearer {api_key.strip()}"}

            buffers = []
            files = []
            image_field = "image[]" if (_is_gpt_image_model(effective_model) or len(reference_images) > 1) else "image"
            for index, image in enumerate(reference_images):
                image_buffer = io.BytesIO()
                image.save(image_buffer, format="PNG")
                image_buffer.seek(0)
                buffers.append(image_buffer)
                files.append(
                    (
                        image_field,
                        (f"image_{index}.png", image_buffer, "image/png"),
                    )
                )

            if mask is not None:
                mask_pil = mask_to_pil(mask)
                if mask_pil.size != reference_images[0].size:
                    raise ValueError(
                        "[JMCAI OpenAI] Mask dimensions must match the first reference image."
                    )

                import numpy as np
                from PIL import Image as PILImage

                mask_array = np.array(mask_pil.convert("RGBA"))
                mask_array[:, :, 3] = 255 - np.array(mask_pil)
                mask_rgba = PILImage.fromarray(mask_array, "RGBA")
                mask_buffer = io.BytesIO()
                mask_rgba.save(mask_buffer, format="PNG")
                mask_buffer.seek(0)
                buffers.append(mask_buffer)
                files.append(("mask", ("mask.png", mask_buffer, "image/png")))

            data = {
                "model": effective_model,
                "prompt": prompt,
                "n": str(num_images),
                "size": effective_size,
            }
            _add_request_options(
                data,
                effective_model,
                quality,
                output_format,
                output_compression,
                background,
                moderation,
            )

            try:
                response = requests.post(url, headers=headers, data=data, files=files, timeout=300)
            except requests.exceptions.Timeout:
                raise RuntimeError(
                    "[JMCAI OpenAI] Editing request timed out after 300 seconds."
                )
            except requests.exceptions.ConnectionError as e:
                raise RuntimeError(
                    f"[JMCAI OpenAI] Connection failed to {clean_url}. Error: {e}"
                )
            except Exception as e:
                raise RuntimeError(f"[JMCAI OpenAI] Editing request failed: {e}")

        else:
            # === GENERATION MODE: /v1/images/generations with JSON ===
            url = f"{clean_url}/v1/images/generations"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key.strip()}"
            }
            payload = {
                "model": effective_model,
                "prompt": prompt,
                "n": num_images,
                "size": effective_size,
            }
            _add_request_options(
                payload,
                effective_model,
                quality,
                output_format,
                output_compression,
                background,
                moderation,
            )

            try:
                response = requests.post(url, headers=headers, json=payload, timeout=300)
            except requests.exceptions.Timeout:
                raise RuntimeError(
                    "[JMCAI OpenAI] Request timed out after 300 seconds."
                )
            except requests.exceptions.ConnectionError as e:
                raise RuntimeError(
                    f"[JMCAI OpenAI] Connection failed to {clean_url}. Error: {e}"
                )
            except Exception as e:
                raise RuntimeError(f"[JMCAI OpenAI] Request failed: {e}")

        # --- Handle HTTP Errors ---
        if response.status_code != 200:
            try:
                error_body = response.json()
                error_msg = error_body.get("error", {}).get("message", response.text[:500])
            except Exception:
                error_msg = response.text[:500]

            if response.status_code == 401:
                raise RuntimeError(
                    f"[JMCAI OpenAI] Authentication failed (401). Please check your API key."
                )
            elif response.status_code == 404:
                raise RuntimeError(
                    f"[JMCAI OpenAI] Endpoint not found (404). Please verify base URL: {clean_url}"
                )
            elif response.status_code == 429:
                raise RuntimeError(
                    f"[JMCAI OpenAI] Rate limit exceeded (429). Detail: {error_msg}"
                )
            else:
                raise RuntimeError(
                    f"[JMCAI OpenAI] API Error ({response.status_code}): {error_msg}"
                )

        # --- Parse Response ---
        try:
            json_response = response.json()
        except Exception:
            raise RuntimeError("[JMCAI OpenAI] Invalid JSON response from server.")

        images_data = []
        data_list = json_response.get("data") or []

        for item in data_list:
            b64_data = item.get("b64_json")
            if b64_data:
                try:
                    images_data.append(base64.b64decode(b64_data))
                except Exception as e:
                    logger.error(f"[JMCAI OpenAI] Failed to decode base64: {e}")
            elif item.get("url"):
                try:
                    img_resp = requests.get(item["url"], timeout=60)
                    img_resp.raise_for_status()
                    images_data.append(img_resp.content)
                except Exception as e:
                    logger.error(f"[JMCAI OpenAI] Failed to download image from URL: {e}")

        if not images_data:
            raise RuntimeError(
                f"[JMCAI OpenAI] No image data returned. Raw response: {str(json_response)[:500]}"
            )

        result_tensor = bytes_to_tensor(images_data)

        logger.info(
            f"[JMCAI OpenAI] Success | Model: {effective_model} | "
            f"Images: {result_tensor.shape[0]} | Size: {result_tensor.shape[1]}x{result_tensor.shape[2]}"
        )

        return (result_tensor,)


NODE_CLASS_MAPPINGS = {
    "JMCAI_OpenAIImageGenerate": JMCAI_OpenAIImageGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JMCAI_OpenAIImageGenerate": "JMCAI❤ OpenAI/GPT-Image-2 生图",
}
