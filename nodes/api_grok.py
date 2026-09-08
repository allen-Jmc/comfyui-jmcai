"""
JMCAI Grok Image Generation node for ComfyUI.

Uses xAI xai_sdk protobuf API directly for text-to-image and multi-image editing.
"""
import os
import base64
import logging
import io

from .base.api_image_config import get_api_config, get_model_list, BUILTIN_MODELS
from .base.api_image_utils import (
    tensor_to_pil,
    mask_to_pil,
    bytes_to_tensor,
    detect_mime,
    sanitize_url,
    validate_ref_images,
)

logger = logging.getLogger("JMCAI-APIImage")

MODEL_REF_IMAGE_LIMITS = {
    "grok-imagine-image-quality": (0, 3),
    "grok-imagine-image": (0, 1),
    "grok-imagine-image-pro": (0, 1),
}

SUPPORTED_ASPECT_RATIOS = [
    "1:1 (正方形)",
    "16:9 (横屏宽屏)",
    "9:16 (手机长屏)",
    "4:3 (标准横屏)",
    "3:4 (标准竖屏)",
    "3:2 (摄影横屏)",
    "2:3 (摄影竖屏)",
    "2:1 (电影宽屏)",
    "1:2 (竖版长屏)",
    "19.5:9 (超宽横屏)",
    "9:19.5 (超长竖屏)",
    "20:9 (全景横屏)",
    "9:20 (全景竖屏)",
    "auto (模型自适应)",
]

ASPECT_RATIO_TO_PB = {
    "1:1":    "IMG_ASPECT_RATIO_1_1",
    "16:9":   "IMG_ASPECT_RATIO_16_9",
    "9:16":   "IMG_ASPECT_RATIO_9_16",
    "4:3":    "IMG_ASPECT_RATIO_4_3",
    "3:4":    "IMG_ASPECT_RATIO_3_4",
    "3:2":    "IMG_ASPECT_RATIO_3_2",
    "2:3":    "IMG_ASPECT_RATIO_2_3",
    "2:1":    "IMG_ASPECT_RATIO_2_1",
    "1:2":    "IMG_ASPECT_RATIO_1_2",
    "19.5:9": "IMG_ASPECT_RATIO_19_5_9",
    "9:19.5": "IMG_ASPECT_RATIO_9_19_5",
    "20:9":   "IMG_ASPECT_RATIO_20_9",
    "9:20":   "IMG_ASPECT_RATIO_9_20",
    "auto":   "IMG_ASPECT_RATIO_AUTO",
}

RESOLUTION_TO_PB = {
    "1k": "IMG_RESOLUTION_1K",
    "2k": "IMG_RESOLUTION_2K",
}

MAX_REF_IMAGE_DIMENSION = 1536
GRPC_MESSAGE_LIMIT = 4194304  # 4MB


class JMCAI_GrokImageGenerate:
    """
    Generate or edit images using xAI Grok Imagine API.
    """

    CATEGORY = "JMCAI❤/API生图 (xAI Grok)"
    FUNCTION = "generate"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        models = get_model_list("Grok API")
        if not models:
            models = BUILTIN_MODELS.get("Grok API", ["grok-imagine-image-quality"])

        saved_config = get_api_config("Grok API")
        saved_key = saved_config.get("api_key", "")
        saved_url = saved_config.get("base_url", "https://api.x.ai")

        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your image generation prompt here..."
                }),
                "api_key": ("STRING", {
                    "default": saved_key,
                    "placeholder": "xai-... (xAI API Key)"
                }),
                "model_name": (models, {
                    "default": models[0] if models else "grok-imagine-image-quality"
                }),
            },
            "optional": {
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                }),
                "aspect_ratio": (SUPPORTED_ASPECT_RATIOS, {
                    "default": "1:1 (正方形)",
                }),
                "resolution": (["1k (标清 1K)", "2k (高清 2K)"], {
                    "default": "1k (标清 1K)",
                }),
                "base_url": ("STRING", {
                    "default": saved_url,
                    "placeholder": "https://api.x.ai (default)"
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
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0x7FFFFFFF,
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @staticmethod
    def _resize_if_needed(pil_img, max_dim, idx=0):
        w, h = pil_img.size
        longest = max(w, h)
        if longest <= max_dim:
            return pil_img, False

        from PIL import Image as PILImage
        scale = max_dim / longest
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
        return resized, True

    @staticmethod
    def _encode_to_data_uri(pil_img, jpeg_quality=85):
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=jpeg_quality)
        img_bytes = buf.getvalue()
        b64_str = base64.b64encode(img_bytes).decode("utf-8")
        uri = f"data:image/jpeg;base64,{b64_str}"
        return uri, len(b64_str)

    @staticmethod
    def _get_aspect_ratio_pb(aspect_ratio_str):
        try:
            from xai_sdk.proto import image_pb2
            clean_ar = (aspect_ratio_str or "").split()[0].strip().lower()
            pb_name = ASPECT_RATIO_TO_PB.get(clean_ar)
            if pb_name is None:
                return None
            ar_field = image_pb2.GenerateImageRequest.DESCRIPTOR.fields_by_name['aspect_ratio']
            enum_type = ar_field.enum_type
            value_desc = enum_type.values_by_name.get(pb_name)
            return value_desc.number if value_desc else None
        except Exception:
            return None

    @staticmethod
    def _get_resolution_pb(resolution_str):
        try:
            from xai_sdk.proto import image_pb2
            clean_res = (resolution_str or "").split()[0].strip().lower()
            pb_name = RESOLUTION_TO_PB.get(clean_res)
            if pb_name is None:
                return None
            res_field = image_pb2.GenerateImageRequest.DESCRIPTOR.fields_by_name['resolution']
            enum_type = res_field.enum_type
            value_desc = enum_type.values_by_name.get(pb_name)
            return value_desc.number if value_desc else None
        except Exception:
            return None

    def generate(self, prompt, api_key, model_name, num_images=1,
                 aspect_ratio="1:1", resolution="1k",
                 base_url="", ref_images=None,
                 image1=None, image2=None, image3=None,
                 mask=None, custom_model="", seed=0):
        if not prompt or not prompt.strip():
            raise ValueError("[JMCAI Grok] Prompt is empty.")
        if not api_key or not api_key.strip():
            raise ValueError("[JMCAI Grok] API Key is not set.")

        effective_model = custom_model.strip() if custom_model and custom_model.strip() else model_name

        try:
            import xai_sdk
            from xai_sdk.proto import image_pb2
            from xai_sdk.image import convert_image_format_to_pb
        except ImportError as e:
            raise ImportError(
                f"[JMCAI Grok] 缺少依赖包: {e}。请先运行: pip install xai_sdk"
            )

        os.environ["XAI_API_KEY"] = api_key.strip()
        effective_url = sanitize_url(base_url)
        if effective_url:
            os.environ["XAI_API_BASE"] = effective_url
        elif "XAI_API_BASE" in os.environ:
            del os.environ["XAI_API_BASE"]

        extra_img_count = sum(1 for s in [image1, image2, image3] if s is not None)
        validate_ref_images("Grok", effective_model, ref_images, MODEL_REF_IMAGE_LIMITS, extra_count=extra_img_count)

        all_ref_pils = []
        if ref_images is not None:
            try:
                all_ref_pils.extend(tensor_to_pil(ref_images))
            except Exception as e:
                logger.warning(f"[JMCAI Grok] Failed to process ref_images: {e}")
        for slot_name, slot_val in [("image1", image1), ("image2", image2), ("image3", image3)]:
            if slot_val is not None:
                try:
                    all_ref_pils.extend(tensor_to_pil(slot_val))
                except Exception as e:
                    logger.warning(f"[JMCAI Grok] Failed to process {slot_name}: {e}")

        model_limits = MODEL_REF_IMAGE_LIMITS.get(effective_model, (0, 3))
        max_refs = model_limits[1]
        if len(all_ref_pils) > max_refs:
            all_ref_pils = all_ref_pils[:max_refs]

        reference_uris = []
        jpeg_quality = 85 if len(all_ref_pils) > 1 else 90

        for idx, ref_img in enumerate(all_ref_pils):
            try:
                ref_img, was_resized = self._resize_if_needed(ref_img, MAX_REF_IMAGE_DIMENSION, idx)
                uri, b64_len = self._encode_to_data_uri(ref_img, jpeg_quality)
                reference_uris.append(uri)
            except Exception as e:
                logger.warning(f"[JMCAI Grok] Failed to process reference image {idx + 1}: {e}")

        has_reference = len(reference_uris) > 0
        is_multi_image = len(reference_uris) > 1

        mask_uri = None
        if mask is not None and has_reference:
            try:
                mask_pil = mask_to_pil(mask)
                buf = io.BytesIO()
                mask_pil.save(buf, format="PNG")
                mask_bytes = buf.getvalue()
                b64_mask = base64.b64encode(mask_bytes).decode("utf-8")
                mask_uri = f"data:image/png;base64,{b64_mask}"
            except Exception as e:
                logger.warning(f"[JMCAI Grok] Failed to process mask: {e}")

        try:
            client = xai_sdk.Client()
            request_kwargs = {
                "prompt": prompt,
                "model": effective_model,
                "n": num_images,
                "format": convert_image_format_to_pb("base64"),
            }

            ar_pb = self._get_aspect_ratio_pb(aspect_ratio)
            if ar_pb is not None:
                request_kwargs["aspect_ratio"] = ar_pb

            res_pb = self._get_resolution_pb(resolution)
            if res_pb is not None:
                request_kwargs["resolution"] = res_pb

            if has_reference:
                if is_multi_image:
                    image_contents = [
                        image_pb2.ImageUrlContent(image_url=uri) for uri in reference_uris
                    ]
                    request_kwargs["images"] = image_contents
                else:
                    image_content = image_pb2.ImageUrlContent(image_url=reference_uris[0])
                    request_kwargs["image"] = image_content
                    if mask_uri:
                        mask_content = image_pb2.ImageUrlContent(image_url=mask_uri)
                        request_kwargs["mask"] = mask_content

            request = image_pb2.GenerateImageRequest(**request_kwargs)
            response_pb = client.image._stub.GenerateImage(request)

            from xai_sdk.sync.image import ImageResponse as SdkImageResponse
            images_data = []
            actual_count = len(response_pb.images)

            for i in range(min(num_images, actual_count)):
                try:
                    img_resp = SdkImageResponse(response_pb, i)
                    images_data.append(img_resp.image)
                except Exception as e:
                    logger.error(f"[JMCAI Grok] Failed to extract image {i}: {e}")

        except Exception as e:
            error_str = str(e)
            error_lower = error_str.lower()
            if "401" in error_str or "UNAUTHENTICATED" in error_str:
                raise RuntimeError("[JMCAI Grok] Authentication failed. Check your xAI API key.")
            elif "message length too large" in error_lower or "message_length" in error_lower:
                raise RuntimeError("[JMCAI Grok] Request payload too large (gRPC 4MB limit). Use smaller reference images.")
            elif "rate_limit" in error_lower or "resource_exhausted" in error_lower:
                raise RuntimeError("[JMCAI Grok] Rate limit exceeded. Please wait and retry.")
            else:
                raise RuntimeError(f"[JMCAI Grok] API Error: {error_str}")

        if not images_data:
            raise RuntimeError("[JMCAI Grok] No images returned from the API.")

        result_tensor = bytes_to_tensor(images_data)
        return (result_tensor,)


NODE_CLASS_MAPPINGS = {
    "JMCAI_GrokImageGenerate": JMCAI_GrokImageGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JMCAI_GrokImageGenerate": "JMCAI❤ xAI Grok 生图",
}
