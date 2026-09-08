"""
Shared utility functions for JMCAI API Image nodes.

Handles conversions between:
- ComfyUI IMAGE tensor [B,H,W,C] float32 0-1
- PIL.Image objects
- Raw image bytes (PNG/JPEG)
- Base64 encoded strings
- Mask tensors
- URL sanitation and validation
"""
import base64
import io
import logging
import re
import numpy as np

logger = logging.getLogger("JMCAI-APIImage")

try:
    import torch
except ImportError:
    torch = None

try:
    from PIL import Image
except ImportError:
    Image = None


def tensor_to_pil(tensor):
    """
    Convert ComfyUI IMAGE tensor [B,H,W,C] to list of PIL Images.
    ComfyUI IMAGE format: torch.Tensor, shape [B, H, W, C], dtype float32, range [0, 1].
    """
    if tensor is None:
        return []
    if Image is None:
        raise ImportError("Pillow is required: pip install Pillow")

    images = []
    if hasattr(tensor, 'cpu'):
        tensor = tensor.cpu().detach()

    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    for i in range(tensor.shape[0]):
        img_np = tensor[i].numpy()
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np, mode="RGB")
        images.append(img)

    return images


def pil_to_tensor(pil_images):
    """
    Convert list of PIL Images to ComfyUI IMAGE tensor [B,H,W,C].
    When images have different sizes, all images are resized to match the
    first image's dimensions using Lanczos resampling.
    """
    if torch is None:
        raise ImportError("PyTorch is required")

    if not isinstance(pil_images, (list, tuple)):
        pil_images = [pil_images]

    if not pil_images:
        raise ValueError("pil_to_tensor received an empty image list")

    tensors = []
    target_size = None
    for idx, img in enumerate(pil_images):
        if img.mode != "RGB":
            img = img.convert("RGB")

        if target_size is None:
            target_size = img.size
        elif img.size != target_size:
            logger.info(
                f"[pil_to_tensor] Resizing image {idx} from {img.size} "
                f"to {target_size} to match batch dimensions"
            )
            img = img.resize(target_size, Image.LANCZOS)

        img_np = np.array(img).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(img_np))

    return torch.stack(tensors, dim=0)


def mask_to_pil(mask_tensor):
    """
    Convert ComfyUI MASK tensor to PIL Image (grayscale).
    ComfyUI MASK format: torch.Tensor, shape [H, W] or [B, H, W].
    """
    if mask_tensor is None:
        return None
    if Image is None:
        raise ImportError("Pillow is required: pip install Pillow")

    if hasattr(mask_tensor, 'cpu'):
        mask_tensor = mask_tensor.cpu().detach()

    if mask_tensor.dim() == 3:
        mask_tensor = mask_tensor[0]

    mask_np = mask_tensor.numpy()
    mask_np = np.clip(mask_np * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(mask_np, mode="L")


def tensor_to_bytes(tensor, format="PNG"):
    """Convert ComfyUI IMAGE tensor to list of image bytes."""
    pil_images = tensor_to_pil(tensor)
    results = []
    for img in pil_images:
        buf = io.BytesIO()
        img.save(buf, format=format)
        results.append(buf.getvalue())
    return results


def tensor_to_base64(tensor, format="PNG"):
    """Convert ComfyUI IMAGE tensor to list of base64 encoded strings."""
    byte_list = tensor_to_bytes(tensor, format=format)
    return [base64.b64encode(b).decode("utf-8") for b in byte_list]


def bytes_to_tensor(image_bytes_list):
    """Convert list of raw image bytes to ComfyUI IMAGE tensor [B,H,W,C]."""
    if Image is None:
        raise ImportError("Pillow is required: pip install Pillow")

    pil_images = []
    for img_bytes in image_bytes_list:
        try:
            img = Image.open(io.BytesIO(img_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            pil_images.append(img)
        except Exception as e:
            logger.error(f"Failed to decode image bytes: {e}")
            continue

    if not pil_images:
        raise ValueError("No valid images could be decoded from the provided bytes")

    return pil_to_tensor(pil_images)


def detect_mime(data):
    """Detect image MIME type from bytes header."""
    if not data or len(data) < 12:
        return "image/png"
    header = bytes(data[:12])
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def sanitize_url(url, strip_v1=False):
    """
    Validate and clean a base_url value.
    Returns cleaned URL if valid (starts with http:// or https://),
    or empty string if invalid.
    If strip_v1=True, strips trailing /v1 or /v1/ to prevent double /v1 path duplication.
    """
    if not url or not isinstance(url, str):
        return ""
    cleaned = url.strip().rstrip("/")
    if not cleaned:
        return ""
    if not cleaned.startswith("http://") and not cleaned.startswith("https://"):
        logger.warning(
            f"[JMCAI API] Invalid base_url '{cleaned}' - must start with "
            f"http:// or https://. Using default endpoint instead."
        )
        return ""
    if strip_v1:
        cleaned = re.sub(r"/v1/?$", "", cleaned).rstrip("/")
    return cleaned


def validate_ref_images(provider, model_name, ref_images, limits_map, extra_count=0):
    """
    Validate reference images against per-model limits.
    """
    ref_count = 0
    if ref_images is not None:
        try:
            ref_pils = tensor_to_pil(ref_images)
            ref_count = len(ref_pils)
        except Exception as e:
            logger.warning(f"[{provider}] Failed to count ref images: {e}")

    ref_count += extra_count

    limit = None
    for known_model, bounds in limits_map.items():
        if known_model == model_name or known_model in model_name.lower():
            limit = bounds
            break

    if limit is None:
        if ref_count > 0:
            logger.warning(
                f"[{provider}] Model '{model_name}' is not in the known model list. "
                f"Reference image support is unknown - proceeding with {ref_count} image(s)."
            )
        return ref_count

    min_required, max_allowed = limit

    if max_allowed == 0 and ref_count > 0:
        supported = [m for m, (mn, mx) in limits_map.items() if mx > 0]
        alt_text = f"Try: {', '.join(supported)}" if supported else "Use a different provider"
        raise ValueError(
            f"[JMCAI API {provider}] Model '{model_name}' is text-to-image only "
            f"and does not support reference images. "
            f"Please disconnect ref_images/image1-3 inputs. {alt_text}."
        )

    if min_required > 0 and ref_count == 0:
        raise ValueError(
            f"[JMCAI API {provider}] Model '{model_name}' is an editing model "
            f"that requires {min_required}-{max_allowed} reference image(s), "
            f"but none were provided."
        )

    if ref_count > max_allowed > 0:
        logger.warning(
            f"[{provider}] Model '{model_name}' supports max {max_allowed} reference image(s), "
            f"but {ref_count} provided. Extra images may be ignored or cause API errors."
        )

    return ref_count
