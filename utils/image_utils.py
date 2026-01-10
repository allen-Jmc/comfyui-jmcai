import io
import base64
import numpy as np
import torch
from PIL import Image
from ..core.logger import jm_log

def tensor_to_pil(image):
    """将ComfyUI张量转换为PIL图像"""
    if image is None:
        return None
    
    # ComfyUI张量是 [B, H, W, C] 或 [H, W, C]，范围 [0, 1]
    if len(image.shape) == 4:  # 批量图像 [B, H, W, C]
        img_np = (image[0] * 255).cpu().numpy().astype(np.uint8)
        pil_image = Image.fromarray(img_np)
    else: # [H, W, C]
        img_np = (image * 255).cpu().numpy().astype(np.uint8)
        pil_image = Image.fromarray(img_np)
    
    return pil_image

def pil_to_tensor(pil_image):
    """将PIL图像转换为ComfyUI张量 [1, H, W, 3]"""
    img_np = np.array(pil_image.convert("RGB"), dtype=np.float32) / 255.0
    # 增加 Batch 维度
    return torch.from_numpy(img_np).unsqueeze(0)

def encode_image_to_base64(pil_image, format="PNG"):
    """将PIL图像进行Base64编码，确认为 PNG 以保持最高质量"""
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"

def process_single_image(image):
    """封装：张量 -> Base64 (DataURI)"""
    if image is None:
        return None
    pil_img = tensor_to_pil(image)
    return encode_image_to_base64(pil_img)

def process_batch_images(images):
    """封装：批量张量 -> Base64 (DataURI) 列表"""
    if images is None:
        return None
    
    base64_list = []
    # 如果是张量且有 Batch 维度
    if hasattr(images, "shape") and len(images.shape) == 4:
        for i in range(images.shape[0]):
            img_np = (images[i] * 255).cpu().numpy().astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            base64_list.append(encode_image_to_base64(pil_img))
    else:
        # 处理单张或列表（如果后续有这种输入）
        base64_list.append(process_single_image(images))
        
    return base64_list

def smart_resize(pil_image, max_width, max_height, method="fit"):
    """
    智能缩放图像逻辑
    - fit: 等比缩放以适应 max_width/max_height (返回比例缩放后的图，尺寸可能不等于目标值)
    - resize: 等比缩放 (同 fit，为了符合 JS 习惯命名)
    - stretch: 拉伸到指定宽高
    - pad: 等比缩放后补黑边到指定宽高 (推荐用于批次)
    - crop: 缩放并裁剪，填满指定宽高 (推荐用于批次)
    """
    w, h = pil_image.size
    
    if method == "stretch":
        return pil_image.resize((max_width, max_height), Image.Resampling.LANCZOS)
    
    if method == "crop":
        # 计算覆盖比例
        scale = max(max_width / w, max_height / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized_img = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 居中裁剪
        left = (new_w - max_width) // 2
        top = (new_h - max_height) // 2
        return resized_img.crop((left, top, left + max_width, top + max_height))

    # 计算等比缩放比例 (fit / resize / pad)
    scale = min(max_width / w, max_height / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    resized_img = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    if method == "fit" or method == "resize":
        return resized_img
    
    if method == "pad":
        # 创建黑色背景板
        new_img = Image.new("RGB", (max_width, max_height), (0, 0, 0))
        # 居中粘贴
        paste_x = (max_width - new_w) // 2
        paste_y = (max_height - new_h) // 2
        new_img.paste(resized_img, (paste_x, paste_y))
        return new_img
        
    return resized_img

def unify_image_sizes(image_list, target_w, target_h, method="fit"):
    """
    统一一组图像的尺寸
    返回 PIL 图像列表
    """
    return [smart_resize(img, target_w, target_h, method) for img in image_list]
