import hashlib
import os
import torch
import numpy as np
from PIL import Image, ImageOps

import folder_paths

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}


class JMCSafeLoadImage:
    """
    ComfyUI 安全加载图片节点。
    用于接管前端空图或无效图传入的情况。如果找不到图片或者遇到约定占位符，
    不会抛出致命错误，而是返回一张纯透明（黑底）的安全占位张量，并附带 Boolean 值指示是否真的有一张有效图。
    支持标准文件路径、MaskEditor 产生的带标注路径（如 clipspace-xxx.png [input]）以及空占位符。
    """
    
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
            and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]
        # 将我们和前端约定的暗号显式加入白名单中
        if "EMPTY_REJECT_IN_UI.png" not in files:
            files.append("EMPTY_REJECT_IN_UI.png")
        
        return {"required": {
            "image": (sorted(files), {"image_upload": True})
        }}

    CATEGORY = "JMCAI❤/图像工具"
    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN")
    RETURN_NAMES = ("image", "mask", "has_image")
    FUNCTION = "load_image_safely"

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        """
        接管 ComfyUI 默认的严格枚举检查。
        无论是 MaskEditor 产生的 'clipspace-xxx.png [input]'、前端占位符还是普通文件名，
        均允许通过校验，进入节点内部进行安全加载或优雅降级，防止整个工作流报 'Value not in list'。
        """
        if not image or image == "EMPTY_REJECT_IN_UI.png" or str(image).strip() == "":
            return True
        return True

    @classmethod
    def IS_CHANGED(cls, image):
        """
        动态计算图片哈希以支持缓存侦测。当用户在画布或 MaskEditor 中重新编辑图片时自动触发重绘。
        """
        if not image or image == "EMPTY_REJECT_IN_UI.png" or str(image).strip() == "":
            return ""
        try:
            image_path = folder_paths.get_annotated_filepath(image)
            m = hashlib.sha256()
            with open(image_path, "rb") as f:
                m.update(f.read())
            return m.digest().hex()
        except Exception:
            return float("nan")

    def get_placeholder(self):
        """生成一个极小的透明图张量占位符"""
        # 生成一个 64x64 的全黑图像数组作为占位
        image = Image.new("RGB", (64, 64), color="black")
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        # 对于透明通道的蒙版也用纯黑（代表完全不可见），为了防止形状不匹配，mask用(1, 64, 64)
        mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")
        return (image, mask, False)

    def load_image_safely(self, image):
        # 1. 前端约定好的 Bypass 占位符拦截，或者传了空字符串
        if not image or image == "EMPTY_REJECT_IN_UI.png" or str(image).strip() == "":
            print(f"\033[93m[JMCAI SafeLoadImage] 检测到空输入占位 '{image}', 返回透明图占位和 has_image=False...\033[0m")
            return self.get_placeholder()

        # 2. 正常路径读取逻辑 (支持带 [input] / [temp] 等标注的文件)
        try:
            image_path = folder_paths.get_annotated_filepath(image)
            # 文件不存在时捕获报错
            if not os.path.exists(image_path):
                print(f"\033[91m[JMCAI SafeLoadImage] 无法在物理路径中找到图片 '{image_path}', 已自动劫持处理！\033[0m")
                return self.get_placeholder()
                
            # 正常使用 PIL 读取图像
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            
            # 以下复用了 ComfyUI 原生 LoadImage 的标准化处理
            if 'A' in img.getbands():
                mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - mask
            else:
                mask = np.zeros((img.height, img.width), dtype=np.float32)
                
            image_out = img.convert("RGB")
            image_out = np.array(image_out).astype(np.float32) / 255.0
            image_out = torch.from_numpy(image_out)[None,]
            
            mask_out = torch.from_numpy(mask)[None,]
            
            return (image_out, mask_out, True)

        except Exception as e:
            # 任何其它读取损坏引发的异常处理，绝不让 ComfyUI 断在这里
            print(f"\033[91m[JMCAI Error] 安全图片节点尝试载入文件 '{image}' 时出错: {e} \n已安全降级！\033[0m")
            return self.get_placeholder()


# 将该节点注册到插件内
NODE_CLASS_MAPPINGS = {
    "JMCSafeLoadImage": JMCSafeLoadImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JMCSafeLoadImage": "JmcAI Safe Load Image 🖼️"
}
