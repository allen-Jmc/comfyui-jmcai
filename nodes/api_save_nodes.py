"""
Image Save node for JMCAI API Image plugin.

Saves generated images to ComfyUI's output directory with customizable prefix.
Compatible with ComfyUI's image preview and gallery system.
"""
import os
import logging
import json
from datetime import datetime

import numpy as np

from .base.api_image_utils import tensor_to_pil

logger = logging.getLogger("JMCAI-APIImage")

# Try to import ComfyUI's folder_paths for output directory
try:
    import folder_paths
except ImportError:
    folder_paths = None


class JMCAI_APIImageSave:
    """
    保存 API 生成的图像到 ComfyUI 输出目录，并在画布节点中实时显示预览。
    """

    CATEGORY = "JMCAI❤/图像工具"
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    RETURN_TYPES = ()  # Output node, no return types needed

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "文件名保存前缀": ("STRING", {
                    "default": "JMCAI_API",
                    "placeholder": "保存文件名前缀"
                }),
            },
            "optional": {
                "自定义输出目录": ("STRING", {
                    "default": "",
                    "placeholder": "留空默认保存至 ComfyUI output 目录"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def save_images(self, 图像=None, 文件名保存前缀="JMCAI_API",
                    自定义输出目录="", prompt=None, extra_pnginfo=None, **kwargs):
        """
        保存图像至输出目录，同时嵌入 Prompt 与 PNG 元数据信息。
        支持中英文参数无缝兼容。
        """
        images = kwargs.get("images", 图像)
        filename_prefix = kwargs.get("filename_prefix", 文件名保存前缀)
        output_dir = kwargs.get("output_dir", 自定义输出目录)

        if images is None:
            raise ValueError("[JMCAI Save] 没有传入有效图像数据。")

        # Determine output directory
        if output_dir and output_dir.strip():
            out_dir = output_dir.strip()
        elif folder_paths is not None:
            out_dir = folder_paths.get_output_directory()
        else:
            out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

        os.makedirs(out_dir, exist_ok=True)

        # Convert tensor to PIL images
        pil_images = tensor_to_pil(images)

        results = []
        for i, img in enumerate(pil_images):
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}_{i:04d}.png"
            filepath = os.path.join(out_dir, filename)

            # Add metadata if available
            from PIL import PngImagePlugin
            metadata = PngImagePlugin.PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for k, v in extra_pnginfo.items():
                    metadata.add_text(k, json.dumps(v))

            # Save image
            img.save(filepath, pnginfo=metadata)
            logger.info(f"[JMCAI Save] 图像已成功保存至: {filepath}")

            results.append({
                "filename": filename,
                "subfolder": "",
                "type": "output",
            })

        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {
    "JMCAI_APIImageSave": JMCAI_APIImageSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JMCAI_APIImageSave": "JMCAI❤ API图像保存与预览",
}
