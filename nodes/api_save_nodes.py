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
    Save API-generated images to ComfyUI output directory.

    Saves images with timestamps and optional prefix.
    Compatible with ComfyUI's image preview system.
    """

    CATEGORY = "JMCAI❤/图像工具"
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    RETURN_TYPES = ()  # Output node, no return types needed

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {
                    "default": "JMCAI_API",
                    "placeholder": "Filename prefix"
                }),
            },
            "optional": {
                "output_dir": ("STRING", {
                    "default": "",
                    "placeholder": "Custom output directory (leave empty for default)"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def save_images(self, images, filename_prefix="JMCAI_API",
                    output_dir="", prompt=None, extra_pnginfo=None):
        """
        Save images to output directory with metadata.

        Args:
            images: ComfyUI IMAGE tensor [B, H, W, C]
            filename_prefix: prefix for saved filenames
            output_dir: custom output directory (uses ComfyUI default if empty)
            prompt: ComfyUI hidden prompt data
            extra_pnginfo: ComfyUI hidden PNG metadata
        """
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
            logger.info(f"[JMCAI Save] Image saved: {filepath}")

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
