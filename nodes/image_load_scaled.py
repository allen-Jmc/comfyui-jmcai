"""
JMCAI 加载图片并按总像素缩放。

把 ComfyUI 官方 Load Image 与 Scale Image to Total Pixels 的常用组合
合并为一个节点，同时保留图片批次、透明通道遮罩和文件缓存检测。
"""

import hashlib
import math
import os

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths

try:
    import node_helpers
except Exception:
    node_helpers = None


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


def _list_input_images():
    input_dir = folder_paths.get_input_directory()
    files = [
        name
        for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name))
    ]

    filter_files = getattr(folder_paths, "filter_files_content_types", None)
    if callable(filter_files):
        try:
            files = filter_files(files, ["image"])
        except Exception:
            files = [
                name
                for name in files
                if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
            ]
    else:
        files = [
            name
            for name in files
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
        ]

    return sorted(files)


def _calculate_target_size(width, height, megapixels, resolution_steps):
    total_pixels = max(float(megapixels), 0.01) * 1024 * 1024
    scale_by = math.sqrt(total_pixels / max(width * height, 1))
    steps = max(int(resolution_steps), 1)

    target_width = max(
        steps,
        int(round(width * scale_by / steps)) * steps,
    )
    target_height = max(
        steps,
        int(round(height * scale_by / steps)) * steps,
    )
    return target_width, target_height


class JMCAI_LoadImageScaled:
    upscale_methods = ["lanczos", "bicubic", "bilinear", "area", "nearest-exact"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    _list_input_images(),
                    {
                        "image_upload": True,
                        "display_name": "图像",
                        "tooltip": "选择或上传输入图片，支持 PNG/JPG/WebP/GIF 等格式。",
                    },
                ),
                "缩放方法": (
                    cls.upscale_methods,
                    {
                        "default": "lanczos",
                        "tooltip": "lanczos 画质最好；nearest-exact 与原版缩放节点默认值一致。",
                    },
                ),
                "百万像素": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 16.0,
                        "step": 0.01,
                        "tooltip": "目标总像素，1.0 表示约 1024×1024 像素。",
                    },
                ),
                "对齐倍数": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "advanced": True,
                        "tooltip": "把输出宽高对齐到该倍数；默认 1 不额外对齐。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("图像", "遮罩")
    FUNCTION = "load_and_scale"
    CATEGORY = "JMCAI❤/图像工具"
    DESCRIPTION = "加载图片并按目标总像素等比缩放，替代 Load Image + Scale Image to Total Pixels 的常用组合。"
    SEARCH_ALIASES = [
        "load image",
        "scale image",
        "total pixels",
        "加载图片",
        "总像素缩放",
    ]

    @classmethod
    def VALIDATE_INPUTS(cls, image, **kwargs):
        if not image:
            return "请选择一张输入图片。"
        try:
            image_path = folder_paths.get_annotated_filepath(image)
        except Exception:
            return f"无效的图片路径: {image}"
        if not os.path.isfile(image_path):
            return f"图片不存在: {image}"
        return True

    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        try:
            image_path = folder_paths.get_annotated_filepath(image)
            with open(image_path, "rb") as file:
                return hashlib.sha256(file.read()).hexdigest()
        except Exception:
            return float("nan")

    def load_and_scale(self, image, 缩放方法, 百万像素, 对齐倍数=1):
        import comfy.utils as comfy_utils

        image_path = folder_paths.get_annotated_filepath(image)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"找不到输入图片: {image_path}")

        if node_helpers is not None and hasattr(node_helpers, "pillow"):
            source = node_helpers.pillow(Image.open, image_path)
        else:
            source = Image.open(image_path)
        images = []
        masks = []
        source_size = None

        for frame in ImageSequence.Iterator(source):
            frame = ImageOps.exif_transpose(frame)
            rgb_frame = frame.convert("RGB")

            if source_size is None:
                source_size = rgb_frame.size
            elif rgb_frame.size != source_size:
                continue

            image_array = np.array(rgb_frame, dtype=np.float32, copy=True) / 255.0
            images.append(torch.from_numpy(image_array).unsqueeze(0))

            if "A" in frame.getbands():
                alpha = (
                    np.array(frame.getchannel("A"), dtype=np.float32, copy=True)
                    / 255.0
                )
                frame_mask = 1.0 - torch.from_numpy(alpha)
            else:
                frame_mask = torch.zeros(
                    (rgb_frame.height, rgb_frame.width),
                    dtype=torch.float32,
                )
            masks.append(frame_mask.unsqueeze(0))

        if not images:
            raise ValueError(f"图片中没有可读取的帧: {image_path}")

        image_batch = torch.cat(images, dim=0)
        mask_batch = torch.cat(masks, dim=0)

        samples = image_batch.movedim(-1, 1)
        target_width, target_height = _calculate_target_size(
            samples.shape[3],
            samples.shape[2],
            百万像素,
            对齐倍数,
        )

        scaled = comfy_utils.common_upscale(
            samples,
            target_width,
            target_height,
            缩放方法,
            "disabled",
        ).movedim(1, -1)

        scaled_mask = torch.nn.functional.interpolate(
            mask_batch.unsqueeze(1),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).clamp(0.0, 1.0)

        return (scaled, scaled_mask)


NODE_CLASS_MAPPINGS = {
    "JMCAI_LoadImageScaled": JMCAI_LoadImageScaled,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JMCAI_LoadImageScaled": "JMCAI❤ 加载图片（总像素缩放）",
}
