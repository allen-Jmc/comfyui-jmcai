import torch
import numpy as np
import folder_paths
import random
import os
from ..utils.image_utils import tensor_to_pil, pil_to_tensor, unify_image_sizes
from ..core.logger import jm_log

class JmcAI_ImageBatch_Multi:
    """
    图像组合批次 (多重)
    支持动态数量的图像输入，并进行规格化处理（缩放+对齐）
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "输入图像数量": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1}),
                "最大宽度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "最大高度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "缩放模式": (["pad", "crop", "fit", "resize", "stretch"], {"default": "pad"}),
                "采样方法": (["lanczos", "nearest-exact", "bilinear", "area", "bicubic"], {"default": "lanczos"}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "combine_images"
    CATEGORY = "JMCAI❤/图像工具"

    def combine_images(self, 输入图像数量, 最大宽度, 最大高度, 缩放模式, 采样方法, **kwargs):
        input_count = 输入图像数量
        max_width = 最大宽度
        max_height = 最大高度
        method = 缩放模式
        upscale_method = 采样方法
        
        images = []
        
        # 1. 按照顺序收集输入的图像
        for i in range(1, input_count + 1):
            key = f"image_{i}"
            if key in kwargs and kwargs[key] is not None:
                images.append(kwargs[key])
        
        if not images:
            # 返回一个黑块占位
            jm_log("WARNING", "ImageBatchMulti", "没有接收到任何图像输入")
            return (torch.zeros((1, max_height, max_width, 3)),)

        # 2. 将 Tensor 转换为 PIL 进行处理
        pil_images = []
        for img_batch in images:
            # 如果输入本身就是批次，则遍历处理
            for i in range(img_batch.shape[0]):
                pil_images.append(tensor_to_pil(img_batch[i:i+1]))

        # 3. 统一尺寸规格
        jm_log("INFO", "ImageBatchMulti", f"正在统一 {len(pil_images)} 张图像的规格为 {max_width}x{max_height}, 模式: {method}")
        unified_pils = unify_image_sizes(pil_images, max_width, max_height, method)
        
        # 3.5 批次对齐检测：ComfyUI 的 Batch 要求所有图像尺寸必须完全一致
        # 如果用户选择了 resize/fit 且图像比例不同，会导致输出尺寸不一，拼接会报错。
        # 此处强制进行二次校准，确保输出张量能够成功 cat。
        first_size = unified_pils[0].size
        consistent = all(p.size == first_size for p in unified_pils)
        
        if not consistent:
            jm_log("WARNING", "ImageBatchMulti", "检测到批次内图像比例不一致，正在强制补齐（Pad）以防止拼接错误")
            from ..utils.image_utils import smart_resize
            aligned_pils = []
            for p in unified_pils:
                if p.size != (max_width, max_height):
                    aligned_pils.append(smart_resize(p, max_width, max_height, method="pad"))
                else:
                    aligned_pils.append(p)
            unified_pils = aligned_pils

        # 4. 转换回 Tensor 并合成 Batch
        # pil_to_tensor 返回的是 [1, H, W, 3]
        tensor_list = [pil_to_tensor(p) for p in unified_pils]
        
        # 合并所有图像到一个 Batch [N, H, W, 3]
        final_batch = torch.cat(tensor_list, dim=0)
        
        jm_log("SUCCESS", "ImageBatchMulti", f"图像组合完成，输出批次大小: {final_batch.shape}")
        
        return (final_batch,)

class JmcAI_LocalCropPreprocess:
    """
    局部裁切预处理 (增强版)
    根据遮罩自动裁切图像区域，并进行比例、对齐、缩放以及可视化涂鸦处理。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "画布图像": ("IMAGE",),
                "mask": ("MASK",),
                "裁切范围": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "涂鸦透明度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "涂鸦颜色": ("STRING", {"default": "#000000"}),
                "目标宽度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "目标高度": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 64}),
                "强制正方形": ("BOOLEAN", {"default": True}),
                "对齐倍数": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
            },
            "optional": {
                "回贴边界微调": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BANANACROP_DATA")
    RETURN_NAMES = ("裁切图像", "裁切遮罩", "裁切数据")
    FUNCTION = "preprocess"
    CATEGORY = "JMCAI❤/图像工具"

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return (0, 0, 0)
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def preprocess(self, 画布图像, mask, 裁切范围, 涂鸦透明度, 涂鸦颜色, 目标宽度, 目标高度, 强制正方形, 对齐倍数, 回贴边界微调=0):
        # 内部逻辑更新
        image = 画布图像
        crop_factor = 裁切范围
        overlay_opacity = 涂鸦透明度
        overlay_color = 涂鸦颜色
        padding = 回贴边界微调
        round_to_multiple = 对齐倍数
        
        # ComfyUI Image: [B, H, W, C], Mask: [B, H, W]
        img_h, img_w = image.shape[1], image.shape[2]
        
        # 1. 寻找遮罩边界
        mask_np = mask.cpu().numpy()
        if len(mask_np.shape) == 3:
            mask_np = np.max(mask_np, axis=0)
        
        non_zero = np.nonzero(mask_np)
        if len(non_zero[0]) == 0:
            jm_log("WARNING", "LocalCrop", "未在遮罩中发现有效区域，将返回全图")
            crop_data = {"x1": 0, "y1": 0, "x2": img_w, "y2": img_h, "orig_w": img_w, "orig_h": img_h}
            return (image, mask.unsqueeze(-1) if len(mask.shape)==3 else mask, crop_data)

        # 核心区域
        y_min, y_max = np.min(non_zero[0]), np.max(non_zero[1]) # 注意：此处原逻辑可能有误，应为 non_zero[0] 的 min/max
        y_min, y_max = np.min(non_zero[0]), np.max(non_zero[0])
        x_min, x_max = np.min(non_zero[1]), np.max(non_zero[1])
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # 2. 计算裁切范围
        mask_w = x_max - x_min
        mask_h = y_max - y_min
        
        if 强制正方形:
            # 取长/宽最大值作为基础边长，保证输出为正方形
            side_len = max(mask_w, mask_h)
            target_w = target_h = side_len * crop_factor + padding * 2
            
            # 限制裁切尺寸不能超过原图最小边，防止比例被破坏
            max_allowed = min(img_w, img_h)
            if target_w > max_allowed:
                target_w = target_h = max_allowed
        else:
            # 按照目标宽高比例来确定裁切框
            target_aspect = 目标宽度 / 目标高度
            # 以 mask 的宽或高为基准锁定裁切框大小
            if mask_w / mask_h > target_aspect:
                # 遮罩更宽，按宽拉伸
                target_w = mask_w * crop_factor + padding * 2
                target_h = target_w / target_aspect
            else:
                # 遮罩更高或比例一致，按高拉伸
                target_h = mask_h * crop_factor + padding * 2
                target_w = target_h * target_aspect
        
        # 3. 边界计算与平移对齐
        x1 = int(max(0, center_x - target_w / 2))
        y1 = int(max(0, center_y - target_h / 2))
        x2 = int(min(img_w, x1 + target_w))
        y2 = int(min(img_h, y1 + target_h))
        
        # 关键修正：如果右侧/下方溢出，则反向推算左侧/上方坐标，以维持目标尺寸
        x1 = int(max(0, x2 - target_w))
        y1 = int(max(0, y2 - target_h))
        # 再次确认右侧/下方边界以防极端情况
        x2 = int(min(img_w, x1 + target_w))
        y2 = int(min(img_h, y1 + target_h))
        
        # 4. 对齐 round_to_multiple
        if round_to_multiple > 1:
            cw, ch = x2 - x1, y2 - y1
            new_cw = (cw // round_to_multiple) * round_to_multiple
            new_ch = (ch // round_to_multiple) * round_to_multiple
            # 缩小到倍数（居中）
            diff_w = cw - new_cw
            diff_h = ch - new_ch
            x1 += diff_w // 2
            x2 = x1 + new_cw
            y1 += diff_h // 2
            y2 = y1 + new_ch

        # 5. 执行裁切 (张量操作)
        crop_image = image[:, y1:y2, x1:x2, :]
        if len(mask.shape) == 3:
            crop_mask = mask[:, y1:y2, x1:x2]
        else:
            crop_mask = mask[y1:y2, x1:x2]

        # 6. 最终缩放处理：缩放到用户指定的目标宽度和高度
        if 目标宽度 > 0 and 目标高度 > 0:
            new_h, new_w = 目标高度, 目标宽度
            # 对齐倍数
            if round_to_multiple > 1:
                new_h = (new_h // round_to_multiple) * round_to_multiple
                new_w = (new_w // round_to_multiple) * round_to_multiple
            
            jm_log("INFO", "LocalCrop", f"正在缩放裁切区域至指定尺寸: {new_w}x{new_h}")
            # [B, H, W, C] -> [B, C, H, W]
            tmp_img = crop_image.permute(0, 3, 1, 2)
            tmp_img = torch.nn.functional.interpolate(tmp_img, size=(new_h, new_w), mode='bilinear', align_corners=False)
            crop_image = tmp_img.permute(0, 2, 3, 1).contiguous()
            
            # 同样缩放 MASK [B, H, W] -> [B, 1, H, W]
            tmp_mask = crop_mask.unsqueeze(1) if len(crop_mask.shape) == 3 else crop_mask.unsqueeze(0).unsqueeze(0)
            tmp_mask = torch.nn.functional.interpolate(tmp_mask.float(), size=(new_h, new_w), mode='nearest')
            crop_mask = tmp_mask.squeeze(1) if len(crop_mask.shape) == 3 else tmp_mask.squeeze(0).squeeze(0)

        # 6.5 涂鸦叠加 (Visual Overlay)
        if overlay_opacity > 0:
            rgb = self.hex_to_rgb(overlay_color)
            c_overlay = torch.tensor([rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0], device=crop_image.device).view(1, 1, 1, 3)
            # 确保 mask 维度匹配 [B, H, W] -> [B, H, W, 1]
            m = crop_mask if len(crop_mask.shape) == 3 else crop_mask.unsqueeze(0)
            m = m.unsqueeze(-1).float()
            
            # 只在遮罩非零区域进行融合
            alpha_mask = m * overlay_opacity
            crop_image = crop_image * (1.0 - alpha_mask) + c_overlay * alpha_mask

        crop_data = {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "orig_w": int(img_w),
            "orig_h": int(img_h)
        }

        jm_log("SUCCESS", "LocalCrop", f"裁切完成: 原位({x1},{y1})->({x2},{y2}), 输出尺寸 {crop_image.shape[2]}x{crop_image.shape[1]}")
        
        # 7. 预览处理：保存第一张图到临时目录以在 Node UI 中显示
        temp_dir = folder_paths.get_temp_directory()
        filename = f"jmcai_crop_preview_{random.randint(100000, 999999)}.png"
        full_path = os.path.join(temp_dir, filename)
        
        # 转换为 PIL 并保存
        p_img = tensor_to_pil(crop_image[0:1])
        p_img.save(full_path, format="PNG")
        
        # ComfyUI 标准返回格式: {"ui": {"images": [...]}, "result": (...)}
        return {
            "ui": {
                "images": [
                    {
                        "filename": filename,
                        "subfolder": "",
                        "type": "temp"
                    }
                ]
            },
            "result": (crop_image, crop_mask, crop_data)
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan") # 强制每次运行

class JmcAI_LocalCropPaste:
    """
    局部裁切贴回
    将处理后的图像精确贴回到原图位置
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "原图图像": ("IMAGE",),
                "裁切图像": ("IMAGE",),
                "裁切数据": ("BANANACROP_DATA",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "paste_back"
    CATEGORY = "JMCAI❤/图像工具"

    def paste_back(self, 原图图像, 裁切图像, 裁切数据):
        # original_image: [B, H, W, 3]
        # crop_image: [B, cH, cW, 3]
        original_image = 原图图像
        crop_image = 裁切图像
        crop_data = 裁切数据
        
        output_image = original_image.clone()

        # 如果是预览模式的数据 (有效性检查) 或标记了预览，直接返回原图
        if (裁切图像 is None or 裁切图像.shape[0] == 0 or 
            裁切数据 is None or 裁切数据.get("is_preview", False)):
            jm_log("INFO", "LocalCropPaste", "检测到预览模式或无效输入，跳过贴回操作")
            return (output_image,)
        
        x1, y1, x2, y2 = crop_data["x1"], crop_data["y1"], crop_data["x2"], crop_data["y2"]
        target_w = x2 - x1
        target_h = y2 - y1

        # 检查裁切图是否被缩放过
        batch, ch, cw, cc = crop_image.shape
        if ch != target_h or cw != target_w:
            jm_log("INFO", "LocalCropPaste", f"自动对齐规格: 裁切图({cw}x{ch}) -> 目标区({target_w}x{target_h})")
            # [B, H, W, C] -> [B, C, H, W]
            tmp_img = crop_image.permute(0, 3, 1, 2)
            tmp_img = torch.nn.functional.interpolate(tmp_img, size=(target_h, target_w), mode='bilinear', align_corners=False)
            crop_image = tmp_img.permute(0, 2, 3, 1).contiguous()

        # 贴回 (支持 Batch)
        # 确保像素值范围一致
        b_count = min(output_image.shape[0], crop_image.shape[0])
        
        # 关键修复：确保切片赋值正确
        output_image[:b_count, y1:y1+target_h, x1:x1+target_w, :] = crop_image[:b_count, :target_h, :target_w, :]

        jm_log("SUCCESS", "LocalCropPaste", f"贴回完成，区域覆盖: ({x1}, {y1}) 到 ({x1+target_w}, {y1+target_h})")
        return (output_image,)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "JMCAI_ImageBatch_Multi": JmcAI_ImageBatch_Multi,
    "JMCAI_LocalCropPreprocess": JmcAI_LocalCropPreprocess,
    "JMCAI_LocalCropPaste": JmcAI_LocalCropPaste,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JMCAI_ImageBatch_Multi": "JMCAI❤ 图像组合批次 (多重)",
    "JMCAI_LocalCropPreprocess": "JMCAI❤ 局部裁切预处理",
    "JMCAI_LocalCropPaste": "JMCAI❤ 局部裁切贴回",
}
