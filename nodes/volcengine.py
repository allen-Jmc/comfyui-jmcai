import json
import requests
import torch
import numpy as np
import base64
import io
from PIL import Image
from .base import VolcengineImageGenerationBase
from ..core.logger import jm_log, log_full_request, log_full_response, mask_key
from ..utils.image_utils import process_single_image, process_batch_images

class VolcengineChat(VolcengineImageGenerationBase):
    """火山引擎对话(Chat) API节点"""
    
    def __init__(self):
        super().__init__()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": "doubao-pro-4k"}),
                "系统提示词": ("STRING", {"default": "", "multiline": True}),
                "提示词": ("STRING", {"default": "", "multiline": True}),
                "采样温度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "最大长度": ("INT", {"default": 1024, "min": 1, "max": 4096}),
            },
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "image_2": ("IMAGE", {"default": None}),
                "image_3": ("IMAGE", {"default": None}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_chat"
    CATEGORY = "JMCAI❤/火山引擎"
    
    def _process_image_content(self, image):
        """处理 ComfyUI 图像并返回 API 需要的 image_url 字典"""
        if image is None:
            return None
        
        # 使用工具类处理并获取 base64 (DataURI)
        data_uri = process_single_image(image)
        if not data_uri:
            return None
            
        return {
            "type": "image_url",
            "image_url": {
                "url": data_uri,
                "detail": "high"
            }
        }

    def generate_chat(self, api_key, model, 提示词, 系统提示词="", 采样温度=0.7, 最大长度=1024, image=None, image_2=None, image_3=None):
        prompt = 提示词
        system_prompt = 系统提示词
        temperature = 采样温度
        max_tokens = 最大长度
        url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        
        user_content = []
        if prompt:
            user_content.append({"type": "text", "text": prompt})
        
        for img in [image, image_2, image_3]:
            img_content = self._process_image_content(img)
            if img_content:
                user_content.append(img_content)
        
        if not user_content:
            jm_log("ERROR", "VolcengineChat", "Prompt不能为空")
            return ("Error: Prompt cannot be empty.",)

        jm_log("INFO", "VolcengineChat", f"准备调用火山对话: 模型={model}, Key={mask_key(api_key)}")
        messages.append({"role": "user", "content": user_content})
        
        data = {
            "model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens
        }
        
        try:
            response = self.post_request(url, headers=headers, data=data)
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                jm_log("SUCCESS", "VolcengineChat", "对话请求完成")
                return (result["choices"][0]["message"]["content"],)
            return ("Error: No response content",)
        except Exception as e:
            jm_log("ERROR", "VolcengineChat", f"请求异常: {str(e)}")
            return (f"Error: {str(e)}",)

class VolcengineTextToImageSingle(VolcengineImageGenerationBase):
    """火山引擎Seedream 4.0/4.5 文生图-生成单张图"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": ""}),
                "提示词": ("STRING", {"default": "星际穿越，黑洞，黑洞里冲出一辆快支离破碎的复古列车，抢视觉冲击力，电影大片，末日既视感，动感，对比色，oc渲染，光线追踪，动态模糊，景深，超 surrealist，深蓝，画面通过细腻的丰富的色彩层次塑造主体与场景，质感真实，暗黑风背景的光影效果营造出氛围，整体兼具艺术幻想感，夸张的广角透视效果，耀光，反射，极致的光影，强引力，吞噬", "multiline": True}),
                "aspect_ratio": (cls.get_aspect_ratio_options(), {"default": "1:1"}),
                "size_level": (cls.get_size_level_options(), {"default": "2k"}),
                "optimize_prompt": (["disabled", "standard", "fast"], {"default": "disabled"}),
                "携带水印": ("BOOLEAN", {"default": False}),
                "sequential_gen": (["disabled", "auto"], {"default": "disabled"}),
                "stream": ("BOOLEAN", {"default": False}),
                "response_format": (["b64_json", "url"], {"default": "url"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("图像", "信息",)
    FUNCTION = "generate_image"
    CATEGORY = "JMCAI❤/火山引擎"
    
    def generate_image(self, api_key, model, 提示词, aspect_ratio, size_level, optimize_prompt="disabled", 携带水印=False, sequential_gen="disabled", stream=False, response_format="url"):
        prompt = 提示词
        watermark = 携带水印
        width, height = self.calculate_size(aspect_ratio, size_level, model)
        jm_log("INFO", "VolcengineTextToImageSingle", f"调用单图文生图: 模型={model}, 尺寸={width}x{height}")
        
        url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        
        data = {
            "model": model, "prompt": prompt, "size": f"{width}x{height}",
            "response_format": response_format, "stream": stream,
            "watermark": watermark,
            "sequential_image_generation": sequential_gen if sequential_gen != "disabled" else None,
            "optimize_prompt_options": {"mode": optimize_prompt} if optimize_prompt != "disabled" else None
        }
        
        data = {k: v for k, v in data.items() if v is not None}
        log_full_request("VolcengineTextToImageSingle", url, data)
        
        try:
            response = self.post_request(url, headers=headers, data=data, stream=True)
            log_full_response("VolcengineTextToImageSingle", response)
            return self.handle_response(response, "VolcengineTextToImageSingle")
        except Exception as e:
            jm_log("ERROR", "VolcengineTextToImageSingle", f"核心链路异常: {str(e)}")
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (placeholder, f"Error: {str(e)}")

class VolcengineTextToImageBatch(VolcengineImageGenerationBase):
    """火山引擎Seedream 4.0/4.5 文生图-生成一组图"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": ""}),
                "提示词": ("STRING", {"default": "生成一组共4张连贯插画，核心为同一庭院一角的四季变迁，以统一风格展现四季独特色彩、元素与氛围", "multiline": True}),
                "aspect_ratio": (cls.get_aspect_ratio_options(), {"default": "1:1"}),
                "size_level": (cls.get_size_level_options(), {"default": "2k"}),
                "optimize_prompt": (["disabled", "standard", "fast"], {"default": "disabled"}),
                "生成数量": ("INT", {"default": 4, "min": 2, "max": 15}),
                "携带水印": ("BOOLEAN", {"default": False}),
                "sequential_gen": (["disabled", "auto"], {"default": "auto"}),
                "stream": ("BOOLEAN", {"default": True}),
                "response_format": (["b64_json", "url"], {"default": "b64_json"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("图像", "信息",)
    FUNCTION = "generate_image"
    CATEGORY = "JMCAI❤/火山引擎"
    
    def generate_image(self, api_key, model, 提示词, aspect_ratio, size_level, optimize_prompt="disabled", 生成数量=4, 携带水印=False, sequential_gen="auto", stream=True, response_format="b64_json"):
        prompt = 提示词
        batch_size = 生成数量
        watermark = 携带水印
        width, height = self.calculate_size(aspect_ratio, size_level, model)
        jm_log("INFO", "VolcengineTextToImageBatch", f"调用批量文生图: 模型={model}, 数量={batch_size}")
        
        url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        
        data = {
            "model": model, "prompt": prompt, "size": f"{width}x{height}",
            "response_format": response_format, "stream": stream,
            "watermark": watermark,
            "sequential_image_generation": "auto",
            "sequential_image_generation_options": {"max_images": batch_size},
            "optimize_prompt_options": {"mode": optimize_prompt} if optimize_prompt != "disabled" else None
        }
        
        data = {k: v for k, v in data.items() if v is not None}
        log_full_request("VolcengineTextToImageBatch", url, data)
        
        try:
            response = self.post_request(url, headers=headers, data=data, stream=True)
            log_full_response("VolcengineTextToImageBatch", response)
            return self.handle_response(response, "VolcengineTextToImageBatch")
        except Exception as e:
            jm_log("ERROR", "VolcengineTextToImageBatch", f"生成失败: {str(e)}")
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (placeholder, f"Error: {str(e)}")

class VolcengineImageToImageSingle(VolcengineImageGenerationBase):
    """火山引擎Seedream 4.0/4.5 图生图-单张图生成单张图"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": ""}),
                "提示词": ("STRING", {"default": "生成狗狗趴在草地上的近景画面", "multiline": True}),
                "image": ("IMAGE", {"default": None}),
                "aspect_ratio": (cls.get_aspect_ratio_options(), {"default": "1:1"}),
                "size_level": (cls.get_size_level_options(), {"default": "2k"}),
                "optimize_prompt": (["disabled", "automotic"],),
                "携带水印": ("BOOLEAN", {"default": False}),
                "sequential_gen": (["disabled", "auto"],),
                "stream": ("BOOLEAN", {"default": False}),
                "response_format": (["b64_json", "url"], {"default": "url"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("图像", "信息",)
    FUNCTION = "generate_image"
    CATEGORY = "JMCAI❤/火山引擎"
    
    def generate_image(self, api_key, model, image, 提示词, aspect_ratio, size_level, optimize_prompt, 携带水印, sequential_gen, stream, response_format):
        prompt = 提示词
        watermark = 携带水印
        if not api_key: return (None, "Error: API Key is missing")
        width, height = self.calculate_size(aspect_ratio, size_level, model)
        jm_log("INFO", "VolcengineImageToImageSingle", f"调用单图图生图: 模型={model}, 尺寸={width}x{height}")
        
        url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        
        data = {
            "model": model, "prompt": prompt, "size": f"{width}x{height}",
            "response_format": response_format, "stream": stream,
            "image": self.process_image(image),
            "watermark": watermark,
            "sequential_image_generation": sequential_gen if sequential_gen != "disabled" else None,
            "optimize_prompt_options": {"mode": optimize_prompt} if optimize_prompt != "disabled" else None
        }
        
        data = {k: v for k, v in data.items() if v is not None}
        log_full_request("VolcengineImageToImageSingle", url, data)
        
        try:
            response = self.post_request(url, headers=headers, data=data, stream=True)
            log_full_response("VolcengineImageToImageSingle", response)
            return self.handle_response(response, "VolcengineImageToImageSingle")
        except Exception as e:
            jm_log("ERROR", "VolcengineImageToImageSingle", f"核心链路异常: {str(e)}")
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (placeholder, f"Error: {str(e)}")

class VolcengineImageToImageBatch(VolcengineImageGenerationBase):
    """火山引擎Seedream 4.0/4.5 图生图-单张图生成一组图"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": ""}),
                "提示词": ("STRING", {"default": "参考这个LOGO，做一套户外运动品牌视觉设计，品牌名称为GREEN，包括包装袋、帽子、纸盒、手环、挂绳等。绿色视觉主色调，趣味、简约现代风格", "multiline": True}),
                "image": ("IMAGE", {"default": None}),
                "aspect_ratio": (cls.get_aspect_ratio_options(), {"default": "1:1"}),
                "size_level": (cls.get_size_level_options(), {"default": "2k"}),
                "optimize_prompt": (["disabled", "standard", "fast"], {"default": "disabled"}),
                "生成数量": ("INT", {"default": 5, "min": 2, "max": 15}),
                "携带水印": ("BOOLEAN", {"default": False}),
                "sequential_gen": (["disabled", "auto"], {"default": "auto"}),
                "stream": ("BOOLEAN", {"default": True}),
                "response_format": (["b64_json", "url"], {"default": "b64_json"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("图像", "信息",)
    FUNCTION = "generate_image"
    CATEGORY = "JMCAI❤/火山引擎"
    
    def generate_image(self, api_key, model, image, 提示词, 生成数量, aspect_ratio, size_level, optimize_prompt, 携带水印, sequential_gen, stream, response_format):
        prompt = 提示词
        batch_size = 生成数量
        watermark = 携带水印
        if not api_key: return (None, "Error: API Key is missing")
        width, height = self.calculate_size(aspect_ratio, size_level, model)
        jm_log("INFO", "VolcengineImageToImageBatch", f"调用单图批量图生图: 模型={model}, 数量={batch_size}")
        
        url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        
        data = {
            "model": model, "prompt": prompt, "size": f"{width}x{height}",
            "response_format": response_format, "stream": stream,
            "image": self.process_image(image),
            "watermark": watermark,
            "sequential_image_generation": "auto",
            "sequential_image_generation_options": {"max_images": batch_size},
            "optimize_prompt_options": {"mode": optimize_prompt} if optimize_prompt != "disabled" else None
        }
        
        data = {k: v for k, v in data.items() if v is not None}
        log_full_request("VolcengineImageToImageBatch", url, data)
        
        try:
            response = self.post_request(url, headers=headers, data=data, stream=True)
            log_full_response("VolcengineImageToImageBatch", response)
            return self.handle_response(response, "VolcengineImageToImageBatch")
        except Exception as e:
            jm_log("ERROR", "VolcengineImageToImageBatch", f"生成失败: {str(e)}")
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (placeholder, f"Error: {str(e)}")

class VolcengineMultiImageFusionSingle(VolcengineImageGenerationBase):
    """火山引擎Seedream 4.0/4.5 图生图-多张参考图生成单张图"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": ""}),
                "提示词": ("STRING", {"default": "将图1的服装换为图2的服装", "multiline": True}),
                "images": ("IMAGE", {"default": None}),
                "aspect_ratio": (cls.get_aspect_ratio_options(), {"default": "1:1"}),
                "size_level": (cls.get_size_level_options(), {"default": "2k"}),
                "optimize_prompt": (["disabled", "standard", "fast"], {"default": "disabled"}),
                "携带水印": ("BOOLEAN", {"default": False}),
                "sequential_gen": (["disabled", "auto"], {"default": "disabled"}),
                "stream": ("BOOLEAN", {"default": False}),
                "response_format": (["b64_json", "url"], {"default": "url"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("图像", "信息",)
    FUNCTION = "generate_image"
    CATEGORY = "JMCAI❤/火山引擎"
    
    def generate_image(self, api_key, model, 提示词, images, aspect_ratio, size_level, optimize_prompt="disabled", 携带水印=False, sequential_gen="disabled", stream=False, response_format="url"):
        prompt = 提示词
        watermark = 携带水印
        width, height = self.calculate_size(aspect_ratio, size_level, model)
        jm_log("INFO", "VolcengineMultiImageFusionSingle", f"调用多图生单图: 模型={model}, 尺寸={width}x{height}")
        
        url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        base64_list = self.process_images(images)
        
        data = {
            "model": model, "prompt": prompt, "size": f"{width}x{height}",
            "response_format": response_format, "stream": stream,
            "image": base64_list,
            "watermark": watermark,
            "sequential_image_generation": sequential_gen if sequential_gen != "disabled" else None,
            "optimize_prompt_options": {"mode": optimize_prompt} if optimize_prompt != "disabled" else None
        }
        
        data = {k: v for k, v in data.items() if v is not None}
        log_full_request("VolcengineMultiImageFusionSingle", url, data)
        
        try:
            response = self.post_request(url, headers=headers, data=data, stream=True)
            log_full_response("VolcengineMultiImageFusionSingle", response)
            return self.handle_response(response, "VolcengineMultiImageFusionSingle")
        except Exception as e:
            jm_log("ERROR", "VolcengineMultiImageFusionSingle", f"核心链路异常: {str(e)}")
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (placeholder, f"Error: {str(e)}")

class VolcengineMultiImageFusionBatch(VolcengineImageGenerationBase):
    """火山引擎Seedream 4.0/4.5 图生图-多张参考图生成一组图"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": ""}),
                "提示词": ("STRING", {"default": "生成3张女孩和奶牛玩偶在游乐园开心地坐过山车的图片，涵盖早晨、中午、晚上", "multiline": True}),
                "images": ("IMAGE", {"default": None}),
                "aspect_ratio": (cls.get_aspect_ratio_options(), {"default": "1:1"}),
                "size_level": (cls.get_size_level_options(), {"default": "2k"}),
                "optimize_prompt": (["disabled", "standard", "fast"], {"default": "disabled"}),
                "生成数量": ("INT", {"default": 3, "min": 2, "max": 15}),
                "携带水印": ("BOOLEAN", {"default": False}),
                "sequential_gen": (["auto", "disabled"],),
                "stream": ("BOOLEAN", {"default": True}),
                "response_format": (["b64_json", "url"], {"default": "b64_json"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("图像", "信息",)
    FUNCTION = "generate_image"
    CATEGORY = "JMCAI❤/火山引擎"
    
    def generate_image(self, api_key, model, 提示词, images, aspect_ratio, size_level, optimize_prompt="disabled", 生成数量=3, 携带水印=False, sequential_gen="auto", stream=True, response_format="b64_json"):
        prompt = 提示词
        batch_size = 生成数量
        watermark = 携带水印
        width, height = self.calculate_size(aspect_ratio, size_level, model)
        jm_log("INFO", "VolcengineMultiImageFusionBatch", f"调用多图生一组图: 模型={model}, 数量={batch_size}")
        
        url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        base64_list = self.process_images(images)
        
        data = {
            "model": model, "prompt": prompt, "size": f"{width}x{height}",
            "response_format": response_format, "stream": stream,
            "image": base64_list,
            "watermark": watermark,
            "sequential_image_generation": "auto",
            "sequential_image_generation_options": {"max_images": batch_size},
            "optimize_prompt_options": {"mode": optimize_prompt} if optimize_prompt != "disabled" else None
        }
        
        data = {k: v for k, v in data.items() if v is not None}
        log_full_request("VolcengineMultiImageFusionBatch", url, data)
        
        try:
            response = self.post_request(url, headers=headers, data=data, stream=True)
            log_full_response("VolcengineMultiImageFusionBatch", response)
            return self.handle_response(response, "VolcengineMultiImageFusionBatch")
        except Exception as e:
            jm_log("ERROR", "VolcengineMultiImageFusionBatch", f"生成失败: {str(e)}")
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (placeholder, f"Error: {str(e)}")
