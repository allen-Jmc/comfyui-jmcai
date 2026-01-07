"""
ComfyUI-Doubao 节点定义
"""

import json
import requests
import base64
import io
from PIL import Image
import numpy as np
import torch
import time
import datetime
try:
    from volcengine.ark.ArkService import ArkService
except ImportError:
    ArkService = None


class VolcengineChat:
    """火山引擎对话(Chat) API节点"""
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": "doubao-pro-4k"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
            },
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "image_2": ("IMAGE", {"default": None}),
                "image_3": ("IMAGE", {"default": None}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate_chat"
    CATEGORY = "JMCAI❤/火山引擎"
    
    def _process_image_content(self, image):
        """处理 ComfyUI 图像并返回 API 需要的 image_url 字典"""
        if image is None:
            return None
            
        # 将ComfyUI图像格式转换为PIL图像
        if len(image.shape) == 4:  # 批量图像，只取第一张
            img_np = (image[0] * 255).cpu().numpy().astype(np.uint8)
        else:
            img_np = (image * 255).cpu().numpy().astype(np.uint8)
            
        pil_image = Image.fromarray(img_np)
        
        # 将PIL图像转换为Base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG") # 统一转PNG
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}",
                "detail": "high" # 保持高详情，后续可配置
            }
        }

    def generate_chat(self, api_key, model, prompt, system_prompt="", temperature=0.7, max_tokens=1024, image=None, image_2=None, image_3=None):
        url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        messages = []
        
        # 1. 优化：仅在有内容时添加 System Message
        if system_prompt and system_prompt.strip():
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 2. 构建 User Message 内容
        user_content = []
        
        # 2.1 添加文本 (优化：如果 prompt 为空则不添加，但通常 prompt 必填)
        if prompt:
            user_content.append({
                "type": "text",
                "text": prompt
            })
        
        # 2.2 处理图片 list
        input_images = [image, image_2, image_3]
        for img in input_images:
            img_content = self._process_image_content(img)
            if img_content:
                user_content.append(img_content)
        
        # 3. 组装消息
        # 优化：如果是纯文本且没有图片，可以使用简化的字符串 content 格式（虽然 list 格式也兼容）
        # 但为了统一多模态逻辑，保持 list 格式
        
        if not user_content:
            return ("Error: Prompt cannot be empty.",)

        messages.append({
            "role": "user",
            "content": user_content
        })
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return (result["choices"][0]["message"]["content"],)
            else:
                return ("Error: No response content",)
        except Exception as e:
            return (f"Error: {str(e)}",)


class VolcengineImageGenerationBase:
    """火山引擎图片生成API基类"""
    
    @classmethod
    def get_size_options_4_0(cls):
        """Seedream 4.0系列的尺寸选项（统一保留4.0口径）"""
        return [
            "1K",
            "2K",
            "4K",
            "2048×2048 (1:1)",
            "2304×1728 (4:3)",
            "1728×2304 (3:4)",
            "2560×1440 (16:9)",
            "1440×2560 (9:16)",
            "2496×1664 (3:2)",
            "1664×2496 (2:3)",
            "3024×1296 (21:9)",
            "4096×4096 (1:1)",
        ]

    # 3.0系列已移除，不再提供尺寸选项
    def process_image(self, image):
        """处理ComfyUI图像为Base64编码"""
        if image is None:
            return None
            
        # 将ComfyUI图像格式转换为PIL图像
        if len(image.shape) == 4:  # 批量图像
            img_np = (image[0] * 255).cpu().numpy().astype(np.uint8)
            pil_image = Image.fromarray(img_np)
        else:
            img_np = (image * 255).cpu().numpy().astype(np.uint8)
            pil_image = Image.fromarray(img_np)
        
        # 将PIL图像转换为Base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{img_base64}"
    
    def process_images(self, images):
        """处理多个ComfyUI图像为Base64编码列表"""
        if images is None:
            return None
            
        base64_images = []
        
        # 处理批量图像
        for i in range(images.shape[0]):
            img_np = (images[i] * 255).cpu().numpy().astype(np.uint8)
            pil_image = Image.fromarray(img_np)
            
            # 将PIL图像转换为Base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_images.append(f"data:image/png;base64,{img_base64}")
        
        return base64_images
    
    def normalize_size(self, size):
        s = str(size).strip()
        # 去掉括号中的比例说明
        if " (" in s:
            s = s.split(" (", 1)[0]
        # 将乘号×或大写X统一替换为小写x
        s = s.replace("×", "x").replace("X", "x")
        return s
    
    def handle_response(self, response):
        """处理API响应并返回ComfyUI期望的IMAGE张量。
        同时兼容两种返回格式：url 与 b64_json。
        """
        try:
            response.raise_for_status()
            result = response.json()

            images = []
            image_sources = []

            # Ark Images API通常在 result["data"] 中返回图片列表
            data_items = result.get("data", [])

            for item in data_items:
                # b64_json 直接解码
                if "b64_json" in item and item["b64_json"]:
                    try:
                        img_bytes = base64.b64decode(item["b64_json"])
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        img_np = np.array(img, dtype=np.float32) / 255.0
                        images.append(img_np)
                        image_sources.append("b64_json")
                    except Exception as _:
                        pass

                # url 下载
                elif "url" in item and item["url"]:
                    try:
                        img_response = requests.get(item["url"])
                        img_response.raise_for_status()
                        img = Image.open(io.BytesIO(img_response.content)).convert("RGB")
                        img_np = np.array(img, dtype=np.float32) / 255.0
                        images.append(img_np)
                        image_sources.append(item["url"])
                    except Exception as _:
                        pass

            if len(images) == 0:
                # 尝试其它可能的字段，例如 result["images"]
                fallback_images = result.get("images") or result.get("output")
                if isinstance(fallback_images, list):
                    for b64 in fallback_images:
                        try:
                            if isinstance(b64, str):
                                # 兼容可能带有前缀的 base64
                                if b64.startswith("data:image"):
                                    b64 = b64.split(",", 1)[1]
                                img_bytes = base64.b64decode(b64)
                                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                img_np = np.array(img, dtype=np.float32) / 255.0
                                images.append(img_np)
                                image_sources.append("inline_b64")
                        except Exception:
                            pass

            if len(images) > 0:
                batch_np = np.stack(images, axis=0)
                # 转换为torch张量，符合ComfyUI的IMAGE类型
                batch_tensor = torch.from_numpy(batch_np).float()
                info = json.dumps({
                    "count": len(images),
                    "sources": image_sources
                }, ensure_ascii=False)
                return (batch_tensor, info)

            # 没有成功解析出图片，返回一个占位图避免下游节点报错
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            err_msg = json.dumps({
                "error": "No images in response",
                "raw": result
            }, ensure_ascii=False)
            return (placeholder, err_msg)
        except Exception as e:
            # 异常时同样返回占位图，附带错误信息
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (placeholder, f"Error: {str(e)}")


class VolcengineTextToImage(VolcengineImageGenerationBase):
    """火山引擎Seedream 4.0文生图节点"""
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": "doubao-seedream-4-0-250828"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "size": (cls.get_size_options_4_0(), {"default": "2048×2048 (1:1)"}),
                "sequential_image_generation": (["auto", "disabled"], {"default": "disabled"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "watermark": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "info",)
    FUNCTION = "generate_image"
    CATEGORY = "JMCAI❤/火山引擎"
    
    def generate_image(self, api_key, model, prompt, size="2048×2048 (1:1)", sequential_image_generation="disabled", seed=-1, watermark=False, max_images=1):
        size_norm = self.normalize_size(size)
        url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        data = {"model": model, "prompt": prompt, "size": size_norm, "sequential_image_generation": sequential_image_generation, "response_format": "b64_json", "seed": seed, "watermark": watermark}
        if sequential_image_generation == "auto":
            data["sequential_image_generation_options"] = {"max_images": max_images}
        response = requests.post(url, headers=headers, json=data)
        return self.handle_response(response)


class VolcengineImageToImage(VolcengineImageGenerationBase):
    """火山引擎Seedream 4.0图生图节点"""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": "doubao-seedream-4-0-250828"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE", {"default": None}),
                "size": (cls.get_size_options_4_0(), {"default": "2048×2048 (1:1)"}),
                "sequential_image_generation": (["auto", "disabled"], {"default": "disabled"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "watermark": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "info",)
    FUNCTION = "generate_image"
    CATEGORY = "JMCAI❤/火山引擎"
    
    def generate_image(self, api_key, model, prompt, image, size="2048×2048 (1:1)", sequential_image_generation="disabled", seed=-1, watermark=False, max_images=1):
        size_norm = self.normalize_size(size)
        url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        data = {
            "model": model,
            "prompt": prompt,
            "size": size_norm,
            "sequential_image_generation": sequential_image_generation,
            "image": self.process_image(image),
            "response_format": "b64_json",
            "seed": seed,
            "watermark": watermark,
        }
        if sequential_image_generation == "auto":
            data["sequential_image_generation_options"] = {"max_images": max_images}
        response = requests.post(url, headers=headers, json=data)
        return self.handle_response(response)


class VolcengineMultiImageFusion(VolcengineImageGenerationBase):
    """火山引擎Seedream 4.0多图融合节点"""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": "doubao-seedream-4-0-250828"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "images": ("IMAGE", {"default": None}),
                "size": (cls.get_size_options_4_0(), {"default": "2048×2048 (1:1)"}),
                "sequential_image_generation": (["auto", "disabled"], {"default": "disabled"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "watermark": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "info",)
    FUNCTION = "generate_image"
    CATEGORY = "JMCAI❤/火山引擎"
    
    def generate_image(self, api_key, model, prompt, images, size="2048×2048 (1:1)", sequential_image_generation="disabled", seed=-1, watermark=False, max_images=1):
        size_norm = self.normalize_size(size)
        url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        base64_images = self.process_images(images)
        data = {
            "model": model,
            "prompt": prompt,
            "size": size_norm,
            "sequential_image_generation": sequential_image_generation,
            "image": base64_images,
            "response_format": "b64_json",
            "seed": seed,
            "watermark": watermark,
        }
        if sequential_image_generation == "auto":
            data["sequential_image_generation_options"] = {"max_images": max_images}
        response = requests.post(url, headers=headers, json=data)
        return self.handle_response(response)


class VolcengineStreamOutput(VolcengineImageGenerationBase):
    """火山引擎Seedream 4.0流式输出节点（简化为顺序多图）"""
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": "doubao-seedream-4-0-250828"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "size": (cls.get_size_options_4_0(), {"default": "2048×2048 (1:1)"}),
                "sequential_image_generation": (["auto", "disabled"], {"default": "auto"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "watermark": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "max_images": ("INT", {"default": 3, "min": 1, "max": 15}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "info",)
    FUNCTION = "generate_image"
    CATEGORY = "JMCAI❤/火山引擎"
    
    def generate_image(self, api_key, model, prompt, size="2048×2048 (1:1)", sequential_image_generation="auto", seed=-1, watermark=False, max_images=3):
        size_norm = self.normalize_size(size)
        url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        data = {
            "model": model,
            "prompt": prompt,
            "size": size_norm,
            "sequential_image_generation": sequential_image_generation,
            "response_format": "b64_json",
            "seed": seed,
            "watermark": watermark,
        }
        if sequential_image_generation == "auto":
            data["sequential_image_generation_options"] = {"max_images": max_images}
        response = requests.post(url, headers=headers, json=data)
        return self.handle_response(response)




class VolcengineUsage:
    """火山引擎用量查询节点"""
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "access_key": ("STRING", {"default": "", "multiline": False}),
                "secret_key": ("STRING", {"default": "", "multiline": False}),
                "days": ("INT", {"default": 7, "min": 1, "max": 30}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("usage_info",)
    FUNCTION = "query_usage"
    CATEGORY = "JMCAI❤/火山引擎"
    
    def query_usage(self, access_key, secret_key, days=7):
        if ArkService is None:
            return ("Error: volcengine-python-sdk not installed or import failed.",)
            
        try:
            # 初始化服务
            ark_service = ArkService()
            ark_service.set_ak(access_key)
            ark_service.set_sk(secret_key)
            ark_service.set_region("cn-beijing") # 默认北京 region
            
            # 计算时间范围
            end_time = int(time.time())
            start_time = end_time - (days * 86400)
            
            # 构建请求参数
            # 注意：GetUsage 具体参数结构可能随版本变化
            # 这里参考搜索结果：StartTime, EndTime, Interval
            
            params = {
                "StartTime": start_time,
                "EndTime": end_time,
                "Interval": 3600, # 1小时粒度
                "ProjectName": "default"
            }
            
            # 尝试调用 SDK 的 GetUsage 方法 (通常 SDK 会将 API 映射为 snake_case 方法)
            # 如果不存在，尝试使用通用的 json 调用
            if hasattr(ark_service, "get_usage"):
                response = ark_service.get_usage(params)
            else:
                # 通用调用: action, params, body
                # GetUsage 参数通常在 query 或 body 中，这里尝试 query
                # 注意：具体是在 params 还是 body 取决于 API 定义，GetUsage 看起来像查询，可能在 params
                # 但 search result 说 "sent in the request body ... for a POST request"
                # 所以我们用 json 方法 (POST)
                response = ark_service.json("GetUsage", {}, params)
                
            # 处理响应
            if "Result" in response:
                return (json.dumps(response["Result"], indent=2, ensure_ascii=False),)
            elif "ResponseMetadata" in response and "Error" in response["ResponseMetadata"]:
                return (f"Error: {response['ResponseMetadata']['Error']['Message']}",)
            else:
                return (json.dumps(response, indent=2, ensure_ascii=False),)
                
        except Exception as e:
            return (f"Error: {str(e)}",)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "JMCAI_Volcengine_Chat": VolcengineChat,
    "JMCAI_Volcengine_TextToImage": VolcengineTextToImage,
    "JMCAI_Volcengine_ImageToImage": VolcengineImageToImage,
    "JMCAI_Volcengine_MultiImageFusion": VolcengineMultiImageFusion,
    "JMCAI_Volcengine_StreamOutput": VolcengineStreamOutput,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "JMCAI_Volcengine_Chat": "JMCAI❤ 火山引擎 对话",
    "JMCAI_Volcengine_TextToImage": "JMCAI❤ 火山引擎 文生图",
    "JMCAI_Volcengine_ImageToImage": "JMCAI❤ 火山引擎 图生图",
    "JMCAI_Volcengine_MultiImageFusion": "JMCAI❤ 火山引擎 多图融合",
    "JMCAI_Volcengine_StreamOutput": "JMCAI❤ 火山引擎 流式输出",
}