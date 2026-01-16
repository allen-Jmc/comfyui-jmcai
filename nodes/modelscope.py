import json
import requests
import torch
import numpy as np
import base64
import io
import time
from PIL import Image
from ..core.logger import jm_log, mask_key
from .base.api_base import RemoteAPIBase

class ModelScopeTextToImage(RemoteAPIBase):
    """魔搭 (ModelScope) 文生图节点储备"""
    
    def __init__(self):
        super().__init__()
        
    def log(self, level, message):
        jm_log(level, "ModelScopeTextToImage", message)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "提示词": ("STRING", {"default": "A beautiful landscape", "multiline": True}),
                "model": (["Tongyi-MAI/Z-Image-Turbo"], {"default": "Tongyi-MAI/Z-Image-Turbo"}),
                "custom_model": ("STRING", {"default": ""}),
                "负面提示词": ("STRING", {"default": "", "multiline": True}),
                "宽度": ("INT", {"default": 1280, "min": 256, "max": 2048, "step": 64}),
                "高度": ("INT", {"default": 1280, "min": 256, "max": 2048, "step": 64}),
                "迭代步数": ("INT", {"default": 10, "min": 1, "max": 100}),
                "引导系数 (CFG)": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "随机种子": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "批量生成数量": ("INT", {"default": 1, "min": 1, "max": 4}),
                "max_concurrency": ("INT", {"default": 4, "min": 1, "max": 32}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("图像", "信息",)
    FUNCTION = "generate_image"
    CATEGORY = "JMCAI❤/魔搭社区"

    def generate_image(self, api_key, 提示词, 负面提示词, model, custom_model, 宽度, 高度, 迭代步数=10, **kwargs):
        prompt = 提示词
        negative_prompt = 负面提示词
        width = 宽度
        height = 高度
        steps = 迭代步数
        guidance = kwargs.get("引导系数 (CFG)", 1.5)
        seed = kwargs.get("随机种子", -1)
        batch_size = kwargs.get("批量生成数量", 1)
        max_concurrency = kwargs.get("max_concurrency", 4)
        if not api_key: return (torch.zeros((1, 64, 64, 3)), "Error: API Key is required.")
        base_url = "https://api-inference.modelscope.cn/"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "X-ModelScope-Async-Mode": "true"}
        
        target_model = custom_model.strip() if custom_model and custom_model.strip() else model
        size_str = f"{width}x{height}"
        self.log("INFO", f"准备调用 ModelScope 文生图: 模型={target_model}, 批量={batch_size}")

        data = {
            "model": target_model, "prompt": prompt, "negative_prompt": negative_prompt,
            "n": batch_size, "size": size_str, "steps": int(steps), "guidance": float(guidance),
        }
        if seed != -1: data["seed"] = int(seed)

        try:
            response = self.post_request(f"{base_url}v1/images/generations", headers=headers, data=data)
            response.raise_for_status()
            res_json = response.json()
            task_id = res_json.get("task_id")
            if not task_id: return self._process_image_result(res_json)

            poll_headers = {"Authorization": f"Bearer {api_key}", "X-ModelScope-Task-Type": "image_generation"}
            self.log("INFO", f"任务已提交，ID: {task_id}，开始轮询状态...")

            for i in range(60):
                poll_resp = self.get_request(f"{base_url}v1/tasks/{task_id}", headers=poll_headers)
                poll_resp.raise_for_status()
                data_json = poll_resp.json()
                status = data_json.get("task_status")
                if status == "SUCCEED": return self._process_image_result(data_json)
                elif status == "FAILED": return (torch.zeros((1, 64, 64, 3)), f"Error: {data_json.get('message', '')}")
                time.sleep(2)
            return (torch.zeros((1, 64, 64, 3)), "Error: Task timed out.")
        except Exception as e:
            return (torch.zeros((1, 64, 64, 3)), f"Error: {str(e)}")

    def _process_image_result(self, result):
        images = []
        image_sources = []
        urls = result.get("output_images", [])
        if not urls:
            data_items = result.get("data", [])
            urls = [item.get("url") for item in data_items if item.get("url")]
        
        for url in urls:
            try:
                img_resp = requests.get(url)
                img_resp.raise_for_status()
                img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                img_np = np.array(img, dtype=np.float32) / 255.0
                images.append(img_np)
                image_sources.append(url)
            except: pass
        
        if images:
            batch_np = np.stack(images, axis=0)
            batch_tensor = torch.from_numpy(batch_np).float()
            info = json.dumps({"count": len(images), "sources": image_sources}, ensure_ascii=False)
            return (batch_tensor, info)
        return (torch.zeros((1, 64, 64, 3)), "Error: No images found.")

class ModelScopeChat(RemoteAPIBase):
    """魔搭 (ModelScope) 多模态 LLM 节点"""
    
    def __init__(self):
        super().__init__()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ([
                    "Qwen/Qwen3-VL-8B-Instruct",
                    "Qwen/Qwen3-VL-235B-A22B-Instruct",
                    "Qwen/Qwen3-VL-8B-Thinking"
                ], {"default": "Qwen/Qwen3-VL-8B-Instruct"}),
                "系统提示词": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "提示词": ("STRING", {"default": "请详细描述这张图片的内容。", "multiline": True}),
                "采样温度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "采样阈值 (Top P)": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "最大输出长度": ("INT", {"default": 1024, "min": 1, "max": 4096}),
            },
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "image_2": ("IMAGE", {"default": None}),
                "image_3": ("IMAGE", {"default": None}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("信息",)
    FUNCTION = "generate_chat"
    CATEGORY = "JMCAI❤/魔搭社区"

    def _process_image_to_base64(self, image):
        if image is None: return None
        img_np = (image[0] if len(image.shape) == 4 else image) * 255
        pil_image = Image.fromarray(img_np.cpu().numpy().astype(np.uint8))
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate_chat(self, api_key, model, 提示词, 系统提示词="You are a helpful assistant.", 采样温度=0.7, image=None, image_2=None, image_3=None, **kwargs):
        prompt = 提示词
        system_prompt = 系统提示词
        temperature = 采样温度
        top_p = kwargs.get("采样阈值 (Top P)", 0.9)
        max_tokens = kwargs.get("最大输出长度", 1024)
        if not api_key: return ("Error: API Key is required.",)
        url = "https://api-inference.modelscope.cn/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        messages = []
        # 添加系统提示词
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
            
        user_content = [{"type": "text", "text": prompt}] if prompt else []
        for img in [image, image_2, image_3]:
            if img is not None:
                b64 = self._process_image_to_base64(img)
                if b64: user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        
        if not user_content: return ("Error: Prompt or Image is required.",)
        
        messages.append({"role": "user", "content": user_content})
        
        data = {
            "model": model, 
            "messages": messages, 
            "temperature": temperature, 
            "top_p": top_p,
            "max_tokens": max_tokens, 
            "stream": False
        }

        try:
            response = self.post_request(url, headers=headers, data=data)
            response.raise_for_status()
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                jm_log("SUCCESS", "ModelScopeChat", "对话请求完成")
                return (result["choices"][0]["message"]["content"],)
            return (f"Error: {result}",)
        except Exception as e:
            jm_log("ERROR", "ModelScopeChat", f"请求异常: {str(e)}")
            return (f"Error: {str(e)}",)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "JMCAI_ModelScope_TextToImage": ModelScopeTextToImage,
    "JMCAI_ModelScope_Chat": ModelScopeChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JMCAI_ModelScope_TextToImage": "JMCAI❤ 魔搭 文生图",
    "JMCAI_ModelScope_Chat": "JMCAI❤ 魔搭 多模态 LLM",
}
