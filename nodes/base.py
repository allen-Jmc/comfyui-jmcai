import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import base64
import io
import time
import numpy as np
import torch
from PIL import Image
from ..core.logger import jm_log
from ..utils.image_utils import process_single_image, process_batch_images

class VolcengineImageGenerationBase:
    """火山引擎图片生成API基类"""
    
    def __init__(self):
        self.session = self._create_session()

    def _create_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def post_request(self, url, headers, data, stream=False, timeout=60):
        """统一的 API 请求处理，带重试和超时"""
        try:
            return self.session.post(url, headers=headers, json=data, stream=stream, timeout=timeout)
        except requests.exceptions.SSLError as e:
            jm_log("ERROR", "API", f"SSL 握手失败: {str(e)}。尝试降低 SSL 验证安全性（注意：仅作为紧急补丁）")
            # 某些环境下系统证书链有问题，尝试跳过验证（慎用）
            return self.session.post(url, headers=headers, json=data, stream=stream, timeout=timeout, verify=False)
        except Exception as e:
            raise e
    
    @classmethod
    def get_size_level_options(cls):
        # 简化为通用规格: 1k (Only 4.0), 2k (1.0x), 4k (2.0x) 并基于 2K 基准表
        return ["1k (Only 4.0)", "2k", "4k"]

    @classmethod
    def get_aspect_ratio_options(cls):
        return ["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9"]

    def calculate_size(self, aspect_ratio, size_level, model_name):
        """基于官方 2K 基准表进行倍率缩放计算"""
        # 1. 定义 2K 基准表
        base_2k_map = {
            "1:1":  (2048, 2048),
            "4:3":  (2304, 1728),
            "3:4":  (1728, 2304),
            "16:9": (2560, 1440),
            "9:16": (1440, 2560),
            "3:2":  (2496, 1664),
            "2:3":  (1664, 2496),
            "21:9": (3024, 1296),
        }
        
        base_w, base_h = base_2k_map.get(aspect_ratio, (2048, 2048))
        
        # 2. 确定缩放倍率
        if "1k" in size_level.lower():
            target_level = "1k"
        elif "2k" in size_level.lower():
            target_level = "2k"
        elif "4k" in size_level.lower():
            target_level = "4k"
        else:
            target_level = "2k"

        scale_map = {
            "1k": 0.5,
            "2k": 1.0,
            "4k": 2.0
        }
        scale = scale_map.get(target_level, 1.0)
        
        width = int(base_w * scale)
        height = int(base_h * scale)
        
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        jm_log("INFO", "SizeCalc", f"尺寸计算: 比例={aspect_ratio}, 规格={size_level}({target_level} -> {scale}x) -> {width}x{height} (基准: {base_w}x{base_h})")
        return width, height

    def process_image(self, image):
        """兼容层：调用 utils 的图像处理"""
        return process_single_image(image)
    
    def process_images(self, images):
        """兼容层：调用 utils 的批量图像处理"""
        return process_batch_images(images)
    
    def normalize_size(self, size):
        s = str(size).strip()
        if " (" in s:
            s = s.split(" (", 1)[0]
        s = s.replace("×", "x").replace("X", "x")
        return s
    
    def handle_response(self, response, node_name="VolcengineImageGen"):
        """处理API响应(包含流式和非流式)并返回ComfyUI期望的IMAGE张量。"""
        images = []
        image_sources = []
        
        try:
            content_type = response.headers.get("Content-Type", "")
            
            if "text/event-stream" in content_type:
                for line in response.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        data_content = line_str[6:].strip()
                        if data_content == "[DONE]":
                            break
                        data_json = json.loads(data_content)
                        event_type = data_json.get("type")
                        
                        b64_list = []
                        if event_type == "image_generation.partial_succeeded":
                            if data_json.get("b64_json"):
                                b64_list.append(data_json.get("b64_json"))
                        
                        choices = data_json.get("choices", [])
                        for choice in choices:
                            if choice.get("image_base64"):
                                b64_list.append(choice.get("image_base64"))
                            elif choice.get("b64_json"):
                                b64_list.append(choice.get("b64_json"))
                            elif choice.get("message") and isinstance(choice.get("message"), dict):
                                msg = choice["message"]
                                if msg.get("content") and isinstance(msg["content"], list):
                                    for part in msg["content"]:
                                        if part.get("type") == "image_url" and part.get("image_url"):
                                            url_data = part["image_url"].get("url", "")
                                            if url_data.startswith("data:image"):
                                                b64_list.append(url_data.split(",", 1)[1])

                        for b64 in b64_list:
                            try:
                                img_bytes = base64.b64decode(b64)
                                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                img_np = np.array(img, dtype=np.float32) / 255.0
                                images.append(img_np)
                                image_sources.append("sse_b64")
                            except Exception as e:
                                jm_log("ERROR", node_name, f"Base64解析失败: {str(e)}")

                        if event_type == "image_generation.completed":
                            jm_log("INFO", node_name, f"流式传输完成，用量统计: {data_json.get('usage', {})}")
            else:
                response.raise_for_status()
                result = response.json()
                data_items = result.get("data", [])
                for item in data_items:
                    if "b64_json" in item and item["b64_json"]:
                        b64 = item["b64_json"]
                        try:
                            img_bytes = base64.b64decode(b64)
                            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                            img_np = np.array(img, dtype=np.float32) / 255.0
                            images.append(img_np)
                            image_sources.append("b64_json")
                        except Exception: pass
                    elif "url" in item and item["url"]:
                        try:
                            print(f"\033[32m[Volcengine]\033[0m 图像下载链接: {item['url']}")
                            img_response = requests.get(item["url"])
                            img_response.raise_for_status()
                            img = Image.open(io.BytesIO(img_response.content)).convert("RGB")
                            img_np = np.array(img, dtype=np.float32) / 255.0
                            images.append(img_np)
                            image_sources.append(item["url"])
                        except Exception: pass

            if len(images) > 0:
                jm_log("SUCCESS", node_name, f"图片生成任务完成，获得 {len(images)} 张图片")
                batch_np = np.stack(images, axis=0)
                batch_tensor = torch.from_numpy(batch_np).float()
                info_dict = {"count": len(images), "sources": image_sources, "node": node_name}
                info = json.dumps(info_dict, ensure_ascii=False)
                return (batch_tensor, info)

            jm_log("ERROR", node_name, f"响应中未发现有效图片数据")
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (placeholder, "Error: No images found")
            
        except Exception as e:
            jm_log("ERROR", node_name, f"处理响应异常: {str(e)}")
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (placeholder, f"Error: {str(e)}")
