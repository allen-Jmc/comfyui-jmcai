"""
JMCAI Image Size Tool (Ported from Image-Size-Tools & Expanded)

为 AI 图像生成模型提供分辨率计算节点。
零依赖，内置数据。
"""

# SD 1.5 分辨率数据 - 512x512 为原生
SD15_RESOLUTIONS = {
    "512×512 (1:1) - 原生": {"width": 512, "height": 512, "ratio": "1:1"},
    "768×512 (3:2) - 横屏": {"width": 768, "height": 512, "ratio": "3:2"},
    "512×768 (2:3) - 竖屏": {"width": 512, "height": 768, "ratio": "2:3"},
    "768×576 (4:3) - 标准": {"width": 768, "height": 576, "ratio": "4:3"},
    "576×768 (3:4) - 标准竖屏": {"width": 576, "height": 768, "ratio": "3:4"},
    "912×512 (16:9) - 宽屏": {"width": 912, "height": 512, "ratio": "16:9"},
    "512×912 (9:16) - 手机壁纸": {"width": 512, "height": 912, "ratio": "9:16"},
    "768×768 (1:1) - 大正方形": {"width": 768, "height": 768, "ratio": "1:1"},
}

# SDXL 分辨率数据 - 尺寸可被 64 整除，目标约 100万像素
SDXL_RESOLUTIONS = {
    "1024×1024 (1:1) - 原生": {"width": 1024, "height": 1024, "ratio": "1:1", "pixels": 1048576},
    "1152×896 (9:7) - 横屏": {"width": 1152, "height": 896, "ratio": "9:7", "pixels": 1032192},
    "896×1152 (7:9) - 竖屏": {"width": 896, "height": 1152, "ratio": "7:9", "pixels": 1032192},
    "1344×768 (7:4) - 宽屏": {"width": 1344, "height": 768, "ratio": "7:4", "pixels": 1032192},
    "768×1344 (4:7) - 长屏": {"width": 768, "height": 1344, "ratio": "4:7", "pixels": 1032192},
    "1216×832 (19:13) - 电影": {"width": 1216, "height": 832, "ratio": "19:13", "pixels": 1011712},
    "832×1216 (13:19) - 电影长屏": {"width": 832, "height": 1216, "ratio": "13:19", "pixels": 1011712},
    "1280×768 (5:3) - 超宽": {"width": 1280, "height": 768, "ratio": "5:3", "pixels": 983040},
    "768×1280 (3:5) - 超长": {"width": 768, "height": 1280, "ratio": "3:5", "pixels": 983040},
    "1536×640 (12:5) - 横幅": {"width": 1536, "height": 640, "ratio": "12:5", "pixels": 983040},
    "640×1536 (5:12) - 摩天大楼": {"width": 640, "height": 1536, "ratio": "5:12", "pixels": 983040},
    "1600×640 (5:2) - 极限宽": {"width": 1600, "height": 640, "ratio": "5:2", "pixels": 1024000},
    "640×1600 (2:5) - 极限长": {"width": 640, "height": 1600, "ratio": "2:5", "pixels": 1024000},
}

# Flux.1 Dev 分辨率数据
FLUX_RESOLUTIONS = {
    "1024×1024 (1:1) - 原生": {"width": 1024, "height": 1024, "ratio": "1:1"},
    "1920×1080 (16:9) - Full HD": {"width": 1920, "height": 1080, "ratio": "16:9"},
    "1080×1920 (9:16) - 竖屏 HD": {"width": 1080, "height": 1920, "ratio": "9:16"},
    "1536×640 (12:5) - 超宽屏": {"width": 1536, "height": 640, "ratio": "12:5"},
    "640×1536 (5:12) - 超长屏": {"width": 640, "height": 1536, "ratio": "5:12"},
    "1600×1600 (1:1) - 高清正方形": {"width": 1600, "height": 1600, "ratio": "1:1"},
    "1280×720 (16:9) - HD": {"width": 1280, "height": 720, "ratio": "16:9"},
    "720×1280 (9:16) - 竖屏 HD": {"width": 720, "height": 1280, "ratio": "9:16"},
    "1366×768 (16:9) - 笔记本": {"width": 1366, "height": 768, "ratio": "16:9"},
    "2560×1440 (16:9) - 2K": {"width": 2560, "height": 1440, "ratio": "16:9"},
}

# Qwen-Image 基础分辨率数据 (基于训练分布)
QWEN_RESOLUTIONS = {
    "1328×1328 (1:1) - Qwen2.5-VL 推荐": {"width": 1328, "height": 1328, "ratio": "1:1"},
    "1024×1024 (1:1) - 标准高清": {"width": 1024, "height": 1024, "ratio": "1:1"},
    "1664×928 (16:9) - 宽屏": {"width": 1664, "height": 928, "ratio": "16:9"},
    "928×1664 (9:16) - 竖屏": {"width": 928, "height": 1664, "ratio": "9:16"},
    "1472×1104 (4:3) - 标准": {"width": 1472, "height": 1104, "ratio": "4:3"},
    "1104×1472 (3:4) - 竖屏标准": {"width": 1104, "height": 1472, "ratio": "3:4"},
    "1584×1056 (3:2) - 横屏": {"width": 1584, "height": 1056, "ratio": "3:2"},
    "1056×1584 (2:3) - 竖屏": {"width": 1056, "height": 1584, "ratio": "2:3"},
    "640×640 (1:1) - 效率模式": {"width": 640, "height": 640, "ratio": "1:1"},
}

# Qwen-Image-Edit 专用 (对齐 1024x1024 分布)
QWEN_EDIT_RESOLUTIONS = {
    "1024×1024 (1:1) - 默认推荐": {"width": 1024, "height": 1024, "ratio": "1:1"},
    "1280×720 (16:9) - HD 横屏": {"width": 1280, "height": 720, "ratio": "16:9"},
    "720×1280 (9:16) - HD 竖屏": {"width": 720, "height": 1280, "ratio": "9:16"},
    "1216×832 (3:2) - 稳定横屏": {"width": 1216, "height": 832, "ratio": "3:2"},
    "832×1216 (2:3) - 稳定竖屏": {"width": 832, "height": 1216, "ratio": "2:3"},
}

# Z-Image 高分辨率比例数据
ZIMAGE_RESOLUTIONS = {
    "1024×1024 (1:1) - 标准": {"width": 1024, "height": 1024, "ratio": "1:1"},
    "1536×1024 (3:2) - 写实横屏": {"width": 1536, "height": 1024, "ratio": "3:2"},
    "1024×1536 (2:3) - 写实竖屏": {"width": 1024, "height": 1536, "ratio": "2:3"},
    "1920×1080 (16:9) - 壁纸宽屏": {"width": 1920, "height": 1080, "ratio": "16:9"},
    "1080×1920 (9:16) - 手机竖屏": {"width": 1080, "height": 1920, "ratio": "9:16"},
    "1280×1280 (1:1) - 大图": {"width": 1280, "height": 1280, "ratio": "1:1"},
}

# WAN2.1 基础分辨率数据
WAN21_SIMPLE_RESOLUTIONS = {
    "832×480 (16:9) - 480p 标准": {"width": 832, "height": 480, "ratio": "16:9"},
    "1280×720 (16:9) - 720p 高清": {"width": 1280, "height": 720, "ratio": "16:9"},
    "480×832 (9:16) - 480p 竖屏": {"width": 480, "height": 832, "ratio": "9:16"},
    "720×1280 (9:16) - 720p 竖屏": {"width": 720, "height": 1280, "ratio": "9:16"},
}

# WAN2.1 进阶分辨率数据
WAN21_ADVANCED_RESOLUTIONS = {
    "832×480 (16:9) - 480p 标准": {"width": 832, "height": 480, "ratio": "16:9"},
    "1280×720 (16:9) - 720p 高清": {"width": 1280, "height": 720, "ratio": "16:9"},
    "480×832 (9:16) - 480p 竖屏": {"width": 480, "height": 832, "ratio": "9:16"},
    "720×1280 (9:16) - 720p 竖屏": {"width": 720, "height": 1280, "ratio": "9:16"},
    "640×480 (4:3) - 经典 480p": {"width": 640, "height": 480, "ratio": "4:3"},
    "960×720 (4:3) - 经典 720p": {"width": 960, "height": 720, "ratio": "4:3"},
    "480×640 (3:4) - 经典竖屏": {"width": 480, "height": 640, "ratio": "3:4"},
    "720×960 (3:4) - 经典竖屏": {"width": 720, "height": 960, "ratio": "3:4"},
    "720×720 (1:1) - 正方形 720p": {"width": 720, "height": 720, "ratio": "1:1"},
    "480×480 (1:1) - 正方形 480p": {"width": 480, "height": 480, "ratio": "1:1"},
    "1024×576 (16:9) - 中等": {"width": 1024, "height": 576, "ratio": "16:9"},
    "576×1024 (9:16) - 中等竖屏": {"width": 576, "height": 1024, "ratio": "9:16"},
}


class JmcAI_SD15ResolutionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "像素尺寸": (list(SD15_RESOLUTIONS.keys()),),
            },
        }
    
    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("宽", "高", "宽高比")
    FUNCTION = "get_resolution"
    CATEGORY = "JMCAI❤/尺寸工具"
    
    def get_resolution(self, 像素尺寸):
        res_data = SD15_RESOLUTIONS[像素尺寸]
        return (res_data["width"], res_data["height"], res_data["ratio"])


class JmcAI_SDXLResolutionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "像素尺寸": (list(SDXL_RESOLUTIONS.keys()),),
            },
        }
    
    RETURN_TYPES = ("INT", "INT", "STRING", "INT")
    RETURN_NAMES = ("宽", "高", "宽高比", "总像素")
    FUNCTION = "get_resolution"
    CATEGORY = "JMCAI❤/尺寸工具"
    
    def get_resolution(self, 像素尺寸):
        res_data = SDXL_RESOLUTIONS[像素尺寸]
        return (res_data["width"], res_data["height"], res_data["ratio"], res_data["pixels"])


class JmcAI_FluxResolutionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "像素尺寸": (list(FLUX_RESOLUTIONS.keys()),),
            },
        }
    
    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("宽", "高", "宽高比")
    FUNCTION = "get_resolution"
    CATEGORY = "JMCAI❤/尺寸工具"
    
    def get_resolution(self, 像素尺寸):
        res_data = FLUX_RESOLUTIONS[像素尺寸]
        return (res_data["width"], res_data["height"], res_data["ratio"])


class JmcAI_QwenImageResolutionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "像素尺寸": (list(QWEN_RESOLUTIONS.keys()),),
            },
        }
    
    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("宽", "高", "宽高比")
    FUNCTION = "get_resolution"
    CATEGORY = "JMCAI❤/尺寸工具"
    
    def get_resolution(self, 像素尺寸):
        res_data = QWEN_RESOLUTIONS[像素尺寸]
        return (res_data["width"], res_data["height"], res_data["ratio"])


class JmcAI_QwenImageEditResolutionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "像素尺寸": (list(QWEN_EDIT_RESOLUTIONS.keys()),),
            },
        }
    
    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("宽", "高", "宽高比")
    FUNCTION = "get_resolution"
    CATEGORY = "JMCAI❤/尺寸工具"
    
    def get_resolution(self, 像素尺寸):
        res_data = QWEN_EDIT_RESOLUTIONS[像素尺寸]
        return (res_data["width"], res_data["height"], res_data["ratio"])


class JmcAI_ZImageResolutionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "像素尺寸": (list(ZIMAGE_RESOLUTIONS.keys()),),
            },
        }
    
    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("宽", "高", "宽高比")
    FUNCTION = "get_resolution"
    CATEGORY = "JMCAI❤/尺寸工具"
    
    def get_resolution(self, 像素尺寸):
        res_data = ZIMAGE_RESOLUTIONS[像素尺寸]
        return (res_data["width"], res_data["height"], res_data["ratio"])


class JmcAI_ImageSizeDetectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("宽", "高")
    FUNCTION = "detect_size"
    CATEGORY = "JMCAI❤/尺寸工具"
    
    def detect_size(self, 图像):
        # Image tensor shape is [B, H, W, C]
        batch_size, height, width, channels = 图像.shape
        return (int(width), int(height))


class JmcAI_WAN21ResolutionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "像素尺寸": (list(WAN21_SIMPLE_RESOLUTIONS.keys()),),
                "尺寸减半": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("INT", "INT", "STRING", "INT", "INT")
    RETURN_NAMES = ("宽", "高", "宽高比", "减半宽", "减半高")
    FUNCTION = "get_resolution"
    CATEGORY = "JMCAI❤/尺寸工具"
    
    def get_resolution(self, 像素尺寸, 尺寸减半):
        res_data = WAN21_SIMPLE_RESOLUTIONS[像素尺寸]
        width = res_data["width"]
        height = res_data["height"]
        
        halved_width = int(width // 2)
        halved_height = int(height // 2)
        
        # 确保偶数，以便视频编码兼容
        if halved_width % 2 != 0: halved_width += 1
        if halved_height % 2 != 0: halved_height += 1
        
        return (width, height, res_data["ratio"], halved_width, halved_height)


class JmcAI_WAN21AdvancedResolutionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "像素尺寸": (list(WAN21_ADVANCED_RESOLUTIONS.keys()),),
                "尺寸减半": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("INT", "INT", "STRING", "INT", "INT")
    RETURN_NAMES = ("宽", "高", "宽高比", "减半宽", "减半高")
    FUNCTION = "get_resolution"
    CATEGORY = "JMCAI❤/尺寸工具"
    
    def get_resolution(self, 像素尺寸, 尺寸减半):
        res_data = WAN21_ADVANCED_RESOLUTIONS[像素尺寸]
        width = res_data["width"]
        height = res_data["height"]
        
        halved_width = int(width // 2)
        halved_height = int(height // 2)
        
        if halved_width % 2 != 0: halved_width += 1
        if halved_height % 2 != 0: halved_height += 1
        
        return (width, height, res_data["ratio"], halved_width, halved_height)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "JmcAI_SD15ResolutionNode": JmcAI_SD15ResolutionNode,
    "JmcAI_SDXLResolutionNode": JmcAI_SDXLResolutionNode,
    "JmcAI_FluxResolutionNode": JmcAI_FluxResolutionNode,
    "JmcAI_QwenImageResolutionNode": JmcAI_QwenImageResolutionNode,
    "JmcAI_QwenImageEditResolutionNode": JmcAI_QwenImageEditResolutionNode,
    "JmcAI_ZImageResolutionNode": JmcAI_ZImageResolutionNode,
    "JmcAI_ImageSizeDetectorNode": JmcAI_ImageSizeDetectorNode,
    "JmcAI_WAN21ResolutionNode": JmcAI_WAN21ResolutionNode,
    "JmcAI_WAN21AdvancedResolutionNode": JmcAI_WAN21AdvancedResolutionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JmcAI_SD15ResolutionNode": "JMCAI❤ SD 1.5 尺寸助手",
    "JmcAI_SDXLResolutionNode": "JMCAI❤ SDXL 尺寸助手", 
    "JmcAI_FluxResolutionNode": "JMCAI❤ Flux.1 尺寸助手",
    "JmcAI_QwenImageResolutionNode": "JMCAI❤ Qwen-Image 尺寸助手",
    "JmcAI_QwenImageEditResolutionNode": "JMCAI❤ Qwen-Image-Edit 尺寸助手",
    "JmcAI_ZImageResolutionNode": "JMCAI❤ Z-Image 尺寸助手",
    "JmcAI_ImageSizeDetectorNode": "JMCAI❤ 图像尺寸探测",
    "JmcAI_WAN21ResolutionNode": "JMCAI❤ WAN2.1 尺寸助手",
    "JmcAI_WAN21AdvancedResolutionNode": "JMCAI❤ WAN2.1 尺寸助手 (进阶版)",
}
