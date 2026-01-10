from .volcengine import (
    VolcengineChat,
    VolcengineTextToImageSingle,
    VolcengineTextToImageBatch,
    VolcengineImageToImageSingle,
    VolcengineImageToImageBatch,
    VolcengineMultiImageFusionSingle,
    VolcengineMultiImageFusionBatch
)
from .modelscope import (
    ModelScopeTextToImage,
    ModelScopeChat
)
from .image_tools import (
    JmcAI_ImageBatch_Multi,
    JmcAI_LocalCropPreprocess,
    JmcAI_LocalCropPaste
)

NODE_CLASS_MAPPINGS = {
    # 图像工具
    "JMCAI_ImageBatch_Multi": JmcAI_ImageBatch_Multi,
    "JMCAI_LocalCropPreprocess": JmcAI_LocalCropPreprocess,
    "JMCAI_LocalCropPaste": JmcAI_LocalCropPaste,
    
    # 火山引擎
    "JMCAI_Volcengine_Chat": VolcengineChat,
    "JMCAI_Volcengine_TextToImage_Single": VolcengineTextToImageSingle,
    "JMCAI_Volcengine_TextToImage_Batch": VolcengineTextToImageBatch,
    "JMCAI_Volcengine_ImageToImage_Single": VolcengineImageToImageSingle,
    "JMCAI_Volcengine_ImageToImage_Batch": VolcengineImageToImageBatch,
    "JMCAI_Volcengine_MultiImageFusion_Single": VolcengineMultiImageFusionSingle,
    "JMCAI_Volcengine_MultiImageFusion_Batch": VolcengineMultiImageFusionBatch,
    
    # 魔搭社区
    "JMCAI_ModelScope_TextToImage": ModelScopeTextToImage,
    "JMCAI_ModelScope_Chat": ModelScopeChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JMCAI_Volcengine_Chat": "JMCAI❤ 火山引擎 对话",
    "JMCAI_Volcengine_TextToImage_Single": "JMCAI❤ 火山引擎 文生图-生成单张图",
    "JMCAI_Volcengine_TextToImage_Batch": "JMCAI❤ 火山引擎 文生图-生成一组图",
    "JMCAI_Volcengine_ImageToImage_Single": "JMCAI❤ 火山引擎 图生图-单张图生成单张图",
    "JMCAI_Volcengine_ImageToImage_Batch": "JMCAI❤ 火山引擎 图生图-单张图生成一组图",
    "JMCAI_Volcengine_MultiImageFusion_Single": "JMCAI❤ 火山引擎 图生图-多张参考图生成单张图",
    "JMCAI_Volcengine_MultiImageFusion_Batch": "JMCAI❤ 火山引擎 图生图-多张参考图生成一组图",
    
    "JMCAI_ModelScope_TextToImage": "JMCAI❤ 魔搭 文生图",
    "JMCAI_ModelScope_Chat": "JMCAI❤ 魔搭 多模态 LLM",
    
    "JMCAI_ImageBatch_Multi": "JMCAI❤ 图像组合批次 (多重)",
    "JMCAI_LocalCropPreprocess": "JMCAI❤ 局部裁切预处理",
    "JMCAI_LocalCropPaste": "JMCAI❤ 局部裁切贴回",
}
