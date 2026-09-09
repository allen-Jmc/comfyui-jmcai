"""
JMCAI API Configuration nodes for ComfyUI.

Provides nodes for loading and saving API configurations persistently.
Supports adding custom models dynamically.
"""
import logging

from .base.api_image_config import (
    get_api_config,
    set_api_config,
    get_model_list,
    add_custom_model,
    remove_custom_model,
    load_config,
    BUILTIN_MODELS,
)

logger = logging.getLogger("JMCAI-APIImage")

API_TYPES = list(BUILTIN_MODELS.keys())


class JMCAI_APIImageConfigLoader:
    """
    从本地持久化配置文件 (api_config.json) 读取 API 密钥与地址配置。
    输出 API密钥、接口地址和模型名称，直连下游 API 生图节点。
    """

    CATEGORY = "JMCAI❤/API配置"
    FUNCTION = "load"
    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("API密钥", "接口地址", "模型名称",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "平台类型": (API_TYPES, {
                    "default": "OpenAI Compatible"
                }),
            },
            "optional": {
                "覆盖API密钥": ("STRING", {
                    "default": "",
                    "placeholder": "留空使用已保存配置；填入可临时覆盖"
                }),
                "覆盖接口地址": ("STRING", {
                    "default": "",
                    "placeholder": "留空使用已保存地址；填入可临时覆盖"
                }),
                "覆盖模型名称": ("STRING", {
                    "default": "",
                    "placeholder": "留空使用已保存模型；填入可临时覆盖"
                }),
            },
        }

    def load(self, 平台类型="OpenAI Compatible", 覆盖API密钥="", 覆盖接口地址="", 覆盖模型名称="", **kwargs):
        """加载 API 配置，支持中英文双入参兼容。"""
        api_type = kwargs.get("api_type", 平台类型)
        api_key_override = kwargs.get("api_key_override", 覆盖API密钥)
        base_url_override = kwargs.get("base_url_override", 覆盖接口地址)
        model_override = kwargs.get("model_override", 覆盖模型名称)

        cfg = get_api_config(api_type)

        api_key = api_key_override.strip() if api_key_override and api_key_override.strip() else cfg.get("api_key", "")
        base_url = base_url_override.strip() if base_url_override and base_url_override.strip() else cfg.get("base_url", "")
        model_name = model_override.strip() if model_override and model_override.strip() else cfg.get("model_name", "")

        if not api_key:
            logger.warning(f"[JMCAI ConfigLoader] 未检测到 {api_type} 的 API Key，请先配置。")
        else:
            logger.info(f"[JMCAI ConfigLoader] 已成功载入 {api_type} 配置 | 模型: {model_name}")

        return (api_key, base_url, model_name,)


class JMCAI_APIImageConfigSaver:
    """
    将 API 密钥、接口 Base URL 和自定义模型持久化保存至 api_config.json，
    重启 ComfyUI 依然有效。
    """

    CATEGORY = "JMCAI❤/API配置"
    FUNCTION = "save"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("保存状态",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "平台类型": (API_TYPES, {
                    "default": "OpenAI Compatible"
                }),
                "API密钥": ("STRING", {
                    "default": "",
                    "placeholder": "输入需要持久化保存的 API Key (如 sk-...)"
                }),
            },
            "optional": {
                "接口地址": ("STRING", {
                    "default": "",
                    "placeholder": "输入接口 Base URL（留空则保持现有保存值）"
                }),
                "模型名称": ("STRING", {
                    "default": "",
                    "placeholder": "输入默认选择的模型（留空则保持现有保存值）"
                }),
                "新增自定义模型": ("STRING", {
                    "default": "",
                    "placeholder": "输入要加入下拉模型列表的新模型名称"
                }),
                "移除自定义模型": ("STRING", {
                    "default": "",
                    "placeholder": "输入要从下拉列表中移除的自定义模型名称"
                }),
            },
        }

    def save(self, 平台类型="OpenAI Compatible", API密钥="",
             接口地址="", 模型名称="",
             新增自定义模型="", 移除自定义模型="", **kwargs):
        """保存 API 配置并动态管理自定义模型列表。"""
        api_type = kwargs.get("api_type", 平台类型)
        api_key = kwargs.get("api_key", API密钥)
        base_url = kwargs.get("base_url", 接口地址)
        model_name = kwargs.get("model_name", 模型名称)
        add_custom_model_name = kwargs.get("add_custom_model_name", 新增自定义模型)
        remove_custom_model_name = kwargs.get("remove_custom_model_name", 移除自定义模型)

        status_parts = []

        # 保存主配置
        save_kwargs = {}
        if api_key and api_key.strip():
            save_kwargs["api_key"] = api_key.strip()
        if base_url and base_url.strip():
            save_kwargs["base_url"] = base_url.strip()
        if model_name and model_name.strip():
            save_kwargs["model_name"] = model_name.strip()

        if save_kwargs:
            set_api_config(api_type, **save_kwargs)
            status_parts.append(f"已成功保存 {api_type} 配置: {list(save_kwargs.keys())}")

        # 添加自定义模型
        if add_custom_model_name and add_custom_model_name.strip():
            added = add_custom_model(api_type, add_custom_model_name.strip())
            if added:
                status_parts.append(f"已新增模型: {add_custom_model_name.strip()}")
            else:
                status_parts.append(f"模型已存在: {add_custom_model_name.strip()}")

        # 移除自定义模型
        if remove_custom_model_name and remove_custom_model_name.strip():
            removed = remove_custom_model(api_type, remove_custom_model_name.strip())
            if removed:
                status_parts.append(f"已移除模型: {remove_custom_model_name.strip()}")
            else:
                status_parts.append(f"未找到该模型: {remove_custom_model_name.strip()}")

        status = " | ".join(status_parts) if status_parts else "未发生任何配置更改"
        logger.info(f"[JMCAI ConfigSaver] {status}")
        return (status,)


NODE_CLASS_MAPPINGS = {
    "JMCAI_APIImageConfigLoader": JMCAI_APIImageConfigLoader,
    "JMCAI_APIImageConfigSaver": JMCAI_APIImageConfigSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JMCAI_APIImageConfigLoader": "JMCAI❤ API配置加载器",
    "JMCAI_APIImageConfigSaver": "JMCAI❤ API配置保存器",
}
