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
    Load API configuration from persistent storage.

    Outputs api_key, base_url, and model_name for connecting to downstream
    generation nodes. Configuration is stored in api_config.json.
    """

    CATEGORY = "JMCAI❤/API配置"
    FUNCTION = "load"
    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("api_key", "base_url", "model_name",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_type": (API_TYPES, {
                    "default": "OpenAI Compatible"
                }),
            },
            "optional": {
                "api_key_override": ("STRING", {
                    "default": "",
                    "placeholder": "Override saved API key (leave empty to use saved)"
                }),
                "base_url_override": ("STRING", {
                    "default": "",
                    "placeholder": "Override saved base URL (leave empty to use saved)"
                }),
                "model_override": ("STRING", {
                    "default": "",
                    "placeholder": "Override saved model (leave empty to use saved)"
                }),
            },
        }

    def load(self, api_type, api_key_override="", base_url_override="", model_override=""):
        """Load API configuration, applying any overrides."""
        cfg = get_api_config(api_type)

        api_key = api_key_override.strip() if api_key_override and api_key_override.strip() else cfg.get("api_key", "")
        base_url = base_url_override.strip() if base_url_override and base_url_override.strip() else cfg.get("base_url", "")
        model_name = model_override.strip() if model_override and model_override.strip() else cfg.get("model_name", "")

        if not api_key:
            logger.warning(f"[JMCAI ConfigLoader] No API key found for {api_type}. Please configure it.")
        else:
            logger.info(f"[JMCAI ConfigLoader] Loaded config for {api_type} | Model: {model_name}")

        return (api_key, base_url, model_name,)


class JMCAI_APIImageConfigSaver:
    """
    Save API configuration to persistent storage.

    Persists api_key, base_url, model_name, and optional custom models
    to api_config.json for use across ComfyUI sessions.
    """

    CATEGORY = "JMCAI❤/API配置"
    FUNCTION = "save"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_type": (API_TYPES, {
                    "default": "OpenAI Compatible"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "placeholder": "API Key to save"
                }),
            },
            "optional": {
                "base_url": ("STRING", {
                    "default": "",
                    "placeholder": "Base URL to save (leave empty to keep current)"
                }),
                "model_name": ("STRING", {
                    "default": "",
                    "placeholder": "Model name to save (leave empty to keep current)"
                }),
                "add_custom_model_name": ("STRING", {
                    "default": "",
                    "placeholder": "New custom model to add to the list"
                }),
                "remove_custom_model_name": ("STRING", {
                    "default": "",
                    "placeholder": "Custom model to remove from the list"
                }),
            },
        }

    def save(self, api_type, api_key,
             base_url="", model_name="",
             add_custom_model_name="", remove_custom_model_name=""):
        """Save API configuration and manage custom models."""
        status_parts = []

        # Save main config
        save_kwargs = {}
        if api_key and api_key.strip():
            save_kwargs["api_key"] = api_key.strip()
        if base_url and base_url.strip():
            save_kwargs["base_url"] = base_url.strip()
        if model_name and model_name.strip():
            save_kwargs["model_name"] = model_name.strip()

        if save_kwargs:
            set_api_config(api_type, **save_kwargs)
            status_parts.append(f"Saved config for {api_type}: {list(save_kwargs.keys())}")

        # Add custom model
        if add_custom_model_name and add_custom_model_name.strip():
            added = add_custom_model(api_type, add_custom_model_name.strip())
            if added:
                status_parts.append(f"Added custom model: {add_custom_model_name.strip()}")
            else:
                status_parts.append(f"Model already exists: {add_custom_model_name.strip()}")

        # Remove custom model
        if remove_custom_model_name and remove_custom_model_name.strip():
            removed = remove_custom_model(api_type, remove_custom_model_name.strip())
            if removed:
                status_parts.append(f"Removed custom model: {remove_custom_model_name.strip()}")
            else:
                status_parts.append(f"Model not found: {remove_custom_model_name.strip()}")

        status = " | ".join(status_parts) if status_parts else "No changes to save"
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
