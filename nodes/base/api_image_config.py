"""
Persistent API configuration management for JMCAI API Image nodes.

Stores configuration in api_config.json within the plugin root directory.
Supports multiple API types, custom model lists, and automatic migration.
"""
import json
import os
import logging

logger = logging.getLogger("JMCAI-APIImage")

# Locate plugin root directory:
# Current file is in comfyui-jmcai/nodes/base/api_image_config.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
CONFIG_FILE = os.path.join(PLUGIN_ROOT, "api_config.json")

# Built-in model lists for each API type
BUILTIN_MODELS = {
    "Gemini Native": [
        "gemini-3-pro-image-preview",
        "gemini-2.5-flash-image",
    ],
    "Grok API": [
        "grok-imagine-image-quality",
        "grok-imagine-image",
        "grok-imagine-image-pro",
    ],
    "OpenAI Compatible": [
        "gpt-image-2",
        "gpt-image-2-2026-04-21",
        "gemini-3-pro-image-preview",
        "gemini-2.5-flash-image",
        "dall-e-3",
        "dall-e-2",
        "gpt-image-1",
    ],
    "BytePlus ModelArk": [
        "dola-seedream-5-0-pro-260628",
        "seedream-5-0-260128",
        "seedream-5-0-lite",
        "seedream-4-5",
        "seedream-4-0",
        "seedream-3-0-t2i",
    ],
    "Qwen Image": [
        "qwen-image-plus",
        "qwen-image-edit",
    ],
    "GLM Image": [
        "glm-image",
        "cogview-4-250304",
    ],
}

DEFAULT_CONFIG = {
    "api_configs": {
        "Gemini Native": {
            "api_key": "",
            "base_url": "(SDK - Automatic)",
            "model_name": "gemini-3-pro-image-preview",
            "custom_models": [],
        },
        "Grok API": {
            "api_key": "",
            "base_url": "https://api.x.ai",
            "model_name": "grok-imagine-image-quality",
            "custom_models": [],
        },
        "OpenAI Compatible": {
            "api_key": "",
            "base_url": "https://api.openai.com",
            "model_name": "gpt-image-2",
            "custom_models": [],
        },
        "BytePlus ModelArk": {
            "api_key": "",
            "base_url": "https://ark.ap-southeast.bytepluses.com/api/v3",
            "model_name": "dola-seedream-5-0-pro-260628",
            "custom_models": [],
        },
        "Qwen Image": {
            "api_key": "",
            "base_url": "https://dashscope.aliyuncs.com/api/v1",
            "model_name": "qwen-image-plus",
            "custom_models": [],
        },
        "GLM Image": {
            "api_key": "",
            "base_url": "https://open.bigmodel.cn/api",
            "model_name": "glm-image",
            "custom_models": [],
        },
    }
}


def load_config():
    """Load configuration from api_config.json, creating default if missing."""
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)

        if "api_configs" not in config:
            config["api_configs"] = DEFAULT_CONFIG["api_configs"].copy()
        else:
            for api_type, defaults in DEFAULT_CONFIG["api_configs"].items():
                if api_type not in config["api_configs"]:
                    config["api_configs"][api_type] = defaults.copy()
                else:
                    if "custom_models" not in config["api_configs"][api_type]:
                        config["api_configs"][api_type]["custom_models"] = []
                    for k, v in defaults.items():
                        if k not in config["api_configs"][api_type]:
                            config["api_configs"][api_type][k] = v

        return config
    except Exception as e:
        logger.error(f"[JMCAI Config] Failed to load config: {e}")
        return DEFAULT_CONFIG.copy()


def save_config(config):
    """Save configuration to api_config.json."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        logger.info(f"[JMCAI Config] Config saved to {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"[JMCAI Config] Failed to save config: {e}")


def get_api_config(api_type):
    """Get configuration for a specific API type."""
    config = load_config()
    return config.get("api_configs", {}).get(api_type, {})


def set_api_config(api_type, api_key=None, base_url=None, model_name=None):
    """Update configuration for a specific API type and persist."""
    config = load_config()
    if api_type not in config.get("api_configs", {}):
        config.setdefault("api_configs", {})[api_type] = DEFAULT_CONFIG["api_configs"].get(
            api_type, {"api_key": "", "base_url": "", "model_name": "", "custom_models": []}
        ).copy()

    cfg = config["api_configs"][api_type]
    if api_key is not None:
        cfg["api_key"] = api_key
    if base_url is not None:
        cfg["base_url"] = base_url
    if model_name is not None:
        cfg["model_name"] = model_name

    save_config(config)
    return cfg


def get_model_list(api_type):
    """Get combined list of built-in + custom models for an API type."""
    builtin = BUILTIN_MODELS.get(api_type, [])
    cfg = get_api_config(api_type)
    custom = cfg.get("custom_models", [])
    all_models = list(builtin)
    for m in custom:
        if m and m not in all_models:
            all_models.append(m)
    return all_models


def add_custom_model(api_type, model_name):
    """Add a custom model to the persistent config."""
    if not model_name or not model_name.strip():
        return False
    model_name = model_name.strip()
    config = load_config()
    cfg = config.setdefault("api_configs", {}).setdefault(api_type, {})
    customs = cfg.setdefault("custom_models", [])
    if model_name not in customs:
        customs.append(model_name)
        save_config(config)
        logger.info(f"[JMCAI Config] Added custom model '{model_name}' to {api_type}")
        return True
    return False


def remove_custom_model(api_type, model_name):
    """Remove a custom model from the persistent config."""
    if not model_name or not model_name.strip():
        return False
    model_name = model_name.strip()
    config = load_config()
    cfg = config.get("api_configs", {}).get(api_type, {})
    customs = cfg.get("custom_models", [])
    if model_name in customs:
        customs.remove(model_name)
        save_config(config)
        logger.info(f"[JMCAI Config] Removed custom model '{model_name}' from {api_type}")
        return True
    return False
