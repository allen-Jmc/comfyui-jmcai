# ComfyUI-JMCAI

<div align="center">

**❤ 多平台全能云端 API 生图与工作流增强套件 for ComfyUI**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## 📖 简介

**ComfyUI-JMCAI** 是专为 ComfyUI 设计的高性能、现代化多平台 AI 生图与工作流增强套件。

全面支持通过**自定义 API 渠道（官方接口或中转站）**接入全球顶尖生图模型，包括 **GPT-Image-2**、**Nano Banana (Gemini 3 Pro / 2.5 Flash Image)**、**xAI Grok**、**火山引擎 / BytePlus Seedream 5.0 Pro**、**阿里通义万相** 以及 **智谱 GLM-Image**。

插件同时内置了完善的 API 密钥持久化管理、URL 容错脱水、多图参考融合、高精度局部裁切重绘、全主流基模尺寸助手，以及与 JMCAI 桌面端无缝协同的远程媒体资产上传服务。

---

## ✨ 核心能力与节点矩阵

### 🎨 1. 多平台 API 生图节点 (`JMCAI❤/API生图`)

| 节点名称 | 目标模型 / 系列 | 支持能力与协议特点 |
|:---|:---|:---|
| **JMCAI❤ OpenAI/GPT-Image-2 生图** | `gpt-image-2`, `gpt-image-1`, `dall-e-3`, 中转站 `gemini-3-pro-image-preview` / `nano-banana` 等 | 支持官方或任意第三方中转站 OpenAI 格式接口；文生图 (`/v1/images/generations`) 与最多 16 张参考图编辑重绘 (`/v1/images/edits`)；内置 URL 自动脱水容错（自动裁切多余 `/v1` 避免 404）。 |
| **JMCAI❤ Gemini/Nano Banana 原生生图** | `gemini-3-pro-image-preview` (Nano Banana Pro), `gemini-2.5-flash-image` (Nano Banana 2) | 基于官方 Google GenAI SDK；支持原生文生图、最高 14 张参考图多图融合理解、遮罩编辑以及多轮对话图像生成。 |
| **JMCAI❤ 火山/Seedream 5.0 生图** | `dola-seedream-5-0-pro-260628`, `seedream-5-0-260128`, `seedream-4-5`, `seedream-4-0` | 直连 BytePlus / 火山方舟 ModelArk 最新 API；支持文生图、单/多参考图生图、超高分辨率与灵活长宽比控制。 |
| **JMCAI❤ xAI Grok 生图** | `grok-imagine-image-quality`, `grok-imagine-image`, `grok-imagine-image-pro` | 直连 xAI 原生 Protobuf gRPC 接口；支持最高 3 张高质量参考图融合与遮罩编辑。 |
| **JMCAI❤ 阿里/通义万相生图** | `qwen-image-plus`, `qwen-image-edit` | 支持阿里 DashScope 通义万相文生图与最高 3 图编辑重绘。 |
| **JMCAI❤ 智谱/GLM-Image 生图** | `glm-image`, `cogview-4-250304` | 智谱 AI CogView / GLM-Image 快速文本生成高质量中文构图。 |

---

### ⚙️ 2. API 配置与持久化管理 (`JMCAI❤/API配置`)

| 节点名称 | 功能说明 | 特性 |
|:---|:---|:---|
| **JMCAI❤ API配置加载器** (`JMCAI_APIImageConfigLoader`) | 一键读取保存的各平台 API 配置 | 输出 `api_key`、`base_url`、`model_name`，直连下游生图节点；支持节点内局部覆盖 (Override)。 |
| **JMCAI❤ API配置保存器** (`JMCAI_APIImageConfigSaver`) | 持久化保存 API Key 与中转站配置 | 配置写入插件根目录 `api_config.json`，跨 ComfyUI 重启生效；支持动态增加或删除自定义模型名称。 |

---

### 🛠 3. 图像与工作流增强工具 (`JMCAI❤/图像工具`)

| 节点名称 | 功能说明 | 特性 |
|:---|:---|:---|
| **JMCAI❤ API图像保存与预览** (`JMCAI_APIImageSave`) | 保存 API 生成图片并生成预览 | 自动附加生成 Prompt 与元数据，兼容 ComfyUI 画布实时预览。 |
| **JmcAI Safe Load Image** (`JMCSafeLoadImage`) | 客户端资产安全加载节点 | 防御前端 `EMPTY_REJECT_IN_UI.png` 占位符引发的崩溃，安全回退。 |
| **JMCAI❤ 局部裁切预处理** (`JMCAI_LocalCropPreprocess`) | 自动提取遮罩区域并缩放对齐 | 支持 **部分运行 (Partial Run)** 实时预览构图；提供 4 组输出端口（`裁切图像`、`裁切遮罩`、`裁切数据`，以及全新的 `裁切图像(无遮罩)` 便于图像对比）。 |
| **JMCAI❤ 局部裁切贴回** (`JMCAI_LocalCropPaste`) | 局部重绘精准还原回原图 | 支持平滑羽化边缘，确保重绘区域与原始图像过渡自然。 |
| **JMCAI❤ 图像组合批次 (多重)** (`JMCAI_ImageBatch_Multi`) | 动态多图合并为批次张量 | 自由调整输入端口数量，自动完成图像规格化与比例对齐。 |

---

### 📐 4. 尺寸助手系列 (`JMCAI❤/尺寸工具`)

内置全主流文生图基模推荐分辨率，一键输出符合模型训练分布的最佳宽高：
- **SD 1.5 尺寸助手** (`JmcAI_SD15ResolutionNode`)
- **SDXL 尺寸助手** (`JmcAI_SDXLResolutionNode`)
- **Flux.1 尺寸助手** (`JmcAI_FluxResolutionNode`)
- **Qwen-Image 尺寸助手** (`JmcAI_QwenImageResolutionNode`)
- **Qwen-Image-Edit 尺寸助手** (`JmcAI_QwenImageEditResolutionNode`)
- **Z-Image 尺寸助手** (`JmcAI_ZImageResolutionNode`)
- **WAN2.1 尺寸助手** (`JmcAI_WAN21ResolutionNode`)
- **WAN2.1 尺寸助手 (进阶版)** (`JmcAI_WAN21AdvancedResolutionNode`)
- **图像尺寸探测** (`JmcAI_ImageSizeDetectorNode`)

---

### 🌐 5. JMCAI 桌面端伴侣能力 (Companion Routes)

插件内置了 JMCAI 桌面端通信路由，支持远程 ComfyUI 运行环境自动探测与媒体资产直传：
- `GET /jmcai/capabilities`：能力握手检测
- `POST /jmcai/upload/audio`：音频素材上传
- `POST /jmcai/upload/video`：视频素材上传
- `POST /jmcai/upload/file`：工作流参考文件上传

---

## 🚀 安装与环境准备

### 方法 1: 放入 custom_nodes 目录

```bash
cd <ComfyUI路径>/custom_nodes
git clone https://github.com/allen-Jmc/comfyui-jmcai.git
```

### 方法 2: 安装依赖

基础生图仅需 Python 标准三方库：
```bash
pip install -r comfyui-jmcai/requirements.txt
```

> **可选原生 SDK 说明：**
> - 如果通过第三方中转站（OpenAI 兼容格式）调用 **GPT-Image-2**、**Nano Banana**、**Seedream**，**无需安装任何重型外部 SDK**，开箱即用！
> - 若需要直连官方原生 SDK 模式，可按需安装：
>   - Gemini 原生直连：`pip install google-genai`
>   - xAI Grok 原生直连：`pip install xai_sdk`
>   - 阿里通义万相原生直连：`pip install dashscope`

---

## 💡 使用中转站 / 自定义 API 调用指南

### 场景 A：使用中转站调用 GPT-Image-2
1. 在画布中添加 **`JMCAI❤ OpenAI/GPT-Image-2 生图`** 节点。
2. `api_key`：填写您从中转站获取的密钥（如 `sk-xxxx`）。
3. `base_url`：填写中转站地址（例如 `https://api.your-relay.com`，即使填入带有 `/v1` 的路径，节点内置的清洗器也会自动智能处理）。
4. `model_name`：选择或手动输入 `gpt-image-2`。
5. 连接 `prompt` 即可直接生成；若需要多图参考或垫图，连接至 `ref_images` 端口。

### 场景 B：使用中转站以 OpenAI 兼容格式调用 Nano Banana
许多 API 中转站将 Google 的 Nano Banana 系列（`gemini-3-pro-image-preview` / `gemini-2.5-flash-image`）封装为标准 OpenAI 图像协议。
1. 直接使用 **`JMCAI❤ OpenAI/GPT-Image-2 生图`** 节点。
2. 填写中转站的 `api_key` 与 `base_url`。
3. `model_name` 下拉选择 `gemini-3-pro-image-preview`（或输入中转站指定的模型别名）。
4. 节点已对非 OpenAI 专有参数进行了自动兼容适配，中转站可无缝出图。

### 场景 C：持久化保存 API 配置
1. 在画布中放置 **`JMCAI❤ API配置保存器`** 节点。
2. 选择平台类型（如 `OpenAI Compatible`），填入您的 `api_key`、`base_url`，点击运行一次即可写入 `api_config.json`。
3. 之后在任何工作流中，直接使用 **`JMCAI❤ API配置加载器`** 即可将配置注入生图节点，无需每次重复输入密钥。

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。
