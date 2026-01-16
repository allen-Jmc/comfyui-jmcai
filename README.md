# ComfyUI-JMCAI

<div align="center">

**❤ 多平台 AI 节点集合 for ComfyUI**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## 📖 简介

ComfyUI-JMCAI 是一个为 ComfyUI 设计的多平台 AI 节点插件，主打**高性能、低侵入、自动化**。目前集成并优化了**火山引擎 (Volcengine) Doubao/Seedream** 以及 **魔搭社区 (ModelScope)** 的多项 AI 能力，并提供高效的图像增强与处理工具。

---

## ✨ 功能特性

### 🔥 火山引擎节点 (`JMCAI❤/火山引擎`)

| 节点名称 | 功能描述 | 特性 |
|:---|:---|:---|
| **火山引擎 对话** | 多模态对话 | 支持最多 3 张图片输入，支持文字/图像混合理解 |
| **火山引擎 文生图 (单/组)** | 文本生成图片 | 支持 **Seedream 4.0/4.5**，可生成单张或连贯的一组图像 |
| **火山引擎 图生图 (单/组)** | 图片参考生成 | 基于单张参考图，生成单张或连贯的一组相关图像 |
| **火山引擎 多图融合 (单/组)** | 多图特征融合 | 融合多张图片特征，生成单张或一组新的创意图像 |

> [!TIP]
> 批量 (Batch) 节点均支持 **流式输出 (Stream)**，可在生成过程中实时反馈进度。

### 🤖 魔搭社区节点 (`JMCAI❤/魔搭社区`)

| 节点名称 | 功能描述 | 特性 |
|:---|:---|:---|
| **魔搭 文生图** | 高速文本生图 | 支持 **Z-Image-Turbo** 等模型，主打超快推理速度 |
| **魔搭 多模态 LLM** | 视觉语言模型 | 支持 **Qwen-VL** 系列，支持最多 3 张图像的多轮理解 |

### 🛠 图像工具 (`JMCAI❤/图像工具`)

| 节点名称 | 功能描述 | 特性 |
|:---|:---|:---|
| **局部裁切预处理** | 自动提取遮罩区域 | 支持 **部分运行 (Partial Run)** 预览，秒级反馈裁切效果 |
| **局部裁切贴回** | 精确还原图像 | 处理生成后的差异，并将其完美贴合回原图位置 |
| **图像组合批次 (多重)** | 动态多图合并 | 自由调整输入端口数量，自动完成图像规格化与对齐 |

---

## 🚀 安装

### 方法 1: 通过 ComfyUI-Manager (推荐)

1. 打开 ComfyUI-Manager → `Install custom nodes`
2. 搜索 `JMCAI` 或直接粘贴仓库地址：
   ```
   https://github.com/allen-Jmc/comfyui-jmcai
   ```
3. 安装完成后重启 ComfyUI

### 方法 2: 手动安装

```bash
# 克隆到 custom_nodes 目录
cd <ComfyUI路径>/custom_nodes
git clone https://github.com/allen-Jmc/comfyui-jmcai

# 安装依赖
pip install -r comfyui-jmcai/requirements.txt
```

---

## 📝 使用说明

### 1. 配置 API 密钥

- **火山引擎**: 访问 [Ark 控制台](https://console.volcengine.com/ark) 获取 API Key 和推理接入点 ID。
- **魔搭社区**: 访问 [ModelScope 控制台](https://modelscope.cn/my/personal/index) 获取 SDK 令牌。

### 2. 在 ComfyUI 中使用

1. 在节点库搜索 `JMCAI` 即可看到所有相关分类。
2. 将节点拖入画布，填入您的 API Key 并根据提示配置参数。
3. **重要**：对于火山引擎节点，`model` 参数应填写控制台提供的 **推理接入点 ID**。

### 3. 特色：局部裁切 (Partial Run)

在 `局部裁切预处理` 节点中，开启 **“仅预览”** 选项后：
- **实时预览**：点击运行即仅执行上游必要节点，跳过漫长的 KSampler 或云端生成。
- **参数微调**：可以在节点 UI 中直接看到裁切后的构图和遮罩效果，确认无误后再关闭“仅预览”正式运行。

---

## � 依赖项

```text
requests>=2.25.0
Pillow>=9.0.0
numpy>=1.20.0
torch>=2.0.0
```

---

## 📄 许可证

本项目采用 [MIT License](LICENSE)。

---

<div align="center">

**如果觉得有用，请给个 ⭐ Star！**

Made with ❤ by JMCAI

</div>