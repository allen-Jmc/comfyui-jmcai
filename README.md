# ComfyUI-JMCAI

<div align="center">

**❤ 多平台 AI 节点集合 for ComfyUI**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## 📖 简介

ComfyUI-JMCAI 是一个为 ComfyUI 设计的多平台 AI 节点插件，目前支持**火山引擎 Doubao/Seedream 4.0** API。所有节点统一使用 `JMCAI❤/火山引擎` 分类，方便识别和管理。

---

## ✨ 功能特性

### 🔥 火山引擎节点

| 节点名称 | 功能描述 | 特性 |
|:---|:---|:---|
| **JMCAI❤ 火山引擎 对话** | 多模态对话 | 支持最多 3 张图片输入 |
| **JMCAI❤ 火山引擎 文生图** | 文本生成图片 | Seedream 4.0，支持多种尺寸 |
| **JMCAI❤ 火山引擎 图生图** | 图片转换图片 | 基于参考图生成 |
| **JMCAI❤ 火山引擎 多图融合** | 多图融合生成 | 融合多张图片特征 |
| **JMCAI❤ 火山引擎 流式输出** | 顺序生成多图 | 渐进式图像生成 |

---

## 🚀 安装

### 方法 1: 通过 ComfyUI-Manager (推荐)

1. 打开 ComfyUI-Manager → `Install custom nodes`
2. 粘贴仓库地址：
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

### 1. 获取 API 密钥

- 访问 [火山引擎控制台](https://console.volcengine.com/ark)
- 获取您的 **API Key** (Bearer Token)
- 获取对应的 **Model ID** (推理接入点 ID)

### 2. 在 ComfyUI 中使用

1. 在节点库搜索 `JMCAI` 或 `火山引擎`
2. 添加所需节点到工作流
3. **手动填入** API Key 和 Model ID
4. 配置参数并运行

### 3. 示例工作流

在 `example_workflows/` 目录下提供了完整的示例：

- `jmcai_volcengine_chat.json` - 对话示例
- `jmcai_volcengine_text_to_image.json` - 文生图示例
- `jmcai_volcengine_image_to_image.json` - 图生图示例
- `jmcai_volcengine_multi_image_fusion.json` - 多图融合示例
- `jmcai_volcengine_stream_output.json` - 流式输出示例

---

## 🎨 Seedream 4.0 尺寸说明

支持多种像素比例：
- `2048×2048 (1:1)`
- `2560×1440 (16:9)`
- `1440×2560 (9:16)`
- 更多尺寸见节点下拉菜单

---

## 🔧 依赖项

```
requests>=2.25.0
Pillow>=9.0.0
numpy>=1.20.0
```

---

## 💡 高级特性

### 多模态对话 (3 图支持)

`JMCAI❤ 火山引擎 对话` 节点支持同时输入最多 3 张图片。未连接的端口不会发送数据，自动优化 Token 消耗。

### 终端启动信息

插件加载时会在终端显示彩色 Banner 和节点列表。

---

## 🐛 常见问题

### Q: 如何区分 System Prompt 和 User Prompt？
**A**: `system_prompt` 用于设定 AI 的角色或规则，`prompt` 用于输入用户的具体问题。

### Q: 模型名称填什么？
**A**: 填写火山引擎控制台生成的**推理接入点 ID**。

---

## 📄 许可证

本项目采用 [MIT License](LICENSE)。

---

## 🙏 致谢

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [火山引擎](https://www.volcengine.com/)

---

<div align="center">

**如果觉得有用，请给个 ⭐ Star！**

Made with ❤ by JMCAI

</div>