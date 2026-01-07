# 示例工作流说明

本目录包含 ComfyUI-JMCAI 插件的示例工作流文件。

## 📁 文件列表

| 文件名 | 功能说明 |
|:---|:---|
| `jmcai_volcengine_chat.json` | 对话示例 - 演示基础文本对话功能 |
| `jmcai_volcengine_text_to_image.json` | 文生图示例 - Seedream 4.0 文本生成图片 |
| `jmcai_volcengine_image_to_image.json` | 图生图示例 - 基于参考图生成新图片 |
| `jmcai_volcengine_multi_image_fusion.json` | 多图融合示例 - 融合多张图片特征 |
| `jmcai_volcengine_stream_output.json` | 流式输出示例 - 顺序生成多张图片 |

## 🚀 使用方法

1. **加载工作流**
   - 在 ComfyUI 中点击 `Load` 按钮
   - 选择对应的 `.json` 文件

2. **配置 API Key**
   - 找到节点中的 `api_key` 字段
   - 填入您的火山引擎 API Key

3. **运行测试**
   - 点击 `Queue Prompt` 运行工作流
   - 查看输出结果

## ⚠️ 注意事项

- 所有示例均使用 `JMCAI_Volcengine_*` 节点
- 请确保已正确安装插件并重启 ComfyUI
- API Key 需要在火山引擎控制台获取

## 💡 提示

- 可以根据需要修改节点参数
- 建议先从简单的对话示例开始测试
- 图片生成需要消耗较多 Token，请注意账户余额

---

**更多信息请查看主 README.md**