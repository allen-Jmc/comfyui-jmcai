import json
import os

# 定义工作流目录
workflow_dir = r"d:\Trea项目存放\comfyui插件\comfyui-jmcai\example_workflows"

# 定义节点名称映射
node_mappings = {
    "VolcengineChatAPI": "JMCAI_Volcengine_Chat",
    "VolcengineSeedream4TextToImage": "JMCAI_Volcengine_TextToImage",
    "VolcengineSeedream4ImageToImage": "JMCAI_Volcengine_ImageToImage",
    "VolcengineSeedream4MultiImageFusion": "JMCAI_Volcengine_MultiImageFusion",
    "VolcengineSeedream4StreamOutput": "JMCAI_Volcengine_StreamOutput"
}

# 遍历所有JSON文件
for filename in os.listdir(workflow_dir):
    if filename.endswith('.json') and filename != 'README.md':
        filepath = os.path.join(workflow_dir, filename)
        
        try:
            # 读取JSON文件
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替换所有旧节点名称
            updated_content = content
            for old_name, new_name in node_mappings.items():
                updated_content = updated_content.replace(old_name, new_name)
            
            # 写回文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"✓ Updated: {filename}")
        except Exception as e:
            print(f"✗ Failed: {filename} - {e}")

print("\n工作流更新完成!")
