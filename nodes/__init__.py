import importlib
import os
import glob

# 动态加载所有节点
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 获取当前目录下所有的 .py 文件
current_dir = os.path.dirname(__file__)
node_files = glob.glob(os.path.join(current_dir, "*.py"))

for file_path in node_files:
    file_name = os.path.basename(file_path)
    # 排除不需要自动加载的文件
    if file_name in ["__init__.py", "base.py"]:
        continue
    
    module_name = f".{file_name[:-3]}"
    try:
        # 使用相对导入
        module = importlib.import_module(module_name, package=__package__)
        
        # 合并 NODE_CLASS_MAPPINGS
        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            
        # 合并 NODE_DISPLAY_NAME_MAPPINGS
        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            
    except Exception as e:
        print(f"\033[31m[JMCAI Error]\033[0m 无法加载节点模块 {module_name}: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
