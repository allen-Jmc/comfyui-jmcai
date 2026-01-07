"""
ComfyUI-JmcAI - 多平台 API 插件
当前包含 Doubao/Seedream 4.0 节点，后续可扩展其他平台。
"""


from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# --------------------------------------------------------------------------------
# Terminal Startup Info
# --------------------------------------------------------------------------------

class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_startup_info():
    version = "1.1.0"
    
    # Banner
    print(f"{TerminalColors.HEADER}================================================================{TerminalColors.ENDC}")
    print(f"{TerminalColors.HEADER} ❤ JMCAI ComfyUI Plugin {TerminalColors.ENDC}{TerminalColors.OKCYAN}v{version}{TerminalColors.ENDC}")
    print(f"{TerminalColors.HEADER}================================================================{TerminalColors.ENDC}")

    # Node List
    print(f"{TerminalColors.OKBLUE}[JMCAI] Loading nodes...{TerminalColors.ENDC}")
    
    loaded_count = 0
    for node_name in NODE_DISPLAY_NAME_MAPPINGS.values():
        print(f"{TerminalColors.OKGREEN} -> Loaded: {node_name}{TerminalColors.ENDC}")
        loaded_count += 1
        
    print(f"{TerminalColors.HEADER}----------------------------------------------------------------{TerminalColors.ENDC}")
    print(f"{TerminalColors.OKCYAN}[JMCAI] Total {loaded_count} nodes loaded successfully. Enjoy! ❤{TerminalColors.ENDC}")
    print(f"{TerminalColors.HEADER}================================================================{TerminalColors.ENDC}")

# Execute output
print_startup_info()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]