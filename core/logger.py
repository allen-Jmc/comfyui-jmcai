import json
import time

def jm_log(level, node_name, message):
    """插件全局统一日志输出"""
    curr_time = time.strftime("%H:%M:%S")
    thread_id = "T-10" # 模拟固定线程ID
    
    colors = {
        "INFO": "\033[1;34m",   # 蓝色
        "SUCCESS": "\033[1;32m", # 绿色
        "ERROR": "\033[1;31m",   # 红色
        "RESET": "\033[0m"
    }
    
    icons = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "ERROR": "❌"
    }
    
    color = colors.get(level, colors["RESET"])
    icon = icons.get(level, "")
    
    print(f"{color}[{curr_time}][{thread_id}] {icon} [{node_name}] {message}{colors['RESET']}")


def log_full_request(node_name, url, data):
    """Log full request details (truncating Base64)"""
    try:
        # Create a shallow copy for logging to avoid modifying actual data
        log_data = data.copy()
        if "image" in log_data:
            if isinstance(log_data["image"], str) and len(log_data["image"]) > 200:
                log_data["image"] = log_data["image"][:50] + "...[Base64 len=" + str(len(log_data["image"])) + "]..." + log_data["image"][-20:]
            elif isinstance(log_data["image"], list):
                log_data["image"] = [
                    (img[:50] + f"...[Base64 len={len(img)}]..." + img[-20:] if isinstance(img, str) and len(img) > 200 else img)
                    for img in log_data["image"]
                ]
        
        print(f"\n================ [REQUEST DEBUG: {node_name}] ================")
        print(f"URL: {url}")
        print(f"Data:\n{json.dumps(log_data, indent=2, ensure_ascii=False)}")
        print("==============================================================\n")
    except Exception as e:
        print(f"[Log Error] Failed to log request: {e}")

def log_full_response(node_name, response):
    """Log full response details"""
    try:
        print(f"\n================ [RESPONSE DEBUG: {node_name}] ================")
        print(f"Status Code: {response.status_code}")
        print("Headers:")
        print(json.dumps(dict(response.headers), indent=2, ensure_ascii=False))
        
        try:
            resp_json = response.json()
            # Deep copy or partial copy to truncate b64 in response log
            log_json = json.loads(json.dumps(resp_json)) 
            if "data" in log_json and isinstance(log_json["data"], list):
                for item in log_json["data"]:
                    if "b64_json" in item and item["b64_json"] and len(item["b64_json"]) > 200:
                        item["b64_json"] = item["b64_json"][:50] + f"...[Base64 len={len(item['b64_json'])}]..."
            
            print("Body (JSON):")
            print(json.dumps(log_json, indent=2, ensure_ascii=False))
        except:
            print("Body (Text):")
            print(response.text[:2000] + ("..." if len(response.text)>2000 else ""))
            
        print("===============================================================\n")
    except Exception as e:
        print(f"[Log Error] Failed to log response: {e}")


def mask_key(key):
    """API Key 脱敏处理"""
    if not key or not isinstance(key, str):
        return "***"
    if len(key) <= 6:
        return "***"
    return f"{key[:3]}***{key[-3:]}"
