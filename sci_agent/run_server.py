"""
启动SciAgent Web服务器
"""
import os
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载 .env 文件
def load_env():
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key and value and key not in os.environ:
                        os.environ[key] = value
        print("[Info] 已加载 .env 配置")

load_env()

# 调试：确认 API Key 已加载
api_key = os.environ.get("SILICONFLOW_API_KEY", "")
if api_key:
    print(f"[Info] SILICONFLOW_API_KEY 已加载 (长度: {len(api_key)})")
else:
    print("[Warning] SILICONFLOW_API_KEY 未设置!")


def main():
    import uvicorn
    
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8080))
    reload = os.environ.get("RELOAD", "false").lower() == "true"
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                    SciAgent 启动中                        ║
╠══════════════════════════════════════════════════════════╣
║  地址: http://{host}:{port}                              
║  API文档: http://{host}:{port}/docs                      
║  前端界面: http://{host}:{port}/                         
╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "sci_agent.api.app:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    main()
