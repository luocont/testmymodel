"""
验证环境变量加载是否正确
"""
import os
from pathlib import Path

def load_env_file():
    """从 .env 文件加载环境变量 - 与 inference-parallel-magnet-cn.py 相同的逻辑"""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print(f"正在加载环境变量配置: {env_file}")
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # 去除键值对两端的空格和引号
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
        print("环境变量加载完成\n")
        return True
    else:
        print("警告: 未找到 .env 文件\n")
        return False

def main():
    print("=" * 60)
    print("  验证环境变量加载")
    print("=" * 60)
    print()

    # 加载环境变量
    load_env_file()

    # 检查关键配置
    print("=" * 60)
    print("  检查加载的配置")
    print("=" * 60)
    print()

    vars_to_check = [
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_TEMPERATURE",
        "LLM_MAX_TOKENS",
    ]

    for var in vars_to_check:
        value = os.getenv(var)
        if value:
            if "API_KEY" in var:
                # 显示API密钥的前10个和后4个字符
                if len(value) > 14:
                    masked = value[:10] + "..." + value[-4:]
                else:
                    masked = "***"
                print(f"{var}: {masked}")
            else:
                print(f"{var}: {value}")

            # 检查值是否包含引号
            if value.startswith('"') or value.startswith("'"):
                print(f"  ⚠️  警告: 值包含引号!")
            if value.endswith('"') or value.endswith("'"):
                print(f"  ⚠️  警告: 值包含引号!")
        else:
            print(f"{var}: (未设置)")

    print()
    print("=" * 60)

    # 特别检查模型名称
    model = os.getenv("LLM_MODEL")
    if model == "qwen2.5-7b-instruct":
        print("⚠️  模型名称 'qwen2.5-7b-instruct' 不被阿里云OpenAI兼容模式支持!")
        print("   请将 LLM_MODEL 改为: qwen-plus, qwen-turbo 或 qwen-max")
    elif model in ["qwen-plus", "qwen-turbo", "qwen-max", "qwen-max-longcontext"]:
        print(f"✅ 模型名称 '{model}' 有效")
    else:
        print(f"⚠️  未知的模型名称: '{model}'")

    print("=" * 60)

if __name__ == "__main__":
    main()
    input("\n按回车键退出...")
