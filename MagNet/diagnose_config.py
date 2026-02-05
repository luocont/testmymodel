"""
诊断脚本 - 检查API配置是否正确加载
"""
import os
from pathlib import Path
import sys

# 设置输出编码为UTF-8
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def load_env_file():
    """从 .env 文件加载环境变量 - 复制自 inference-parallel-magnet-cn.py"""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print(f"✅ 找到 .env 文件: {env_file}")
        print("\n=== 正在加载环境变量 ===")
        with open(env_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # 移除可能的引号
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value
                    # 打印加载的内容(隐藏API密钥的中间部分)
                    if "API_KEY" in key:
                        masked_value = value[:10] + "..." if len(value) > 10 else "***"
                        print(f"  行 {line_num}: {key}={masked_value}")
                    else:
                        print(f"  行 {line_num}: {key}={value}")
        print("\n✅ 环境变量加载完成\n")
        return True
    else:
        print(f"❌ 未找到 .env 文件: {env_file}")
        return False

def check_env_vars():
    """检查关键的环境变量"""
    print("=== 检查关键环境变量 ===\n")

    required_vars = {
        "LLM_PROVIDER": "API提供商",
        "LLM_API_KEY": "API密钥",
        "LLM_BASE_URL": "API基础URL",
        "LLM_MODEL": "模型名称",
    }

    all_present = True
    for var, desc in required_vars.items():
        value = os.getenv(var)
        if value:
            if "API_KEY" in var:
                masked = value[:10] + "..." + value[-4:] if len(value) > 14 else "***"
                print(f"✅ {var} ({desc}): {masked}")
            else:
                print(f"✅ {var} ({desc}): {value}")
        else:
            print(f"❌ {var} ({desc}): 未设置")
            all_present = False

    print()
    return all_present

def test_llm_client():
    """测试 LLM 客户端创建"""
    print("=== 测试 LLM 客户端创建 ===\n")

    try:
        from llm_client import create_client_from_env

        print("正在创建 LLM 客户端...")
        client = create_client_from_env()

        print(f"✅ 客户端创建成功!\n")
        print(f"配置信息:")
        print(f"  提供商: {client.config.provider}")
        print(f"  模型: {client.config.model}")
        print(f"  Base URL: {client.config.base_url}")
        print(f"  温度: {client.config.temperature}")
        print(f"  最大tokens: {client.config.max_tokens}")

        # 验证模型名称
        print(f"\n=== 模型名称验证 ===")
        model = client.config.model
        print(f"当前模型: {model}")

        # 阿里云支持的模型列表
        aliyun_supported_models = [
            "qwen-turbo",
            "qwen-plus",
            "qwen-max",
            "qwen-max-longcontext",
        ]

        if client.config.provider.value == "aliyun":
            if model in aliyun_supported_models:
                print(f"✅ 模型名称在阿里云OpenAI兼容模式支持列表中")
            else:
                print(f"❌ 模型名称 '{model}' 不在阿里云OpenAI兼容模式支持列表中!")
                print(f"\n阿里云OpenAI兼容模式支持的模型:")
                for m in aliyun_supported_models:
                    print(f"  - {m}")
                print(f"\n建议: 将模型名称改为 'qwen-plus' 或 'qwen-turbo'")

        return client

    except Exception as e:
        print(f"❌ 客户端创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 60)
    print("  MagNet API 配置诊断工具")
    print("=" * 60)
    print()

    # 步骤1: 加载 .env 文件
    if not load_env_file():
        print("\n❌ 无法继续 - 请确保 .env 文件存在")
        return False

    # 步骤2: 检查环境变量
    if not check_env_vars():
        print("\n❌ 缺少必需的环境变量")
        return False

    # 步骤3: 测试客户端创建
    client = test_llm_client()
    if not client:
        print("\n❌ 客户端创建失败")
        return False

    print("\n" + "=" * 60)
    print("✅ 所有配置检查通过!")
    print("=" * 60)
    print("\n如果仍然遇到 'Unsupported model' 错误,请:")
    print("1. 检查 .env 文件中的 LLM_MODEL 值")
    print("2. 将其改为 'qwen-plus' 或 'qwen-turbo'")
    print("3. 重新运行程序")

    return True

if __name__ == "__main__":
    success = main()
    input("\n按回车键退出...")
    sys.exit(0 if success else 1)
