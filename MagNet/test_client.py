"""
测试 LLM 客户端连接
"""
import os
from pathlib import Path
import sys

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_client import create_client_from_env, create_aliyun_client

def load_env_file():
    """从 .env 文件加载环境变量"""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print(f"正在加载环境变量配置: {env_file}")
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # 移除引号
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value
        print("环境变量加载完成")
    else:
        print("警告: 未找到 .env 文件")
        return False
    return True

def test_client():
    """测试客户端连接"""
    # 加载环境变量
    if not load_env_file():
        return False

    print("\n=== 环境变量检查 ===")
    print(f"LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
    print(f"LLM_MODEL: {os.getenv('LLM_MODEL')}")
    print(f"LLM_BASE_URL: {os.getenv('LLM_BASE_URL')}")
    api_key = os.getenv('LLM_API_KEY')
    print(f"LLM_API_KEY: {api_key[:10]}..." if api_key else "LLM_API_KEY: None")

    try:
        print("\n=== 创建客户端 ===")
        client = create_client_from_env()
        print(f"客户端创建成功!")
        print(f"使用模型: {client.config.model}")
        print(f"API 提供商: {client.config.provider}")

        print("\n=== 测试简单生成 ===")
        test_prompt = "你好,请用一句话介绍你自己。"
        response = client.completion(prompt=test_prompt)
        print(f"响应: {response.choices[0].text}")

        print("\n✅ 测试成功!")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_client()
    sys.exit(0 if success else 1)
