"""
测试阿里云API调用 - 模拟官方调用方式
"""
import os
from pathlib import Path
import sys

# 设置输出编码
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def load_env_file():
    """加载环境变量"""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print(f"正在加载环境变量: {env_file}")
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
        print("环境变量加载完成\n")
    else:
        print("警告: 未找到 .env 文件\n")

def test_official_method():
    """测试官方调用方式"""
    print("=" * 60)
    print("测试1: 官方调用方式 (chat.completions)")
    print("=" * 60)
    print()

    try:
        from openai import OpenAI

        # 获取API密钥
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            print("错误: 未设置 LLM_API_KEY")
            return False

        print(f"API密钥: {api_key[:10]}...{api_key[-4:]}")
        print(f"模型: {os.getenv('LLM_MODEL', 'qwen2.5-7b-instruct')}")

        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        print("\n正在发送测试请求...")
        completion = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "qwen2.5-7b-instruct"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "你好，请用一句话介绍你自己。"},
            ],
            stream=False  # 先用非流式测试
        )

        response_text = completion.choices[0].message.content
        print(f"\n响应: {response_text}")
        print("\n✅ 官方调用方式测试成功!")
        return True

    except Exception as e:
        print(f"\n❌ 官方调用方式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_our_method():
    """测试我们的LLMClient调用方式"""
    print("\n" + "=" * 60)
    print("测试2: LLMClient调用方式 (completion方法)")
    print("=" * 60)
    print()

    try:
        from llm_client import create_client_from_env

        print("正在创建LLMClient...")
        llm_client = create_client_from_env()

        print(f"提供商: {llm_client.config.provider}")
        print(f"模型: {llm_client.config.model}")
        print(f"Base URL: {llm_client.config.base_url}")

        print("\n正在发送测试请求...")
        response = llm_client.completion(
            prompt="你好，请用一句话介绍你自己。"
        )

        response_text = response.choices[0].message.content
        print(f"\n响应: {response_text}")
        print("\n✅ LLMClient调用方式测试成功!")
        return True

    except Exception as e:
        print(f"\n❌ LLMClient调用方式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 60)
    print("  阿里云API调用测试")
    print("=" * 60)
    print()

    # 加载环境变量
    load_env_file()

    # 测试官方方式
    result1 = test_official_method()

    # 测试我们的方式
    result2 = test_our_method()

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"官方调用方式: {'✅ 通过' if result1 else '❌ 失败'}")
    print(f"LLMClient方式: {'✅ 通过' if result2 else '❌ 失败'}")
    print("=" * 60)

    if result1 and result2:
        print("\n所有测试通过! 代码已修复成功!")
        return True
    else:
        print("\n部分测试失败,请检查错误信息")
        return False

if __name__ == "__main__":
    success = main()
    input("\n按回车键退出...")
    sys.exit(0 if success else 1)
