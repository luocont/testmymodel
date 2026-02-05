"""
API 使用示例脚本
演示如何为 MagNet 多智能体框架和评估系统配置不同的 API 提供商
"""

import os
from pathlib import Path

# 添加父目录到路径以导入模块
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent / "evaluation"))

from llm_client import (
    LLMClient,
    create_client_from_env,
    create_aliyun_client,
    create_openrouter_client,
    create_local_client,
    APIConfig,
    APIProvider,
)

from llm_evaluator import (
    EvalLLMClient,
    create_eval_client_from_env,
    create_eval_aliyun_client,
    create_eval_openrouter_client,
    create_eval_azure_client,
    EvalAPIConfig,
    EvalAPIProvider,
)


def example_1_aliyun_only():
    """示例 1: 全部使用阿里云百炼"""
    print("\n=== 示例 1: 全部使用阿里云百炼 ===\n")

    # 多智能体框架使用阿里云
    client = create_aliyun_client(
        api_key=os.getenv("DASHSCOPE_API_KEY", "your_key_here"),
        model="qwen2.5-7b-instruct",
        temperature=0.7,
        max_tokens=512
    )
    print(f"多智能体客户端: {client.config.provider.value} - {client.config.model}")

    # 评估系统使用阿里云
    evaluator = create_eval_aliyun_client(
        api_key=os.getenv("DASHSCOPE_API_KEY", "your_key_here"),
        model="qwen-max",
        temperature=0,
        max_tokens=4096
    )
    print(f"评估客户端: {evaluator.config.provider.value} - {evaluator.config.model}")


def example_2_openrouter_only():
    """示例 2: 全部使用 OpenRouter"""
    print("\n=== 示例 2: 全部使用 OpenRouter ===\n")

    # 多智能体框架使用 OpenRouter
    client = create_openrouter_client(
        api_key=os.getenv("OPENROUTER_API_KEY", "your_key_here"),
        model="openai/chatgpt-4o-latest",
        temperature=0.7,
        max_tokens=512
    )
    print(f"多智能体客户端: {client.config.provider.value} - {client.config.model}")

    # 评估系统使用 OpenRouter
    evaluator = create_eval_openrouter_client(
        api_key=os.getenv("OPENROUTER_API_KEY", "your_key_here"),
        model="openai/gpt-4o",
        temperature=0,
        max_tokens=4096
    )
    print(f"评估客户端: {evaluator.config.provider.value} - {evaluator.config.model}")


def example_3_mixed_apis():
    """示例 3: 混合使用不同 API"""
    print("\n=== 示例 3: 多智能体使用阿里云，评估使用 OpenRouter ===\n")

    # 多智能体框架使用阿里云（成本较低）
    client = create_aliyun_client(
        api_key=os.getenv("DASHSCOPE_API_KEY", "your_key_here"),
        model="qwen2.5-7b-instruct",
    )
    print(f"多智能体客户端: {client.config.provider.value} - {client.config.model}")

    # 评估系统使用 OpenRouter（质量更高）
    evaluator = create_eval_openrouter_client(
        api_key=os.getenv("OPENROUTER_API_KEY", "your_key_here"),
        model="openai/gpt-4o",
    )
    print(f"评估客户端: {evaluator.config.provider.value} - {evaluator.config.model}")


def example_4_local_with_cloud_eval():
    """示例 4: 本地模型 + 云端评估"""
    print("\n=== 示例 4: 本地 vLLM + 阿里云评估 ===\n")

    # 多智能体框架使用本地 vLLM（免费）
    client = create_local_client(
        base_url="http://localhost:8000/v1",
        model="model_name",
    )
    print(f"多智能体客户端: {client.config.provider.value} - {client.config.model}")

    # 评估系统使用阿里云（保证质量）
    evaluator = create_eval_aliyun_client(
        api_key=os.getenv("DASHSCOPE_API_KEY", "your_key_here"),
        model="qwen-max",
    )
    print(f"评估客户端: {evaluator.config.provider.value} - {evaluator.config.model}")


def example_5_from_env():
    """示例 5: 从环境变量加载配置"""
    print("\n=== 示例 5: 从环境变量加载配置 ===\n")

    # 设置环境变量
    os.environ["LLM_PROVIDER"] = "aliyun"
    os.environ["LLM_API_KEY"] = "your_key_here"
    os.environ["LLM_MODEL"] = "qwen2.5-7b-instruct"

    os.environ["EVAL_LLM_PROVIDER"] = "openrouter"
    os.environ["EVAL_LLM_API_KEY"] = "your_key_here"
    os.environ["EVAL_LLM_MODEL"] = "openai/gpt-4o"

    # 从环境变量创建客户端
    client = create_client_from_env()
    print(f"多智能体客户端: {client.config.provider.value} - {client.config.model}")

    evaluator = create_eval_client_from_env()
    print(f"评估客户端: {evaluator.config.provider.value} - {evaluator.config.model}")


def example_6_custom_config():
    """示例 6: 使用 APIConfig 自定义配置"""
    print("\n=== 示例 6: 自定义配置 ===\n")

    # 自定义阿里云配置
    config = APIConfig(
        provider=APIProvider.ALIYUN,
        api_key="your_key_here",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen2.5-14b-instruct",
        temperature=0.8,
        max_tokens=1024,
    )
    client = LLMClient(config)
    print(f"自定义多智能体客户端: {client.config.provider.value} - {client.config.model}")
    print(f"温度: {client.config.temperature}, 最大 tokens: {client.config.max_tokens}")


def example_7_chat_completion():
    """示例 7: 发送聊天请求"""
    print("\n=== 示例 7: 发送聊天请求 ===\n")

    client = create_aliyun_client(
        api_key=os.getenv("DASHSCOPE_API_KEY", "your_key_here"),
        model="qwen2.5-7b-instruct",
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，请介绍一下你自己。"},
    ]

    print("发送消息...")
    response = client.chat_completion(messages=.messages)

    if hasattr(response, 'choices'):
        print(f"响应: {response.choices[0].message.content}")
    else:
        print("注意: 需要有效的 API Key 才能发送实际请求")


def example_8_cost_estimation():
    """示例 8: 成本估算"""
    print("\n=== 示例 8: 成本估算 ===\n")

    from llm_evaluator import get_eval_cost

    models = [
        "gpt-4o",
        "gpt-4o-mini",
        "qwen-max",
        "openai/gpt-4o",
    ]

    print("模型成本对比 (每 100 万 tokens):")
    print("-" * 50)
    for model in models:
        input_cost, output_cost = get_eval_cost(model)
        print(f"{model:25} | 输入: ${input_cost:5.2f} | 输出: ${output_cost:5.2f}")


def main():
    """运行所有示例"""
    print("=" * 60)
    print("MagNet API 使用示例")
    print("=" * 60)

    # 运行所有示例
    example_1_aliyun_only()
    example_2_openrouter_only()
    example_3_mixed_apis()
    example_4_local_with_cloud_eval()
    example_5_from_env()
    example_6_custom_config()
    example_7_chat_completion()
    example_8_cost_estimation()

    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)

    print("\n提示: 将示例中的 'your_key_here' 替换为实际的 API Key 即可使用。")
    print("\n更多信息请参考: docs/API_CONFIG_GUIDE.md")


if __name__ == "__main__":
    main()
