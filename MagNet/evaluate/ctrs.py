#!/usr/bin/env python3
"""
CTRS (认知治疗评分量表) 评估

用法:
    python ctrs.py \
        --input output/sessions \
        --output results/ctrs \
        --max_iter 3
"""

import argparse
import json
import os
from pathlib import Path
from langchain.prompts import PromptTemplate
from openai import AzureOpenAI, OpenAI
import tiktoken


# ========== 配置 ==========
# Azure OpenAI 凭据
AZURE_ENDPOINT = "your_azure_endpoint"
AZURE_API_KEY = "your_subscription_key"
AZURE_API_VERSION = "2024-02-01"
AZURE_DEPLOYMENT = "gpt-4o"

# 阿里云 DashScope API 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL = "qwen2.5-7b-instruct"

# OpenRouter API 配置 (用于评估)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/gpt-4o"
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "")

# 默认使用的API提供商
# 可选值: "azure", "dashscope"
EVALUATION_PROVIDER = "azure"


def num_tokens_from_messages(messages, model="gpt-4o"):
    """计算消息的token数量"""
    enc = tiktoken.encoding_for_model(model)
    tokens_per_message = 3
    total_tokens = 0

    for message in messages:
        total_tokens += tokens_per_message + len(enc.encode(message["content"]))

    total_tokens += 3
    return total_tokens


def num_tokens_from_string(string: str, model="gpt-4o") -> int:
    """计算字符串的token数量"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(string))


def calculate_cost(input_tokens, output_tokens):
    """计算API成本 (美元)"""
    input_cost = (input_tokens * 5) / 1000000  # $5 per million input tokens
    output_cost = (output_tokens * 20) / 1000000  # $20 per million output tokens
    return input_cost + output_cost


def format_history(history: list) -> str:
    """格式化对话历史"""
    return '\n'.join([
        f"{msg['role'].capitalize()}: {msg['message']}"
        for msg in history
    ])


def load_prompt(prompts_dir: str, filename: str) -> str:
    """加载prompt模板"""
    file_path = os.path.join(prompts_dir, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def evaluate_ctrs_single_session(
    session_data: dict,
    evaluator,
    prompts_dir: str,
    max_iter: int = 3,
    model_name: str = AZURE_DEPLOYMENT
) -> dict:
    """
    评估单个会话的CTRS分数

    CTRS包含6个维度：
    - 通用技能 (3个): understanding, interpersonal_effectiveness, collaboration
    - CBT技能 (3个): guided_discovery, focus, strategy
    """
    ctrs_dimensions = [
        "general_1_understanding",
        "general_2_interpersonal_effectiveness",
        "general_3_collaboration",
        "CBT_1_guided_discovery",
        "CBT_2_focus",
        "CBT_3_strategy"
    ]

    history = format_history(session_data["history"])
    scores = {}
    total_cost = 0

    for dimension in ctrs_dimensions:
        # 加载对应维度的prompt
        prompt_text = load_prompt(prompts_dir, f"{dimension}.txt")
        prompt_template = PromptTemplate(
            input_variables=["conversation"],
            template=prompt_text
        )

        prompt = prompt_template.format(conversation=history)
        messages = [{"role": "user", "content": prompt}]

        # 调用LLM评估，多次运行取平均
        response = evaluator.chat.completions.create(
            messages=messages,
            temperature=0,
            model=model_name,
            n=max_iter
        )

        # 计算分数和成本
        dimension_scores = []
        for choice in response.choices:
            output_text = choice.message.content
            # 提取分数 (格式: "分数, 解释")
            score = int(output_text.split(",")[0].strip())
            dimension_scores.append(score)

            # 计算成本
            input_tokens = num_tokens_from_messages(messages)
            output_tokens = num_tokens_from_string(output_text)
            total_cost += calculate_cost(input_tokens, output_tokens)

        # 取平均分
        avg_score = sum(dimension_scores) / len(dimension_scores)
        scores[dimension] = avg_score

    scores["cost"] = total_cost
    return scores


def main():
    parser = argparse.ArgumentParser(description="CTRS (认知治疗评分量表) 评估")
    parser.add_argument("--input", type=str, required=True,
                        help="输入目录路径，包含session_X.json文件")
    parser.add_argument("--output", type=str, required=True,
                        help="输出目录路径")
    parser.add_argument("--max_iter", type=int, default=3,
                        help="每个维度评估的次数，取平均值（默认3）")
    parser.add_argument("--prompts_dir", type=str,
                        default="prompts/ctrs",
                        help="CTRS prompt模板目录")
    parser.add_argument("--use_dashscope", action="store_true",
                        help="使用阿里云DashScope API进行评估")
    parser.add_argument("--dashscope_model", type=str, default=DASHSCOPE_MODEL,
                        help=f"阿里云模型名称（默认: {DASHSCOPE_MODEL}）")
    parser.add_argument("--use_openrouter", action="store_true",
                        help="使用OpenRouter API进行评估（推荐）")
    parser.add_argument("--openrouter_model", type=str, default=OPENROUTER_MODEL,
                        help=f"OpenRouter模型名称（默认: {OPENROUTER_MODEL}）")

    args = parser.parse_args()

    # 初始化评估客户端
    if args.use_openrouter:
        if not OPENROUTER_API_KEY:
            print("错误: 使用OpenRouter API需要设置 OPENROUTER_API_KEY 环境变量")
            print("请在终端中运行: set OPENROUTER_API_KEY=sk-or-xxx  (Windows)")
            print("或: export OPENROUTER_API_KEY=sk-or-xxx  (Linux/Mac)")
            return
        extra_headers = {}
        if OPENROUTER_SITE_URL:
            extra_headers["HTTP-Referer"] = OPENROUTER_SITE_URL
        if OPENROUTER_SITE_NAME:
            extra_headers["X-Title"] = OPENROUTER_SITE_NAME

        evaluator = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            default_headers=extra_headers if extra_headers else None
        )
        model_name = args.openrouter_model
        print(f"Using OpenRouter API with model: {model_name}")
    elif args.use_dashscope:
        if not DASHSCOPE_API_KEY:
            print("错误: 使用阿里云API需要设置 DASHSCOPE_API_KEY 环境变量")
            print("请在终端中运行: set DASHSCOPE_API_KEY=sk-xxx  (Windows)")
            print("或: export DASHSCOPE_API_KEY=sk-xxx  (Linux/Mac)")
            return
        evaluator = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL
        )
        model_name = args.dashscope_model
        print(f"Using 阿里云DashScope API with model: {model_name}")
    else:
        evaluator = AzureOpenAI(
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
        )
        model_name = AZURE_DEPLOYMENT
        print(f"Using Azure OpenAI with model: {model_name}")

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有session文件
    session_files = sorted(input_dir.glob("session_*.json"))

    if not session_files:
        print(f"Error: No session files found in {input_dir}")
        return

    print(f"Found {len(session_files)} session files")
    print(f"Evaluating with {args.max_iter} iterations per dimension...\n")

    # 评估每个会话
    all_scores = {}
    for i, session_file in enumerate(session_files, 1):
        print(f"[{i}/{len(session_files)}] Processing {session_file.name}...")

        with open(session_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)

        try:
            scores = evaluate_ctrs_single_session(
                session_data,
                evaluator,
                args.prompts_dir,
                args.max_iter,
                model_name
            )
            all_scores[session_file.name] = scores

            # 打印结果
            print(f"  Understanding: {scores['general_1_understanding']:.2f}")
            print(f"  Interpersonal: {scores['general_2_interpersonal_effectiveness']:.2f}")
            print(f"  Collaboration: {scores['general_3_collaboration']:.2f}")
            print(f"  Guided Discovery: {scores['CBT_1_guided_discovery']:.2f}")
            print(f"  Focus: {scores['CBT_2_focus']:.2f}")
            print(f"  Strategy: {scores['CBT_3_strategy']:.2f}")
            print(f"  Cost: ${scores['cost']:.4f}")
            print()

        except Exception as e:
            print(f"  ✗ Error: {e}")
            all_scores[session_file.name] = {"error": str(e)}

    # 计算平均分
    valid_scores = {k: v for k, v in all_scores.items() if "error" not in v}

    if valid_scores:
        avg_scores = {}
        for key in valid_scores[list(valid_scores.keys())[0]].keys():
            if key != "cost":
                avg_scores[key] = sum(v[key] for v in valid_scores.values()) / len(valid_scores)

        avg_scores["total_cost"] = sum(v.get("cost", 0) for v in valid_scores.values())

        # 打印汇总
        print("=" * 60)
        print("AVERAGE SCORES:")
        print("-" * 60)
        print("General Skills:")
        print(f"  Understanding: {avg_scores['general_1_understanding']:.2f}/6")
        print(f"  Interpersonal: {avg_scores['general_2_interpersonal_effectiveness']:.2f}/6")
        print(f"  Collaboration: {avg_scores['general_3_collaboration']:.2f}/6")
        print("\nCBT Skills:")
        print(f"  Guided Discovery: {avg_scores['CBT_1_guided_discovery']:.2f}/6")
        print(f"  Focus: {avg_scores['CBT_2_focus']:.2f}/6")
        print(f"  Strategy: {avg_scores['CBT_3_strategy']:.2f}/6")
        print("-" * 60)
        print(f"Total Cost: ${avg_scores['total_cost']:.4f}")
        print("=" * 60)

        # 保存详细结果
        results = {
            "average": avg_scores,
            "per_session": all_scores,
            "num_sessions": len(all_scores),
            "num_valid": len(valid_scores)
        }

        output_file = output_dir / "ctrs_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Results saved to {output_file}")
    else:
        print("No valid evaluations completed!")


if __name__ == "__main__":
    main()
