#!/usr/bin/env python3
"""
PANAS (积极消极情感量表) 评估

评估咨询后客户的情感状态，需要结合原始数据集中的intake_form

用法:
    python panas.py \
        --input output/sessions \
        --dataset dataset/test_data.json \
        --output results/panas \
        --max_iter 3
"""

import argparse
import json
import os
import re
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

# 20种情感
FEELINGS = [
    'interested', 'excited', 'strong', 'enthusiastic', 'proud',
    'alert', 'inspired', 'determined', 'attentive', 'active',
    'distressed', 'upset', 'guilty', 'scared', 'hostile',
    'irritable', 'ashamed', 'nervous', 'jittery', 'afraid'
]


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
    """计算API成本"""
    input_cost = (input_tokens * 5) / 1000000
    output_cost = (output_tokens * 20) / 1000000
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


def parse_panas_response(response_text: str) -> dict:
    """解析PANAS评估响应"""
    scores = {feeling: [] for feeling in FEELINGS}

    lines = response_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 解析格式: "Feeling: explanation, score"
        parts = line.split(',')
        if len(parts) >= 2:
            feeling_part = parts[0].strip()
            score_part = parts[-1].strip()

            # 提取情感名称
            feeling = re.sub(r'[^a-zA-Z]', '', feeling_part).lower()

            # 提取分数
            try:
                score = int(re.sub(r'[^0-9]', '', score_part))
                if feeling in scores:
                    scores[feeling].append(score)
            except ValueError:
                pass

    return scores


def evaluate_panas_single_session(
    session_data: dict,
    intake_form: str,
    attitude: str,
    evaluator,
    prompts_dir: str,
    max_iter: int = 3,
    model_name: str = AZURE_DEPLOYMENT
) -> dict:
    """
    评估单个会话的PANAS分数（咨询后）

    返回20种情感的分数，每种1-5分
    """
    prompt_text = load_prompt(prompts_dir, "panas_after.txt")
    prompt_template = PromptTemplate(
        input_variables=["intake_form", "conversation"],
        template=prompt_text
    )

    prompt = prompt_template.format(
        intake_form=intake_form,
        conversation=format_history(session_data["history"])
    )

    messages = [{"role": "user", "content": prompt}]

    response = evaluator.chat.completions.create(
        messages=messages,
        temperature=0,
        model=model_name,
        n=max_iter
    )

    # 收集所有响应的分数
    all_scores = {feeling: [] for feeling in FEELINGS}
    total_cost = 0

    for choice in response.choices:
        output_text = choice.message.content
        scores = parse_panas_response(output_text)

        for feeling in FEELINGS:
            if feeling in scores and scores[feeling]:
                all_scores[feeling].extend(scores[feeling])

        # 计算成本
        input_tokens = num_tokens_from_messages(messages)
        output_tokens = num_tokens_from_string(output_text)
        total_cost += calculate_cost(input_tokens, output_tokens)

    # 计算平均分
    avg_scores = {}
    for feeling in FEELINGS:
        if all_scores[feeling]:
            avg_scores[feeling] = sum(all_scores[feeling]) / len(all_scores[feeling])
        else:
            avg_scores[feeling] = 0.0

    avg_scores["cost"] = total_cost
    avg_scores["attitude"] = attitude

    return avg_scores


def main():
    parser = argparse.ArgumentParser(description="PANAS (积极消极情感量表) 评估")
    parser.add_argument("--input", type=str, required=True,
                        help="输入目录路径，包含session_X.json文件")
    parser.add_argument("--dataset", type=str, required=True,
                        help="原始数据集JSON文件路径，包含intake_form")
    parser.add_argument("--output", type=str, required=True,
                        help="输出目录路径")
    parser.add_argument("--max_iter", type=int, default=3,
                        help="评估次数，取平均值（默认3）")
    parser.add_argument("--prompts_dir", type=str,
                        default="prompts/panas",
                        help="PANAS prompt模板目录")
    parser.add_argument("--use_dashscope", action="store_true",
                        help="使用阿里云DashScope API进行评估")
    parser.add_argument("--dashscope_model", type=str, default=DASHSCOPE_MODEL,
                        help=f"阿里云模型名称（默认: {DASHSCOPE_MODEL}）")
    parser.add_argument("--use_openrouter", action="store_true",
                        help="使用OpenRouter API进行评估（推荐）")
    parser.add_argument("--openrouter_model", type=str, default=OPENROUTER_MODEL,
                        help=f"OpenRouter模型名称（默认: {OPENROUTER_MODEL}）")

    args = parser.parse_args()

    # 加载原始数据集
    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} client profiles")

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
    print(f"Evaluating PANAS with {args.max_iter} iterations...\n")

    # 评估每个会话
    all_scores = {}
    for i, session_file in enumerate(session_files, 1):
        print(f"[{i}/{len(session_files)}] Processing {session_file.name}...")

        with open(session_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)

        # 从文件名提取索引
        match = re.search(r'(\d+)', session_file.stem)
        if match:
            index = int(match.group(1)) - 1
        else:
            print(f"  ✗ Cannot extract index from filename")
            continue

        if index >= len(dataset):
            print(f"  ✗ Index {index} out of range")
            continue

        # 获取客户信息
        intake_form = dataset[index]['AI_client']['intake_form']
        attitude = dataset[index]['AI_client']['attitude']

        try:
            scores = evaluate_panas_single_session(
                session_data,
                intake_form,
                attitude,
                evaluator,
                args.prompts_dir,
                args.max_iter,
                model_name
            )
            all_scores[session_file.name] = scores

            # 打印结果
            positive_avg = sum(scores[f] for f in FEELINGS[:10]) / 10
            negative_avg = sum(scores[f] for f in FEELINGS[10:]) / 10

            print(f"  Positive Affect: {positive_avg:.2f}/5")
            print(f"  Negative Affect: {negative_avg:.2f}/5")
            print(f"  Attitude: {attitude}")
            print(f"  Cost: ${scores['cost']:.4f}")
            print()

        except Exception as e:
            print(f"  ✗ Error: {e}")
            all_scores[session_file.name] = {"error": str(e)}

    # 计算平均分
    valid_scores = {k: v for k, v in all_scores.items() if "error" not in v}

    if valid_scores:
        avg_scores = {}
        for key in list(valid_scores.values())[0].keys():
            if key not in ["cost", "attitude"]:
                avg_scores[key] = sum(v[key] for v in valid_scores.values()) / len(valid_scores)

        avg_scores["total_cost"] = sum(v.get("cost", 0) for v in valid_scores.values())

        # 计算积极和消极情感平均分
        avg_scores["positive_affect"] = sum(avg_scores[f] for f in FEELINGS[:10]) / 10
        avg_scores["negative_affect"] = sum(avg_scores[f] for f in FEELINGS[10:]) / 10

        # 打印汇总
        print("=" * 60)
        print("AVERAGE PANAS SCORES (AFTER COUNSELING):")
        print("-" * 60)
        print(f"Positive Affect: {avg_scores['positive_affect']:.2f}/5")
        print(f"Negative Affect: {avg_scores['negative_affect']:.2f}/5")
        print("-" * 60)

        # 按态度分组
        attitude_groups = {}
        for session_name, scores in valid_scores.items():
            attitude = scores.get("attitude", "unknown")
            if attitude not in attitude_groups:
                attitude_groups[attitude] = []
            attitude_groups[attitude].append(scores)

        print("By Client Attitude:")
        for attitude, group_scores in attitude_groups.items():
            pos_avg = sum(s[f] for s in group_scores for f in FEELINGS[:10]) / (len(group_scores) * 10)
            neg_avg = sum(s[f] for s in group_scores for f in FEELINGS[10:]) / (len(group_scores) * 10)
            print(f"  {attitude.capitalize()}: Positive={pos_avg:.2f}, Negative={neg_avg:.2f}")

        print("-" * 60)
        print(f"Total Cost: ${avg_scores['total_cost']:.4f}")
        print("=" * 60)

        # 保存结果
        results = {
            "average": avg_scores,
            "per_session": all_scores,
            "by_attitude": {
                att: {
                    "positive_affect": sum(s[f] for s in scores for f in FEELINGS[:10]) / (len(scores) * 10),
                    "negative_affect": sum(s[f] for s in scores for f in FEELINGS[10:]) / (len(scores) * 10)
                }
                for att, scores in attitude_groups.items()
            },
            "num_sessions": len(all_scores),
            "num_valid": len(valid_scores)
        }

        output_file = output_dir / "panas_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
