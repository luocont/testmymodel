#!/usr/bin/env python3
"""
WAI (工作联盟量表) 评估

用法:
    python wai.py \
        --input output/sessions \
        --output results/wai \
        --max_iter 3
"""

import argparse
import json
import os
from pathlib import Path
from langchain.prompts import PromptTemplate
from openai import AzureOpenAI
import tiktoken


# ========== 配置 ==========
AZURE_ENDPOINT = "your_azure_endpoint"
AZURE_API_KEY = "your_subscription_key"
AZURE_API_VERSION = "2024-02-01"
AZURE_DEPLOYMENT = "gpt-4o"


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


def evaluate_wai_single_session(
    session_data: dict,
    evaluator: AzureOpenAI,
    prompts_dir: str,
    max_iter: int = 3
) -> dict:
    """
    评估单个会话的WAI分数

    WAI包含12个项目，评估治疗联盟的三个维度：
    - Goal: 目标一致性 (WAI-4, 5, 6, 7)
    - Task: 任务一致性 (WAI-1, 2, 12)
    - Bond: 情感纽带 (WAI-3, 8, 9, 10, 11)
    """
    scores = {}
    total_cost = 0
    history = format_history(session_data["history"])

    # 评估12个WAI项目
    for i in range(1, 13):
        prompt_text = load_prompt(prompts_dir, f"wai{i}.txt")
        prompt_template = PromptTemplate(
            input_variables=["conversation"],
            template=prompt_text
        )

        prompt = prompt_template.format(conversation=history)
        messages = [{"role": "user", "content": prompt}]

        response = evaluator.chat.completions.create(
            messages=messages,
            temperature=0,
            model=AZURE_DEPLOYMENT,
            n=max_iter
        )

        # 计算分数
        item_scores = []
        for choice in response.choices:
            output_text = choice.message.content
            # 提取分数 (格式: "分数, 解释")
            score = int(output_text.split(",")[0].strip())
            item_scores.append(score)

            # 计算成本
            input_tokens = num_tokens_from_messages(messages)
            output_tokens = num_tokens_from_string(output_text)
            total_cost += calculate_cost(input_tokens, output_tokens)

        # 取平均分
        avg_score = sum(item_scores) / len(item_scores)
        scores[f"wai{i}"] = avg_score

    scores["cost"] = total_cost
    return scores


def main():
    parser = argparse.ArgumentParser(description="WAI (工作联盟量表) 评估")
    parser.add_argument("--input", type=str, required=True,
                        help="输入目录路径，包含session_X.json文件")
    parser.add_argument("--output", type=str, required=True,
                        help="输出目录路径")
    parser.add_argument("--max_iter", type=int, default=3,
                        help="每个项目评估的次数，取平均值（默认3）")
    parser.add_argument("--prompts_dir", type=str,
                        default="prompts/wai",
                        help="WAI prompt模板目录")

    args = parser.parse_args()

    # 初始化Azure OpenAI客户端
    evaluator = AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    )

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有session文件
    session_files = sorted(input_dir.glob("session_*.json"))

    if not session_files:
        print(f"Error: No session files found in {input_dir}")
        return

    print(f"Found {len(session_files)} session files")
    print(f"Evaluating {12} WAI items with {args.max_iter} iterations each...\n")

    # 评估每个会话
    all_scores = {}
    for i, session_file in enumerate(session_files, 1):
        print(f"[{i}/{len(session_files)}] Processing {session_file.name}...")

        with open(session_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)

        try:
            scores = evaluate_wai_single_session(
                session_data,
                evaluator,
                args.prompts_dir,
                args.max_iter
            )
            all_scores[session_file.name] = scores

            # 打印结果
            goal_avg = sum(scores[f"wai{i}"] for i in [4, 5, 6, 7]) / 4
            task_avg = sum(scores[f"wai{i}"] for i in [1, 2, 12]) / 3
            bond_avg = sum(scores[f"wai{i}"] for i in [3, 8, 9, 10, 11]) / 5
            overall_avg = sum(scores[f"wai{i}"] for i in range(1, 13)) / 12

            print(f"  Goal: {goal_avg:.2f}/7")
            print(f"  Task: {task_avg:.2f}/7")
            print(f"  Bond: {bond_avg:.2f}/7")
            print(f"  Overall: {overall_avg:.2f}/7")
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
            if key != "cost":
                avg_scores[key] = sum(v[key] for v in valid_scores.values()) / len(valid_scores)

        avg_scores["total_cost"] = sum(v.get("cost", 0) for v in valid_scores.values())

        # 计算子维度平均分
        avg_scores["goal_avg"] = sum(avg_scores[f"wai{i}"] for i in [4, 5, 6, 7]) / 4
        avg_scores["task_avg"] = sum(avg_scores[f"wai{i}"] for i in [1, 2, 12]) / 3
        avg_scores["bond_avg"] = sum(avg_scores[f"wai{i}"] for i in [3, 8, 9, 10, 11]) / 5
        avg_scores["overall_avg"] = sum(avg_scores[f"wai{i}"] for i in range(1, 13)) / 12

        # 打印汇总
        print("=" * 60)
        print("AVERAGE WAI SCORES:")
        print("-" * 60)
        print(f"Goal Dimension: {avg_scores['goal_avg']:.2f}/7")
        print(f"Task Dimension: {avg_scores['task_avg']:.2f}/7")
        print(f"Bond Dimension: {avg_scores['bond_avg']:.2f}/7")
        print(f"Overall: {avg_scores['overall_avg']:.2f}/7")
        print("-" * 60)
        print(f"Total Cost: ${avg_scores['total_cost']:.4f}")
        print("=" * 60)

        # 保存结果
        results = {
            "average": avg_scores,
            "per_session": all_scores,
            "num_sessions": len(all_scores),
            "num_valid": len(valid_scores)
        }

        output_file = output_dir / "wai_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
