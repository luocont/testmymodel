#!/usr/bin/env python3
"""
计算生成对话的多样性指标

用法:
    python diversity.py \
        --input output/sessions \
        --output results/diversity.json
"""

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import List
import re


def extract_ngrams(text: str, n: int) -> List[str]:
    """从文本中提取n-gram"""
    # 分词（简单实现，可根据需要改进）
    tokens = text.lower().split()
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def calculate_distinct(history: List[dict], n: int) -> float:
    """
    计算Distinct-n指标

    Args:
        history: 对话历史列表，每个元素包含role和message
        n: n-gram的大小

    Returns:
        Distinct-n分数
    """
    # 提取所有咨询师的回复
    counselor_messages = [
        msg["message"] for msg in history if msg["role"] == "counselor"
    ]

    if not counselor_messages:
        return 0.0

    # 提取所有n-gram
    all_ngrams = []
    for msg in counselor_messages:
        ngrams = extract_ngrams(msg, n)
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    # 计算不同的n-gram数量
    unique_ngrams = set(all_ngrams)

    # Distinct-n = |unique n-grams| / |total n-grams|
    return len(unique_ngrams) / len(all_ngrams)


def calculate_ead(history: List[dict], n: int) -> float:
    """
    计算EAD (Expectation Adjusted Distinct)

    EAD是对Distinct指标的调整，考虑了生成长度的影响
    """
    counselor_messages = [
        msg["message"] for msg in history if msg["role"] == "counselor"
    ]

    if not counselor_messages:
        return 0.0

    # 计算每条消息的n-gram多样性
    message_diversities = []
    total_ngrams = 0
    unique_ngrams = set()

    for msg in counselor_messages:
        ngrams = extract_ngrams(msg, n)
        if ngrams:
            message_diversities.append(len(set(ngrams)) / len(ngrams))
            total_ngrams += len(ngrams)
            unique_ngrams.update(ngrams)

    if not message_diversities:
        return 0.0

    # EAD = 平均多样性 * (1 - 长度惩罚)
    avg_diversity = sum(message_diversities) / len(message_diversities)

    # 简单的长度惩罚：当消息数量增加时，多样性自然下降
    # 这里使用log来平滑这个惩罚
    length_penalty = 1.0
    if len(counselor_messages) > 1:
        length_penalty = 1.0 - (1.0 / len(counselor_messages))

    return avg_diversity * length_penalty


def evaluate_session(session_data: dict) -> dict:
    """评估单个会话的多样性指标"""
    history = session_data.get("history", [])

    return {
        "distinct_1": calculate_distinct(history, 1),
        "distinct_2": calculate_distinct(history, 2),
        "distinct_3": calculate_distinct(history, 3),
        "ead": calculate_ead(history, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="计算对话的多样性指标")
    parser.add_argument("--input", type=str, required=True,
                        help="输入目录路径，包含session_X.json文件")
    parser.add_argument("--output", type=str, required=True,
                        help="输出JSON文件路径")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取所有session文件
    session_files = sorted(input_dir.glob("session_*.json"))

    if not session_files:
        print(f"Error: No session files found in {input_dir}")
        return

    print(f"Found {len(session_files)} session files")

    # 评估每个会话
    all_results = {}
    for session_file in session_files:
        print(f"Processing {session_file.name}...")

        with open(session_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)

        session_result = evaluate_session(session_data)
        all_results[session_file.name] = session_result

        print(f"  Distinct-1: {session_result['distinct_1']:.4f}")
        print(f"  Distinct-2: {session_result['distinct_2']:.4f}")
        print(f"  Distinct-3: {session_result['distinct_3']:.4f}")
        print(f"  EAD: {session_result['ead']:.4f}")

    # 计算平均值
    avg_results = {
        "distinct_1": sum(r["distinct_1"] for r in all_results.values()) / len(all_results),
        "distinct_2": sum(r["distinct_2"] for r in all_results.values()) / len(all_results),
        "distinct_3": sum(r["distinct_3"] for r in all_results.values()) / len(all_results),
        "ead": sum(r["ead"] for r in all_results.values()) / len(all_results)
    }

    # 保存结果
    results = {
        "average": avg_results,
        "per_session": all_results,
        "num_sessions": len(all_results)
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("Average Results:")
    print(f"  Distinct-1: {avg_results['distinct_1']:.4f}")
    print(f"  Distinct-2: {avg_results['distinct_2']:.4f}")
    print(f"  Distinct-3: {avg_results['distinct_3']:.4f}")
    print(f"  EAD: {avg_results['ead']:.4f}")
    print(f"{'='*60}")
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
