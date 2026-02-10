# -*- coding: utf-8 -*-
"""
统一评估脚本
线性测量 CTRS、Diversity、WAI 四个指标
支持多主题多对话的批量评估
"""

import sys
import os
import json
import re
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# 添加 evaluation 目录到路径（用于导入 llm_evaluator）
evaluation_dir = os.path.join(parent_dir, "evaluation")
sys.path.insert(0, evaluation_dir)

from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from llm_evaluator import (
    create_eval_client_from_env,
    get_eval_cost,
)

try:
    import tiktoken
except ImportError:
    tiktoken = None


# ============================================================================
# Token 计算函数
# ============================================================================

def normalize_model_name(model: str) -> str:
    """将模型名转换为 tiktoken 支持的格式"""
    # 处理 OpenRouter 格式: openai/gpt-4o -> gpt-4o
    if "/" in model:
        model = model.split("/")[-1]
    # 处理其他可能的格式
    model = model.replace("openai-", "")
    return model


def num_tokens_from_string(string: str, model: str = "gpt-4o") -> int:
    """Returns the number of tokens in a text string."""
    if tiktoken is None:
        # 简单估算：中文约1.5字符=1token，英文约4字符=1token
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', string))
        other_chars = len(string) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    normalized_model = normalize_model_name(model)
    try:
        encoding = tiktoken.encoding_for_model(normalized_model)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    except Exception:
        # 如果映射失败，使用 cl100k_base (gpt-4o 的编码)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(string))


def num_tokens_from_message(messages, model: str = "gpt-4o"):
    if tiktoken is None:
        total = 0
        for msg in messages:
            total += num_tokens_from_string(msg.get("content", ""), model)
        return total + 6

    normalized_model = normalize_model_name(model)
    try:
        enc = tiktoken.encoding_for_model(normalized_model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    tokens_per_message = 3
    total_tokens = 0
    for message in messages:
        total_tokens += tokens_per_message + len(enc.encode(message.get("content", "")))
    total_tokens += 3
    return total_tokens


def calculate_cost(input_message, output: str, model: str = "gpt-4o"):
    """计算评估成本"""
    input_cost_per_m, output_cost_per_m = get_eval_cost(model)
    input_cost = (num_tokens_from_message(input_message, model) * input_cost_per_m) / 1000000
    output_cost = (num_tokens_from_string(output, model) * output_cost_per_m) / 1000000
    return input_cost + output_cost


# ============================================================================
# 文本处理函数
# ============================================================================

def remove_unwanted(document: str) -> str:
    """清理文本：移除 mentions、URL、hashtags、标点等"""
    document = re.sub(r"@[A-Za-z0-9_]+", " ", document)
    document = re.sub(r'http\S+', ' ', document)
    document = re.sub(r"#[A-Za-z0-9_]+", "", document)
    document = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", document)
    document = document.replace('  ', " ")
    return document.strip()


def simple_tokenize(text: str) -> List[str]:
    """简单的中文分词（按字符和空格）"""
    words = text.split()
    tokens = []
    for word in words:
        if re.search(r'[\u4e00-\u9fff]', word):
            tokens.extend(list(word))
        else:
            tokens.extend(word.split())
    return [t for t in tokens if t.strip()]


def generate_conversation_text(turns: List[Dict[str, str]]) -> str:
    """将对话转换为文本格式"""
    lines = []
    for turn in turns:
        user_input = turn.get('user_input', '')
        model_reply = turn.get('model_reply', '')
        if user_input:
            lines.append(f"User: {user_input}")
        if model_reply:
            lines.append(f"Assistant: {model_reply}")
    return '\n'.join(lines)


# ============================================================================
# Diversity 评估
# ============================================================================

def calculate_diversity_for_conversation(turns: List[Dict[str, str]]) -> Dict[str, float]:
    """计算单个对话的 Diversity 指标"""
    # 生成对话文本
    conv_text = generate_conversation_text(turns)
    conv_mod = remove_unwanted(conv_text)
    conv_tok = simple_tokenize(conv_mod)

    if len(conv_tok) == 0:
        return {
            "Distinct-1": 0.0,
            "Distinct-2": 0.0,
            "Distinct-3": 0.0,
            "EAD": 0.0,
            "total_tokens": 0,
            "vocabulary_size": 0
        }

    # 生成 n-grams
    trigram = [(conv_tok[i], conv_tok[i + 1], conv_tok[i + 2])
               for i in range(len(conv_tok) - 2)]
    bigram = [(conv_tok[i], conv_tok[i + 1])
              for i in range(len(conv_tok) - 1)]
    unigram = [conv_tok[i] for i in range(len(conv_tok))]

    # 计算 Diversity 指标
    dist_1 = len(set(unigram)) / len(unigram) if unigram else 0
    dist_2 = len(set(bigram)) / len(bigram) if bigram else 0
    dist_3 = len(set(trigram)) / len(trigram) if trigram else 0

    # EAD (Expectation Adjusted Distinct)
    vocab_size = len(set(unigram))
    if vocab_size > 0 and len(unigram) > 0:
        ead = len(set(unigram)) / (
            vocab_size * (1 - ((vocab_size - 1) / vocab_size) ** len(unigram))
        )
    else:
        ead = 0

    return {
        "Distinct-1": dist_1,
        "Distinct-2": dist_2,
        "Distinct-3": dist_3,
        "EAD": ead,
        "total_tokens": len(unigram),
        "vocabulary_size": vocab_size
    }


# ============================================================================
# CTRS 评估
# ============================================================================

def evaluate_ctrs_for_conversation(
    turns: List[Dict[str, str]],
    evaluator_client,
    prompt_dir: str,
    model_for_cost: str,
    max_iter: int = 1
) -> tuple[Dict[str, float], float]:
    """评估单个对话的 CTRS 指标"""
    ctrs_list = [
        "general_1_understanding",
        "general_2_interpersonal_effectiveness",
        "general_3_collaboration",
        "CBT_1_guided_discovery",
        "CBT_2_focus",
        "CBT_3_strategy"
    ]

    conv_text = generate_conversation_text(turns)
    score_dict = {}
    total_cost = 0

    for ctrs_item in ctrs_list:
        prompt_path = os.path.join(prompt_dir, f"{ctrs_item}.txt")
        if not os.path.exists(prompt_path):
            print(f"  警告：未找到提示词文件 {prompt_path}")
            score_dict[ctrs_item] = 0
            continue

        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read()

        prompt_template = PromptTemplate(
            input_variables=["conversation"],
            template=prompt_text
        )
        prompt = prompt_template.format(conversation=conv_text)
        messages = [{'role': 'user', 'content': prompt}]

        score = 0
        max_retries = 3
        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            try:
                if retry_count > 0:
                    print(f"    重试 {retry_count}/{max_retries} - {ctrs_item}...")
                    time.sleep(2)

                response = evaluator_client.chat_completion(
                    messages=messages,
                    temperature=0,
                    n=max_iter
                )

                for j in range(max_iter):
                    total_cost += calculate_cost(messages, response.choices[j].message.content, model_for_cost)
                    txt_response = response.choices[j].message.content
                    score += int(txt_response.split(',')[0])

                avg_score = score / max_iter
                score_dict[ctrs_item] = avg_score
                print(f"    {ctrs_item}: {avg_score:.2f}")
                success = True

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"    错误 - {ctrs_item}: {e}")
                    score_dict[ctrs_item] = 0
                else:
                    print(f"    错误 - {ctrs_item}: {e}, 重试中...")

    return score_dict, total_cost


# ============================================================================
# WAI 评估
# ============================================================================

def evaluate_wai_for_conversation(
    turns: List[Dict[str, str]],
    evaluator_client,
    prompt_dir: str,
    model_for_cost: str,
    max_iter: int = 1
) -> tuple[Dict[str, float], float]:
    """评估单个对话的 WAI 指标"""
    conv_text = generate_conversation_text(turns)
    score_dict = {}
    total_cost = 0

    # WAI 有 12 个评估项目
    for i in range(12):
        prompt_path = os.path.join(prompt_dir, f"wai{i+1}.txt")
        if not os.path.exists(prompt_path):
            print(f"    警告：未找到提示词文件 {prompt_path}")
            score_dict[f"wai{i+1}"] = 0
            continue

        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read()

        prompt_template = PromptTemplate(
            input_variables=["conversation"],
            template=prompt_text
        )
        prompt = prompt_template.format(conversation=conv_text)
        messages = [{'role': 'user', 'content': prompt}]

        score = 0
        max_retries = 3
        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            try:
                if retry_count > 0:
                    print(f"    重试 {retry_count}/{max_retries} - wai{i+1}...")
                    time.sleep(2)

                response = evaluator_client.chat_completion(
                    messages=messages,
                    temperature=0,
                    n=max_iter
                )

                for j in range(max_iter):
                    total_cost += calculate_cost(messages, response.choices[j].message.content, model_for_cost)
                    txt_response = response.choices[j].message.content
                    score += int(txt_response.split(',')[0])

                avg_score = score / max_iter
                score_dict[f"wai{i+1}"] = avg_score
                print(f"    wai{i+1}: {avg_score:.2f}")
                success = True

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"    错误 - wai{i+1}: {e}")
                    score_dict[f"wai{i+1}"] = 0
                else:
                    print(f"    错误 - wai{i+1}: {e}, 重试中...")

    return score_dict, total_cost


# ============================================================================
# 批量评估函数
# ============================================================================

def run_batch_evaluation(
    input_dir: str,
    output_dir: str,
    ctrs_prompt_dir: str,
    wai_prompt_dir: str,
    max_iter: int = 1,
    pattern: str = "conversation_history_*.json"
):
    """
    批量评估多个 JSON 文件（12个主题，60个对话）

    Args:
        input_dir: 输入目录（包含 conversation_history_*.json 文件）
        output_dir: 输出目录
        ctrs_prompt_dir: CTRS 提示词目录
        wai_prompt_dir: WAI 提示词目录
        max_iter: 每个指标评估次数
        pattern: 文件匹配模式
    """
    print("=" * 80)
    print("批量统一评估脚本")
    print("=" * 80)

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载环境变量
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"已加载环境变量: {env_file}")

    # 创建评估客户端
    if os.getenv("EVAL_LLM_PROVIDER") and os.getenv("EVAL_LLM_API_KEY"):
        print(f"评估 API: {os.getenv('EVAL_LLM_PROVIDER')}")
        evaluator_client = create_eval_client_from_env()
        model_for_cost = evaluator_client.model
        print(f"评估模型: {model_for_cost}")
    else:
        print("错误：未设置 EVAL_LLM_PROVIDER 和 EVAL_LLM_API_KEY")
        sys.exit(1)

    # 查找所有输入文件
    input_path = Path(input_dir)
    input_files = sorted(input_path.glob(pattern))

    if not input_files:
        print(f"错误：在 {input_dir} 中未找到匹配 {pattern} 的文件")
        return

    print(f"\n找到 {len(input_files)} 个文件")
    for f in input_files:
        print(f"  - {f.name}")

    # 全局评估结果存储
    all_conversation_scores = []
    all_topic_conversations: Dict[str, List[Dict]] = {}
    total_cost = 0
    total_conversations = 0

    # 处理每个文件
    for file_idx, input_file in enumerate(input_files, 1):
        print(f"\n{'#' * 80}")
        print(f"[文件 {file_idx}/{len(input_files)}] {input_file.name}")
        print(f"{'#' * 80}")

        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        conversations = data.get("conversations", [])
        print(f"共 {len(conversations)} 个对话")

        metadata = data.get("evaluation_metadata", {})
        input_filename = input_file.stem

        # 逐个评估对话
        for conv_idx, conv in enumerate(conversations, 1):
            conv_id = conv.get("conversation_id", f"{file_idx}-{conv_idx}")
            source_file = conv.get("source_file", input_file.name)
            turns = conv.get("turns", [])

            # 提取主题名（从文件名中提取，去掉 conversation_history_ 和 _5 等后缀）
            topic = input_filename.replace("conversation_history_", "").replace("_5", "")

            print(f"\n{'=' * 60}")
            print(f"[{total_conversations + conv_idx}/{60}] 对话 ID: {conv_id}, 主题: {topic}")
            print(f"{'=' * 60}")

            # 初始化主题分组
            if topic not in all_topic_conversations:
                all_topic_conversations[topic] = []

            # Diversity 评估（本地计算）
            print("  [Diversity] 计算...")
            diversity_scores = calculate_diversity_for_conversation(turns)
            print(f"    Distinct-1: {diversity_scores['Distinct-1']:.4f}")
            print(f"    Distinct-2: {diversity_scores['Distinct-2']:.4f}")
            print(f"    Distinct-3: {diversity_scores['Distinct-3']:.4f}")
            print(f"    EAD: {diversity_scores['EAD']:.4f}")

            # CTRS 评估
            print("  [CTRS] 评估...")
            ctrs_scores, ctrs_cost = evaluate_ctrs_for_conversation(
                turns, evaluator_client, ctrs_prompt_dir, model_for_cost, max_iter
            )
            total_cost += ctrs_cost

            # WAI 评估
            print("  [WAI] 评估...")
            wai_scores, wai_cost = evaluate_wai_for_conversation(
                turns, evaluator_client, wai_prompt_dir, model_for_cost, max_iter
            )
            total_cost += wai_cost

            # 保存单个对话结果
            conv_result = {
                "conversation_id": conv_id,
                "topic": topic,
                "source_file": source_file,
                "ctrs": ctrs_scores,
                "diversity": diversity_scores,
                "wai": wai_scores
            }
            all_conversation_scores.append(conv_result)
            all_topic_conversations[topic].append(conv_result)

        total_conversations += len(conversations)

    # 计算主题平均分
    print(f"\n{'#' * 80}")
    print("计算主题平均分...")
    print(f"{'#' * 80}")

    topic_averages = {}

    for topic, convs in all_topic_conversations.items():
        print(f"\n主题: {topic} ({len(convs)} 个对话)")

        # CTRS 平均
        ctrs_avg = {}
        ctrs_keys = convs[0]["ctrs"].keys()
        for key in ctrs_keys:
            values = [c["ctrs"][key] for c in convs]
            ctrs_avg[key] = sum(values) / len(values)

        # Diversity 平均
        diversity_avg = {}
        diversity_keys = ["Distinct-1", "Distinct-2", "Distinct-3", "EAD"]
        for key in diversity_keys:
            values = [c["diversity"][key] for c in convs]
            diversity_avg[key] = sum(values) / len(values)

        # WAI 平均
        wai_avg = {}
        wai_keys = convs[0]["wai"].keys()
        for key in wai_keys:
            values = [c["wai"][key] for c in convs]
            wai_avg[key] = sum(values) / len(values)

        topic_averages[topic] = {
            "conversation_count": len(convs),
            "ctrs": ctrs_avg,
            "diversity": diversity_avg,
            "wai": wai_avg
        }

        print(f"  CTRS 平均: {sum(ctrs_avg.values()) / len(ctrs_avg):.2f}")
        print(f"  Diversity 平均: D1={diversity_avg['Distinct-1']:.4f}")
        print(f"  WAI 平均: {sum(wai_avg.values()) / len(wai_avg):.2f}")

    # 计算总体平均分
    print(f"\n{'#' * 80}")
    print("计算总体平均分...")
    print(f"{'#' * 80}")

    # CTRS 总体平均
    ctrs_overall = {}
    ctrs_keys = all_conversation_scores[0]["ctrs"].keys()
    for key in ctrs_keys:
        values = [c["ctrs"][key] for c in all_conversation_scores]
        ctrs_overall[key] = sum(values) / len(values)

    # Diversity 总体平均
    diversity_overall = {}
    diversity_keys = ["Distinct-1", "Distinct-2", "Distinct-3", "EAD"]
    for key in diversity_keys:
        values = [c["diversity"][key] for c in all_conversation_scores]
        diversity_overall[key] = sum(values) / len(values)

    # WAI 总体平均
    wai_overall = {}
    wai_keys = all_conversation_scores[0]["wai"].keys()
    for key in wai_keys:
        values = [c["wai"][key] for c in all_conversation_scores]
        wai_overall[key] = sum(values) / len(values)

    overall_average = {
        "conversation_count": len(all_conversation_scores),
        "topic_count": len(topic_averages),
        "ctrs": ctrs_overall,
        "diversity": diversity_overall,
        "wai": wai_overall
    }

    print(f"  总对话数: {len(all_conversation_scores)}")
    print(f"  主题数: {len(topic_averages)}")
    print(f"  CTRS 总平均: {sum(ctrs_overall.values()) / len(ctrs_overall):.2f}")
    print(f"  Diversity 总平均: D1={diversity_overall['Distinct-1']:.4f}")
    print(f"  WAI 总平均: {sum(wai_overall.values()) / len(wai_overall):.2f}")

    # 构建最终结果
    result = {
        "evaluation_metadata": {
            "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_directory": str(input_dir),
            "total_files_evaluated": len(input_files),
            "total_conversations_evaluated": len(all_conversation_scores),
            "topics_evaluated": list(topic_averages.keys()),
            "topic_count": len(topic_averages),
            "evaluation_model": model_for_cost,
            "total_cost_usd": round(total_cost, 4)
        },
        "conversation_scores": all_conversation_scores,
        "topic_averages": topic_averages,
        "overall_average": overall_average
    }

    # 保存结果文件
    print(f"\n{'#' * 80}")
    print("保存结果文件...")
    print(f"{'#' * 80}")

    # 总体结果
    overall_file = output_path / "all_topics_evaluated.json"
    with open(overall_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"总体结果: {overall_file}")

    # 单独保存各指标结果
    ctrs_file = output_path / "all_topics_ctrs.json"
    diversity_file = output_path / "all_topics_diversity.json"
    wai_file = output_path / "all_topics_wai.json"

    # CTRS 结果
    ctrs_result = {
        "metadata": result["evaluation_metadata"],
        "conversation_scores": [
            {
                "conversation_id": c["conversation_id"],
                "topic": c["topic"],
                "scores": c["ctrs"]
            }
            for c in all_conversation_scores
        ],
        "topic_averages": {
            t: v["ctrs"]
            for t, v in topic_averages.items()
        },
        "overall_average": ctrs_overall
    }
    with open(ctrs_file, 'w', encoding='utf-8') as f:
        json.dump(ctrs_result, f, indent=2, ensure_ascii=False)
    print(f"CTRS 结果: {ctrs_file}")

    # Diversity 结果
    diversity_result = {
        "metadata": result["evaluation_metadata"],
        "conversation_scores": [
            {
                "conversation_id": c["conversation_id"],
                "topic": c["topic"],
                "scores": c["diversity"]
            }
            for c in all_conversation_scores
        ],
        "topic_averages": {
            t: v["diversity"]
            for t, v in topic_averages.items()
        },
        "overall_average": diversity_overall
    }
    with open(diversity_file, 'w', encoding='utf-8') as f:
        json.dump(diversity_result, f, indent=2, ensure_ascii=False)
    print(f"Diversity 结果: {diversity_file}")

    # WAI 结果
    wai_result = {
        "metadata": result["evaluation_metadata"],
        "conversation_scores": [
            {
                "conversation_id": c["conversation_id"],
                "topic": c["topic"],
                "scores": c["wai"]
            }
            for c in all_conversation_scores
        ],
        "topic_averages": {
            t: v["wai"]
            for t, v in topic_averages.items()
        },
        "overall_average": wai_overall
    }
    with open(wai_file, 'w', encoding='utf-8') as f:
        json.dump(wai_result, f, indent=2, ensure_ascii=False)
    print(f"WAI 结果: {wai_file}")

    print(f"\n{'#' * 80}")
    print("批量评估完成！")
    print(f"总成本: ${total_cost:.4f}")
    print(f"{'#' * 80}")

    return result


# ============================================================================
# 单文件评估函数（保留原有功能）
# ============================================================================

def run_unified_evaluation(
    input_file: str,
    output_dir: str,
    ctrs_prompt_dir: str,
    wai_prompt_dir: str,
    max_iter: int = 1
):
    """
    运行单文件统一评估

    Args:
        input_file: 输入 JSON 文件路径
        output_dir: 输出目录
        ctrs_prompt_dir: CTRS 提示词目录
        wai_prompt_dir: WAI 提示词目录
        max_iter: 每个指标评估次数
    """
    print("=" * 80)
    print("统一评估脚本")
    print("=" * 80)

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载环境变量
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"已加载环境变量: {env_file}")

    # 创建评估客户端
    if os.getenv("EVAL_LLM_PROVIDER") and os.getenv("EVAL_LLM_API_KEY"):
        print(f"评估 API: {os.getenv('EVAL_LLM_PROVIDER')}")
        evaluator_client = create_eval_client_from_env()
        model_for_cost = evaluator_client.model
        print(f"评估模型: {model_for_cost}")
    else:
        print("错误：未设置 EVAL_LLM_PROVIDER 和 EVAL_LLM_API_KEY")
        sys.exit(1)

    # 读取输入文件
    print(f"\n读取输入文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    conversations = data.get("conversations", [])
    print(f"共 {len(conversations)} 个对话")

    # 提取元数据
    metadata = data.get("evaluation_metadata", {})
    input_filename = Path(input_file).stem

    # 评估结果存储
    conversation_scores = []
    topic_conversations: Dict[str, List[Dict]] = {}

    total_cost = 0

    # 逐个评估对话
    for idx, conv in enumerate(conversations, 1):
        conv_id = conv.get("conversation_id", idx)
        source_file = conv.get("source_file", "unknown")
        turns = conv.get("turns", [])
        topic = source_file.replace(".json", "")

        print(f"\n{'=' * 80}")
        print(f"[{idx}/{len(conversations)}] 对话 ID: {conv_id}, 主题: {topic}")
        print(f"{'=' * 80}")

        # 初始化主题分组
        if topic not in topic_conversations:
            topic_conversations[topic] = []

        # Diversity 评估（本地计算）
        print("  [Diversity] 计算...")
        diversity_scores = calculate_diversity_for_conversation(turns)
        print(f"    Distinct-1: {diversity_scores['Distinct-1']:.4f}")
        print(f"    Distinct-2: {diversity_scores['Distinct-2']:.4f}")
        print(f"    Distinct-3: {diversity_scores['Distinct-3']:.4f}")
        print(f"    EAD: {diversity_scores['EAD']:.4f}")

        # CTRS 评估
        print("  [CTRS] 评估...")
        ctrs_scores, ctrs_cost = evaluate_ctrs_for_conversation(
            turns, evaluator_client, ctrs_prompt_dir, model_for_cost, max_iter
        )
        total_cost += ctrs_cost

        # WAI 评估
        print("  [WAI] 评估...")
        wai_scores, wai_cost = evaluate_wai_for_conversation(
            turns, evaluator_client, wai_prompt_dir, model_for_cost, max_iter
        )
        total_cost += wai_cost

        # 保存单个对话结果
        conv_result = {
            "conversation_id": conv_id,
            "topic": topic,
            "source_file": source_file,
            "ctrs": ctrs_scores,
            "diversity": diversity_scores,
            "wai": wai_scores
        }
        conversation_scores.append(conv_result)
        topic_conversations[topic].append(conv_result)

    # 计算主题平均分
    print(f"\n{'=' * 80}")
    print("计算主题平均分...")
    print(f"{'=' * 80}")

    topic_averages = {}

    for topic, convs in topic_conversations.items():
        print(f"\n主题: {topic} ({len(convs)} 个对话)")

        # CTRS 平均
        ctrs_avg = {}
        ctrs_keys = convs[0]["ctrs"].keys()
        for key in ctrs_keys:
            values = [c["ctrs"][key] for c in convs]
            ctrs_avg[key] = sum(values) / len(values)

        # Diversity 平均
        diversity_avg = {}
        diversity_keys = ["Distinct-1", "Distinct-2", "Distinct-3", "EAD"]
        for key in diversity_keys:
            values = [c["diversity"][key] for c in convs]
            diversity_avg[key] = sum(values) / len(values)

        # WAI 平均
        wai_avg = {}
        wai_keys = convs[0]["wai"].keys()
        for key in wai_keys:
            values = [c["wai"][key] for c in convs]
            wai_avg[key] = sum(values) / len(values)

        topic_averages[topic] = {
            "conversation_count": len(convs),
            "ctrs": ctrs_avg,
            "diversity": diversity_avg,
            "wai": wai_avg
        }

        print(f"  CTRS 平均: {sum(ctrs_avg.values()) / len(ctrs_avg):.2f}")
        print(f"  Diversity 平均: D1={diversity_avg['Distinct-1']:.4f}")
        print(f"  WAI 平均: {sum(wai_avg.values()) / len(wai_avg):.2f}")

    # 计算总体平均分
    print(f"\n{'=' * 80}")
    print("计算总体平均分...")
    print(f"{'=' * 80}")

    # CTRS 总体平均
    ctrs_overall = {}
    ctrs_keys = conversation_scores[0]["ctrs"].keys()
    for key in ctrs_keys:
        values = [c["ctrs"][key] for c in conversation_scores]
        ctrs_overall[key] = sum(values) / len(values)

    # Diversity 总体平均
    diversity_overall = {}
    diversity_keys = ["Distinct-1", "Distinct-2", "Distinct-3", "EAD"]
    for key in diversity_keys:
        values = [c["diversity"][key] for c in conversation_scores]
        diversity_overall[key] = sum(values) / len(values)

    # WAI 总体平均
    wai_overall = {}
    wai_keys = conversation_scores[0]["wai"].keys()
    for key in wai_keys:
        values = [c["wai"][key] for c in conversation_scores]
        wai_overall[key] = sum(values) / len(values)

    overall_average = {
        "conversation_count": len(conversation_scores),
        "topic_count": len(topic_averages),
        "ctrs": ctrs_overall,
        "diversity": diversity_overall,
        "wai": wai_overall
    }

    print(f"  总对话数: {len(conversation_scores)}")
    print(f"  主题数: {len(topic_averages)}")
    print(f"  CTRS 总平均: {sum(ctrs_overall.values()) / len(ctrs_overall):.2f}")
    print(f"  Diversity 总平均: D1={diversity_overall['Distinct-1']:.4f}")
    print(f"  WAI 总平均: {sum(wai_overall.values()) / len(wai_overall):.2f}")

    # 构建最终结果
    result = {
        "evaluation_metadata": {
            **metadata,
            "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_file": input_file,
            "total_conversations_evaluated": len(conversation_scores),
            "topics_evaluated": len(topic_averages),
            "evaluation_model": model_for_cost,
            "total_cost_usd": round(total_cost, 4)
        },
        "conversation_scores": conversation_scores,
        "topic_averages": topic_averages,
        "overall_average": overall_average
    }

    # 保存结果文件
    print(f"\n{'=' * 80}")
    print("保存结果文件...")
    print(f"{'=' * 80}")

    output_file = output_path / f"{input_filename}_evaluated.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"结果已保存到: {output_file}")

    # 单独保存各指标结果
    ctrs_file = output_path / f"{input_filename}_ctrs.json"
    diversity_file = output_path / f"{input_filename}_diversity.json"
    wai_file = output_path / f"{input_filename}_wai.json"

    # CTRS 结果
    ctrs_result = {
        "metadata": result["evaluation_metadata"],
        "conversation_scores": [
            {
                "conversation_id": c["conversation_id"],
                "topic": c["topic"],
                "scores": c["ctrs"]
            }
            for c in conversation_scores
        ],
        "topic_averages": {
            t: v["ctrs"]
            for t, v in topic_averages.items()
        },
        "overall_average": ctrs_overall
    }
    with open(ctrs_file, 'w', encoding='utf-8') as f:
        json.dump(ctrs_result, f, indent=2, ensure_ascii=False)
    print(f"CTRS 结果: {ctrs_file}")

    # Diversity 结果
    diversity_result = {
        "metadata": result["evaluation_metadata"],
        "conversation_scores": [
            {
                "conversation_id": c["conversation_id"],
                "topic": c["topic"],
                "scores": c["diversity"]
            }
            for c in conversation_scores
        ],
        "topic_averages": {
            t: v["diversity"]
            for t, v in topic_averages.items()
        },
        "overall_average": diversity_overall
    }
    with open(diversity_file, 'w', encoding='utf-8') as f:
        json.dump(diversity_result, f, indent=2, ensure_ascii=False)
    print(f"Diversity 结果: {diversity_file}")

    # WAI 结果
    wai_result = {
        "metadata": result["evaluation_metadata"],
        "conversation_scores": [
            {
                "conversation_id": c["conversation_id"],
                "topic": c["topic"],
                "scores": c["wai"]
            }
            for c in conversation_scores
        ],
        "topic_averages": {
            t: v["wai"]
            for t, v in topic_averages.items()
        },
        "overall_average": wai_overall
    }
    with open(wai_file, 'w', encoding='utf-8') as f:
        json.dump(wai_result, f, indent=2, ensure_ascii=False)
    print(f"WAI 结果: {wai_file}")

    print(f"\n{'=' * 80}")
    print("评估完成！")
    print(f"总成本: ${total_cost:.4f}")
    print(f"{'=' * 80}")

    return result


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="统一评估脚本 - 线性测量 CTRS、Diversity、WAI 指标"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="输入 JSON 文件路径（单文件模式）"
    )
    parser.add_argument(
        "-d", "--input_dir",
        type=str,
        help="输入目录路径（批量模式，处理目录下所有 conversation_history_*.json 文件）"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output_results",
        help="输出目录（默认：output_results）"
    )
    parser.add_argument(
        "--ctrs_prompts",
        type=str,
        default="evaluation/CTRS/prompts",
        help="CTRS 提示词目录（相对于 MagNet 目录）"
    )
    parser.add_argument(
        "--wai_prompts",
        type=str,
        default="evaluation/WAI/prompts",
        help="WAI 提示词目录（相对于 MagNet 目录）"
    )
    parser.add_argument(
        "-m_iter", "--max_iter",
        type=int,
        default=1,
        help="每个指标评估次数（默认：1）"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="conversation_history_*.json",
        help="批量模式下的文件匹配模式（默认：conversation_history_*.json）"
    )

    args = parser.parse_args()

    # 转换为绝对路径
    magent_dir = Path(__file__).parent.parent
    ctrs_prompt_dir = magent_dir / args.ctrs_prompts
    wai_prompt_dir = magent_dir / args.wai_prompts

    # 判断运行模式
    if args.input_dir:
        # 批量模式
        print("运行模式：批量评估（处理目录下所有匹配文件）")
        run_batch_evaluation(
            input_dir=args.input_dir,
            output_dir=args.output,
            ctrs_prompt_dir=str(ctrs_prompt_dir),
            wai_prompt_dir=str(wai_prompt_dir),
            max_iter=args.max_iter,
            pattern=args.pattern
        )
    elif args.input:
        # 单文件模式
        print("运行模式：单文件评估")
        run_unified_evaluation(
            input_file=args.input,
            output_dir=args.output,
            ctrs_prompt_dir=str(ctrs_prompt_dir),
            wai_prompt_dir=str(wai_prompt_dir),
            max_iter=args.max_iter
        )
    else:
        parser.print_help()
        print("\n错误：必须指定 --input（单文件）或 --input_dir（批量模式）")
        sys.exit(1)


if __name__ == "__main__":
    main()
