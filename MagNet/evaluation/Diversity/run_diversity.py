"""
运行 Diversity 评估
测量生成对话的词汇多样性：Distinct-1, Distinct-2, Distinct-3, EAD
"""
import os
import json
import re
from pathlib import Path


def remove_unwanted(document):
    """清理文本：移除 mentions、URL、hashtags、标点等"""
    # remove user mentions
    document = re.sub("@[A-Za-z0-9_]+", " ", document)
    # remove URLS
    document = re.sub(r'http\S+', ' ', document)
    # remove hashtags
    document = re.sub("#[A-Za-z0-9_]+", "", document)
    # remove punctuation
    document = re.sub("[^0-9A-Za-z\u4e00-\u9fff]", "", document)
    # remove double spaces
    document = document.replace('  ', " ")

    return document.strip()


def generate_history(history):
    """从对话历史生成文本"""
    history_text = '\n'.join(
        [
            f"{message['message']}"
            for message in history
        ]
    )
    return history_text


def simple_tokenize(text):
    """简单的中文分词（按字符和空格）"""
    # 先按空格分词
    words = text.split()
    # 对每个词进一步按中文字符分词
    tokens = []
    for word in words:
        if re.search(r'[\u4e00-\u9fff]', word):  # 包含中文
            # 中文按字符分词
            tokens.extend(list(word))
        else:
            # 英文按词分词
            tokens.extend(word.split())
    return [t for t in tokens if t.strip()]


def calculate_diversity(input_dir, output_file):
    """
    计算 Diversity 指标

    Args:
        input_dir: 输入目录（包含 session_*.json 文件）
        output_file: 输出结果文件
    """
    input_path = Path(input_dir)

    # 收集所有对话
    trigram = []
    bigram = []
    unigram = []

    session_files = sorted(input_path.glob("session_*.json"))

    if not session_files:
        print(f"错误：在 {input_dir} 中未找到 session_*.json 文件")
        return

    print(f"找到 {len(session_files)} 个会话文件")
    print("=" * 50)

    for idx, file_path in enumerate(session_files, 1):
        print(f"处理文件 [{idx}/{len(session_files)}]: {file_path.name}")

        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # 生成对话文本
        conv_str = generate_history(json_data["history"])
        # 清理文本
        conv_mod = remove_unwanted(conv_str)
        # 分词
        conv_tok = simple_tokenize(conv_mod)

        # 生成 n-grams
        conv_trigram = [(conv_tok[i], conv_tok[i + 1], conv_tok[i + 2])
                       for i in range(len(conv_tok) - 2)]
        conv_bigram = [(conv_tok[i], conv_tok[i + 1])
                      for i in range(len(conv_tok) - 1)]
        conv_unigram = [(conv_tok[i])
                       for i in range(len(conv_tok))]

        trigram.extend(conv_trigram)
        bigram.extend(conv_bigram)
        unigram.extend(conv_unigram)

    print("=" * 50)
    print(f"总 token 数: {len(unigram)}")
    print(f"总 unigram 数: {len(unigram)}")
    print(f"总 bigram 数: {len(bigram)}")
    print(f"总 trigram 数: {len(trigram)}")

    # 计算 Diversity 指标
    dist_1 = len(set(unigram)) / len(unigram) if unigram else 0
    dist_2 = len(set(bigram)) / len(bigram) if bigram else 0
    dist_3 = len(set(trigram)) / len(trigram) if trigram else 0

    # 估算词汇表大小（用于计算 EAD）
    vocab_size = len(set(unigram))

    # EAD (Expectation Adjusted Distinct)
    # EAD = |unique unigrams| / (V * (1 - ((V-1)/V)^N))
    # 其中 V 是词汇表大小，N 是总 token 数
    if vocab_size > 0 and len(unigram) > 0:
        ead = len(set(unigram)) / (
            vocab_size * (1 - ((vocab_size - 1) / vocab_size) ** len(unigram))
        )
    else:
        ead = 0

    # 输出结果
    results = {
        "input_directory": str(input_dir),
        "total_sessions": len(session_files),
        "total_tokens": len(unigram),
        "vocabulary_size": vocab_size,
        "diversity_metrics": {
            "Distinct-1": dist_1,
            "Distinct-2": dist_2,
            "Distinct-3": dist_3,
            "EAD": ead
        }
    }

    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=" * 50)
    print("Diversity 评估结果:")
    print(f"  Distinct-1: {dist_1:.4f}")
    print(f"  Distinct-2: {dist_2:.4f}")
    print(f"  Distinct-3: {dist_3:.4f}")
    print(f"  EAD:        {ead:.4f}")
    print("=" * 50)
    print(f"结果已保存到: {output_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="计算对话的 Diversity 指标"
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        default="../output-cn",
        help="包含 session_*.json 文件的输入目录"
    )
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        default="../output-diversity/diversity_results.json",
        help="输出结果文件路径"
    )

    args = parser.parse_args()

    calculate_diversity(args.input_dir, args.output_file)
