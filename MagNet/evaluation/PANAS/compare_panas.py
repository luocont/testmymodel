# -*- coding: utf-8 -*-
"""
PANAS 前后对比脚本
计算咨询前后情绪的变化
"""
import json
import os
from pathlib import Path


def load_panas_results(directory):
    """加载 PANAS 结果"""
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 提取 session 编号
                session_num = filename.replace('session_', '').replace('.json', '')
                results[session_num] = data
    return results


def calculate_average_scores(score_dict):
    """计算平均分数"""
    positive_emotions = ['interested', 'excited', 'strong', 'enthusiastic', 'proud', 'alert', 'inspired', 'determined', 'attentive', 'active']
    negative_emotions = ['distressed', 'upset', 'guilty', 'scared', 'hostile', 'irritable', 'ashamed', 'nervous', 'jittery', 'afraid']

    pos_scores = []
    neg_scores = []

    for emotion in positive_emotions:
        if emotion in score_dict and isinstance(score_dict[emotion], list):
            scores = score_dict[emotion]
            if scores:
                pos_scores.append(sum(scores) / len(scores))

    for emotion in negative_emotions:
        if emotion in score_dict and isinstance(score_dict[emotion], list):
            scores = score_dict[emotion]
            if scores:
                neg_scores.append(sum(scores) / len(scores))

    avg_positive = sum(pos_scores) / len(pos_scores) if pos_scores else 0
    avg_negative = sum(neg_scores) / len(neg_scores) if neg_scores else 0

    return avg_positive, avg_negative


def compare_panas(before_dir, after_dir, output_file):
    """
    对比 PANAS 前后结果

    Args:
        before_dir: 咨询前结果目录
        after_dir: 咨询后结果目录
        output_file: 输出文件路径
    """
    before_results = load_panas_results(before_dir)
    after_results = load_panas_results(after_dir)

    comparison = {
        "summary": {
            "total_sessions": len(before_results),
            "positive_emotions": {
                "before_total": 0,
                "after_total": 0,
                "average_change": 0
            },
            "negative_emotions": {
                "before_total": 0,
                "after_total": 0,
                "average_change": 0
            }
        },
        "sessions": []
    }

    positive_changes = []
    negative_changes = []

    for session_num in sorted(before_results.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        if session_num not in after_results:
            continue

        before_data = before_results[session_num]
        after_data = after_results[session_num]

        # 计算前后平均分
        before_pos, before_neg = calculate_average_scores(before_data)
        after_pos, after_neg = calculate_average_scores(after_data)

        # 计算变化（后 - 前）
        pos_change = after_pos - before_pos
        neg_change = after_neg - before_neg

        # 理想情况：正向情绪增加，负向情绪减少
        # 所以正向变化越大越好，负向变化越小越好（负值表示减少）
        improvement_score = pos_change - neg_change  # 改善分数

        session_result = {
            "session": f"session_{session_num}",
            "attitude": before_data.get('attitude', 'unknown'),
            "before": {
                "positive": round(before_pos, 2),
                "negative": round(before_neg, 2)
            },
            "after": {
                "positive": round(after_pos, 2),
                "negative": round(after_neg, 2)
            },
            "change": {
                "positive": round(pos_change, 2),
                "negative": round(neg_change, 2)
            },
            "improvement": round(improvement_score, 2)
        }

        comparison["sessions"].append(session_result)

        positive_changes.append(pos_change)
        negative_changes.append(neg_change)

    # 计算总体统计
    comparison["summary"]["positive_emotions"]["before_total"] = round(
        sum([s["before"]["positive"] for s in comparison["sessions"]]) / len(comparison["sessions"]), 2)
    comparison["summary"]["positive_emotions"]["after_total"] = round(
        sum([s["after"]["positive"] for s in comparison["sessions"]]) / len(comparison["sessions"]), 2)
    comparison["summary"]["positive_emotions"]["average_change"] = round(
        sum(positive_changes) / len(positive_changes), 2)

    comparison["summary"]["negative_emotions"]["before_total"] = round(
        sum([s["before"]["negative"] for s in comparison["sessions"]]) / len(comparison["sessions"]), 2)
    comparison["summary"]["negative_emotions"]["after_total"] = round(
        sum([s["after"]["negative"] for s in comparison["sessions"]]) / len(comparison["sessions"]), 2)
    comparison["summary"]["negative_emotions"]["average_change"] = round(
        sum(negative_changes) / len(negative_changes), 2)

    # 按 attitude 分组统计
    attitude_groups = {}
    for session in comparison["sessions"]:
        attitude = session["attitude"]
        if attitude not in attitude_groups:
            attitude_groups[attitude] = {
                "count": 0,
                "positive_changes": [],
                "negative_changes": [],
                "improvements": []
            }
        attitude_groups[attitude]["count"] += 1
        attitude_groups[attitude]["positive_changes"].append(session["change"]["positive"])
        attitude_groups[attitude]["negative_changes"].append(session["change"]["negative"])
        attitude_groups[attitude]["improvements"].append(session["improvement"])

    comparison["by_attitude"] = {}
    for attitude, data in attitude_groups.items():
        comparison["by_attitude"][attitude] = {
            "count": data["count"],
            "avg_positive_change": round(sum(data["positive_changes"]) / len(data["positive_changes"]), 2),
            "avg_negative_change": round(sum(data["negative_changes"]) / len(data["negative_changes"]), 2),
            "avg_improvement": round(sum(data["improvements"]) / len(data["improvements"]), 2)
        }

    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("PANAS 前后对比结果")
    print("=" * 70)

    print("\n[总体统计]")
    print(f"  会话数量: {comparison['summary']['total_sessions']}")
    print(f"\n  [正向情绪]:")
    print(f"    咨询前: {comparison['summary']['positive_emotions']['before_total']:.2f}")
    print(f"    咨询后: {comparison['summary']['positive_emotions']['after_total']:.2f}")
    pos_change = comparison['summary']['positive_emotions']['average_change']
    print(f"    变化:   {pos_change:+.2f} ({'改善' if pos_change > 0 else '下降'})")

    print(f"\n  [负向情绪]:")
    print(f"    咨询前: {comparison['summary']['negative_emotions']['before_total']:.2f}")
    print(f"    咨询后: {comparison['summary']['negative_emotions']['after_total']:.2f}")
    neg_change = comparison['summary']['negative_emotions']['average_change']
    print(f"    变化:   {neg_change:+.2f} ({'改善' if neg_change < 0 else '增加'})")

    print("\n[按态度分组]")
    for attitude, stats in comparison["by_attitude"].items():
        print(f"\n  {attitude.upper()} (n={stats['count']}):")
        print(f"    正向情绪变化: {stats['avg_positive_change']:+.2f}")
        print(f"    负向情绪变化: {stats['avg_negative_change']:+.2f}")
        print(f"    总体改善:     {stats['avg_improvement']:+.2f}")

    print("\n[各会话详细结果]")
    for session in comparison["sessions"]:
        print(f"\n  {session['session']} ({session['attitude']}):")
        print(f"    正向: {session['before']['positive']:.2f} -> {session['after']['positive']:.2f} ({session['change']['positive']:+.2f})")
        print(f"    负向: {session['before']['negative']:.2f} -> {session['after']['negative']:.2f} ({session['change']['negative']:+.2f})")
        print(f"    改善: {session['improvement']:+.2f}")

    print("\n" + "=" * 70)
    print(f"结果已保存到: {output_file}")
    print("=" * 70)

    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="对比 PANAS 前后结果"
    )
    parser.add_argument(
        "--before_dir",
        type=str,
        default="../../output-panas-before",
        help="咨询前 PANAS 结果目录"
    )
    parser.add_argument(
        "--after_dir",
        type=str,
        default="../../output-panas-after",
        help="咨询后 PANAS 结果目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../../output-panas-comparison/comparison.json",
        help="输出对比结果文件"
    )

    args = parser.parse_args()

    compare_panas(args.before_dir, args.after_dir, args.output)
