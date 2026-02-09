"""
MultiAgentESC 与 MagNet Client 集成脚本

这个脚本让 MultiAgentESC 作为咨询师，与 MagNet 的 ClientAgent 进行对话，
生成符合 MagNet 测评格式的对话文件。

注意：MultiAgentESC 使用自己的提示词系统，不使用 MagNet 预设的提示词。
"""

import argparse
import json
import multiprocessing
import traceback
import sys
import os
from pathlib import Path
from langchain.prompts import PromptTemplate

# 添加 MultiAgentESC 到路径
multiagent_path = Path(__file__).parent.parent.parent / "MultiAgentESC"
sys.path.insert(0, str(multiagent_path))

# 导入 MultiAgentESC 的模块
from multiagent import (
    single_agent_response,
    get_emotion,
    get_cause,
    get_intention,
    get_strategy,
    response_with_strategy,
    debate,
    reflect,
    vote,
    judge,
    self_reflection
)
from prompt import get_prompt
from strategy import strategy_definitions

# 导入 MagNet 的客户端和通用模块
sys.path.insert(0, str(Path(__file__).parent))
from inference_parallel_magnet import ClientAgent, generate, get_llm_client

import autogen
import heapq
import numpy as np
from sentence_transformers import util
from collections import Counter
import re


class MultiAgentESCCounselor:
    """
    MultiAgentESC 咨询师适配器

    这个类将 MultiAgentESC 的多智能体系统适配为 MagNet 的咨询师接口。
    使用 MultiAgentESC 自己的提示词和策略系统。
    """

    def __init__(self, config_list, cache_path_root="", model_path="all-roberta-large-v1"):
        self.config_list = config_list
        self.cache_path_root = cache_path_root
        self.model_path = model_path

        # 延迟加载模型（在需要时）
        self.model = None
        self.quadruple = None

    def _load_model_and_data(self):
        """延迟加载模型和数据"""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_path)

        if self.quadruple is None:
            # 加载 MultiAgentESC 的嵌入数据
            quadruple_path = multiagent_path / "embeddings.txt"
            if quadruple_path.exists():
                with open(quadruple_path, "r") as txt:
                    self.quadruple = txt.readlines()
            else:
                print(f"警告: 未找到 embeddings.txt 文件，将使用空参考数据")
                self.quadruple = []

    def is_complex(self, context):
        """判断对话是否足够复杂需要多智能体协作"""
        prompt = get_prompt("behavior_control").format(context=context)
        from multiagent import is_complex
        return is_complex(prompt, self.config_list, self.cache_path_root)

    def generate_response(self, context):
        """
        生成咨询师响应

        Args:
            context: 对话上下文（格式：自然的对话历史）

        Returns:
            response: 咨询师的响应
        """
        self._load_model_and_data()

        # 检查是否需要多智能体协作
        if not self.is_complex(context):
            # 简单情况：使用单智能体零样本生成
            prompt = get_prompt("zero_shot").format(context=context)
            response = single_agent_response(prompt, self.config_list, self.cache_path_root)
            return response

        # 复杂情况：使用 MultiAgentESC 的完整流程
        try:
            # 1. 情感分析
            emotion_prompt = get_prompt("get_emotion").format(context=context)
            emotion, emo_and_reason = get_emotion(emotion_prompt, self.config_list, self.cache_path_root)

            # 2. 原因分析
            cause_prompt = get_prompt("get_cause").format(emo_and_reason=emo_and_reason, context=context)
            cause, cau_and_reason = get_cause(cause_prompt, self.config_list, self.cache_path_root)

            # 3. 意图分析
            intention_prompt = get_prompt("get_intention").format(
                emo_and_reason=emo_and_reason,
                cau_and_reason=cau_and_reason,
                context=context
            )
            intention, int_and_reason = get_intention(intention_prompt, self.config_list, self.cache_path_root)

            # 4. 策略选择
            # 获取用户最后一条消息
            post = context.split("Assistant:")[-1].split("User:")[-1].strip() if "User:" in context else ""
            if not post:
                # 尝试其他格式
                lines = context.strip().split('\n')
                for line in reversed(lines):
                    if line.strip():
                        post = line.split(":", 1)[-1].strip()
                        break

            pred_strategy, pairs = get_strategy(
                emo_and_reason, cau_and_reason, int_and_reason,
                context, post, self.quadruple, self.model, self.config_list
            )

            # 清理策略名称
            pred_strategy = self._clean_strategy(pred_strategy)

            if len(pred_strategy) == 0:
                # 没有合适的策略，使用零样本
                prompt = get_prompt("zero_shot").format(context=context)
                response = single_agent_response(prompt, self.config_list, self.cache_path_root)
                return response

            # 5. 生成响应
            if len(pred_strategy) == 1:
                # 单策略
                examples = ""
                for pair in pairs:
                    strat = pair[1].split("]", 1)[0].strip("[").strip()
                    if strat == pred_strategy[0]:
                        examples += f"{pair[0]}\n{pair[1]}\n\n"
                examples = examples.strip()

                response = response_with_strategy(
                    context, emo_and_reason, cau_and_reason, int_and_reason,
                    pred_strategy[0], examples, self.config_list, self.cache_path_root
                )
                pred_strategy = pred_strategy[0]
            else:
                # 多策略：辩论+投票
                responses = []
                for strat in pred_strategy:
                    examples = ""
                    for pair in pairs:
                        p_strat = pair[1].split("]", 1)[0].strip("[").strip()
                        if p_strat == strat:
                            examples += f"{pair[0]}\n{pair[1]}\n\n"
                    examples = examples.strip()

                    resp = response_with_strategy(
                        context, emo_and_reason, cau_and_reason, int_and_reason,
                        strat, examples, self.config_list, self.cache_path_root
                    )
                    responses.append(f'[{strat}] {resp}')

                # 辩论
                debate_history = debate(
                    context, emo_and_reason, cau_and_reason, int_and_reason,
                    responses, self.config_list
                )

                # 反思
                reflection_result = reflect(
                    context, emo_and_reason, cau_and_reason, int_and_reason,
                    debate_history, responses, self.config_list
                )

                # 投票
                strats, responses = vote(reflection_result)

                if len(strats) == 1 and strats[0] != "None":
                    pred_strategy, response = strats[0].strip(), responses[0].strip()
                else:
                    # 最终裁决
                    pred_strategy, response = judge(
                        context, strats, responses, self.config_list, self.cache_path_root
                    )

            # 6. 自我反思
            final_strategy, response = self_reflection(
                context, pred_strategy, response, self.config_list, self.cache_path_root
            )

            return response

        except Exception as e:
            print(f"MultiAgentESC 生成出错，使用零样本备选: {e}")
            prompt = get_prompt("zero_shot").format(context=context)
            return single_agent_response(prompt, self.config_list, self.cache_path_root)

    def _clean_strategy(self, strategy):
        """清理策略名称"""
        cleaned_strategy = set()
        strategy_map = {
            "question": "Question",
            "restatement or paraphrasing": "Restatement or Paraphrasing",
            "reflection of feelings": "Reflection of feelings",
            "self-disclosure": "Self-disclosure",
            "affirmation and reassurance": "Affirmation and Reassurance",
            "providing suggestions": "Providing Suggestions",
            "information": "Information",
            "others": "Others"
        }

        for s in strategy:
            s_lower = s.lower()
            for key, value in strategy_map.items():
                if key in s_lower:
                    cleaned_strategy.add(value)

        return list(cleaned_strategy)


def convert_history_to_natural(history):
    """将 MagNet 历史格式转换为自然语言格式"""
    lines = []
    for msg in history:
        role = msg['role']
        message = msg['message']
        if role == "counselor":
            lines.append(f"Assistant: {message}")
        elif role == "client":
            lines.append(f"User: {message}")
    return ' '.join(lines)


def run_therapy_session(index, example, output_dir, total, max_turns,
                       config_list, cache_path_root, model_path):
    """
    运行一个咨询会话

    Args:
        index: 样本索引
        example: 样本数据
        output_dir: 输出目录
        total: 总样本数
        max_turns: 最大轮次
        config_list: AutoGen 配置列表
        cache_path_root: 缓存路径
        model_path: 嵌入模型路径
    """
    output_dir = Path(output_dir)
    file_number = index + 1

    try:
        print(f"[MultiAgentESC] 生成第 {file_number}/{total} 个样本")

        # 初始化 MultiAgentESC 咨询师
        counselor = MultiAgentESCCounselor(
            config_list=config_list,
            cache_path_root=cache_path_root,
            model_path=model_path
        )

        # 初始化 MagNet 客户端
        client = ClientAgent(example=example)

        # 初始化对话历史
        history = []
        example_cbt = example['AI_counselor']['CBT']

        # 添加初始对话
        history.append({
            "role": "counselor",
            "message": example_cbt['init_history_counselor']
        })
        history.append({
            "role": "client",
            "message": example_cbt['init_history_client']
        })

        # 对话循环
        for turn in range(max_turns):
            # 咨询师生成响应
            context = convert_history_to_natural(history)
            counselor_response = counselor.generate_response(context)

            # 清理响应
            counselor_response = counselor_response.strip()
            if counselor_response.startswith("Assistant:"):
                counselor_response = counselor_response.split("Assistant:", 1)[-1].strip()

            history.append({
                "role": "counselor",
                "message": counselor_response
            })

            # 客户端生成响应
            client_response = client.generate(history)
            client_response = client_response.strip()
            if client_response.startswith("Client:"):
                client_response = client_response.split("Client:", 1)[-1].strip()

            # 移除 [/END] 标记（不中断对话，确保进行满20轮）
            client_response = client_response.replace('[/END]', '')

            history.append({
                "role": "client",
                "message": client_response
            })

        # 准备输出数据
        session_data = {
            "example": example,
            "cbt_technique": "MultiAgentESC (Strategy-based)",
            "cbt_plan": "MultiAgentESC uses dynamic strategy selection based on emotion, cause, and intention analysis.",
            "cost": 0,  # MultiAgentESC 不计算成本
            "history": history
        }

        # 保存结果
        file_name = f"session_{file_number}.json"
        file_path = output_dir / file_name

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=4)

        print(f"[MultiAgentESC] 完成 {file_number}/{total}")

    except Exception as e:
        error_file_name = f"error_multiagentesc_{file_number}.txt"
        error_file_path = output_dir / error_file_name
        tb = e.__traceback__
        with open(error_file_path, "w", encoding="utf-8") as f:
            f.write(f"Error: {e}\n")
            f.write("".join(traceback.format_exception(type(e), e, tb)))
        print(f"[MultiAgentESC] 错误 {file_number}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="使用 MultiAgentESC 作为咨询师生成对话"
    )
    parser.add_argument("-o", "--output_dir", type=str, default="output-multiagentesc",
                        help="输出目录")
    parser.add_argument("-d", "--dataset", type=str,
                        default="../dataset/data_cn.json",
                        help="数据集文件路径")
    parser.add_argument("-num_pr", "--num_processes", type=int, default=None,
                        help="并行进程数（默认使用所有CPU核心）")
    parser.add_argument("-m_turns", "--max_turns", type=int, default=20,
                        help="最大对话轮次")
    parser.add_argument("--llm_model", type=str, default="qwen2.5:32b",
                        help="LLM 模型名称")
    parser.add_argument("--model_path", type=str, default="all-roberta-large-v1",
                        help="SentenceTransformer 模型路径")
    parser.add_argument("--cache_path", type=str, default="",
                        help="缓存路径根目录")

    args = parser.parse_args()

    # 设置工作目录
    os.chdir(Path(__file__).parent)

    # 加载数据集
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).parent.parent / "dataset" / "data_cn.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 配置 AutoGen
    config_list = autogen.config_list_from_json(
        env_or_file="OAI_CONFIG_LIST",
        file_location=str(multiagent_path),
        filter_dict={
            "model": [args.llm_model],
        }
    )

    total = len(data)
    print(f"[MultiAgentESC] 开始处理 {total} 个样本")
    print(f"[MultiAgentESC] 使用模型: {args.llm_model}")
    print(f"[MultiAgentESC] 输出目录: {output_dir}")

    # 准备参数列表
    args_list = [
        (index, example, output_dir, total, args.max_turns,
         config_list, args.cache_path, args.model_path)
        for index, example in enumerate(data)
    ]

    # 并行处理
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        for i, _ in enumerate(pool.starmap(run_therapy_session, args_list)):
            pass

    print(f"[MultiAgentESC] 全部完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
