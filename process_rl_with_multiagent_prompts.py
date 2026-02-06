"""
使用 MultiAgentESC 提示词系统 + API 处理 RL.json 的脚本

这个脚本会读取 RL.json 文件，对其中每个 message 的 role 为 user 的 content
使用 MultiAgentESC 的提示词系统（通过 API 直接调用）生成回答，保持上下文支持，
最终保存成和 RL.json 相同格式的回答汇总 JSON 文件。

MultiAgentESC 提示词系统包括：
1. 复杂度判断（behavior_control）
2. 情感分析（get_emotion）
3. 原因分析（get_cause）
4. 意图分析（get_intention）
5. 策略选择和响应生成
"""

import json
import sys
import os
from pathlib import Path
from openai import OpenAI
import re
import heapq
import numpy as np
from sentence_transformers import util
from collections import Counter


# MultiAgentESC 的提示词系统
MULTIAGENTESC_PROMPTS = {
    "behavior_control": '''### Instruction
You are a psychological counseling expert. You will be provided with an incomplete conversation between an Assistant and a User.
Please analyze whether this conversation reflects the user's current emotional state, the reason the user is seeking emotional support, and how the user plans to cope with the event.
If all three points are reflected, please reply "YES," otherwise reply "NO."

### Conversation
{context}

Your answer must include two parts:
1. "YES" or "NO"
2. If "YES", briefly explain how the conversation reflects these elements; if "NO", explain which elements are missing.

Your answer must follow this format:
1. [YES or NO]
2. [explaination]
''',

    "zero_shot": '''### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Your task is to play a role as 'Assistant' and generate a response based on the given dialogue context.

### Dialogue context
{context}

Your answer must be fewer than 30 words and must follow this format:
Response: [response]
''',

    "get_emotion": '''### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Please infer the emotional state expressed in the user's last utterance.

### Dialogue context
{context}

Your answer must include the following elements:
Emotion: the emotion user expressed in their last utterance.
Reasoning: the reasoning behind your answer.

Your answer must follow this format:
Emotion: [emotion]
Reasoning: [reasoning]
''',

    "get_cause": '''### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Another agent analyzes the conversation and infers the emotional state expressed by the user in their last utterance.

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

Please infer the specific event that led to the user's emotional state based on the dialogue context. Your answer must include the following elements:
Event: the specific event that led to the user's emotional state.
Reasoning: the reasoning behind your answer.

Your answer must follow this format:
Event: [event]
Reasoning: [reasoning]
''',

    "get_intention": '''### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Other agents have analyzed the conversation, infering the emotional state expressed by the user in their last utterance and the specific event that led to the user's emotional state.

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

### Event
{cau_and_reason}

Please reasonably infer the user's intention based on the dialogue context, with the goal of addressing the event that lead to their emotional state. Your answer must include the following elements:
Intention: user's intention which aims to address the event that lead to their emotional state.
Reasoning: the reasoning behind your answer.

Your answer must follow this format:
Intention: [intention]
Reasoning: [reasoning]
''',

    "response_with_strategy": '''You will be provided with a dialogue context between an 'Assistant' and a 'User'. Psychologists have analyzed the conversation, infering the emotional state expressed by the user in their last utterance, the specific event that led to the user's emotional state and user's intention aiming to address the event that lead to their emotional state.

### Dialogue context
{context}

### Emotional state
{emo_and_reason}

### Event
{cau_and_reason}

### Intention
{int_and_reason}

Please generate a response from the Assistant's perspective using the {strategy} strategy.
The following are examples of this strategy, all presented in the format of <post\n[strategy] response>.

### Examples
{examples}

Your answer must be fewer than 30 words and must follow this format:
Response: [strategy] [response]
''',
}


class MultiAgentESCWithAPI:
    """
    使用 API 实现 MultiAgentESC 提示词系统的处理器
    """

    def __init__(self, api_key, base_url, model_name, cache_path_root="", model_path="all-roberta-large-v1", timeout=300):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout  # 设置超时时间为 300 秒
        )
        self.model_name = model_name
        self.cache_path_root = cache_path_root
        self.model_path = model_path

        # 延迟加载模型
        self.model = None
        self.quadruple = None

    def _load_model_and_data(self):
        """延迟加载模型和数据"""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_path)

        if self.quadruple is None:
            # 加载 MultiAgentESC 的嵌入数据
            quadruple_path = Path(__file__).parent / "MultiAgentESC" / "embeddings.txt"
            if quadruple_path.exists():
                with open(quadruple_path, "r", encoding="utf-8") as txt:
                    self.quadruple = txt.readlines()
                print(f"已加载 {len(self.quadruple)} 条参考数据")
            else:
                print(f"警告: 未找到 embeddings.txt 文件，将使用空参考数据")
                self.quadruple = []

    def _call_api(self, messages, temperature=0.0, max_tokens=400):
        """调用 API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API 调用失败: {e}")
            return ""

    def messages_to_natural(self, messages):
        """将 messages 列表转换为自然语言格式"""
        lines = []
        for msg in messages:
            role = msg.get('role', '').lower()
            content = msg.get('content', '').strip()

            if role in ['user', 'seeker', 'client']:
                lines.append(f"User: {content}")
            elif role in ['assistant', 'supporter', 'counselor']:
                lines.append(f"Assistant: {content}")
            elif role == 'system':
                continue  # system 消息不加入上下文

        return ' '.join(lines)

    def is_complex(self, context):
        """判断对话是否足够复杂需要多智能体协作"""
        prompt = MULTIAGENTESC_PROMPTS["behavior_control"].format(context=context)
        response = self._call_api([{"role": "user", "content": prompt}], max_tokens=200)
        return "yes" in response.lower()

    def get_emotion(self, context):
        """获取用户情感"""
        prompt = MULTIAGENTESC_PROMPTS["get_emotion"].format(context=context)
        response = self._call_api([{"role": "user", "content": prompt}], max_tokens=400)

        try:
            # 尝试多种匹配方式
            # 英文格式: Emotion: xxx
            emotion = re.findall(r'Emotion:\s*(.*?)(?:\n|Reasoning:)', response, re.IGNORECASE)
            if emotion:
                emotion = emotion[0].strip()
            else:
                # 中文格式: 情感：xxx 或 情绪：xxx
                emotion = re.findall(r'情感[：:]\s*(.*?)(?:\n|原因|推理)', response)
                if not emotion:
                    emotion = re.findall(r'情绪[：:]\s*(.*?)(?:\n|原因|推理)', response)
                if emotion:
                    emotion = emotion[0].strip()
                else:
                    # 直接提取第一个有意义的词
                    lines = response.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith(('Emotion', 'Reasoning', '情感', '原因', '推理')):
                            emotion = line.split()[0] if line.split() else "Negative"
                            break
                    else:
                        emotion = "Negative"
        except Exception as e:
            print(f"情感解析失败: {e}, 原始响应: {response[:100]}")
            emotion = "Negative"

        return emotion, response

    def get_cause(self, context, emo_and_reason):
        """获取事件原因"""
        prompt = MULTIAGENTESC_PROMPTS["get_cause"].format(context=context, emo_and_reason=emo_and_reason)
        response = self._call_api([{"role": "user", "content": prompt}], max_tokens=400)

        try:
            # 尝试多种匹配方式
            # 英文格式: Event: xxx
            cause = re.findall(r'Event:\s*(.*?)(?:\n|Reasoning:)', response, re.IGNORECASE)
            if cause:
                cause = cause[0].strip()
            else:
                # 中文格式: 事件：xxx 或 原因：xxx
                cause = re.findall(r'事件[：:]\s*(.*?)(?:\n|推理|分析)', response)
                if not cause:
                    cause = re.findall(r'原因[：:]\s*(.*?)(?:\n|推理|分析)', response)
                if cause:
                    cause = cause[0].strip()
                else:
                    # 如果没有匹配，尝试提取第一句话
                    lines = response.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and len(line) > 5 and not line.startswith(('Event', 'Reasoning', '事件', '原因', '推理')):
                            cause = line[:100]  # 限制长度
                            break
                    else:
                        cause = "Not mention"
        except Exception as e:
            print(f"原因解析失败: {e}, 原始响应: {response[:100]}")
            cause = "Not mention"

        return cause, response

    def get_intention(self, context, emo_and_reason, cau_and_reason):
        """获取用户意图"""
        prompt = MULTIAGENTESC_PROMPTS["get_intention"].format(
            context=context,
            emo_and_reason=emo_and_reason,
            cau_and_reason=cau_and_reason
        )
        response = self._call_api([{"role": "user", "content": prompt}], max_tokens=400)

        try:
            # 尝试多种匹配方式
            # 英文格式: Intention: xxx
            intention = re.findall(r'Intention:\s*(.*?)(?:\n|Reasoning:)', response, re.IGNORECASE)
            if intention:
                intention = intention[0].strip()
            else:
                # 中文格式: 意图：xxx 或 目的：xxx
                intention = re.findall(r'意图[：:]\s*(.*?)(?:\n|推理|分析)', response)
                if not intention:
                    intention = re.findall(r'目的[：:]\s*(.*?)(?:\n|推理|分析)', response)
                if intention:
                    intention = intention[0].strip()
                else:
                    # 如果没有匹配，尝试提取第一句话
                    lines = response.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and len(line) > 5 and not line.startswith(('Intention', 'Reasoning', '意图', '目的', '推理')):
                            intention = line[:100]  # 限制长度
                            break
                    else:
                        intention = "Not mention"
        except Exception as e:
            print(f"意图解析失败: {e}, 原始响应: {response[:100]}")
            intention = "Not mention"

        return intention, response

    def single_agent_response(self, context):
        """单智能体零样本生成"""
        prompt = MULTIAGENTESC_PROMPTS["zero_shot"].format(context=context)
        response = self._call_api([{"role": "user", "content": prompt}], max_tokens=100)

        try:
            # 尝试提取 Response: 后面的内容
            match = re.search(r'Response:\s*(.*?)(?:\n|$)', response, re.IGNORECASE)
            if match:
                response = match.group(1).strip()
            else:
                # 如果没有 Response: 前缀，直接使用返回的内容
                response = response.strip()
        except Exception as e:
            print(f"响应解析失败: {e}")
            response = "I understand. Please continue."

        return response

    def get_strategy(self, emo_and_reason, cau_and_reason, int_and_reason, context, post):
        """获取策略（简化版，使用默认策略）"""
        # 由于没有完整的策略匹配系统，返回默认策略
        default_strategies = ["Question", "Restatement or Paraphrasing", "Reflection of feelings"]
        return default_strategies

    def response_with_strategy(self, context, emo_and_reason, cau_and_reason, int_and_reason, strategy):
        """使用指定策略生成响应"""
        # 简化的示例
        examples = f"User: Hello\n[{strategy}] Hi there! How can I help you today?"

        prompt = MULTIAGENTESC_PROMPTS["response_with_strategy"].format(
            context=context,
            emo_and_reason=emo_and_reason,
            cau_and_reason=cau_and_reason,
            int_and_reason=int_and_reason,
            strategy=strategy,
            examples=examples
        )
        response = self._call_api([{"role": "user", "content": prompt}], max_tokens=100)

        try:
            # 尝试多种匹配方式
            # 格式: Response: [Strategy] content
            match = re.search(r'Response:\s*\[[^\]]+\]\s*(.*?)(?:\n|$)', response, re.IGNORECASE)
            if match:
                response = match.group(1).strip()
            elif strategy in response:
                # 如果包含策略名称，尝试提取策略后面的内容
                response = response.split(strategy, 1)[1].strip()
                if response.startswith(']'):
                    response = response[1:].strip()
            else:
                # 直接提取 Response: 后面的内容
                match = re.search(r'Response:\s*(.*)', response, re.IGNORECASE)
                if match:
                    response = match.group(1).strip()
                else:
                    response = response.strip()
        except Exception as e:
            print(f"策略响应解析失败: {e}")
            response = "I understand. Please continue."

        return response

    def generate_response_with_analysis(self, messages):
        """
        为给定的消息列表生成响应，并返回分析信息

        Args:
            messages: 消息列表，包含完整的对话历史

        Returns:
            (response, analysis): 响应文本和分析信息字典
        """
        # 将消息转换为自然语言格式
        context = self.messages_to_natural(messages)

        if not context.strip():
            context = "User: [新的对话开始]"

        analysis = {}

        # 检查是否需要多智能体协作
        try:
            if not self.is_complex(context):
                # 简单情况：使用零样本
                response = self.single_agent_response(context)
                analysis["emotion"] = "Not analyzed"
                analysis["cause"] = "Not analyzed"
                analysis["intention"] = "Not analyzed"
                analysis["strategy"] = "Zero-shot"
                return response, analysis
        except Exception as e:
            print(f"复杂度检查失败: {e}")

        # 复杂情况：使用 MultiAgentESC 流程
        try:
            # 1. 情感分析
            emotion, emo_and_reason = self.get_emotion(context)
            analysis["emotion"] = emotion
            print(f"    情感: {emotion}")

            # 2. 原因分析
            cause, cau_and_reason = self.get_cause(context, emo_and_reason)
            analysis["cause"] = cause
            print(f"    事件: {cause}")

            # 3. 意图分析
            intention, int_and_reason = self.get_intention(context, emo_and_reason, cau_and_reason)
            analysis["intention"] = intention
            print(f"    意图: {intention}")

            # 4. 获取策略
            strategies = self.get_strategy(emo_and_reason, cau_and_reason, int_and_reason, context, "")
            strategy = strategies[0]
            analysis["strategy"] = strategy

            # 5. 使用第一个策略生成响应
            response = self.response_with_strategy(
                context, emo_and_reason, cau_and_reason, int_and_reason, strategy
            )
            print(f"    策略: {strategy}")

            return response, analysis

        except Exception as e:
            print(f"MultiAgentESC 流程失败: {e}")
            import traceback
            traceback.print_exc()
            # 备选：使用零样本
            response = self.single_agent_response(context)
            analysis["emotion"] = "Error"
            analysis["cause"] = "Error"
            analysis["intention"] = "Error"
            analysis["strategy"] = "Fallback"
            return response, analysis

    def generate_response(self, messages):
        """
        为给定的消息列表生成响应（兼容旧接口）

        Args:
            messages: 消息列表，包含完整的对话历史

        Returns:
            response: 咨询师的响应文本
        """
        response, _ = self.generate_response_with_analysis(messages)
        return response


def process_rl_json(input_path, output_path, processor):
    """
    处理 RL.json 文件，逐个对话写入结果

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        processor: MultiAgentESC 处理器实例
    """
    # 读取输入文件
    print(f"读取输入文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"共 {len(data)} 个对话样本\n")

    # 初始化结果列表（用于追加保存）
    results = []
    total = len(data)

    # 如果输出文件已存在，先读取已有结果（支持断点续传）
    processed_ids = set()
    if Path(output_path).exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                results = existing_results
                processed_ids = {item.get('id') for item in existing_results if item.get('id') is not None}
                print(f"发现已有 {len(processed_ids)} 个处理结果，将跳过")
        except Exception as e:
            print(f"读取已有结果失败: {e}，将重新处理")
            results = []
            processed_ids = set()

    # 处理每个对话
    for idx, item in enumerate(data, 1):
        item_id = item.get('id', idx)

        # 跳过已处理的对话
        if item_id in processed_ids:
            print(f"[{idx}/{total}] 跳过对话 #{item_id} (已处理)")
            continue

        print(f"[{idx}/{total}] 处理对话 #{item_id}")
        result_item = {
            "id": item.get("id"),
            "normalizedTag": item.get("normalizedTag"),
            "messages": []
        }

        messages = item.get("messages", [])
        history = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # 只保留 system 和 user 消息，跳过原始的 assistant 消息
            if role.lower() in ['system', 'user']:
                result_item["messages"].append(msg)

            # 更新历史（用于生成上下文）
            if role.lower() != 'system':
                history.append(msg)

            # 如果是用户消息，生成响应
            if role.lower() in ['user', 'seeker', 'client']:
                print(f"  用户: {content[:50]}...")

                # 获取响应和分析信息
                response, analysis = processor.generate_response_with_analysis(history)

                # 添加响应到结果和历史
                response_msg = {
                    "role": "assistant",
                    "content": response
                }
                # 添加分析信息到消息中
                if analysis:
                    response_msg["emotion"] = analysis.get("emotion", "")
                    response_msg["cause"] = analysis.get("cause", "")
                    response_msg["intention"] = analysis.get("intention", "")
                    response_msg["strategy"] = analysis.get("strategy", "")

                result_item["messages"].append(response_msg)
                history.append(response_msg)

                print(f"  咨询师: {response[:50]}...")
        print()

        # 添加到结果列表
        results.append(result_item)

        # 立即写入文件（每次处理完一个对话就保存）
        print(f"  💾 保存对话 #{item_id} 到 {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n处理完成！共处理 {len(results)} 个对话")
    print(f"结果已保存到: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="使用 MultiAgentESC 提示词系统 + API 处理 RL.json"
    )
    parser.add_argument("-i", "--input", type=str, default="eval.json",
                        help="输入文件路径 (默认: eval.json)")
    parser.add_argument("-o", "--output", type=str, default="RL_result.json",
                        help="输出文件路径 (默认: RL_result.json)")
    parser.add_argument("--api_key", type=str, required=True,
                        help="API Key")
    parser.add_argument("--base_url", type=str, required=True,
                        help="API Base URL")
    parser.add_argument("--model", type=str, default="qwen2.5-7b-instruct",
                        help="模型名称 (默认: qwen2.5-7b-instruct)")
    parser.add_argument("--model_path", type=str, default="all-roberta-large-v1",
                        help="SentenceTransformer 模型路径 (用于策略选择)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="API 调用超时时间（秒），默认 300")

    args = parser.parse_args()

    # 设置工作目录
    os.chdir(Path(__file__).parent)

    # 初始化处理器
    print("初始化 MultiAgentESC (API 版本) 处理器...")
    print(f"  API Base URL: {args.base_url}")
    print(f"  模型: {args.model}")
    print()

    processor = MultiAgentESCWithAPI(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model,
        model_path=args.model_path,
        timeout=args.timeout
    )

    # 处理文件
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path(__file__).parent / args.input

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / args.output

    process_rl_json(input_path, output_path, processor)


if __name__ == "__main__":
    main()
