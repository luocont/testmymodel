"""
MultiAgentESC 与 MagNet Client 集成脚本 - 阿里云 API 简化版

这个脚本让 MultiAgentESC 作为咨询师，使用阿里云 API 与 MagNet 的 ClientAgent 进行对话。
使用 MultiAgentESC 的提示词系统，但简化了多智能体流程。
"""

import argparse
import json
import multiprocessing
import sys
import os
from pathlib import Path
from langchain.prompts import PromptTemplate
import re

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入 MagNet 的 LLM 客户端
from llm_client import create_aliyun_client, LLMClient


# ============================================
# MultiAgentESC 提示词（直接嵌入）
# ============================================

PROMPTS = {
    "zero_shot": '''### Instruction
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Your task is to play a role as 'Assistant' and generate a response based on the given dialogue context.

### Dialogue context
{context}

Your answer must be fewer than 50 words and must follow this format:
Response: [response]
''',

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
You are a psychological counseling expert. You will be provided with a dialogue context between an 'Assistant' and a 'User'. Other agents have analyzed the conversation, inferring the emotional state expressed by the user in their last utterance and the specific event that led to the user's emotional state.

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
'''
}


def get_prompt(prompt_name):
    """获取提示词"""
    if prompt_name not in PROMPTS:
        raise ValueError(f"Prompt '{prompt_name}' not found.")
    return PROMPTS[prompt_name]


# ============================================
# MagNet ClientAgent 类定义
# ============================================

class ClientAgent:
    """
    MagNet 的 ClientAgent 类
    使用阿里云 API
    """

    def __init__(self, example, api_key=None, model="qwen2.5-7b-instruct"):
        """
        初始化客户端智能体

        Args:
            example: 样本数据
            api_key: 阿里云 API Key
            model: 模型名称
        """
        self.example = example
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")

        # 创建 LLM 客户端
        self.llm_client = create_aliyun_client(
            api_key=self.api_key,
            model=model,
            temperature=0.7,
            max_tokens=512
        )

        # 加载提示词
        prompt_text = self._load_prompt("agent_client.txt")
        self.attitude = (
            f"{self.example['AI_client']['attitude']}: "
            f"{self.example['AI_client']['attitude_instruction']}"
        )
        self.prompt_template = PromptTemplate(
            input_variables=["intake_form", "attitude", "history"],
            template=prompt_text
        )

    def _load_prompt(self, file_name):
        """加载提示词文件"""
        base_dir = Path(__file__).parent.parent / "prompts"
        file_path = base_dir / file_name
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def generate(self, history):
        """
        生成客户端响应

        Args:
            history: 对话历史列表

        Returns:
            客户端的响应
        """
        history_text = '\n'.join(
            [
                f"{message['role'].capitalize()}: {message['message']}"
                for message in history
            ]
        )

        prompt = self.prompt_template.format(
            intake_form=self.example,
            attitude=self.attitude,
            history=history_text
        )

        response = self.llm_client.completion(prompt=prompt)
        return response.choices[0].message.content


# ============================================
# MultiAgentESC 咨询师类
# ============================================

class MultiAgentESCCounselorAliyun:
    """
    MultiAgentESC 咨询师适配器 - 阿里云 API 版本
    """

    def __init__(self, api_key: str, model: str = "qwen2.5-7b-instruct"):
        """
        初始化咨询师

        Args:
            api_key: 阿里云 API Key
            model: 模型名称
        """
        self.api_key = api_key
        self.model = model

        # 创建 LLM 客户端
        self.llm_client = create_aliyun_client(
            api_key=api_key,
            model=model,
            temperature=0.0,
            max_tokens=512
        )

    def generate_response(self, context: str) -> str:
        """
        生成咨询师响应（使用 MultiAgentESC 的零样本提示词）

        Args:
            context: 对话上下文

        Returns:
            咨询师的响应
        """
        # 使用 MultiAgentESC 的零样本提示词
        prompt = get_prompt("zero_shot").format(context=context)

        response = self.llm_client.completion(prompt=prompt)

        # 提取响应内容
        response_text = response.choices[0].message.content

        try:
            # 尝试提取 Response: 后面的内容
            response_text = re.findall(r'Response:\s*(.*)', response_text)[0].strip()
        except:
            # 如果格式不匹配，直接使用返回的内容
            response_text = response_text.strip()

        return response_text


# ============================================
# 辅助函数
# ============================================

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


# ============================================
# 主处理函数
# ============================================

def run_therapy_session(index, example, output_dir, total, max_turns,
                       api_key, model):
    """
    运行一个咨询会话

    Args:
        index: 样本索引
        example: 样本数据
        output_dir: 输出目录
        total: 总样本数
        max_turns: 最大轮次
        api_key: 阿里云 API Key
        model: 模型名称
    """
    output_dir = Path(output_dir)
    file_number = index + 1

    try:
        print(f"[MultiAgentESC-Aliyun] 生成第 {file_number}/{total} 个样本")

        # 初始化 MultiAgentESC 咨询师
        counselor = MultiAgentESCCounselorAliyun(
            api_key=api_key,
            model=model
        )

        # 初始化 MagNet 客户端
        client = ClientAgent(example=example, api_key=api_key, model=model)

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

            history.append({
                "role": "client",
                "message": client_response
            })

            # 检查是否结束
            if '[/END]' in client_response:
                history[-1]['message'] = history[-1]['message'].replace('[/END]', '')
                break

        # 准备输出数据
        session_data = {
            "example": example,
            "cbt_technique": "MultiAgentESC-Aliyun (Strategy-based)",
            "cbt_plan": f"MultiAgentESC using Aliyun {model} with dynamic strategy selection.",
            "cost": 0,
            "history": history
        }

        # 保存结果
        file_name = f"session_{file_number}.json"
        file_path = output_dir / file_name

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=4)

        print(f"[MultiAgentESC-Aliyun] 完成 {file_number}/{total}")

    except Exception as e:
        import traceback
        error_file_name = f"error_multiagentesc_aliyun_{file_number}.txt"
        error_file_path = output_dir / error_file_name
        tb = e.__traceback__
        with open(error_file_path, "w", encoding="utf-8") as f:
            f.write(f"Error: {e}\n")
            f.write("".join(traceback.format_exception(type(e), e, tb)))
        print(f"[MultiAgentESC-Aliyun] 错误 {file_number}: {e}")


# ============================================
# 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="使用 MultiAgentESC + 阿里云 API 生成对话"
    )
    parser.add_argument("-o", "--output_dir", type=str, default="../output-multiagentesc-aliyun",
                        help="输出目录")
    parser.add_argument("-d", "--dataset", type=str,
                        default="../dataset/data_cn.json",
                        help="数据集文件路径")
    parser.add_argument("-num_pr", "--num_processes", type=int, default=None,
                        help="并行进程数")
    parser.add_argument("-m_turns", "--max_turns", type=int, default=20,
                        help="最大对话轮次")
    parser.add_argument("--model", type=str, default="qwen2.5-7b-instruct",
                        help="阿里云模型名称")
    parser.add_argument("--api_key", type=str, default=None,
                        help="阿里云 API Key（也可通过 DASHSCOPE_API_KEY 环境变量设置）")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="处理样本数量（默认全部）")

    args = parser.parse_args()

    # 获取 API Key
    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 未设置阿里云 API Key")
        print("请通过以下方式之一设置：")
        print("1. 使用 --api_key 参数")
        print("2. 设置 DASHSCOPE_API_KEY 环境变量")
        print("\n例如:")
        print("  $env:DASHSCOPE_API_KEY='your-api-key'  # PowerShell")
        print("  set DASHSCOPE_API_KEY=your-api-key  # CMD")
        sys.exit(1)

    # 设置工作目录
    os.chdir(Path(__file__).parent)

    # 加载数据集
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).parent.parent / "dataset" / "data_cn.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 限制样本数量
    if args.num_samples:
        data = data[:args.num_samples]

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(data)
    print(f"[MultiAgentESC-Aliyun] 开始处理 {total} 个样本")
    print(f"[MultiAgentESC-Aliyun] 使用模型: {args.model}")
    print(f"[MultiAgentESC-Aliyun] 输出目录: {output_dir}")
    print(f"[MultiAgentESC-Aliyun] API Key: {api_key[:10]}...{api_key[-4:]}")

    # 准备参数列表
    args_list = [
        (index, example, output_dir, total, args.max_turns, api_key, args.model)
        for index, example in enumerate(data)
    ]

    # 并行处理
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        for i, _ in enumerate(pool.starmap(run_therapy_session, args_list)):
            pass

    print(f"[MultiAgentESC-Aliyun] 全部完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
