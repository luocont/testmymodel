#!/usr/bin/env python3
"""
使用微调后的模型生成咨询对话

用法:
    python generate_dialogue.py \
        --input dataset/test_data.json \
        --output output/sessions \
        --max_turns 20
"""

import argparse
import json
import multiprocessing
import traceback
from pathlib import Path
from langchain.prompts import PromptTemplate
import openai


# ========== 配置 ==========
# vLLM服务配置
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "dummy-key"

# 模拟客户的LLM配置（可选，如果需要模拟客户）
CLIENT_MODEL = "gpt-4o-mini"  # 或使用其他模型
CLIENT_API_KEY = "your-api-key"  # 如果使用OpenAI/Azure
CLIENT_BASE_URL = None  # 如果使用自定义endpoint


def generate_counselor_response(prompt, client):
    """使用微调后的模型生成咨询师回复"""
    response = client.chat.completions.create(
        model="model",  # vLLM会忽略这个参数
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512
    )
    return response.choices[0].message.content


def generate_client_response(history, intake_form, attitude, attitude_instruction, client=None):
    """
    生成客户回复

    如果提供了client，使用LLM生成；否则使用简单的规则
    """
    if client is not None:
        # 使用LLM模拟客户
        prompt = f"""You are a client in a counseling session.

Client Background:
{intake_form}

Client Attitude: {attitude}
{attitude_instruction}

Conversation History:
{format_history(history)}

Generate your response as the client. Be natural and stay in character.
If you feel the session has reached a natural conclusion, include "[/END]" at the end of your message."""

        response = client.chat.completions.create(
            model=CLIENT_MODEL if hasattr(CLIENT_MODEL, 'strip') else "model",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=300
        )
        return response.choices[0].message.content
    else:
        # 使用占位符 - 实际使用时应该提供真实的客户LLM或使用预设回复
        return "Thank you for that. I appreciate your help.[/END]"


def format_history(history):
    """格式化对话历史为文本"""
    return '\n'.join([
        f"{msg['role'].capitalize()}: {msg['message']}"
        for msg in history
    ])


def generate_single_session(example, max_turns, client_model, client_llm=None):
    """生成单个咨询会话"""
    try:
        # 初始化会话
        history = []

        # 获取初始对话
        init_counselor = example['AI_counselor']['CBT']['init_history_counselor']
        init_client = example['AI_counselor']['CBT']['init_history_client']

        history.append({"role": "counselor", "message": init_counselor})
        history.append({"role": "client", "message": init_client})

        # 获取客户信息
        intake_form = example['AI_client']['intake_form']
        attitude = example['AI_client']['attitude']
        attitude_instruction = example['AI_client']['attitude_instruction']
        reason_counseling = example['AI_counselor']['CBT']['reason_counseling']

        # 对话轮次
        for turn in range(max_turns):
            # 咨询师回复
            prompt = f"""You are a professional counselor. Generate your response to the client.

Client Information:
{example['AI_counselor']['CBT']['client_information']}

Reason for Counseling:
{reason_counseling}

Conversation History:
{format_history(history)}

Generate your counselor response:"""

            counselor_msg = generate_counselor_response(prompt, client_model)
            counselor_msg = counselor_msg.replace("Counselor:", "").strip()
            history.append({"role": "counselor", "message": counselor_msg})

            # 客户回复
            client_msg = generate_client_response(
                history, intake_form, attitude, attitude_instruction, client_llm
            )
            client_msg = client_msg.replace("Client:", "").strip()

            # 检查是否结束
            if "[/END]" in client_msg:
                client_msg = client_msg.replace("[/END]", "").strip()
                history.append({"role": "client", "message": client_msg})
                break

            history.append({"role": "client", "message": client_msg})

        return {
            "example": example,
            "history": history
        }

    except Exception as e:
        raise Exception(f"Error generating session: {str(e)}\n{traceback.format_exc()}")


def process_single_file(index, examples, output_dir, max_turns, vllm_url, client_llm=None):
    """处理单个文件的包装函数（用于多进程）"""
    try:
        # 初始化vLLM客户端
        client_model = openai.OpenAI(
            base_url=vllm_url,
            api_key=VLLM_API_KEY
        )

        example = examples[index]
        session_data = generate_single_session(example, max_turns, client_model, client_llm)

        # 保存结果
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_name = f"session_{index + 1}.json"
        file_path = output_path / file_name

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=4)

        print(f"✓ Generated session_{index + 1}.json")

    except Exception as e:
        error_path = Path(output_dir) / f"error_{index + 1}.txt"
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(f"Error: {str(e)}\n\n{traceback.format_exc()}")
        print(f"✗ Error generating session_{index + 1}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="使用微调后的模型生成咨询对话")
    parser.add_argument("--input", type=str, required=True,
                        help="输入JSON文件路径（客户初始设定）")
    parser.add_argument("--output", type=str, required=True,
                        help="输出目录路径")
    parser.add_argument("--max_turns", type=int, default=20,
                        help="最大对话轮次（默认20）")
    parser.add_argument("--num_processes", type=int, default=1,
                        help="并行进程数（默认1）")
    parser.add_argument("--vllm_url", type=str, default=VLLM_BASE_URL,
                        help=f"vLLM服务URL（默认: {VLLM_BASE_URL}）")
    parser.add_argument("--use_client_llm", action="store_true",
                        help="是否使用LLM模拟客户（需要配置API密钥）")
    parser.add_argument("--client_api_key", type=str, default=None,
                        help="客户模拟LLM的API密钥")

    args = parser.parse_args()

    # 加载输入数据
    print(f"Loading data from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        examples = json.load(f)

    print(f"Loaded {len(examples)} examples")

    # 可选：初始化客户LLM
    client_llm = None
    if args.use_client_llm:
        if args.client_api_key:
            client_llm = openai.OpenAI(
                api_key=args.client_api_key,
                base_url=CLIENT_BASE_URL
            )
            print("Using LLM to simulate client responses")
        else:
            print("Warning: --use_client_llm specified but no --client_api_key provided")
            print("Client responses will be placeholders. Please provide API key or remove --use_client_llm flag.")

    # 生成会话
    print(f"Generating {len(examples)} sessions with {args.num_processes} processes...")

    if args.num_processes == 1:
        # 单进程
        for i in range(len(examples)):
            process_single_file(i, examples, args.output, args.max_turns, args.vllm_url, client_llm)
    else:
        # 多进程
        with multiprocessing.Pool(args.num_processes) as pool:
            pool.starmap(
                process_single_file,
                [(i, examples, args.output, args.max_turns, args.vllm_url, client_llm)
                 for i in range(len(examples))]
            )

    print(f"\n✓ Done! Sessions saved to {args.output}/")


if __name__ == "__main__":
    main()
