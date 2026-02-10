"""
使用本地 Qwen3 模型处理 eval.json 中的用户输入
读取 eval.json 中 role 为 user 的 content，使用 Qwen3 模型生成回答，并保存到新的 JSON 文件
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import List, Dict, Any
import argparse


def load_qwen_model(model_path: str):
    """
    加载本地 Qwen3 模型和分词器

    Args:
        model_path: 本地模型路径

    Returns:
        model, tokenizer
    """
    print(f"正在加载模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print("模型加载完成!")
    return model, tokenizer


def generate_response(model, tokenizer, user_input: str, system_prompt: str = None) -> str:
    """
    使用 Qwen3 模型生成回答

    Args:
        model: Qwen3 模型
        tokenizer: 分词器
        user_input: 用户输入
        system_prompt: 系统提示词（可选）

    Returns:
        模型生成的回答
    """
    # 构建消息列表
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_input})

    # 使用 chat 模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码输入
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成回答
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # 解码输出
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def process_eval_json(
    input_json_path: str,
    output_json_path: str,
    model_path: str,
    overwrite_assistant: bool = True
):
    """
    处理 eval.json 文件，为每个 user 消息生成模型回答

    Args:
        input_json_path: 输入 JSON 文件路径
        output_json_path: 输出 JSON 文件路径
        model_path: Qwen3 模型路径
        overwrite_assistant: 是否覆盖原有的 assistant 回答
    """
    # 加载模型
    model, tokenizer = load_qwen_model(model_path)

    # 读取输入 JSON 文件
    print(f"正在读取输入文件: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"共 {len(data)} 条对话记录")

    # 处理每一条记录
    processed_data = []

    for idx, item in enumerate(data):
        print(f"\n处理第 {idx + 1}/{len(data)} 条记录 (id: {item.get('id', idx)})")

        messages = item.get('messages', [])
        new_messages = []
        system_prompt = None

        # 遍历消息
        for msg in messages:
            role = msg.get('role')

            if role == 'system':
                # 保存系统提示
                system_prompt = msg.get('content')
                new_messages.append(msg)

            elif role == 'user':
                # 添加用户消息
                new_messages.append(msg)

                # 生成模型回答
                user_content = msg.get('content', '')
                print(f"  用户输入: {user_content}")

                response = generate_response(model, tokenizer, user_content, system_prompt)
                print(f"  模型回答: {response}")

                # 添加 assistant 回答
                new_messages.append({
                    "role": "assistant",
                    "content": response
                })

            elif role == 'assistant':
                # 如果不覆盖原有回答，则保留
                if not overwrite_assistant:
                    new_messages.append(msg)

        # 创建新记录
        new_item = {
            "id": item.get('id', idx),
            "normalizedTag": item.get('normalizedTag', ''),
            "messages": new_messages
        }

        processed_data.append(new_item)

    # 保存到输出文件
    print(f"\n正在保存输出文件: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print("处理完成!")


def main():
    parser = argparse.ArgumentParser(description='使用 Qwen3 模型处理 eval.json')
    parser.add_argument(
        '--input',
        type=str,
        default='e:/GitLoadWareHouse/testmymodel/MagNet/dataset/eval.json',
        help='输入 JSON 文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='e:/GitLoadWareHouse/testmymodel/MagNet/dataset/eval_output.json',
        help='输出 JSON 文件路径'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct',
        help='Qwen3 模型路径（本地路径或 HuggingFace 模型名）'
    )
    parser.add_argument(
        '--keep-original',
        action='store_true',
        help='保留原有的 assistant 回答，不覆盖'
    )

    args = parser.parse_args()

    process_eval_json(
        input_json_path=args.input,
        output_json_path=args.output,
        model_path=args.model,
        overwrite_assistant=not args.keep_original
    )


if __name__ == '__main__':
    main()
