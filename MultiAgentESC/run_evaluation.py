"""
MultiAgentESC框架评估脚本
遍历详细报告目录下的所有JSON文件，使用MultiAgentESC框架生成回复
"""

import json
import os
import glob
import argparse
import autogen
from tqdm import tqdm
from multiagent import single_agent_response
from prompt import get_prompt


def load_config(llm_name):
    """加载LLM配置"""
    config_list = autogen.config_list_from_json(
        env_or_file="OAI_CONFIG_LIST",
        file_location=".",
        filter_dict={
            "model": [llm_name],
        }
    )
    return config_list


def json2natural(history):
    """将历史对话转换为自然语言格式"""
    natural_language = ""
    for u in history:
        content = u["content"].strip()
        role = u["role"].capitalize()
        natural_language += f"{role}: {content} "
    return natural_language.strip()


def process_conversation(conversation, config_list, cache_path_root):
    """处理单个对话，为每个user_input生成回复"""
    history = []  # 对话历史

    for turn in conversation["turns"]:
        user_input = turn["user_input"]

        # 构建当前上下文
        context = json2natural(history + [{"role": "user", "content": user_input}])

        # 使用single_agent_response生成回复
        try:
            response = single_agent_response(
                get_prompt("zero_shot").format(context=context),
                config_list=config_list,
                cache_path_root=cache_path_root
            )
            turn["model_reply"] = response
        except Exception as e:
            print(f"生成回复时出错: {e}")
            turn["model_reply"] = "抱歉，生成回复时出现错误。"

        # 将用户输入和生成的回复添加到历史记录
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": turn["model_reply"]})

    return conversation


def main():
    parser = argparse.ArgumentParser(description="使用MultiAgentESC框架处理评估数据")
    parser.add_argument("--input_dir", type=str, default=r"e:\GitLoadWareHouse\testmymodel\详细报告",
                        help="输入JSON文件目录")
    parser.add_argument("--output_dir", type=str, default="results/evaluation",
                        help="输出结果目录")
    parser.add_argument("--llm_name", type=str, default="qwen2.5-7b-instruct",
                        help="LLM模型名称")
    parser.add_argument("--cache_path_root", type=str, default=".cache",
                        help="缓存路径")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理的对话数量（用于测试）")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载配置
    print(f"加载LLM配置: {args.llm_name}")
    config_list = load_config(args.llm_name)

    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(args.input_dir, "*.json"))
    print(f"找到 {len(json_files)} 个JSON文件")

    # 处理每个文件
    for json_file in tqdm(json_files, desc="处理文件"):
        filename = os.path.basename(json_file)
        output_file = os.path.join(args.output_dir, f"multiagent_{filename}")

        # 跳过已处理的文件
        if os.path.exists(output_file):
            print(f"跳过已处理文件: {filename}")
            continue

        print(f"\n处理文件: {filename}")

        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 更新元数据
        data["evaluation_metadata"]["evaluation_model"] = args.llm_name
        data["evaluation_metadata"]["evaluated_model_framework"] = "MultiAgentESC"

        # 处理每个对话
        conversations_to_process = data["conversations"][:args.limit] if args.limit else data["conversations"]
        print(f"处理 {len(conversations_to_process)} 个对话")

        for i, conv in enumerate(conversations_to_process):
            print(f"  处理对话 {i+1}/{len(conversations_to_process)} (ID: {conv.get('conversation_id', 'N/A')})")
            data["conversations"][i] = process_conversation(conv, config_list, args.cache_path_root)

        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"已保存到: {output_file}")

    print("\n处理完成！")


if __name__ == "__main__":
    main()
