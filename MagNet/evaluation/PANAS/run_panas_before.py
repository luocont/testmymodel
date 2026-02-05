# -*- coding: utf-8 -*-
import sys
import os

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

if __name__ == "__main__":
    import argparse
    from langchain.prompts import PromptTemplate
    import json
    import re
    from pathlib import Path

    from llm_evaluator import (
        create_eval_client_from_env,
        get_eval_cost,
    )
    from dotenv import load_dotenv

    def num_tokens_from_string(string: str) -> int:
        """Returns the number of tokens in a text string."""
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-4o")
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def num_tokens_from_message(messages, model="gpt-4o"):
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        tokens_per_message = 3
        total_tokens = 0

        for message in messages:
            total_tokens += tokens_per_message + len(enc.encode(message["content"]))

        total_tokens += 3
        return total_tokens

    def calculate_cost(input_message, output, model: str = "gpt-4o"):
        """计算评估成本"""
        input_cost_per_m, output_cost_per_m = get_eval_cost(model)
        input_cost = (num_tokens_from_message(input_message) * input_cost_per_m) / 1000000
        output_cost = (num_tokens_from_string(output) * output_cost_per_m) / 1000000
        return input_cost + output_cost

    def score(response, max_iter):
        score_dict = {}
        feelings = ['interested', 'excited', 'strong', 'enthusiastic', 'proud', 'alert', 'inspired', 'determined', 'attentive', 'active', 'distressed', 'upset', 'guilty', 'scared', 'hostile', 'irritable', 'ashamed', 'nervous', 'jittery', 'afraid']
        for i in range(len(feelings)):
            score_dict[feelings[i]] = []
        for i in range(max_iter):
            txt = response.choices[i].message.content
            txt_list = txt.split('\n')
            for j in range(len(txt_list)):
                txt_list_words = txt_list[j].split()
                if len(txt_list_words) >= 2:
                    feel = re.sub(r'[^a-zA-Z0-9]', '', txt_list_words[0]).lower()
                    score_str = re.sub(r'[^a-zA-Z0-9]', '', txt_list_words[-1])
                    try:
                        score = int(score_str)
                        if feel in score_dict:
                            score_dict[feel].append(score)
                    except ValueError:
                        pass
        return score_dict

    def save_as_json(dictionary, filename):
        """Save dictionary to a JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, indent=4, ensure_ascii=False)
        print(f"Dictionary saved to {filename}")

    def load_prompt(file_name):
        base_dir = "prompts/"
        file_path = base_dir + file_name
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    # 解析参数
    parser = argparse.ArgumentParser(description="Evaluate PANAS before results")
    parser.add_argument("-o", "--output_dir", type=str, default="../../output-panas-before",
                        help="Directory to save the results")
    parser.add_argument("-m_iter", "--max_iter", type=int, default=1,
                        help="Number of times GPT-4o is run for scoring")

    args = parser.parse_args()

    # 加载环境变量
    env_path = Path("../../.env")
    if env_path.exists():
        load_dotenv(env_path)
        print(f"已加载环境变量: {env_path}")
    else:
        print("警告：未找到 .env 文件")

    # 创建评估客户端
    if os.getenv("EVAL_LLM_PROVIDER") and os.getenv("EVAL_LLM_API_KEY"):
        print(f"从环境变量加载评估 API 配置: {os.getenv('EVAL_LLM_PROVIDER')}")
        evaluator_client = create_eval_client_from_env()
        model_for_cost = evaluator_client.model
        print(f"使用评估模型: {model_for_cost}")
    else:
        print("错误：未设置评估环境变量 EVAL_LLM_PROVIDER 和 EVAL_LLM_API_KEY")
        sys.exit(1)

    # 加载数据
    with open("../../dataset/data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"开始 PANAS Before 评估")
    print(f"输出目录: {args.output_dir}")
    print(f"评估次数: {args.max_iter}")
    print("=" * 60)

    for number in range(len(data)):
        total_cost = 0
        print(f"\n处理客户 [{number+1}/{len(data)}]")

        prompt_text = load_prompt("panas_before.txt")
        prompt_template = PromptTemplate(
            input_variables=["intake_form"],
            template=prompt_text
        )
        prompt = prompt_template.format(
            intake_form=data[number]['AI_client']['intake_form']
        )
        messages = [{'role': 'user', 'content': prompt}]

        max_retries = 3
        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            try:
                if retry_count > 0:
                    print(f"  重试 {retry_count}/{max_retries}...")

                # 使用 chat_completion 方法
                response = evaluator_client.chat_completion(
                    messages=messages,
                    temperature=0,
                    n=args.max_iter
                )

                score_dict = score(response, args.max_iter)
                score_dict['cost'] = 0
                score_dict['attitude'] = data[number]['AI_client']['attitude']

                for i in range(args.max_iter):
                    score_dict['cost'] = score_dict['cost'] + calculate_cost(messages, response.choices[i].message.content, model_for_cost)

                save_as_json(score_dict, args.output_dir + '/session_' + str(number+1) + '.json')
                print(f"  完成 - session_{number+1}.json")
                success = True

            except Exception as e:
                import traceback
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"  错误 - session_{number+1}: {e}")
                    traceback.print_exc()
                else:
                    print(f"  错误 - session_{number+1}: {e}")
                    import time
                    time.sleep(2)

    print("\n" + "=" * 60)
    print("PANAS Before 评估完成！")
    print(f"结果已保存到: {args.output_dir}")
    print("=" * 60)
