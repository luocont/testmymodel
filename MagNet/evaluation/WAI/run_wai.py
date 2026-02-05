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

    def generate_history(history):
        history_text = '\n'.join(
            [
                f"{message['role'].capitalize()}: {message['message']}"
                for message in history
            ]
        )
        return history_text

    def load_prompt(file_name):
        base_dir = "prompts/"
        file_path = base_dir + file_name
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def save_as_json(dictionary, filename):
        """Save dictionary to a JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, indent=4, ensure_ascii=False)
        print(f"Dictionary saved to {filename}")

    # 解析参数
    parser = argparse.ArgumentParser(description="Evaluate WAI results")
    parser.add_argument("-i", "--input_dir", type=str, default="../../output-cn",
                        help="Directory to read the sessions")
    parser.add_argument("-o", "--output_dir", type=str, default="../../output-wai",
                        help="Directory to save the results")
    parser.add_argument("-m_iter", "--max_iter", type=int, default=1,
                        help="Number of times GPT-4o is run for scoring a single session")

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"开始 WAI 评估")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"评估次数: {args.max_iter}")
    print("=" * 60)

    for idx, filename in enumerate(os.listdir(args.input_dir), 1):
        if filename.endswith('.json'):
            file_path = os.path.join(args.input_dir, filename)
            print(f"\n处理文件 [{idx}]: {filename}")

            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            score_dict = {}
            total_cost = 0

            # WAI 有 12 个评估项目
            for i in range(12):
                score = 0
                prompt_text = load_prompt("wai" + str(i+1) + ".txt")
                prompt_template = PromptTemplate(
                    input_variables=["conversation"],
                    template=prompt_text
                )
                prompt = prompt_template.format(
                    conversation=generate_history(json_data["history"])
                )
                messages = [{'role': 'user', 'content': prompt}]

                max_retries = 3
                retry_count = 0
                success = False

                while retry_count < max_retries and not success:
                    try:
                        if retry_count > 0:
                            print(f"  重试 {retry_count}/{max_retries} - wai{i+1}...")

                        # 使用 chat_completion 方法
                        response = evaluator_client.chat_completion(
                            messages=messages,
                            temperature=0,
                            n=args.max_iter
                        )

                        for j in range(args.max_iter):
                            total_cost = total_cost + calculate_cost(messages, response.choices[j].message.content, model_for_cost)
                            txt_response = response.choices[j].message.content
                            score = score + int(txt_response.split(',')[0])

                        avg_score = score / args.max_iter
                        score_dict["wai" + str(i+1)] = avg_score
                        print(f"  wai{i+1}: {avg_score:.2f}")
                        success = True

                    except Exception as e:
                        import traceback
                        retry_count += 1
                        if retry_count >= max_retries:
                            print(f"  错误 - wai{i+1}: {e}")
                            traceback.print_exc()
                            score_dict["wai" + str(i+1)] = 0
                        else:
                            print(f"  错误 - wai{i+1}: {e}")
                            import time
                            time.sleep(2)

            score_dict["cost"] = total_cost
            save_as_json(score_dict, os.path.join(args.output_dir, filename))

    print("\n" + "=" * 60)
    print("WAI 评估完成！")
    print(f"结果已保存到: {args.output_dir}")
    print("=" * 60)
