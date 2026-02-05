from langchain.prompts import PromptTemplate
import os
import json
import argparse
import os
from openai import AzureOpenAI
import tiktoken
from pathlib import Path

# 导入通用评估客户端
import sys
sys.path.append("..")
from llm_evaluator import (
    create_eval_client_from_env,
    create_eval_aliyun_client,
    create_eval_openrouter_client,
    create_eval_azure_client,
    get_eval_cost,
    EvalLLMClient,
)

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4o")
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to count tokens
def num_tokens_from_message(messages, model="gpt-4o"):
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
    with open(filename, 'w') as file:
        json.dump(dictionary, file, indent=4)
    print(f"Dictionary saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CTRS results")
    parser.add_argument("-i","--input_dir", type=str, default=".",
                        help="Directory to read the sessions")
    parser.add_argument("-o","--output_dir", type=str, default=".",
                        help="Directory to to save the results.")
    parser.add_argument("-m_iter","--max_iter", type=int, default=3,
                        help="Number of times GPT-4o is run for scoring a single session.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================
    # 评估 API 配置
    # ============================================
    # 从环境变量加载配置（推荐）
    # 需要设置的环境变量:
    # EVAL_LLM_PROVIDER=openrouter
    # EVAL_LLM_API_KEY=your_openrouter_api_key
    # EVAL_LLM_MODEL=openai/gpt-4o

    if os.getenv("EVAL_LLM_PROVIDER") and os.getenv("EVAL_LLM_API_KEY"):
        print(f"从环境变量加载评估 API 配置: {os.getenv('EVAL_LLM_PROVIDER')}")
        evaluator_client = create_eval_client_from_env()
        evaluator = evaluator_client.client
        model_for_cost = evaluator_client.model
        print(f"使用评估模型: {model_for_cost}")
    else:
        print("未设置评估环境变量，使用默认 Azure 配置")
        # 兼容旧代码 - 保留原有 Azure 配置
        endpoint = "azure endpoint"
        model_name = "gpt-4o"
        deployment = "gpt-4o"
        model_for_cost = model_name  # 用于成本计算

        subscription_key = "subscription key"
        api_version = "api version"

        evaluator = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )

    # 获取当前模型的成本配置
    input_cost_per_m, output_cost_per_m = get_eval_cost(model_for_cost)

    ctrs_list = ["general_1_understanding", "general_2_interpersonal_effectiveness", "general_3_collaboration", "CBT_1_guided_discovery", "CBT_2_focus", "CBT_3_strategy"]
    
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(args.input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            score_dict = {}
            total_cost = 0
            for i in range(len(ctrs_list)):
                score = 0
                prompt_text = load_prompt(ctrs_list[i] + ".txt")
                prompt_template = PromptTemplate(
                                    input_variables=["conversation"],
                                    template=prompt_text)
                prompt = prompt_template.format(
                                                    conversation = generate_history(json_data["history"])
                                                )
                messages = [{'role': 'user', 'content': prompt}]
                response = evaluator.chat.completions.create(
                            messages=messages,
                            temperature=0,
                            model=model_for_cost,
                            n=args.max_iter
                            )
                for j in range(args.max_iter):
                    total_cost = total_cost + calculate_cost(messages, response.choices[j].message.content, model_for_cost)
                    txt_response = response.choices[j].message.content
                    score = score + int(txt_response.split(',')[0])
                avg_score = score/args.max_iter
                score_dict[ctrs_list[i]] = avg_score
            score_dict["cost"] = total_cost
            save_as_json(score_dict, os.path.join(args.output_dir, filename))
