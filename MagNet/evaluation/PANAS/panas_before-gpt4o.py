from langchain.prompts import PromptTemplate
import json
import argparse
from openai import AzureOpenAI
import tiktoken
import re
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

def load_prompt(file_name):
    base_dir = "prompts/"
    file_path = base_dir + file_name
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

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
            feel = re.sub(r'[^a-zA-Z0-9]', '', txt_list_words[0]).lower()
            score_str = re.sub(r'[^a-zA-Z0-9]', '', txt_list_words[-1])
            score = int(score_str)
            score_dict[feel].append(score)
    return score_dict

def save_as_json(dictionary, filename):
    """Save dictionary to a JSON file"""
    with open(filename, 'w') as file:
        json.dump(dictionary, file, indent=4)
    print(f"Dictionary saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate PANAS results")
    parser.add_argument("-o","--output_dir", type=str, default=".",
                        help="Directory to to save the results.")
    parser.add_argument("-m_iter","--max_iter", type=int, default=3,
                        help="Number of times GPT-4o is run for scoring a single client.")

    args = parser.parse_args()

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

    with open("../../dataset/data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for number in range(len(data)):
        total_cost = 0
        prompt_text = load_prompt("panas_before.txt")
        prompt_template = PromptTemplate(
                            input_variables=["intake_form"],
                            template=prompt_text)
        prompt = prompt_template.format(
                    intake_form = data[number]['AI_client']['intake_form']
                )
        messages = [{'role': 'user', 'content': prompt}]
        response = evaluator.chat.completions.create(
                    messages=messages,
                    temperature=0,
                    model=model_for_cost,
                    n=args.max_iter
                    )
        score_dict = score(response, args.max_iter)
        score_dict['cost'] = 0
        score_dict['attitude'] = data[number]['AI_client']['attitude']
        for i in range(args.max_iter):
            score_dict['cost'] = score_dict['cost'] + calculate_cost(messages, response.choices[i].message.content, model_for_cost)
        save_as_json(score_dict, args.output_dir + 'session_' + str(number+1) + '.json')
