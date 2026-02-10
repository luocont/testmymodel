"""
MultiAgentESC框架评估脚本（纯API版本）
只使用阿里云API，不依赖任何本地模型
使用框架中的原始prompts
"""

import json
import os
import glob
import argparse
from tqdm import tqdm
from openai import OpenAI


def get_prompt(prompt_name):
    """加载框架中的原始prompts"""
    prompts = {
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
'''
    }

    if prompt_name not in prompts:
        raise ValueError(f"Prompt '{prompt_name}' not found.")
    return prompts[prompt_name]


class SimpleMultiAgentESC:
    """简化的MultiAgentESC框架，直接调用LLM API"""

    def __init__(self, api_key, base_url, model_name):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name

    def call_llm(self, prompt):
        """调用LLM API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a psychological counseling expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用错误: {e}")
            return None

    def is_complex(self, context):
        """判断对话是否足够复杂，需要多智能体协作"""
        prompt = get_prompt("behavior_control").format(context=context)
        response = self.call_llm(prompt)
        if response and "YES" in response.upper():
            return True
        return False

    def get_emotion(self, context):
        """提取用户情绪"""
        prompt = get_prompt("get_emotion").format(context=context)
        response = self.call_llm(prompt)
        if response and "Emotion:" in response:
            emotion = response.split("Emotion:")[-1].split("\n")[0].strip()
            return emotion, response
        return "Unknown", response

    def get_cause(self, context, emo_and_reason):
        """提取导致情绪的事件"""
        prompt = get_prompt("get_cause").format(context=context, emo_and_reason=emo_and_reason)
        response = self.call_llm(prompt)
        if response and "Event:" in response:
            cause = response.split("Event:")[-1].split("\n")[0].strip()
            return cause, response
        return "Unknown", response

    def get_intention(self, context, emo_and_reason, cau_and_reason):
        """提取用户意图"""
        prompt = get_prompt("get_intention").format(
            context=context,
            emo_and_reason=emo_and_reason,
            cau_and_reason=cau_and_reason
        )
        response = self.call_llm(prompt)
        if response and "Intention:" in response:
            intention = response.split("Intention:")[-1].split("\n")[0].strip()
            return intention, response
        return "Unknown", response

    def generate_response_zero_shot(self, context):
        """直接生成回复（简单场景）"""
        prompt = get_prompt("zero_shot").format(context=context)
        response = self.call_llm(prompt)
        if response and "Response:" in response:
            return response.split("Response:")[-1].strip()
        return response if response else "抱歉，生成回复时出现错误。"

    def generate_response_with_analysis(self, context):
        """带分析的回复生成（复杂场景）"""
        # 获取情绪
        emotion, emo_and_reason = self.get_emotion(context)
        print(f"    情绪: {emotion}")

        # 获取原因
        cause, cau_and_reason = self.get_cause(context, emo_and_reason)
        print(f"    原因: {cause}")

        # 获取意图
        intention, int_and_reason = self.get_intention(context, emo_and_reason, cau_and_reason)
        print(f"    意图: {intention}")

        # 使用零样本生成回复
        response = self.generate_response_zero_shot(context)
        return response


def load_config(config_file="OAI_CONFIG_LIST"):
    """加载LLM配置"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config[0]


def json2natural(history):
    """将历史对话转换为自然语言格式"""
    natural_language = ""
    for u in history:
        content = u["content"].strip()
        role = "User" if u["role"] == "user" else "Assistant"
        natural_language += f"{role}: {content} "
    return natural_language.strip()


def process_conversation(conversation, agent):
    """处理单个对话，为每个user_input生成回复"""
    history = []  # 对话历史

    for turn in conversation["turns"]:
        user_input = turn["user_input"]

        # 构建当前上下文
        current_context = history + [{"role": "user", "content": user_input}]
        context = json2natural(current_context)

        # 判断是否需要复杂分析
        turn_count = len(history) // 2 + 1

        if turn_count <= 5:
            # 简单场景：直接生成回复
            print(f"    [简单场景] 第{turn_count}轮")
            response = agent.generate_response_zero_shot(context)
        else:
            # 复杂场景：使用分析后生成
            is_complex = agent.is_complex(context)
            if is_complex:
                print(f"    [复杂场景] 第{turn_count}轮 - 需要多智能体分析")
                response = agent.generate_response_with_analysis(context)
            else:
                print(f"    [简单场景] 第{turn_count}轮 - 不需要复杂分析")
                response = agent.generate_response_zero_shot(context)

        turn["model_reply"] = response

        # 将用户输入和生成的回复添加到历史记录
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        print(f"    用户: {user_input[:50]}...")
        print(f"    回复: {response[:50]}...")

    return conversation


def main():
    parser = argparse.ArgumentParser(description="使用MultiAgentESC框架处理评估数据（纯API版本）")
    parser.add_argument("--input_dir", type=str, default=r"e:\GitLoadWareHouse\testmymodel\详细报告",
                        help="输入JSON文件目录")
    parser.add_argument("--output_dir", type=str, default="results/evaluation_api",
                        help="输出结果目录")
    parser.add_argument("--config", type=str, default="OAI_CONFIG_LIST",
                        help="LLM配置文件")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理的对话数量（用于测试）")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载配置
    print("加载LLM配置...")
    llm_config = load_config(args.config)
    print(f"  模型: {llm_config['model']}")
    print(f"  API地址: {llm_config['base_url']}")

    # 初始化agent
    agent = SimpleMultiAgentESC(
        api_key=llm_config['api_key'],
        base_url=llm_config['base_url'],
        model_name=llm_config['model']
    )

    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(args.input_dir, "*.json"))
    print(f"\n找到 {len(json_files)} 个JSON文件")

    # 处理每个文件
    for json_file in tqdm(json_files, desc="处理文件"):
        filename = os.path.basename(json_file)
        output_file = os.path.join(args.output_dir, f"multiagent_{filename}")

        # 跳过已处理的文件
        if os.path.exists(output_file):
            print(f"\n跳过已处理文件: {filename}")
            continue

        print(f"\n{'='*60}")
        print(f"处理文件: {filename}")
        print(f"{'='*60}")

        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 更新元数据
        data["evaluation_metadata"]["evaluation_model"] = llm_config['model']
        data["evaluation_metadata"]["evaluated_model_framework"] = "MultiAgentESC-API"

        # 处理每个对话
        conversations_to_process = data["conversations"][:args.limit] if args.limit else data["conversations"]
        print(f"处理 {len(conversations_to_process)} 个对话\n")

        for i, conv in enumerate(conversations_to_process):
            print(f"  对话 {i+1}/{len(conversations_to_process)} (ID: {conv.get('conversation_id', 'N/A')})")
            data["conversations"][i] = process_conversation(conv, agent)
            print()

        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"已保存到: {output_file}")

    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)


if __name__ == "__main__":
    main()
