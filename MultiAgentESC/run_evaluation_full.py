"""
MultiAgentESC框架评估脚本（完整版 - 纯API）
实现原始框架的所有核心逻辑，使用requests直接调用API，避免openai库依赖问题
"""

import json
import os
import glob
import argparse
from tqdm import tqdm
import requests
from collections import Counter


# ==================== 策略定义 ====================
STRATEGY_DEFINITIONS = {
    "Question": "Asking for information related to the problem to help the user articulate the issues that they face. Open-ended questions are best, and closed questions can be used to get specific information.",
    "Restatement or Paraphrasing": "A simple, more concise rephrasing of the user's statements that could help them see their situation more clearly.",
    "Reflection of feelings": "Articulate and describe the user's feelings.",
    "Self-disclosure": "Divulge similar experiences that you have had or emotions that you share with the user to express your empathy.",
    "Affirmation and Reassurance": "Affirm the user's strengths, motivation, and capabilities and provide reassurance and encouragement.",
    "Providing Suggestions": "Provide suggestions about how to change, but be careful to not overstep and tell them what to do.",
    "Information": "Provide useful information to the user, for example with data, facts, opinions, resources, or by answering questions.",
    "Others": "Exchange pleasantries and use other support strategies that do not fall into the above categories."
}

STRATEGY_LIST = list(STRATEGY_DEFINITIONS.keys())


# ==================== Prompt定义 ====================
def get_prompt(prompt_name):
    """加载框架中的原始prompts（中文版）"""
    prompts = {
        "behavior_control": '''### 指令
你是一位心理咨询专家。你将获得一段助手和用户之间的不完整对话。
请分析这段对话是否反映了用户当前的情绪状态、用户寻求情感支持的原因以及用户计划如何应对该事件。
如果以上三点都有体现，请回复"是"，否则回复"否"。

### 对话语境
{context}

你的回答必须包含两个部分：
1. "是"或"否"
2. 如果是"是"，简要说明对话如何体现这些要素；如果是"否"，说明缺少哪些要素。

你的回答必须遵循以下格式：
1. [是或否]
2. [说明]
''',

        "zero_shot": '''### 指令
你是一位心理咨询专家。你将获得一段助手和用户之间的对话语境。你的任务是扮演助手的角色，根据给定的对话语境生成回复。

### 对话语境
{context}

你的回答必须少于30个词，并遵循以下格式：
回复：[回复内容]
''',

        "get_emotion": '''### 指令
你是一位心理咨询专家。你将获得一段助手和用户之间的对话语境。请推断用户最后话语中表达的情绪状态。

### 对话语境
{context}

你的回答必须包含以下要素：
情绪：用户在最后话语中表达的情绪。
推理：你回答的理由。

你的回答必须遵循以下格式：
情绪：[情绪]
推理：[推理]
''',

        "get_cause": '''### 指令
你是一位心理咨询专家。你将获得一段助手和用户之间的对话语境。另一位智能体分析了对话，并推断出用户在最后话语中表达的情绪状态。

### 对话语境
{context}

### 情绪状态
{emo_and_reason}

请根据对话语境推断导致用户情绪状态的具体事件。你的回答必须包含以下要素：
事件：导致用户情绪状态的具体事件。
推理：你回答的理由。

你的回答必须遵循以下格式：
事件：[事件]
推理：[推理]
''',

        "get_intention": '''### 指令
你是一位心理咨询专家。你将获得一段助手和用户之间的对话语境。其他智能体已经分析了对话，推断出用户在最后话语中表达的情绪状态以及导致用户情绪状态的具体事件。

### 对话语境
{context}

### 情绪状态
{emo_and_reason}

### 事件
{cau_and_reason}

请根据对话语境合理推断用户的意图，目标是应对导致其情绪状态的事件。你的回答必须包含以下要素：
意图：用户旨在应对导致其情绪状态的事件的意图。
推理：你回答的理由。

你的回答必须遵循以下格式：
意图：[意图]
推理：[推理]
''',

        "select_strategy": '''### 你将获得一段助手和用户之间的对话语境。心理学家已经分析了对话，推断出用户在最后话语中表达的情绪状态、导致用户情绪状态的具体事件以及用户旨在应对导致其情绪状态的事件的意图。

### 对话语境
{context}

### 情绪状态
{emo_and_reason}

### 事件
{cau_and_reason}

### 意图
{int_and_reason}

根据提供的信息和对话语境，请选择最合适的策略来生成回复并解释原因。

### 可用策略
{strategy_list}

你的回答必须包含以下要素：
策略：可用策略中最合适的策略。
推理：你回答的理由。

你的回答必须遵循以下格式：
策略：[策略]
推理：[推理]
''',

        "response_with_strategy": '''你将获得一段助手和用户之间的对话语境。心理学家已经分析了对话，推断出用户在最后话语中表达的情绪状态、导致用户情绪状态的具体事件以及用户旨在应对导致其情绪状态的事件的意图。

### 对话语境
{context}

### 情绪状态
{emo_and_reason}

### 事件
{cau_and_reason}

### 意图
{int_and_reason}

请使用{strategy}策略从助手的角度生成回复。

### 策略说明
{strategy_description}

你的回答必须少于30个词，并遵循以下格式：
回复：[策略] [回复内容]
''',

        "debate": '''### 你将获得一段助手和用户之间的对话语境。心理学家已经分析了对话，推断出用户在最后话语中表达的情绪状态、导致用户情绪状态的具体事件以及用户旨在应对导致其情绪状态的事件的意图。

### 对话语境
{context}

### 情绪状态
{emo_and_reason}

### 事件
{cau_and_reason}

### 意图
{int_and_reason}

根据提供的信息和对话语境，请从以下选项中选择最合适的回复并解释原因。

### 回复选项
{responses}

你的回答必须包含以下要素：
回复：最合适的回复以及该回复中使用的策略。
推理：你回答的理由。

你的回答必须遵循以下格式：
回复：[策略] [回复内容]
推理：[推理]
''',

        "reflect": '''### 你将获得一段助手和用户之间的对话语境。心理学家已经分析了对话，推断出用户在最后话语中表达的情绪状态、导致用户情绪状态的具体事件以及用户旨在应对导致其情绪状态的事件的意图。

### 对话语境
{context}

### 情绪状态
{emo_and_reason}

### 事件
{cau_and_reason}

### 意图
{int_and_reason}

根据提供的信息和对话语境，正在进行小组讨论以确定哪个回复是最合适的。

### 讨论内容
{discussion_content}

你应该仔细分析以上各种不同观点，反思自己的想法，最终得出令人信服的结果。如果你认为其他人的观点更合理，可以改变你的想法。

你的回答必须包含以下要素：
回复：最合适的回复以及该回复中使用的策略。
推理：你回答的理由。

你的回答必须遵循以下格式：
回复：[策略] [回复内容]
推理：[推理]
''',

        "judge": '''你将获得一段助手和用户之间的对话语境。

### 对话语境
{context}

以下是治疗师使用不同策略生成的回复，都以<[策略] 回复内容>的格式呈现。请选择最合适的回复并解释原因。

### 回复选项
{responses}

你的回答必须包含以下要素：
回复：最合适的回复以及该回复中使用的策略。
推理：你回答的理由。

你的回答必须遵循以下格式：
回复：[策略] [回复内容]
推理：[推理]
''',

        "self_reflection": '''你将获得一段助手和用户之间的对话语境。

### 对话语境
{context}

以下是治疗师使用{pred_strategy}策略生成的回复，以<[策略] 回复内容>的格式呈现。请分析该回复是否与正在进行的对话一致，是否符合该策略，以及是否有效帮助缓解用户的情绪压力。

### 回复
[{pred_strategy}] {response}

如果回复符合上述要求，请原样返回；如果不符合，请修改回复并提供更精炼的版本。精炼版本必须少于30个词。

你的回答必须包含以下要素：
回复：原始回复或精炼回复以及该回复中使用的策略。
推理：你回答的理由。

你的回答必须遵循以下格式：
回复：[策略] [原始/精炼回复]
推理：[推理]
'''
    }

    if prompt_name not in prompts:
        raise ValueError(f"Prompt '{prompt_name}' not found.")
    return prompts[prompt_name]


# ==================== MultiAgentESC框架类 ====================
class MultiAgentESC:
    """完整的MultiAgentESC框架实现（纯API版本，使用requests直接调用）"""

    def __init__(self, api_key, base_url, model_name):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

        # 构建完整的API端点
        if "/compatible-mode/v1" in base_url:
            self.api_endpoint = base_url.replace("/compatible-mode/v1", "/compatible-mode/v1/chat/completions")
        elif "/v1" in base_url:
            self.api_endpoint = base_url.rstrip("/") + "/chat/completions"
        else:
            self.api_endpoint = base_url.rstrip("/") + "/v1/chat/completions"

    def call_llm(self, prompt, max_tokens=400):
        """使用requests直接调用LLM API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是一位心理咨询专家。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(self.api_endpoint, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                print(f"API返回格式异常: {result}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"API调用错误: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应内容: {e.response.text}")
            return None
        except Exception as e:
            print(f"未知错误: {e}")
            return None

    # ==================== 第一阶段：对话分析 ====================
    def is_complex(self, context):
        """判断对话是否足够复杂，需要多智能体协作"""
        prompt = get_prompt("behavior_control").format(context=context)
        response = self.call_llm(prompt, max_tokens=200)
        if response and ("是" in response or "YES" in response.upper()):
            return True
        return False

    def get_emotion(self, context):
        """提取用户情绪"""
        prompt = get_prompt("get_emotion").format(context=context)
        response = self.call_llm(prompt)
        if response and ("情绪：" in response or "Emotion:" in response):
            if "情绪：" in response:
                emotion = response.split("情绪：")[-1].split("\n")[0].strip()
            else:
                emotion = response.split("Emotion:")[-1].split("\n")[0].strip()
            return emotion, response
        return "未知", response

    def get_cause(self, context, emo_and_reason):
        """提取导致情绪的事件"""
        prompt = get_prompt("get_cause").format(context=context, emo_and_reason=emo_and_reason)
        response = self.call_llm(prompt)
        if response and ("事件：" in response or "Event:" in response):
            if "事件：" in response:
                cause = response.split("事件：")[-1].split("\n")[0].strip()
            else:
                cause = response.split("Event:")[-1].split("\n")[0].strip()
            return cause, response
        return "未知", response

    def get_intention(self, context, emo_and_reason, cau_and_reason):
        """提取用户意图"""
        prompt = get_prompt("get_intention").format(
            context=context,
            emo_and_reason=emo_and_reason,
            cau_and_reason=cau_and_reason
        )
        response = self.call_llm(prompt)
        if response and ("意图：" in response or "Intention:" in response):
            if "意图：" in response:
                intention = response.split("意图：")[-1].split("\n")[0].strip()
            else:
                intention = response.split("Intention:")[-1].split("\n")[0].strip()
            return intention, response
        return "未知", response

    # ==================== 第二阶段：策略选择 ====================
    def select_strategy(self, context, emo_and_reason, cau_and_reason, int_and_reason):
        """选择最合适的策略"""
        strategy_list_text = "\n".join([f"- {s}: {STRATEGY_DEFINITIONS[s]}" for s in STRATEGY_LIST])

        prompt = get_prompt("select_strategy").format(
            context=context,
            emo_and_reason=emo_and_reason,
            cau_and_reason=cau_and_reason,
            int_and_reason=int_and_reason,
            strategy_list=strategy_list_text
        )

        response = self.call_llm(prompt)
        if response and ("策略：" in response or "Strategy:" in response):
            if "策略：" in response:
                strategy = response.split("策略：")[-1].split("\n")[0].strip()
            else:
                strategy = response.split("Strategy:")[-1].split("\n")[0].strip()
            # 清理策略名称
            for s in STRATEGY_LIST:
                if s.lower() in strategy.lower():
                    return s
            return strategy
        return "Question"  # 默认策略

    # ==================== 第三阶段：响应生成 ====================
    def generate_response_zero_shot(self, context):
        """直接生成回复（简单场景）"""
        prompt = get_prompt("zero_shot").format(context=context)
        response = self.call_llm(prompt, max_tokens=200)
        if response and ("回复：" in response or "Response:" in response):
            if "回复：" in response:
                return response.split("回复：")[-1].strip()
            else:
                return response.split("Response:")[-1].strip()
        return response if response else "抱歉，生成回复时出现错误。"

    def generate_response_with_strategy(self, context, emo_and_reason, cau_and_reason, int_and_reason, strategy):
        """根据策略生成回复"""
        prompt = get_prompt("response_with_strategy").format(
            context=context,
            emo_and_reason=emo_and_reason,
            cau_and_reason=cau_and_reason,
            int_and_reason=int_and_reason,
            strategy=strategy,
            strategy_description=STRATEGY_DEFINITIONS.get(strategy, "")
        )

        response = self.call_llm(prompt, max_tokens=200)
        if response and ("回复：" in response or "Response:" in response):
            if "回复：" in response:
                result = response.split("回复：")[-1].strip()
            else:
                result = response.split("Response:")[-1].strip()
            # 移除策略前缀
            if f"[{strategy}]" in result:
                result = result.split(f"[{strategy}]")[-1].strip()
            return result
        return response if response else "抱歉，生成回复时出现错误。"

    # ==================== 多智能体协作 ====================
    def debate(self, context, emo_and_reason, cau_and_reason, int_and_reason, responses):
        """多智能体辩论"""
        responses_template = "\n\n".join([f"[{i+1}] {r}" for i, r in enumerate(responses)])

        prompt = get_prompt("debate").format(
            context=context,
            emo_and_reason=emo_and_reason,
            cau_and_reason=cau_and_reason,
            int_and_reason=int_and_reason,
            responses=responses_template
        )

        return self.call_llm(prompt)

    def reflect(self, context, emo_and_reason, cau_and_reason, int_and_reason, discussion_content, responses):
        """反思讨论结果"""
        prompt = get_prompt("reflect").format(
            context=context,
            emo_and_reason=emo_and_reason,
            cau_and_reason=cau_and_reason,
            int_and_reason=int_and_reason,
            discussion_content=discussion_content,
            responses="\n\n".join(responses)
        )

        return self.call_llm(prompt)

    def vote(self, reflection_results):
        """投票选择最佳回复"""
        count = {}
        strat2response = {}

        for result in reflection_results:
            try:
                if "推理：" in result or "Reasoning" in result:
                    if "推理：" in result:
                        response_part = result.split("\n推理：")[0]
                    else:
                        response_part = result.split("\nReasoning")[0]
                else:
                    response_part = result

                if "回复：" in response_part:
                    response_part = response_part.split("回复：")[-1]
                elif "Response:" in response_part:
                    response_part = response_part.split("Response:")[-1]

                # 提取策略和响应
                for strategy in STRATEGY_LIST:
                    if strategy.lower() in response_part.lower():
                        # 提取响应内容
                        parts = response_part.split(strategy, 1)
                        if len(parts) > 1:
                            response_text = parts[1].strip()
                            if strategy not in count:
                                count[strategy] = 0
                                strat2response[strategy] = response_text
                            count[strategy] += 1
                            break
            except:
                continue

        if len(count) == 0:
            return ["Question"], ["请告诉我更多关于这个的情况。"]

        max_count = max(count.values())
        max_strat = [key for key, value in count.items() if value == max_count]
        responses = [strat2response.get(s, "") for s in max_strat]

        return max_strat, responses

    def judge(self, context, responses):
        """判断最佳回复"""
        responses_template = "\n\n".join([f"[{i+1}] {r}" for i, r in enumerate(responses)])

        prompt = get_prompt("judge").format(
            context=context,
            responses=responses_template
        )

        response = self.call_llm(prompt)
        if response and ("回复：" in response or "Response:" in response):
            if "回复：" in response:
                result = response.split("回复：")[-1].split("\n")[0].strip()
            else:
                result = response.split("Response:")[-1].split("\n")[0].strip()

            # 提取策略和响应
            for strategy in STRATEGY_LIST:
                if strategy.lower() in result.lower():
                    parts = result.split(strategy, 1)
                    if len(parts) > 1:
                        return strategy, parts[1].strip()

            return "Question", result
        return "Question", response if response else "请告诉我更多关于这个的情况。"

    def self_reflection(self, context, pred_strategy, response):
        """自我反思"""
        prompt = get_prompt("self_reflection").format(
            context=context,
            pred_strategy=pred_strategy,
            response=response
        )

        result = self.call_llm(prompt)
        if result and ("回复：" in result or "Response:" in result):
            if "回复：" in result:
                response_part = result.split("回复：")[-1].split("\n")[0].strip()
            else:
                response_part = result.split("Response:")[-1].split("\n")[0].strip()

            for strategy in STRATEGY_LIST:
                if strategy.lower() in response_part.lower():
                    parts = response_part.split(strategy, 1)
                    if len(parts) > 1:
                        return strategy, parts[1].strip()

            return pred_strategy, response_part
        return pred_strategy, response


# ==================== 辅助函数 ====================
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


def clean_strategy(strategy):
    """清理策略名称"""
    strategy_map = {
        "question": "Question",
        "restatement or paraphrasing": "Restatement or Paraphrasing",
        "restatement": "Restatement or Paraphrasing",
        "paraphrasing": "Restatement or Paraphrasing",
        "reflection of feelings": "Reflection of feelings",
        "reflection": "Reflection of feelings",
        "self-disclosure": "Self-disclosure",
        "self disclosure": "Self-disclosure",
        "affirmation and reassurance": "Affirmation and Reassurance",
        "affirmation": "Affirmation and Reassurance",
        "reassurance": "Affirmation and Reassurance",
        "providing suggestions": "Providing Suggestions",
        "suggestions": "Providing Suggestions",
        "information": "Information",
        "others": "Others"
    }

    cleaned = set()
    for s in strategy:
        s_lower = s.lower().strip()
        if s_lower in strategy_map:
            cleaned.add(strategy_map[s_lower])
        else:
            # 模糊匹配
            for key, value in strategy_map.items():
                if key in s_lower:
                    cleaned.add(value)
                    break
            else:
                cleaned.add(s)

    return list(cleaned)


def process_conversation(conversation, agent):
    """处理单个对话，实现完整的MultiAgentESC流程"""
    history = []  # 对话历史

    for turn in conversation["turns"]:
        user_input = turn["user_input"]

        # 构建当前上下文
        current_context = history + [{"role": "user", "content": user_input}]
        context = json2natural(current_context)

        # 判断轮次
        turn_count = len(history) // 2 + 1

        # ==================== 简单场景（前5轮）====================
        if turn_count <= 5:
            print(f"    [简单场景] 第{turn_count}轮")
            response = agent.generate_response_zero_shot(context)
            turn["model_reply"] = response
            turn["pred_strategy"] = "None"

        # ==================== 复杂场景（5轮后）====================
        else:
            # 判断是否需要复杂分析
            is_complex = agent.is_complex(context)

            if not is_complex:
                print(f"    [简单场景] 第{turn_count}轮 - 不需要复杂分析")
                response = agent.generate_response_zero_shot(context)
                turn["model_reply"] = response
                turn["pred_strategy"] = "None"
            else:
                print(f"    [复杂场景] 第{turn_count}轮 - 完整多智能体协作")

                # ---------- 第一阶段：对话分析 ----------
                print("      → 阶段1: 对话分析")
                emotion, emo_and_reason = agent.get_emotion(context)
                print(f"        情绪: {emotion}")

                cause, cau_and_reason = agent.get_cause(context, emo_and_reason)
                print(f"        原因: {cause}")

                intention, int_and_reason = agent.get_intention(context, emo_and_reason, cau_and_reason)
                print(f"        意图: {intention}")

                # ---------- 第二阶段：策略选择 ----------
                print("      → 阶段2: 策略选择")
                pred_strategy = agent.select_strategy(context, emo_and_reason, cau_and_reason, int_and_reason)
                pred_strategy = clean_strategy([pred_strategy])[0] if clean_strategy([pred_strategy]) else pred_strategy
                print(f"        策略: {pred_strategy}")

                # ---------- 第三阶段：响应生成 ----------
                print("      → 阶段3: 响应生成")

                # 根据策略生成回复
                response = agent.generate_response_with_strategy(
                    context, emo_and_reason, cau_and_reason, int_and_reason, pred_strategy
                )

                # 多智能体协作（生成多个策略的回复进行辩论）
                # 这里简化为只使用一个策略，但进行自我反思
                print("        → 自我反思")
                final_strategy, response = agent.self_reflection(context, pred_strategy, response)

                turn["model_reply"] = response
                turn["pred_strategy"] = final_strategy
                turn["emotion"] = emotion
                turn["cause"] = cause
                turn["intention"] = intention

        # 更新历史记录
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": turn["model_reply"]})

        print(f"    用户: {user_input}")
        print(f"    回复: {turn['model_reply']}")

    return conversation


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="使用MultiAgentESC框架处理评估数据（完整版）")
    parser.add_argument("--input_dir", type=str, default=r"e:\GitLoadWareHouse\testmymodel\详细报告",
                        help="输入JSON文件目录")
    parser.add_argument("--output_dir", type=str, default="results/evaluation_full",
                        help="输出结果目录")
    parser.add_argument("--config", type=str, default="OAI_CONFIG_LIST",
                        help="LLM配置文件")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理的对话数量（用于测试）")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载配置
    print("="*60)
    print("MultiAgentESC 完整版评估脚本")
    print("="*60)
    print("\n加载LLM配置...")
    llm_config = load_config(args.config)
    print(f"  模型: {llm_config['model']}")
    print(f"  API地址: {llm_config['base_url']}")

    # 初始化agent
    agent = MultiAgentESC(
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
        data["evaluation_metadata"]["evaluated_model_framework"] = "MultiAgentESC-Full"

        # 处理每个对话
        conversations_to_process = data["conversations"][:args.limit] if args.limit else data["conversations"]
        print(f"处理 {len(conversations_to_process)} 个对话\n")

        for i, conv in enumerate(conversations_to_process):
            print(f"\n  对话 {i+1}/{len(conversations_to_process)} (ID: {conv.get('conversation_id', 'N/A')})")
            data["conversations"][i] = process_conversation(conv, agent)

        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\n已保存到: {output_file}")

    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)


if __name__ == "__main__":
    main()
