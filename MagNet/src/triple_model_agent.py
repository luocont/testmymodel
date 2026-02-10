"""
三联动模型 Agent

整合情感分类模型、顾问模型和主模型的联动逻辑
用于生成心理咨询对话

基于 Gradio 应用的三模型联动逻辑重构
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForCausalLM
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path


# ================================
# Part 1: 情感分类模型
# ================================

class SentimentClassifier(nn.Module):
    """基于BERT的情感分类模型"""

    def __init__(self, bert: BertModel, n_classes=3):
        super().__init__()
        self.bert = bert
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.act = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.drop(out.pooler_output)
        x = self.act(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)


class NeutralSentimentStub:
    """降级占位：情感模型加载失败时使用，永远返回中性"""
    def eval(self): return self
    def to(self, *_args, **_kwargs): return self


# ================================
# Part 2: 大语言模型交互器
# ================================

class LLMInteractor:
    """大语言模型加载器和交互器"""

    def __init__(self, model_path: str, torch_dtype: str = "auto"):
        """
        初始化大语言模型

        Args:
            model_path: 模型路径
            torch_dtype: 数据类型 ("auto", "float16", "bfloat16", "float32")
        """
        print(f"正在从 '{model_path}' 加载大语言模型...")

        try:
            # 获取 GPU 总显存，设置 max_memory 防止 CPU 卸载
            import torch
            max_memory = None
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                max_memory = {
                    0: int(gpu_memory * 0.95),  # 使用 95% 的显存
                    "cpu": 0  # 不使用 CPU 内存
                }

            # 使用 "auto" dtype 以适配不同设备/权重
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                max_memory=max_memory,
                low_cpu_mem_usage=True
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            # 设置 pad_token
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            # 获取设备
            if hasattr(self.model, 'device'):
                self.device = self.model.device
            else:
                self.device = next(self.model.parameters()).device

            # 验证设备是 CUDA
            if self.device.type != "cuda":
                print(f"[警告] 模型不在 CUDA 设备上，当前设备: {self.device}")
                # 如果不在 GPU 上，尝试移到 GPU
                try:
                    if torch.cuda.is_available():
                        self.model = self.model.to("cuda")
                        self.device = torch.device("cuda")
                        print(f"[已修复] 已将模型移至 CUDA")
                except Exception as e:
                    print(f"[错误] 移动模型到 CUDA 失败: {e}")

            print(f"✅ 大语言模型 '{model_path}' 加载成功！设备: {self.device}")
        except Exception as e:
            print(f"加载大语言模型 '{model_path}' 时出错: {e}")
            raise


# ================================
# Part 3: 思考过程提取
# ================================

def extract_thinking_process(raw_text: str) -> str:
    """从顾问模型输出中提取思考过程"""
    # 匹配完整的 <think>...</thinking> 标签
    match_full = re.search(r"<think>(.*?)</thinking>", raw_text, re.DOTALL)
    if match_full:
        return match_full.group(1).strip()

    # 匹配只有 </thinking> 的情况
    match_close_only = re.search(r"(.*?)</thinking>", raw_text, re.DOTALL)
    if match_close_only:
        return match_close_only.group(1).strip()

    # 匹配只有 <think> 的情况
    match_open_only = re.search(r"<think>(.*)", raw_text, re.DOTALL)
    if match_open_only:
        return match_open_only.group(1).strip()

    print("警告: 在顾问模型的输出中未找到任何 <think> 或 </thinking> 标签。")
    return ""


# ================================
# Part 4: 三联动模型 Agent
# ================================

class TripleModelAgent:
    """
    三联动模型 Agent

    整合情感分类、顾问分析和主模型回复的联动逻辑
    """

    def __init__(
        self,
        example: dict,
        sentiment_model_path: str,
        primary_model_path: str,
        consultant_model_path: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.5,
        bert_model_name: str = 'bert-base-chinese',
        max_sentiment_len: int = 128,
        n_sentiment_classes: int = 2
    ):
        """
        初始化三联动模型 Agent

        Args:
            example: 数据样本
            sentiment_model_path: 情感分类模型路径 (.bin文件)
            primary_model_path: 主模型路径
            consultant_model_path: 顾问模型路径
            system_prompt: 主模型系统提示词
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            bert_model_name: BERT模型名称
            max_sentiment_len: 情感分析最大长度
            n_sentiment_classes: 情感分类数量 (2: 消极/积极, 3: 消极/中性/积极)
        """
        self.example = example
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_sentiment_len = max_sentiment_len
        self.n_sentiment_classes = n_sentiment_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载情感分类模型
        print(f"正在加载情感分类模型 ({n_sentiment_classes}类)...")
        try:
            bert, tokenizer = self._load_bert_and_tokenizer(bert_model_name)
            self.sentiment_model = SentimentClassifier(bert=bert, n_classes=n_sentiment_classes)

            ckpt = torch.load(sentiment_model_path, map_location='cpu')
            state = ckpt.get('model_state', ckpt)
            self.sentiment_model.load_state_dict(state, strict=False)

            self.sentiment_model.to(self.device).eval()
            self.sentiment_tokenizer = tokenizer
            print(f"✅ 情感分类模型加载成功！")
        except Exception as e:
            print(f"[警告] 情感分类模型加载失败，将使用中性占位模型继续运行。原因：{e}")
            self.sentiment_model = NeutralSentimentStub()
            self.sentiment_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # 加载主模型
        self.primary_interactor = LLMInteractor(
            model_path=primary_model_path,
            torch_dtype="auto"
        )

        # 加载顾问模型
        self.consultant_interactor = LLMInteractor(
            model_path=consultant_model_path,
            torch_dtype="auto"
        )

    def _load_bert_and_tokenizer(self, model_id: str) -> Tuple[BertModel, BertTokenizer]:
        """稳健地加载 BERT 与 Tokenizer"""
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            tok = BertTokenizer.from_pretrained(model_id)

        bert = BertModel.from_pretrained(model_id)
        return bert, tok

    def _predict_sentiment(self, text: str) -> Tuple[str, float]:
        """
        预测文本情感

        Returns:
            (label, confidence): 情感标签和置信度
            label: '消极', '中性' (仅3类时), '积极'
        """
        if isinstance(self.sentiment_model, NeutralSentimentStub):
            return '中性', 0.50

        try:
            encoding = self.sentiment_tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_sentiment_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            with torch.no_grad():
                logits = self.sentiment_model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)

            # 根据类别数选择标签映射
            if self.n_sentiment_classes == 2:
                label_map = ['消极', '积极']
            else:
                label_map = ['消极', '中性', '积极']

            return label_map[pred.item()], float(conf.item())
        except Exception as e:
            print(f"[警告] 情感预测失败，返回中性。原因：{e}")
            return '中性', 0.50

    def _call_consultant(self, history: List[Dict]) -> str:
        """
        调用顾问模型生成思考过程

        Args:
            history: 对话历史 (OpenAI格式)

        Returns:
            顾问模型的思考过程
        """
        # 构建对话历史文本
        history_text = self.consultant_interactor.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=False
        )

        prompt = f"""##角色
你是一名经验丰富的心理咨询师,擅长运用理情行为疗法等专业技术为来访者提供支持与引导。

##注意事项
-你需解答用户的心理问题并说明思考过程,且确保回复遵循特定原则与格式
当用户征求你的意见时不要在提问,直接根据咨询者的心理问问题给出相关建议。
-你要给出下面的思考过程并且思考聚焦于以下类型:
1.模式识别:需连接当前行为,来访者的具体情绪以及情绪解释与长期关系模式
2.溯源需求:需追溯家庭背景、成长经历、重大事件
3.认知重构:需解构非理性信念
4.危机预警:需触发风险评估与转介流程

##技能
###技能1:开展咨询会话
-温和沟通以营造安全环境,用贴近生活的语言共情来访者的困扰
-按需提问获取信息,及时反馈,并适度普及心理知识。

###技能2:运用核心咨询技能
状况理解:通过积极倾听与适时提问,了解来访者的具体困扰,探索情绪及背后的想法与信念,关注问题背景及相关人际关系。
-认知梳理:帮助来访者觉察并识别可能存在的不合理信念,循序渐进引导理性思考,探讨想法与情绪间的关联。
-方案建议:在充分理解情况后,协助来访者建立理性思维方式,提供系统性的解决方案

###技能3:提出建议与处理特殊情况
-提供可操作的建议,包括实施步骤、应对困难的方法及效果评估方式。
-结合实际情况提建议,通过积极鼓励增强来访者的执行信心。
若遇自伤、严重病症等情况,立即建议寻求线下专业医疗帮助力。

##约束条件
-聊天过程中无需解释自身的咨询技巧与行为,以保持对话自然流畅。对话中保持适度的专业界限,在提供温暖支持的同时,明确咨询的局限性。

---
【完整对话历史】
{history_text}
<|im_start|>assistant
<think>
"""

        # 关键：将输入移到模型设备
        model_inputs = self.consultant_interactor.tokenizer([prompt], return_tensors="pt")
        model_inputs = {k: v.to(self.consultant_interactor.model.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            outputs = self.consultant_interactor.model.generate(
                **model_inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.1,
                top_p=0.9
            )

        response = self.consultant_interactor.tokenizer.decode(
            outputs[0][model_inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return extract_thinking_process(response)

    def _call_primary(self, messages: List[Dict]) -> str:
        """
        调用主模型生成回复

        Args:
            messages: 消息列表 (OpenAI格式)

        Returns:
            主模型的回复
        """
        prompt = self.primary_interactor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 关键：将输入移到模型设备
        model_inputs = self.primary_interactor.tokenizer([prompt], return_tensors="pt")
        model_inputs = {k: v.to(self.primary_interactor.model.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            outputs = self.primary_interactor.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9
            )

        response = self.primary_interactor.tokenizer.decode(
            outputs[0][model_inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response

    def generate(self, history: List[Dict]) -> str:
        """
        生成咨询师回复（三联动逻辑）

        Args:
            history: 对话历史，格式为 [{"role": "counselor"|"client", "message": "..."}]

        Returns:
            咨询师的回复
        """
        if not history:
            # 初始化对话，使用示例中的开场白
            return self.example.get('AI_counselor', {}).get('CBT', {}).get('init_history_counselor', "你好，我是你的心理咨询师，很高兴能和你交流。")

        # 获取最后一条客户端消息
        last_client_message = None
        for msg in reversed(history):
            if msg['role'] == 'client':
                last_client_message = msg['message']
                break

        if not last_client_message:
            return "请告诉我你最近的情况，我会尽力帮助你。"

        # 步骤1: 情感分析
        sentiment_label, sentiment_conf = self._predict_sentiment(last_client_message)
        print(f"    [情感分析]: {sentiment_label} (置信度: {sentiment_conf:.2%})")

        consultant_thinking = ""
        assistant_full_response = ""

        # 步骤2: 如果是消极情绪，调用顾问模型
        if sentiment_label == '消极':
            print(f"    [决策]: 检测到消极情绪，启动顾问模型深度分析...")

            # 转换历史格式
            converted_history = []
            for msg in history:
                role_map = {"counselor": "assistant", "client": "user"}
                converted_history.append({
                    "role": role_map.get(msg['role'], msg['role']),
                    "content": msg['message']
                })

            try:
                consultant_thinking = self._call_consultant(converted_history)
                if consultant_thinking:
                    print(f"    [顾问思路]: {consultant_thinking[:100]}...")
                else:
                    print(f"    [降级处理]: 顾问模型未提供有效思路")
            except Exception as e:
                print(f"    [错误]: 调用顾问模型异常: {e}")
        else:
            print(f"    [决策]: 情绪为{sentiment_label}，主模型将直接回复")

        # 步骤3: 构建主模型提示词
        if consultant_thinking:
            final_user_prompt = (
                f'这是来访者的问题:\n"{last_client_message}"\n\n'
                f'这是心理顾问模型的分析和思考过程，'
                f'请你参考这些思路每次只问一个问题，当遇到极端危机处理时，应立即让来访者转接专业心理咨询。'
                f'当了解了来访者的具体背景和问题后应当及时给出建议就不要再问问题了，'
                f'不需要参考时间而是更多的关注问题的根源，然后直接以友善、专业，富有同情心，'
                f'富含共情的口吻询问或者回答来访者：\n\n'
                f'--- 顾问思路 ---\n{consultant_thinking}\n--- 思路结束 ---'
            )
        else:
            final_user_prompt = last_client_message

        # 构建主模型消息列表
        primary_messages = []
        if self.system_prompt:
            primary_messages.append({"role": "system", "content": self.system_prompt})

        # 添加历史对话（去除 <think> 标签）
        for msg in history[:-1]:  # 排除最后一条（当前要回复的）
            role_map = {"counselor": "assistant", "client": "user"}

            # 清理消息内容：去除 <think> 标签
            message_content = msg['message']
            if '<think>' in message_content:
                # 提取主模型的实际回复（去除思考过程）
                parts = message_content.split("</thinking>")
                if len(parts) > 1:
                    message_content = parts[-1].strip()

            primary_messages.append({
                "role": role_map.get(msg['role'], msg['role']),
                "content": message_content
            })

        # 添加当前用户消息
        primary_messages.append({"role": "user", "content": final_user_prompt})

        try:
            response = self._call_primary(primary_messages)

            # 如果有顾问思路，组合到响应中
            if consultant_thinking:
                assistant_full_response = f"<think>\n{consultant_thinking}\n</thinking>\n\n{response}"
            else:
                assistant_full_response = response

            return assistant_full_response

        except Exception as e:
            print(f"    [错误]: 调用主模型异常: {e}")
            return f"抱歉，生成回复时出现错误: {e}"


# ================================
# 辅助函数
# ================================

def get_triple_model_preset_prompt(prompt_type: str = "cbt") -> str:
    """获取预设系统提示词（与 Gradio 应用保持一致）"""
    prompts = {
        "cbt": """你是一位精通理情行为疗法（Rational Emotive Behavior Therapy，简称REBT）的心理咨询师，能够合理地采用理情行为疗法给来访者提供专业地指导和支持，缓解来访者的负面情绪和行为反应，帮助他们实现个人成长和心理健康。理情行为治疗主要包括以下几个阶段，下面是对话阶段列表，并简要描述了各个阶段的重点。

（1）**检查非理性信念和自我挫败式思维**：理情行为疗法把认知干预视为治疗的"生命"，因此，几乎从治疗一开始，在问题探索阶段，咨询师就以积极的、说服教导式的态度帮助来访者探查隐藏在情绪困扰后面的原因，包括来访者理解事件的思维逻辑，产生情绪的前因后果，借此来明确问题的所在。咨询师坚定地激励来访者去反省自己在遭遇刺激事件后，在感到焦虑、抑郁或愤怒前对自己"说"了些什么。

（2）**与非理性信念辩论**：咨询师运用多种技术（主要是认知技术）帮助来访者向非理性信念和思维质疑发难，证明它们的不现实、不合理之处，认识它们的危害进而产生放弃这些不合理信念的愿望和行为。

（3）**得出合理信念，学会理性思维**：在识别并驳倒非理性信念的基础上，咨询师进一步诱导、帮助来访者找出对于刺激情境和事件的适宜的、理性的反应，找出理性的信念和实事求是的、指向问题解决的思维陈述，以此来替代非理性信念和自我挫bail式思维。为了巩固理性信念，咨询师要向来访者反复教导，证明为什么理性信念是合情合理的，它与非理性信念有什么不同，为什么非理性信念导致情绪失调，而理性信念导致较积极、健康的结果。

（4）**迁移应用治疗收获**：积极鼓励来访者把在治疗中所学到的客观现实的态度，科学合理的思维方式内化成个人的生活态度，并在以后的生活中坚持不懈地按理情行为疗法的教导来解决新的问题。

你需要一步一步来，你一次最多问一个问题。需要富有同情心的回复用户的问题，并且当交流一段过程了解用户的具体情况后应该不要再问问题而是及时给出建议。""",

        "person_centered": """你是一位以人为中心疗法的心理咨询师。你相信每个人都有自我实现的倾向和内在的潜能。你的任务是创造一个温暖、接纳、真诚的治疗氛围，通过无条件的积极关注、共情和一致性来帮助来访者探索自我、接纳自我并实现个人成长。

在与来访者交流时，你应该：
- 表现出真诚和一致性
- 给予无条件的积极关注
- 提供准确的共情理解
- 相信来访者的自我治愈能力
- 不做评判和指导
- 跟随来访者的节奏

请一次只问一个问题，以开放式的探索为主。""",

        "brief": """你是一位短程焦点解决疗法的心理咨询师。你相信来访者是解决自己问题的专家，你的职责是帮助来访者发现和利用自己的资源与优势。

在与来访者交流时，你应该：
- 关注解决方案而非问题
- 探寻例外经验（问题不发生的时候）
- 帮助来访者设定具体、可行的目标
- 使用评量问句、奇迹问句等技术
- 肯定来访者的进步和资源

请保持简洁高效，一次只关注一个核心问题。"""
    }

    return prompts.get(prompt_type, prompts["cbt"])
