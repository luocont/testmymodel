"""
本地强化学习模型咨询师智能体
支持直接加载 Qwen3 等 HuggingFace 格式模型

使用说明：
    from rl_counselor_agent import RLCounselorAgent

    agent = RLCounselorAgent(
        example=example_data,
        model_path="/path/to/your/model",
        system_prompt="你的系统提示词"
    )

    response = agent.generate(history)
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from pathlib import Path


class Agent(ABC):
    """智能体基类（兼容原有框架）"""

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass


class RLCounselorAgent(Agent):
    """
    使用本地强化学习模型的咨询师智能体

    特点：
    1. 直接加载模型权重，无需 API 服务
    2. 预留系统提示词配置接口
    3. 支持 Qwen、Llama 等多种模型格式
    4. 自动处理对话历史格式
    5. 支持显存不足时的自动重试机制
    """

    def __init__(
        self,
        example: Dict,
        model_path: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        torch_dtype: str = "float16"
    ):
        """
        初始化 RL 咨询师智能体

        Args:
            example: 数据样本（包含来访者信息）
            model_path: 本地模型路径（HuggingFace 格式）
            system_prompt: 系统提示词（可选，未提供时使用默认）
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            device_map: 设备映射策略 ("auto", "cpu", 或具体设备ID)
            trust_remote_code: 是否信任远程代码（Qwen 等模型需要）
            torch_dtype: 数据类型 ("float16", "float32", "bfloat16")
        """
        super().__init__()
        self.example = example
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code

        # 设置数据类型
        if torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype == "float32":
            self.torch_dtype = torch.float32
        elif torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float16

        # 系统提示词（如果未提供则使用默认）
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        # 延迟加载（首次调用时才加载模型）
        self.model = None
        self.tokenizer = None

    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        return """你是一位专业的心理咨询师，擅长认知行为疗法(CBT)。

你的任务是：
1. 仔细倾听来访者的陈述，理解他们的情绪和需求
2. 提供共情、专业的回应
3. 运用合适的咨询技术帮助来访者探索问题
4. 每次回应保持在30-80字之间

请直接输出你的回应，不要添加"咨询师："等前缀或格式标记。"""

    def _load_model(self):
        """延迟加载模型和分词器"""
        if self.model is not None:
            return

        print(f"[RLCounselorAgent] 正在加载模型: {self.model_path}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code
        )

        # 设置 pad_token（如果不存在）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self.torch_dtype
        )
        self.model.eval()

        print(f"[RLCounselorAgent] 模型加载完成")
        print(f"[RLCounselorAgent] 设备: {self.model.device}")
        print(f"[RLCounselorAgent] 数据类型: {self.torch_dtype}")

    def _build_conversation(self, history: List[Dict]) -> List[Dict]:
        """
        将历史记录转换为模型对话格式

        Args:
            history: 历史消息列表，格式为 [{"role": "counselor"/"client", "message": "..."}]

        Returns:
            标准格式的消息列表（OpenAI 格式）
        """
        messages = []

        for msg in history:
            role = msg.get("role", "").lower()
            content = msg.get("message", "")

            if role == "counselor":
                messages.append({"role": "assistant", "content": content})
            elif role == "client":
                messages.append({"role": "user", "content": content})

        return messages

    def generate(
        self,
        history: List[Dict],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3
    ) -> str:
        """
        生成咨询师响应

        Args:
            history: 对话历史
            max_new_tokens: 最大生成token数（可选，覆盖初始化时的值）
            temperature: 采样温度（可选，覆盖初始化时的值）
            max_retries: 最大重试次数（用于处理 OOM 等错误）

        Returns:
            咨询师的响应文本
        """
        # 使用实例参数或传入的参数
        max_tokens = max_new_tokens or self.max_new_tokens
        temp = temperature if temperature is not None else self.temperature

        # 延迟加载模型
        if self.model is None:
            self._load_model()

        # 构建对话
        conversation = self._build_conversation(history)
        messages = [
            {"role": "system", "content": self.system_prompt},
            *conversation
        ]

        # 应用聊天模板
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"[警告] 聊天模板应用失败: {e}")
            # 回退到简单格式
            text = f"{self.system_prompt}\n\n"
            for msg in messages[1:]:
                role = "用户" if msg["role"] == "user" else "助手"
                text += f"{role}: {msg['content']}\n"
            text += "助手:"

        # 重试循环
        for attempt in range(max_retries):
            try:
                # 编码输入（与 InternLM2 官方示例对齐）
                inputs = self.tokenizer([text], return_tensors="pt")

                # 将输入移到模型设备（与官方示例对齐）
                for k, v in inputs.items():
                    inputs[k] = v.to(self.model.device)

                # 记录输入长度，用于提取生成的部分
                input_length = inputs['input_ids'].shape[1]

                # 生成参数（与官方示例对齐）
                # 官方: {"max_length": 128, "top_p": 0.8, "temperature": 0.8, "do_sample": True, "repetition_penalty": 1.0}
                gen_kwargs = {
                    "max_new_tokens": max_tokens,
                    "top_p": 0.8,
                    "temperature": temp,
                    "do_sample": temp > 0,
                    "repetition_penalty": 1.0
                }

                # 生成响应
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **gen_kwargs)

                # 解码响应：只取新生成的部分（跳过输入部分）
                response = self.tokenizer.decode(outputs[0][input_length:].tolist(), skip_special_tokens=True)

                # 清理响应
                response = response.strip()

                # 移除可能的前缀
                prefixes_to_remove = [
                    "咨询师：", "咨询师:", "Counselor:", "Assistant:",
                    "助手：", "助手:", "AI:", "ai:"
                ]
                for prefix in prefixes_to_remove:
                    if response.startswith(prefix):
                        response = response[len(prefix):].strip()

                # 如果响应为空或太短，返回默认响应
                if not response or len(response) < 3:
                    return "我理解你的感受。请继续告诉我更多。"

                return response

            except RuntimeError as e:
                error_str = str(e).lower()
                if "out of memory" in error_str or "cuda" in error_str:
                    print(f"[警告] 显存不足，清理缓存后重试 ({attempt + 1}/{max_retries})")
                    torch.cuda.empty_cache()
                    if attempt < max_retries - 1:
                        # 减少生成长度后重试
                        max_tokens = max_tokens // 2
                        continue
                raise
            except Exception as e:
                print(f"[错误] 生成失败: {e}")
                if attempt == max_retries - 1:
                    return "我理解你的感受。请继续告诉我更多。"
                else:
                    print(f"重试中... ({attempt + 1}/{max_retries})")

        return "我理解你的感受。请继续告诉我更多。"


def create_rl_counselor_agent(
    example: Dict,
    model_path: str,
    **kwargs
) -> RLCounselorAgent:
    """
    便捷函数：创建 RL 咨询师智能体

    Args:
        example: 数据样本
        model_path: 模型路径
        **kwargs: 其他参数传递给 RLCounselorAgent

    Returns:
        RLCounselorAgent 实例
    """
    return RLCounselorAgent(
        example=example,
        model_path=model_path,
        **kwargs
    )


# 预设的系统提示词模板
SYSTEM_PROMPTS = {
    "cbt": """你是一位专业的认知行为疗法(CBT)咨询师。

你的核心方法：
1. 识别来访者的负面思维模式
2. 帮助来访者挑战这些不合理的想法
3. 引导来访者发展更平衡的思维方式

回应要求：
- 每次回应30-80字
- 使用温和、探索性的语言
- 避免直接给建议，而是引导思考
- 保持共情和专业边界

直接输出你的回应，不要添加前缀。""",

    "person_centered": """你是一位以人为中心的治疗师。

你的核心理念：
1. 无条件的积极关注
2. 共情理解
3. 真诚一致

回应要求：
- 每次回应30-80字
- 反映来访者的情感和内容
- 避免指导性语言
- 创造安全、接纳的氛围

直接输出你的回应，不要添加前缀。""",

    "brief": """你是一位心理咨询师。

请简洁、共情地回应来访者，每次30-60字。

直接输出你的回应，不要添加前缀。"""
}


def get_preset_prompt(prompt_type: str) -> str:
    """
    获取预设的系统提示词

    Args:
        prompt_type: 提示词类型 ("cbt", "person_centered", "brief")

    Returns:
        系统提示词字符串
    """
    return SYSTEM_PROMPTS.get(prompt_type.lower(), SYSTEM_PROMPTS["brief"])
