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

【重要指示】
1. 你只能输出咨询师（你）的单次回应
2. 禁止重复或转述对话历史
3. 禁止输出来访者的话
4. 禁止添加任何角色标签（如"咨询师："、"来访者："等）
5. 禁止输出"再见"、"期待下次见面"等结束语，除非对话真正结束

你的回应风格：
- 共情、专业、简洁
- 每次回应30-80字
- 直接输出回应内容，不要有任何前缀或格式

现在请直接输出你的咨询师回应："""

    def _load_model(self):
        """延迟加载模型和分词器"""
        if self.model is not None:
            return

        print(f"[RLCounselorAgent] 正在加载模型: {self.model_path}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            use_fast=False  # 使用慢速分词器，更稳定
        )

        # 设置 pad_token（如果不存在）
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # 如果连 eos_token 都没有，添加一个特殊的 pad token
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        print(f"[RLCounselorAgent] 分词器加载完成, vocab_size={len(self.tokenizer)}")

        # 加载模型
        # 如果指定了 device_map="auto"，则让 transformers 自动分配
        # 使用 max_memory 防止模型部分卸载到 CPU
        if self.device_map == "auto":
            import torch
            # 获取 GPU 总显存，留出 2GB 作为缓冲
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                # 设置 max_memory，确保不卸载到 CPU（CPU 设 0MB）
                max_memory = {
                    0: int(gpu_memory * 0.95),  # 使用 95% 的显存
                    "cpu": 0  # 不使用 CPU 内存
                }
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=self.torch_dtype,
                    max_memory=max_memory,
                    low_cpu_mem_usage=True
                )
            else:
                # 如果没有 CUDA，回退到正常加载
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True
                )
        else:
            # 先加载到 CPU，然后移到 GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=self.torch_dtype,
                device_map={"": "cuda"}  # 强制所有层都在 GPU 上
            )

        self.model.eval()

        print(f"[RLCounselorAgent] 模型加载完成")
        # 确保模型在正确的设备上
        if hasattr(self.model, 'device'):
            self.device = self.model.device
        else:
            # 对于多设备分布的情况，获取第一个参数的设备
            self.device = next(self.model.parameters()).device

        # 验证设备是 CUDA
        if self.device.type != "cuda":
            print(f"[RLCounselorAgent] 警告: 模型不在 CUDA 设备上，当前设备: {self.device}")
            # 如果不在 GPU 上，尝试移到 GPU
            try:
                import torch
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                    self.device = torch.device("cuda")
                    print(f"[RLCounselorAgent] 已将模型移至 CUDA")
            except Exception as e:
                print(f"[RLCounselorAgent] 移动模型到 CUDA 失败: {e}")

        print(f"[RLCounselorAgent] 设备: {self.device}")
        print(f"[RLCounselorAgent] 数据类型: {self.torch_dtype}")

    def _build_conversation(self, history: List[Dict]) -> tuple:
        """
        将历史记录转换为 InternLM2 模型的 chat 格式

        Args:
            history: 历史消息列表，格式为 [{"role": "counselor"/"client", "message": "..."}]

        Returns:
            (query, history) 元组，适用于 model.chat() 方法
        """
        # InternLM2 的 chat 方法期望的 history 格式
        # history 是一个列表，每个元素是 (query, response) 元组
        internlm_history = []

        # 如果历史为空，添加系统提示词作为第一条
        if not history:
            return "", []

        # 将系统提示词注入到第一条对话中
        # InternLM2 不支持单独的系统提示词参数，所以我们将其添加到第一条查询前
        first_query_prefix = f"[系统指示]\n{self.system_prompt}\n\n[来访者说]"

        # 历史记录应该是成对的（来访者-咨询师）
        # 跳过最后一条（因为那是当前来访者的话，需要作为 query）
        for i in range(0, len(history) - 1, 2):
            if i + 1 < len(history):
                client_msg = history[i].get("message", "")
                counselor_msg = history[i + 1].get("message", "")
                if client_msg and counselor_msg:
                    # 如果是第一条，添加系统提示词
                    if i == 0:
                        client_msg = f"{first_query_prefix}\n{client_msg}"
                    internlm_history.append((client_msg, counselor_msg))

        # 获取最后一条来访者消息作为 query
        query = ""
        if history:
            last_msg = history[-1]
            if last_msg.get("role") == "client":
                query = last_msg.get("message", "")

        # 如果这是第一条消息，添加系统提示词
        if not internlm_history and query:
            query = f"{first_query_prefix}\n{query}"

        return query, internlm_history

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

        # 构建对话（InternLM2 格式）
        query, internlm_history = self._build_conversation(history)

        # 调试：打印查询和历史（首次生成时）
        if not hasattr(self, '_debug_printed'):
            print(f"\n[调试] InternLM2 输入格式:")
            print(f"  Query: {query[:100]}...")
            print(f"  History length: {len(internlm_history)}")
            if internlm_history:
                print(f"  Last history item: {str(internlm_history[-1][:100])}...")
            self._debug_printed = True

        # 重试循环
        for attempt in range(max_retries):
            try:
                # 使用 InternLM2 的 chat 方法
                response, _ = self.model.chat(
                    self.tokenizer,
                    query,
                    history=internlm_history,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    top_p=0.8,
                    do_sample=temp > 0
                )

                # 清理响应
                response = response.strip()

                # 移除可能的前缀
                prefixes_to_remove = [
                    "咨询师：", "咨询师:", "Counselor:", "Assistant:",
                    "助手：", "助手:", "AI:", "ai:",
                    "心理咨询师：", "心理咨询师:",
                    "\n咨询师：", "\n咨询师:", "\n助手：", "\n助手:",
                    "来访者：", "来访者:", "Client:"
                ]
                for prefix in prefixes_to_remove:
                    if response.startswith(prefix):
                        response = response[len(prefix):].strip()

                # 检测并移除重复的对话内容（模型有时会重复历史对话）
                # 如果响应中包含多轮对话标记，只取最后一行咨询师回复
                if response.count("咨询师") > 1 or response.count("来访者") > 1:
                    # 尝试提取最后一行咨询师回复
                    lines = response.split('\n')
                    filtered_lines = []
                    for line in lines:
                        line = line.strip()
                        # 跳过来访者的话和重复的咨询师标签
                        if not line or line.startswith("来访者") or line.startswith("Client"):
                            continue
                        if line.startswith("咨询师") or line.startswith("助手"):
                            continue
                        filtered_lines.append(line)
                    if filtered_lines:
                        response = filtered_lines[-1]  # 取最后一行

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
            except (TypeError, AttributeError, ValueError) as e:
                # 捕获特定类型的错误并打印详细信息
                import traceback
                print(f"[错误] 生成失败 ({type(e).__name__}): {e}")
                print(f"[详细错误]\n{traceback.format_exc()}")
                if attempt == max_retries - 1:
                    return "我理解你的感受。请继续告诉我更多。"
                else:
                    print(f"重试中... ({attempt + 1}/{max_retries})")
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
