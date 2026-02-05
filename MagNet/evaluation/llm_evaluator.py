"""
评估系统通用 LLM 客户端模块
支持 OpenAI 兼容 API、阿里云百炼、OpenRouter 等多种 API 提供商
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI, AzureOpenAI


class EvalAPIProvider(Enum):
    """评估 API 提供商枚举"""
    OPENAI = "openai"
    AZURE = "azure"
    ALIYUN = "aliyun"
    OPENROUTER = "openrouter"


@dataclass
class EvalAPIConfig:
    """评估 API 配置数据类"""
    provider: EvalAPIProvider
    api_key: str
    base_url: Optional[str] = None
    model: str = "gpt-4o"
    temperature: float = 0
    max_tokens: int = 4096
    # Azure 特定配置
    azure_endpoint: Optional[str] = None
    api_version: Optional[str] = None
    deployment: Optional[str] = None
    # OpenRouter 特定配置
    extra_headers: Optional[Dict[str, str]] = None
    extra_body: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """配置验证和默认值设置"""
        if self.provider == EvalAPIProvider.ALIYUN:
            if self.base_url is None:
                self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            if self.model == "gpt-4o":
                self.model = "qwen-max"  # 阿里云默认使用 qwen-max
        elif self.provider == EvalAPIProvider.OPENROUTER:
            if self.base_url is None:
                self.base_url = "https://openrouter.ai/api/v1"
            if self.extra_headers is None:
                self.extra_headers = {
                    "HTTP-Referer": "https://github.com/your-repo",
                    "X-Title": "MagNet-Evaluation"
                }


class EvalLLMClient:
    """
    评估系统 LLM 客户端类
    支持多种 API 提供商的统一调用接口
    """

    def __init__(self, config: EvalAPIConfig):
        """
        初始化评估客户端

        Args:
            config: EvalAPIConfig 配置对象
        """
        self.config = config
        self.client = self._create_client()
        self.model = config.deployment if config.provider == EvalAPIProvider.AZURE else config.model

    def _create_client(self):
        """根据配置创建对应的客户端"""
        if self.config.provider == EvalAPIProvider.AZURE:
            return AzureOpenAI(
                api_key=self.config.api_key,
                azure_endpoint=self.config.azure_endpoint,
                api_version=self.config.api_version,
            )
        else:
            # OpenRouter 和其他提供商都使用标准 OpenAI 客户端
            # extra_headers 将在每次请求时传递
            import httpx
            # 创建更长的超时时间（解决 SSL 握手超时问题）
            timeout = httpx.Timeout(
                connect=60.0,  # 连接超时 60 秒
                read=120.0,    # 读取超时 120 秒
                write=60.0,    # 写入超时 60 秒
                pool=60.0      # 连接池超时 60 秒
            )
            return OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=timeout,
            )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        **kwargs
    ):
        """
        发送聊天补全请求

        Args:
            messages: 消息列表
            temperature: 温度参数（可选，默认使用配置值）
            max_tokens: 最大 token 数（可选，默认使用配置值）
            n: 生成数量（用于多次评分取平均）
            **kwargs: 其他参数

        Returns:
            API 响应对象
        """
        # 基础参数
        params = {
            "messages": messages,
            "model": self.model,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "n": n,
        }

        # OpenRouter 额外配置 - extra_headers 和 extra_body 需要单独传递
        if self.config.provider == EvalAPIProvider.OPENROUTER:
            extra_headers = self.config.extra_headers or {}
            extra_body = self.config.extra_body or {}
            return self.client.chat.completions.create(
                **params,
                extra_headers=extra_headers,
                extra_body=extra_body,
                **kwargs
            )
        else:
            return self.client.chat.completions.create(**params, **kwargs)


def create_eval_client_from_env() -> EvalLLMClient:
    """
    从环境变量创建评估客户端

    环境变量:
        EVAL_LLM_PROVIDER: API 提供商 (openai/azure/aliyun/openrouter)
        EVAL_LLM_API_KEY: API 密钥
        EVAL_LLM_BASE_URL: API 基础 URL (可选)
        EVAL_LLM_MODEL: 模型名称 (默认: gpt-4o)
        EVAL_LLM_TEMPERATURE: 温度参数 (默认: 0)
        EVAL_LLM_MAX_TOKENS: 最大 token 数 (默认: 4096)
        EVAL_LLM_AZURE_ENDPOINT: Azure 端点 (Azure 必需)
        EVAL_LLM_API_VERSION: API 版本 (Azure 必需)
        EVAL_LLM_DEPLOYMENT: 部署名称 (Azure 必需)

    Returns:
        EvalLLMClient 实例
    """
    provider_str = os.getenv("EVAL_LLM_PROVIDER", "azure")
    provider = EvalAPIProvider(provider_str.lower())

    config = EvalAPIConfig(
        provider=provider,
        api_key=os.getenv("EVAL_LLM_API_KEY", ""),
        base_url=os.getenv("EVAL_LLM_BASE_URL"),
        model=os.getenv("EVAL_LLM_MODEL", "gpt-4o"),
        temperature=float(os.getenv("EVAL_LLM_TEMPERATURE", "0")),
        max_tokens=int(os.getenv("EVAL_LLM_MAX_TOKENS", "4096")),
        azure_endpoint=os.getenv("EVAL_LLM_AZURE_ENDPOINT"),
        api_version=os.getenv("EVAL_LLM_API_VERSION"),
        deployment=os.getenv("EVAL_LLM_DEPLOYMENT"),
    )

    return EvalLLMClient(config)


def create_eval_aliyun_client(
    api_key: Optional[str] = None,
    model: str = "qwen-max",
    temperature: float = 0,
    max_tokens: int = 4096,
) -> EvalLLMClient:
    """
    创建阿里云百炼评估客户端便捷函数

    Args:
        api_key: API 密钥（默认从环境变量 DASHSCOPE_API_KEY 读取）
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大 token 数

    Returns:
        EvalLLMClient 实例
    """
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key is None:
            raise ValueError("阿里云 API Key 未提供，请设置 DASHSCOPE_API_KEY 环境变量")

    config = EvalAPIConfig(
        provider=EvalAPIProvider.ALIYUN,
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return EvalLLMClient(config)


def create_eval_openrouter_client(
    api_key: Optional[str] = None,
    model: str = "openai/gpt-4o",
    temperature: float = 0,
    max_tokens: int = 4096,
    referer: Optional[str] = None,
    title: Optional[str] = None,
) -> EvalLLMClient:
    """
    创建 OpenRouter 评估客户端便捷函数

    Args:
        api_key: API 密钥（默认从环境变量 OPENROUTER_API_KEY 读取）
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大 token 数
        referer: HTTP Referer 头（用于 OpenRouter 排名）
        title: Site Title 头（用于 OpenRouter 排名）

    Returns:
        EvalLLMClient 实例
    """
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError("OpenRouter API Key 未提供，请设置 OPENROUTER_API_KEY 环境变量")

    extra_headers = {
        "HTTP-Referer": referer or "https://github.com/your-repo",
        "X-Title": title or "MagNet-Evaluation",
    }

    config = EvalAPIConfig(
        provider=EvalAPIProvider.OPENROUTER,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_headers=extra_headers,
    )

    return EvalLLMClient(config)


def create_eval_azure_client(
    api_key: str,
    azure_endpoint: str,
    deployment: str,
    api_version: str = "2024-02-15-preview",
    model: str = "gpt-4o",
    temperature: float = 0,
    max_tokens: int = 4096,
) -> EvalLLMClient:
    """
    创建 Azure OpenAI 评估客户端便捷函数

    Args:
        api_key: API 密钥
        azure_endpoint: Azure 端点
        deployment: 部署名称
        api_version: API 版本
        model: 模型名称（用于 token 计算）
        temperature: 温度参数
        max_tokens: 最大 token 数

    Returns:
        EvalLLMClient 实例
    """
    config = EvalAPIConfig(
        provider=EvalAPIProvider.AZURE,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        deployment=deployment,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return EvalLLMClient(config)


# 评估成本计算配置
EVAL_COST_CONFIG = {
    "gpt-4o": {"input": 5, "output": 20},  # 每 100 万 tokens 价格 (美元)
    "gpt-4o-mini": {"input": 0.66, "output": 2.64},
    "qwen-max": {"input": 0.4, "output": 1.2},  # 阿里云 qwen-max 价格 (参考)
    "openai/gpt-4o": {"input": 5, "output": 20},
}


def get_eval_cost(model: str) -> tuple:
    """
    获取评估模型的成本配置

    Args:
        model: 模型名称

    Returns:
        (input_cost, output_cost) 元组
    """
    # 直接匹配
    if model in EVAL_COST_CONFIG:
        config = EVAL_COST_CONFIG[model]
        return config["input"], config["output"]

    # 模糊匹配
    for key, config in EVAL_COST_CONFIG.items():
        if key in model or model in key:
            return config["input"], config["output"]

    # 默认使用 gpt-4o-mini 的价格
    return EVAL_COST_CONFIG["gpt-4o-mini"]["input"], EVAL_COST_CONFIG["gpt-4o-mini"]["output"]
