# Azure OpenAI 配置模板
# 复制此文件为 config.py 并填入您的实际凭据

# Azure OpenAI 凭据
AZURE_ENDPOINT = "https://your-resource-name.openai.azure.com/"
AZURE_API_KEY = "your_subscription_key_here"
AZURE_API_VERSION = "2024-02-01"
AZURE_DEPLOYMENT = "gpt-4o"  # 或 gpt-4o-mini

# vLLM 服务配置
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "dummy-key"

# 阿里云 DashScope API 配置
# 阿里云 API Key (可通过环境变量 DASHSCOPE_API_KEY 设置)
DASHSCOPE_API_KEY ="sk-40fb3997d3ed485ba390a9c4ae3bd2d2"  # 或 "sk-xxx"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL = "qwen2.5-7b-instruct"  # 可选: qwen2.5-72b-instruct, qwen-plus 等

# OpenRouter API 配置 (用于评估)
# OpenRouter API Key (可通过环境变量 OPENROUTER_API_KEY 设置)
OPENROUTER_API_KEY = "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" # 或 "sk-or-xxx"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/gpt-4o"  # 可选: anthropic/claude-3.5-sonnet, google/gemini-pro-1.5 等
OPENROUTER_SITE_URL = ""  # Optional. Site URL for rankings on openrouter.ai
OPENROUTER_SITE_NAME = ""  # Optional. Site title for rankings on openrouter.ai

# 客户模拟LLM配置（可选）
# 如果要使用LLM模拟客户回复，设置以下配置
USE_CLIENT_LLM = False
CLIENT_MODEL = "gpt-4o-mini"
CLIENT_API_KEY = None  # 如果使用OpenAI
CLIENT_BASE_URL = None  # 如果使用自定义endpoint

# LLM 提供商选择
# 可选值: "azure", "vllm", "dashscope"
LLM_PROVIDER = "vllm"

# 评估配置
DEFAULT_MAX_ITER = 3  # 每个评估维度的重复次数
DEFAULT_MAX_TURNS = 20  # 生成的对话最大轮次

# 输出路径配置
OUTPUT_DIR = "output"
RESULTS_DIR = "results"
