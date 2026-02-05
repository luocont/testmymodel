# MultiAgentESC 阿里云 API 配置指南

本指南说明如何配置 MultiAgentESC 使用阿里云百炼 API 与 MagNet ClientAgent 进行对话。

## 快速开始

### 1. 获取阿里云 API Key

1. 访问 [阿里云百炼平台](https://dashscope.aliyun.com/)
2. 注册/登录账号
3. 创建 API Key
4. 保存 API Key（格式类似：`sk-xxxxxxxxxxxxxxxx`）

### 2. 设置环境变量

**Windows (PowerShell)**:
```powershell
$env:DASHSCOPE_API_KEY="your-api-key-here"
```

**Windows (CMD)**:
```cmd
set DASHSCOPE_API_KEY=your-api-key-here
```

**Linux/Mac**:
```bash
export DASHSCOPE_API_KEY=your-api-key-here
```

### 3. 一键运行

```bash
cd MagNet
run_multiagentesc_aliyun.bat
```

## 手动运行

### 基本用法

```bash
cd MagNet/src

# 使用环境变量中的 API Key
python inference-multiagentesc-aliyun.py

# 直接指定 API Key
python inference-multiagentesc-aliyun.py --api_key your-api-key

# 指定模型
python inference-multiagentesc-aliyun.py --model qwen2.5-32b-instruct

# 限制样本数量（测试用）
python inference-multiagentesc-aliyun.py --num_samples 5
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api_key` | 从环境变量读取 | 阿里云 API Key |
| `--model` | qwen2.5-7b-instruct | 模型名称 |
| `-o` | ../output-multiagentesc-aliyun | 输出目录 |
| `-d` | ../dataset/data_cn.json | 数据集文件 |
| `-m_turns` | 20 | 最大对话轮次 |
| `--num_samples` | 全部 | 处理样本数量 |
| `-num_pr` | None | 并行进程数 |

## 支持的模型

阿里云百炼支持多种模型，常用的包括：

| 模型名称 | 说明 | 推荐 |
|---------|------|------|
| qwen2.5-7b-instruct | Qwen2.5 7B | 速度快，成本低 ✅ |
| qwen2.5-14b-instruct | Qwen2.5 14B | 平衡性能和成本 |
| qwen2.5-32b-instruct | Qwen2.5 32B | 更好的效果 |
| qwen2.5-72b-instruct | Qwen2.5 72B | 最佳效果 |
| qwen-max | 通义千问最强 | 最贵但最好 |

## 工作流程

```
1. MultiAgentESC (咨询师，使用阿里云 API)
   ↓
2. 分析情感、原因、意图
   ↓
3. 选择咨询策略
   ↓
4. 生成响应
   ↓
5. MagNet ClientAgent (客户，也使用阿里云 API)
   ↓
6. 生成客户响应
   ↓
7. 重复直到达到最大轮次或客户结束
```

## 输出格式

生成的 `session_N.json` 文件：

```json
{
    "example": { ... },
    "cbt_technique": "MultiAgentESC-Aliyun (Strategy-based)",
    "cbt_plan": "MultiAgentESC using Aliyun qwen2.5-7b-instruct with dynamic strategy selection.",
    "cost": 0,
    "history": [
        {"role": "counselor", "message": "..."},
        {"role": "client", "message": "..."}
    ]
}
```

## 运行测评

对话生成后，使用 MagNet 的测评脚本：

```bash
cd ../evaluation

# CTRS 评估
python run_ctrs.py -i ../output-multiagentesc-aliyun -o ../output-ctrs-multiagentesc-aliyun

# PANAS 评估
python PANAS/run_panas_before.py -i ../output-multiagentesc-aliyun -o ../output-panas-before-multiagentesc-aliyun
python PANAS/run_panas_after.py -i ../output-multiagentesc-aliyun -o ../output-panas-after-multiagentesc-aliyun

# WAI 评估
python WAI/run_wai.py -i ../output-multiagentesc-aliyun -o ../output-wai-multiagentesc-aliyun

# Diversity 评估
python Diversity/run_diversity.py -i ../output-multiagentesc-aliyun -o ../output-diversity-multiagentesc-aliyun
```

## 与原始 MultiAgentESC 的区别

| 特性 | 原始 MultiAgentESC | 阿里云版本 |
|------|-------------------|-----------|
| LLM 后端 | AutoGen + Ollama | 阿里云 API |
| 配置文件 | OAI_CONFIG_LIST | 环境变量或参数 |
| 依赖 | pyautogen | 仅 openai |
| 部署 | 需要 Ollama | 无需本地部署 |

## 常见问题

### Q1: API Key 无效？

**A**: 检查以下几点：
- API Key 是否正确复制
- 是否设置了正确的环境变量
- API Key 是否有足够权限

```bash
# 测试 API Key
echo %DASHSCOPE_API_KEY%
```

### Q2: 模型调用失败？

**A**: 可能的原因：
- 模型名称拼写错误
- API 额度不足
- 网络连接问题

```bash
# 使用其他模型测试
python inference-multiagentesc-aliyun.py --model qwen2.5-32b-instruct
```

### Q3: 速度太慢？

**A**: 优化建议：
- 使用更小的模型：`qwen2.5-7b-instruct`
- 减少对话轮次：`-m_turns 10`
- 减少样本数量：`--num_samples 5`
- 启用多进程：`-num_pr 4`

### Q4: 如何查看详细日志？

**A**: 脚本会输出实时进度：
```
[MultiAgentESC-Aliyun] 生成第 1/10 个样本
[MultiAgentESC-Aliyun] 完成 1/10
```

错误信息会保存在 `error_multiagentesc_aliyun_*.txt` 文件中。

### Q5: 成本估算？

**A**: 以 qwen2.5-7b-instruct 为例：
- 输入: ¥0.0005/千tokens
- 输出: ¥0.002/千tokens
- 每个对话约 2000-5000 tokens
- 100 个对话约 ¥2-5

更详细的定价请查看 [阿里云百炼定价](https://dashscope.aliyun.com/price)

## 配置示例

### 测试配置（快速验证）

```bash
# 使用最小配置测试
python inference-multiagentesc-aliyun.py \
    --model qwen2.5-7b-instruct \
    --num_samples 1 \
    -m_turns 5
```

### 生产配置（完整测评）

```bash
# 使用完整配置
python inference-multiagentesc-aliyun.py \
    --model qwen2.5-32b-instruct \
    -m_turns 20 \
    -num_pr 4
```

### 高性价比配置

```bash
# 平衡速度和效果
python inference-multiagentesc-aliyun.py \
    --model qwen2.5-14b-instruct \
    -m_turns 15 \
    -num_pr 2
```

## 环境要求

```bash
# 必需依赖
pip install openai sentence-transformers numpy

# MagNet 依赖
pip install langchain tqdm

# 可选依赖
pip install torch  # 用于 sentence-transformers 加速
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `inference-multiagentesc-aliyun.py` | 阿里云版本的主脚本 |
| `run_multiagentesc_aliyun.bat` | 一键运行脚本 |
| `ALIYUN_SETUP.md` | 本配置指南 |

## 技术细节

### AliyunLLMWrapper 类

将阿里云 API 包装成统一的接口：

```python
wrapper = AliyunLLMWrapper(
    api_key="your-key",
    model="qwen2.5-7b-instruct"
)
response = wrapper.generate("你的提示")
```

### MultiAgentESC 策略系统

虽然使用了阿里云 API，但完整保留了 MultiAgentESC 的策略选择逻辑：

1. **复杂度判断**: 判断是否需要多智能体
2. **情感分析**: 识别用户情感
3. **原因分析**: 分析事件原因
4. **意图推断**: 理解用户意图
5. **策略选择**: 动态选择策略
6. **响应生成**: 生成最终响应

## 联系支持

如有问题，请检查：
1. API Key 配置
2. 网络连接
3. 阿里云控制台的 API 调用日志
4. 错误日志文件
