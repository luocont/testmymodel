# MultiAgentESC 集成使用指南

本指南详细说明如何使用 MultiAgentESC 框架与 MagNet 的测评系统进行集成测评。

## 目录结构

```
testmymodel/
├── MagNet/
│   ├── src/
│   │   ├── inference-multiagentesc.py    # 主集成脚本
│   │   └── inference-parallel-magnet.py  # MagNet 原始脚本（参考）
│   ├── dataset/
│   │   └── data_cn.json                   # 中文测试数据集
│   ├── evaluation/
│   │   ├── run_ctrs.py                    # CTRS 测评
│   │   ├── PANAS/                         # PANAS 测评
│   │   ├── WAI/                           # WAI 测评
│   │   └── Diversity/                     # Diversity 测评
│   ├── output-multiagentesc/              # MultiAgentESC 对话输出目录
│   ├── run_multiagentesc_eval.bat         # Windows 一键运行脚本
│   └── MULTIAGENTESC_INTEGRATION.md       # 集成说明
└── MultiAgentESC/
    ├── main.py                            # MultiAgentESC 原始脚本
    ├── multiagent.py                      # 多智能体模块
    ├── prompt.py                          # 提示词定义
    ├── strategy.py                        # 策略定义
    ├── dataset/
    │   └── ESConv.json                    # ESConv 数据集
    ├── generate_embeddings.py             # 嵌入生成脚本
    ├── embeddings.txt                     # 句子嵌入（需生成）
    └── OAI_CONFIG_LIST                    # LLM 配置（需创建）
```

## 快速开始

### 方法 1：使用一键脚本（推荐）

1. **配置 LLM**

   在 `MultiAgentESC` 目录下创建 `OAI_CONFIG_LIST` 文件：

   ```json
   [
       {
           "model": "qwen2.5:32b",
           "base_url": "http://localhost:11434/v1",
           "api_type": "openai"
       }
   ]
   ```

2. **运行一键脚本**

   ```bash
   cd MagNet
   run_multiagentesc_eval.bat
   ```

   脚本会自动完成：
   - 检查并生成 embeddings.txt（如果需要）
   - 使用 MultiAgentESC 生成对话
   - 运行各项测评（可选）

### 方法 2：手动运行

#### 步骤 1：生成 embeddings（可选但推荐）

```bash
cd MultiAgentESC
python generate_embeddings.py --dataset dataset/ESConv.json --output embeddings.txt
```

这一步会为 ESConv 数据集中的对话生成句子嵌入，用于 MultiAgentESC 的策略检索。

**注意**：如果没有 embeddings.txt，MultiAgentESC 会自动降级到零样本生成模式。

#### 步骤 2：生成对话

```bash
cd MagNet/src

# 基本用法
python inference-multiagentesc.py

# 自定义参数
python inference-multiagentesc.py \
    -o ../output-multiagentesc \
    -d ../dataset/data_cn.json \
    -m_turns 20 \
    --llm_model "qwen2.5:32b" \
    -num_pr 4
```

参数说明：
- `-o`: 输出目录
- `-d`: 数据集文件路径
- `-m_turns`: 最大对话轮次
- `--llm_model`: 使用的 LLM 模型
- `-num_pr`: 并行进程数（加速处理）

#### 步骤 3：运行测评

```bash
cd ../evaluation

# CTRS 评估（咨询技术评分）
python run_ctrs.py -i ../output-multiagentesc -o ../output-ctrs-multiagentesc

# PANAS 评估（情绪变化）
python PANAS/run_panas_before.py -i ../output-multiagentesc -o ../output-panas-before-multiagentesc
python PANAS/run_panas_after.py -i ../output-multiagentesc -o ../output-panas-after-multiagentesc

# WAI 评估（工作联盟）
python WAI/run_wai.py -i ../output-multiagentesc -o ../output-wai-multiagentesc

# Diversity 评估（响应多样性）
python Diversity/run_diversity.py -i ../output-multiagentesc -o ../output-diversity-multiagentesc
```

## 详细配置

### LLM 配置

MultiAgentESC 使用 AutoGen 框架，支持多种 LLM：

#### Ollama 本地模型

`OAI_CONFIG_LIST`:
```json
[
    {
        "model": "qwen2.5:32b",
        "base_url": "http://localhost:11434/v1",
        "api_type": "openai"
    }
]
```

#### Azure OpenAI

`OAI_CONFIG_LIST`:
```json
[
    {
        "model": "gpt-4o-mini",
        "api_key": "your-api-key",
        "api_base": "https://your-resource.openai.azure.com/",
        "api_type": "azure",
        "api_version": "2024-02-15-preview"
    }
]
```

#### 其他 OpenAI 兼容 API

```json
[
    {
        "model": "model-name",
        "base_url": "https://api.example.com/v1",
        "api_key": "your-api-key",
        "api_type": "openai"
    }
]
```

### 数据集配置

MagNet 提供的数据集格式：

```json
{
    "AI_client": {
        "intake_form": "客户基本信息...",
        "attitude": "positive/neutral/negative",
        "attitude_instruction": "态度描述...",
        "dialogue_history_init": "初始对话...",
        "dialogue_history": null
    },
    "AI_counselor": {
        "CBT": {
            "client_information": "客户信息...",
            "reason_counseling": "咨询原因...",
            "dialogue_history_init": "初始对话...",
            "init_history_counselor": "咨询师初始语...",
            "init_history_client": "客户初始语..."
        }
    },
    "ground_truth_CBT": ["CBT技术类型"]
}
```

## 工作原理

### MultiAgentESC 框架

MultiAgentESC 是一个基于策略选择的多智能体情感支持系统：

1. **复杂度判断**: 判断当前对话是否需要多智能体协作
2. **情感感知**: 识别用户的情感状态
3. **原因分析**: 分析导致情感的具体事件
4. **意图推断**: 理解用户的咨询意图
5. **策略选择**: 基于分析选择合适的咨询策略
6. **响应生成**: 使用选定策略生成响应
7. **多策略协作**: 当多个策略适用时，通过辩论和投票选出最佳响应
8. **自我反思**: 对生成的响应进行优化

### 与 MagNet 的集成

集成脚本 `inference-multiagentesc.py` 实现了：

- **适配器模式**: 将 MultiAgentESC 适配为 MagNet 的咨询师接口
- **格式转换**: 将对话历史转换为 MultiAgentESC 需要的格式
- **兼容输出**: 生成符合 MagNet 测评要求的 JSON 文件

### 提示词系统

**重要**：MultiAgentESC 使用自己的提示词系统（定义在 `MultiAgentESC/prompt.py`），完全不依赖 MagNet 的提示词。

MultiAgentESC 的提示词包括：
- `behavior_control`: 判断对话复杂度
- `zero_shot`: 零样本生成
- `get_emotion`: 情感分析
- `get_cause`: 原因分析
- `get_intention`: 意图分析

## 输出结果

### 对话文件格式

生成的 `session_N.json` 文件：

```json
{
    "example": {
        "AI_client": { ... },
        "AI_counselor": { ... },
        "ground_truth_CBT": [ ... ]
    },
    "cbt_technique": "MultiAgentESC (Strategy-based)",
    "cbt_plan": "MultiAgentESC uses dynamic strategy selection...",
    "cost": 0,
    "history": [
        {"role": "counselor", "message": "你好..."},
        {"role": "client", "message": "谢谢..."},
        ...
    ]
}
```

### 测评结果

每种测评会生成对应的评分文件：

- `output-ctrs-multiagentesc/`: CTRS 技术评分
- `output-panas-before-multiagentesc/`: 咨询前情绪评分
- `output-panas-after-multiagentesc/`: 咨询后情绪评分
- `output-wai-multiagentesc/`: 工作联盟评分
- `output-diversity-multiagentesc/`: 响应多样性评分

## 常见问题

### Q1: 提示词冲突？

**A**: 不会。MultiAgentESC 完全使用自己的提示词系统（在 `MultiAgentESC/prompt.py` 中定义），与 MagNet 的提示词系统完全独立。

### Q2: embeddings.txt 是必须的吗？

**A**: 不是。如果没有 embeddings.txt，MultiAgentESC 会自动降级到零样本生成模式。但为了获得更好的策略选择效果，建议生成 embeddings.txt。

### Q3: 如何加速处理？

**A**:
- 使用 `-num_pr` 参数启用多进程：`-num_pr 4`
- 减少 `-m_turns` 限制对话轮次：`-m_turns 10`
- 使用更快的 LLM 模型

### Q4: 出现错误怎么办？

**A**: 检查以下内容：
1. `OAI_CONFIG_LIST` 文件是否存在且格式正确
2. LLM API 是否可访问
3. 数据集文件路径是否正确
4. 查看 `error_multiagentesc_*.txt` 文件了解详细错误

### Q5: 如何与其他方法对比？

**A**: 可以分别运行不同方法的对话生成，然后使用相同的测评脚本：

```bash
# MultiAgentESC
python inference-multiagentesc.py -o ../output-multiagentesc

# MagNet 原始方法
python inference-parallel-magnet.py -o ../output-magnet

# 运行相同的测评
cd ../evaluation
python run_ctrs.py -i ../output-multiagentesc -o ../output-ctrs-multiagentesc
python run_ctrs.py -i ../output-magnet -o ../output-ctrs-magnet
```

## 技术细节

### MultiAgentESC 的策略系统

MultiAgentESC 支持 7 种咨询策略：

1. **Question**: 提问
2. **Restatement or Paraphrasing**: 重述或改述
3. **Reflection of feelings**: 情感反映
4. **Self-disclosure**: 自我披露
5. **Affirmation and Reassurance**: 肯定和安慰
6. **Providing Suggestions**: 提供建议
7. **Information**: 提供信息

策略选择基于：
- 用户情感状态
- 导致情感的事件
- 用户的咨询意图
- 历史对话的语义相似性

### 与 MagNet 原始方法的区别

| 特性 | MagNet | MultiAgentESC |
|------|--------|---------------|
| 提示词来源 | agent_*.txt | prompt.py |
| 技术选择 | 预设 CBT 技术 | 动态策略选择 |
| 多智能体 | 基于角色的智能体 | 基于功能的智能体 |
| 策略选择 | CBT Agent | 语义检索 + 群体决策 |
| 响应生成 | 多子响应合成 | 辩论 + 投票 + 反思 |

## 引用

如果使用本集成方案，请引用：

```bibtex
@software{magnet2024,
  title = {MagNet: A Multi-Agent Framework for Mental Health Counseling},
  author = {...},
  year = {2024}
}

@software{multiagentesc2024,
  title = {Multi-Agent ESC: Emotional Support Counseling with Strategy Selection},
  author = {...},
  year = {2024}
}
```

## 联系方式

如有问题，请查看：
- MagNet 文档：`MagNet/README.md`
- MultiAgentESC 原始文档
- 集成说明：`MULTIAGENTESC_INTEGRATION.md`
