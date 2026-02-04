# 精简版模型评估框架

这是从 MAGneT 项目中提取的精简版评估框架，专注于**评估微调后模型生成的对话数据**。

## 📁 项目结构

```
simplified/
├── dataset/               # 测试数据集
│   └── test_data.json    # 客户初始设定
├── generate/             # 数据生成脚本
│   └── generate_dialogue.py  # 使用微调后模型生成对话
├── evaluate/             # 评估脚本
│   ├── diversity.py      # 多样性指标
│   ├── ctrs.py           # CTRS评估
│   ├── wai.py            # WAI评估
│   └── panas.py          # PANAS评估
├── prompts/              # 评估提示词
│   ├── ctrs/            # CTRS评估prompts
│   ├── wai/             # WAI评估prompts
│   └── panas/           # PANAS评估prompts
├── requirements.txt      # 依赖包
└── README.md            # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备微调后的模型

使用 vLLM 启动本地服务：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model "/path/to/your/finetuned/model" \
  --dtype float16 \
  --host 0.0.0.0 \
  --port 8000
```

### 3. 生成对话数据

```bash
python generate/generate_dialogue.py \
  --input dataset/test_data.json \
  --output output/sessions \
  --max_turns 20
```

### 4. 评估生成的数据

#### 多样性评估

```bash
python evaluate/diversity.py \
  --input output/sessions \
  --output results/diversity.json
```

#### CTRS评估

```bash
python evaluate/ctrs.py \
  --input output/sessions \
  --output results/ctrs \
  --max_iter 3
```

#### WAI评估

```bash
python evaluate/wai.py \
  --input output/sessions \
  --output results/wai \
  --max_iter 3
```

#### PANAS评估

```bash
# 评估咨询后情感状态
python evaluate/panas.py \
  --input output/sessions \
  --dataset dataset/test_data.json \
  --output results/panas \
  --max_iter 3
```

## 📊 输出数据格式

### 生成的对话 (session_X.json)

```json
{
  "example": { /* 完整的客户初始设定 */ },
  "history": [
    {"role": "counselor", "message": "咨询师说的话"},
    {"role": "client", "message": "客户说的话"},
    ...
  ]
}
```

### 评估结果

每个评估脚本会输出JSON格式的评分结果：

**CTRS/WAI**:
```json
{
  "session_1.json": {
    "metric_1": 5.3,
    "metric_2": 4.7,
    ...
  },
  ...
}
```

**Diversity**:
```json
{
  "distinct_1": 0.85,
  "distinct_2": 0.72,
  "distinct_3": 0.61,
  "ead": 0.79
}
```

## ⚙️ 配置说明

### Azure OpenAI配置

在评估脚本中设置您的Azure OpenAI凭据：

```python
endpoint = "your_azure_endpoint"
api_key = "your_subscription_key"
api_version = "your_api_version"
deployment = "gpt-4o"
```

### vLLM服务配置

如果使用不同端口，修改 `generate_dialogue.py` 中的：

```python
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",  # 修改端口
    api_key="dummy-key"
)
```

## 📈 评估指标说明

### 1. Diversity (多样性)
- **Distinct-1/2/3**: n-gram词汇多样性
- **EAD**: 期望调整后的多样性

### 2. CTRS (认知治疗评分量表)
- **通用技能** (3项): 理解能力、人际效能、协作能力
- **CBT技能** (3项): 引导发现、聚焦能力、策略运用
- 评分范围: 0-6分

### 3. WAI (工作联盟量表)
- 12个项目，评估治疗联盟质量
- 评分范围: 1-7分

### 4. PANAS (积极消极情感量表)
- 20种情感状态（10种积极 + 10种消极）
- 评分范围: 1-5分
- 计算咨询前后的情感变化

## 🔧 常见问题

### Q: 如何使用其他LLM服务？

修改 `generate_dialogue.py` 中的API配置，支持任何OpenAI兼容的API。

### Q: 评估很慢怎么办？

- 减少 `--max_iter` 参数（默认3，可改为1）
- 使用更快的GPT-4o-mini替代GPT-4o
- 并行运行多个评估脚本

### Q: 如何添加新的评估指标？

在 `evaluate/` 目录下创建新的评估脚本，参考现有脚本的格式。

## 📝 许可证

本精简版基于原 MAGneT 项目，遵循相同的许可证。

## 🙏 致谢

- 原始项目: [MAGneT](https://github.com/your-repo/MAGneT)
- 评估指标: CTRS, WAI, PANAS
