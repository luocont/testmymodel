# MagNet 中文版使用指南

## 概述

MagNet 中文版已将多智能体框架的数据和提示词翻译为中文，支持生成中文心理咨询对话。

**注意：** 评估系统的提示词保持英文不变，以确保评估标准的一致性。

---

## 文件结构

### 中文数据文件
- **[dataset/data_cn.json](../dataset/data_cn.json)** - 中文版客户初始信息数据
  - 包含 9 个中文案例
  - 每个案例有 3 种态度变体（积极/中性/消极）

### 中文提示词文件
所有中文提示词存放在 **[prompts/cn/](../prompts/cn/)** 目录：

| 文件 | 说明 |
|------|------|
| `agent_client.txt` | 客户智能体提示词 |
| `agent_cbt.txt` | CBT 智能体提示词 |
| `agent_technique.txt` | 技术选择智能体提示词 |
| `agent_reflections.txt` | 反思技术智能体提示词 |
| `agent_questioning.txt` | 提问技术智能体提示词 |
| `agent_solutions.txt` | 解决方案技术智能体提示词 |
| `agent_normalization.txt` | 正常化技术智能体提示词 |
| `agent_psychoed.txt` | 心理教育技术智能体提示词 |
| `agent_dialogue_gen.txt` | 对话生成智能体提示词 |

---

## 使用方法

### 方式一：修改代码使用中文版（推荐）

#### 1. 修改数据文件路径

在 `src/inference-parallel-magnet.py` 中，修改数据文件路径：

```python
# 原代码（英文版）
with open("../dataset/data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 修改为（中文版）
with open("../dataset/data_cn.json", "r", encoding="utf-8") as f:
    data = json.load(f)
```

#### 2. 修改提示词文件路径

修改 `Agent` 基类的 `load_prompt` 方法：

```python
# 原代码（英文版）
def load_prompt(self, file_name):
    base_dir = "../prompts/"
    file_path = base_dir + file_name
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# 修改为（中文版）
def load_prompt(self, file_name):
    base_dir = "../prompts/cn/"
    file_path = base_dir + file_name
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
```

### 方式二：创建中文版专用脚本

复制 `inference-parallel-magnet.py` 并创建中文版本：

```bash
cp src/inference-parallel-magnet.py src/inference-parallel-magnet-cn.py
```

然后修改以下内容：

1. 修改数据文件路径（第 650 行左右）：
```python
with open("../dataset/data_cn.json", "r", encoding="utf-8") as f:
    data = json.load(f)
```

2. 修改提示词目录（第 100-104 行左右）：
```python
def load_prompt(self, file_name):
    base_dir = "../prompts/cn/"
    file_path = base_dir + file_name
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
```

### 运行中文版

```bash
cd src
python inference-parallel-magnet-cn.py -o ../output-cn -num_pr 4 -m_turns 20
```

---

## API 配置

中文版使用相同的 API 配置，支持阿里云百炼和 OpenRouter。

### 使用阿里云百炼（推荐用于中文）

编辑 `.env` 文件：

```bash
# 多智能体框架配置
LLM_PROVIDER=aliyun
LLM_API_KEY=your_dashscope_api_key
LLM_MODEL=qwen2.5-7b-instruct

# 评估系统配置
EVAL_LLM_PROVIDER=openrouter
EVAL_LLM_API_KEY=your_openrouter_api_key
EVAL_LLM_MODEL=openai/gpt-4o
```

### 推荐的中文模型

| 模型 | 提供商 | 用途 |
|------|--------|------|
| qwen2.5-7b-instruct | 阿里云百炼 | 多智能体对话生成 |
| qwen-max | 阿里云百炼 | 技术智能体（更强） |
| qwen-plus | 阿里云百炼 | 经济实惠的选择 |

---

## 生成示例

### 中文对话示例

```
咨询师：您好劳拉，很高兴认识您。今天我能为您做些什么？
客户：您好，感谢您见我。最近我在跑步方面遇到了一些麻烦。我为自己设定了目标，但我一直觉得自己跑不远，这真的让我很沮丧。

咨询师：我听到了您的困扰。能告诉我更多关于这种感觉吗？
客户：当然。每当我系好跑步鞋出门时，我就会开始担心自己跑不远。这个想法让我失去了继续跑步的动力。
```

---

## 注意事项

### 1. 评估系统仍使用英文

评估系统的提示词保持英文，原因：
- 评估标准需要一致性
- 评估模型（如 GPT-4o）对英文理解更好
- 保持与原研究的可比性

### 2. 模型选择建议

**中文对话生成：**
- 阿里云 qwen2.5-7b-instruct 或 qwen-plus
- 性价比高，中文表现优秀

**技术智能体：**
- 阿里云 qwen-max
- 更强的推理能力

**评估系统：**
- OpenRouter GPT-4o
- 确保评估质量

### 3. 数据格式

中文数据文件保持与英文版相同的 JSON 结构，仅将内容翻译为中文。

---

## 故障排除

### 问题 1: 提示词文件未找到

**错误信息：** `FileNotFoundError: ../prompts/cn/agent_xxx.txt`

**解决方法：** 确保 `prompts/cn/` 目录存在且包含所有提示词文件。

### 问题 2: 中文输出乱码

**解决方法：** 确保所有文件使用 UTF-8 编码：
```python
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()
```

### 问题 3: 模型不支持中文

**解决方法：** 使用支持中文的模型：
- 阿里云百炼 qwen 系列
- 或其他支持中文的模型

---

## 扩展中文数据

如需添加更多中文案例，按照以下格式添加到 `data_cn.json`：

```json
{
    "AI_client": {
        "intake_form": "姓名：\n[姓名]\n年龄：\n[年龄]\n...",
        "attitude": "positive/neutral/negative",
        "attitude_instruction": "[态度描述]",
        "dialogue_history_init": "咨询师：[开场]\n客户：[回应]",
        "dialogue_history": null
    },
    "AI_counselor": {
        "CBT": {
            "client_information": "[客户信息]",
            "reason_counseling": "[咨询原因]",
            "dialogue_history_init": "[初始对话]",
            "init_history_counselor": "[咨询师开场]",
            "init_history_client": "[客户回应]"
        },
        "Response": {
            "client_information": "[客户信息]",
            "reason_counseling": "[咨询原因]",
            "cbt_plan": null,
            "dialogue_history": null
        }
    },
    "ground_truth_CBT": ["技术名称"]
}
```

---

## 相关文档

- [API 配置指南](API_CONFIG_GUIDE.md)
- [快速开始](QUICK_START_CN.md)
- [项目 README](../README.md)
