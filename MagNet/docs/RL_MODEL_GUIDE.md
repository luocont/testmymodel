# MagNet 本地强化学习模型集成指南

本指南介绍如何使用本地 Qwen3 强化学习模型替换 MagNet 框架中的咨询师智能体（Counselor Agent），同时保留原有的来访者智能体（Client Agent）。

## 目录

- [快速开始](#快速开始)
- [详细配置](#详细配置)
- [系统提示词](#系统提示词)
- [运行方式](#运行方式)
- [输出格式](#输出格式)
- [故障排查](#故障排查)

---

## 快速开始

### 1. 环境准备

```bash
cd MagNet

# 安装依赖
pip install -r requirements_inf_eval.txt
pip install transformers>=4.35.0 accelerate torch sentencepiece
```

### 2. 配置环境变量

编辑 `.env` 文件（用于 Client Agent）：

```bash
LLM_PROVIDER=aliyun
LLM_API_KEY=your_dashscope_api_key
LLM_MODEL=qwen2.5-7b-instruct
```

### 3. 配置模型路径

编辑 `config_rl.json` 文件：

```json
{
  "model_path": "E:/Models/your-qwen3-model",
  "system_prompt": "你是一位专业的心理咨询师...",
  "max_turns": 20,
  "max_new_tokens": 512,
  "temperature": 0.7
}
```

### 4. 运行

```bash
cd src
python inference-rl-custom.py --config ../config_rl.json
```

---

## 详细配置

### 配置文件参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `model_path` | string | 本地模型路径（必需） | - |
| `system_prompt` | string | 系统提示词 | 内置默认提示词 |
| `max_turns` | int | 最大对话轮数 | 20 |
| `max_new_tokens` | int | 最大生成token数 | 512 |
| `temperature` | float | 采样温度 (0-1) | 0.7 |
| `device_map` | string | 设备映射 (auto/cpu) | auto |
| `torch_dtype` | string | 数据类型 | float16 |
| `output_dir` | string | 输出目录 | ../output-rl-custom |

### 硬件要求

| 模型大小 | 推荐显存 | 推荐内存 |
|---------|---------|---------|
| 7B 参数 | ~16GB VRAM | ~32GB RAM |
| 14B 参数 | ~32GB VRAM | ~64GB RAM |
| 32B 参数 | ~64GB VRAM | ~128GB RAM |

---

## 系统提示词

### 预设提示词

框架提供了三种预设的系统提示词：

#### 1. CBT 提示词

```bash
python inference-rl-custom.py --model_path /path/to/model --preset_prompt cbt
```

#### 2. 人本主义提示词

```bash
python inference-rl-custom.py --model_path /path/to/model --preset_prompt person_centered
```

#### 3. 简洁提示词

```bash
python inference-rl-custom.py --model_path /path/to/model --preset_prompt brief
```

### 自定义提示词

#### 方式1：命令行直接输入

```bash
python inference-rl-custom.py \
  --model_path /path/to/model \
  --system_prompt "你是一位专业的心理咨询师..."
```

#### 方式2：从文件读取

```bash
python inference-rl-custom.py \
  --model_path /path/to/model \
  --system_prompt_file ../prompts/my_prompt.txt
```

#### 方式3：配置文件

在 `config_rl.json` 中设置 `system_prompt` 字段。

### 提示词设计建议

良好的系统提示词应该：

- **明确角色定位**：清晰定义咨询师的专业身份
- **指定技术框架**：如 CBT、人本主义等
- **约束响应长度**：避免生成过长或过短的回应
- **保持一致性**：与强化学习训练时的提示词风格一致

示例：

```
你是一位专业的心理咨询师，拥有10年的临床经验。

咨询方法：
- 主要使用认知行为疗法(CBT)
- 结合共情倾听和积极关注
- 适度运用开放式提问

回应要求：
- 每次回应30-80字
- 语言温暖、专业、不评判
- 避免说教或过度建议
```

---

## 运行方式

### 方式1：使用配置文件（推荐）

```bash
python inference-rl-custom.py --config ../config_rl.json
```

### 方式2：使用命令行参数

```bash
python inference-rl-custom.py \
  --model_path "E:/Models/qwen3-rl-model" \
  --system_prompt "你的提示词" \
  --output_dir ../output-rl \
  --max_turns 20 \
  --temperature 0.7
```

### 方式3：处理部分样本

```bash
python inference-rl-custom.py \
  --config ../config_rl.json \
  --num_samples 5  # 只处理前5个样本
```

### 方式4：多进程并行

```bash
python inference-rl-custom.py \
  --config ../config_rl.json \
  --num_processes 4  # 使用4个进程并行处理
```

---

## 输出格式

输出文件与原始 MagNet 格式完全一致：

```json
{
    "example": {
        "AI_client": {
            "intake_form": "...",
            "attitude": "positive",
            "attitude_instruction": "...",
            "dialogue_history_init": "..."
        },
        "AI_counselor": {
            "CBT": {
                "client_information": "...",
                "reason_counseling": "...",
                "dialogue_history_init": "...",
                "init_history_counselor": "你好...",
                "init_history_client": "最近..."
            }
        },
        "ground_truth_CBT": ["Alternative Perspective"]
    },
    "cbt_technique": "Custom RL Model (Qwen3)",
    "cbt_plan": "使用本地强化学习模型: E:/Models/qwen3-rl-model",
    "cost": 0,
    "history": [
        {"role": "counselor", "message": "你好..."},
        {"role": "client", "message": "最近..."},
        {"role": "counselor", "message": "我理解..."},
        ...
    ]
}
```

---

## 故障排查

### 常见错误

#### 1. `CUDA out of memory`

**原因**：显存不足

**解决方案**：
- 减小 `max_new_tokens` 参数
- 使用量化模型
- 使用 CPU 推理：`--device_map cpu`

```bash
python inference-rl-custom.py \
  --config ../config_rl.json \
  --max_new_tokens 256 \
  --device_map cpu
```

#### 2. 模型加载失败

**原因**：路径错误或文件缺失

**解决方案**：
- 检查 `model_path` 是否正确
- 确保模型目录包含必要文件（config.json, tokenizer.json 等）

```bash
# 检查模型文件
ls /path/to/your/model/
# 应包含：config.json, tokenizer.json, model-*.safetensors 等
```

#### 3. Client Agent 调用失败

**原因**：`.env` 配置错误

**解决方案**：
- 检查 API 密钥是否正确
- 确保网络连接正常

```bash
# 测试 API 连接
python -c "from llm_client import create_client_from_env; print(create_client_from_env().completion('测试'))"
```

#### 4. 生成内容乱码

**原因**：Tokenizer 不匹配

**解决方案**：确保使用 `trust_remote_code=True`（已默认设置）

---

### 调试技巧

#### 查看详细日志

```bash
python inference-rl-custom.py --config ../config_rl.json 2>&1 | tee run.log
```

#### 测试单个样本

修改 `dataset/data_cn.json`，只保留一个样本进行测试。

#### 检查 GPU 使用

```bash
# Linux
nvidia-smi -l 1

# Windows
while ($true) { nvidia-smi; Start-Sleep -Seconds 1 }
```

---

## 文件结构

```
MagNet/
├── src/
│   ├── inference-rl-custom.py      # 主推理脚本
│   ├── rl_counselor_agent.py       # RL咨询师智能体模块
│   └── llm_client.py                # LLM客户端（原有）
├── prompts/
│   └── cn/
│       └── agent_client.txt        # Client Agent提示词（原有）
├── config_rl.json                  # RL模型配置文件
├── output-rl-custom/               # 输出目录
│   └── session_*.json
├── dataset/
│   └── data_cn.json                # 输入数据
└── .env                            # 环境变量配置
```

---

## 架构说明

### 数据流程

```
┌─────────────────────────────────────────────────────────────┐
│                      数据流程图                              │
└─────────────────────────────────────────────────────────────┘

初始化
   │
   ├─ 加载 data_cn.json
   │
   ├─ ClientAgent (使用框架预设提示词 + API)
   │
   └─ RLCounselorAgent (使用本地RL模型 + 自定义系统提示词)

对话循环
   │
   ├─ counselor_agent.generate(history)
   │  ├─ 加载本地 Qwen3 模型
   │  ├─ 应用系统提示词
   │  └─ 生成响应
   │
   ├─ client_agent.generate(history)
   │  ├─ 使用框架预设提示词
   │  ├─ 调用 LLM API
   │  └─ 生成响应
   │
   └─ 检测 [/END] 或达到 max_turns

输出
   │
   └─ 保存 session_N.json (与原始格式一致)
```

---

## 高级用法

### 自定义提示词模板

创建 `prompts/my_prompt.txt`：

```
你是一位专业的心理咨询师。

你的方法：
1. 共情倾听来访者的困扰
2. 帮助来访者识别负面思维
3. 引导来访者发展新的视角

每次回应控制在50字左右，保持温暖、专业的语调。
```

使用：

```bash
python inference-rl-custom.py \
  --model_path /path/to/model \
  --system_prompt_file ../prompts/my_prompt.txt
```

---

## 联系支持

如有问题，请检查：
1. 模型路径是否正确
2. 环境变量是否配置
3. 硬件资源是否充足
4. 错误日志中的详细信息

祝您使用愉快！
