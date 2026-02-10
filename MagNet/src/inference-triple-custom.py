"""
使用三联动模型生成心理咨询对话

特点：
- 保留 Client Agent 的原有逻辑（使用框架预设提示词）
- 使用三联动模型（情感分析+顾问模型+主模型）作为 Counselor Agent
- 支持自定义系统提示词
- 生成与原始格式一致的 session_*.json 文件

使用示例：
    # 使用配置文件
    python inference-triple-custom.py --config ../config_triple.json

    # 使用命令行参数
    python inference-triple-custom.py --sentiment_model /path/to/sentiment.bin --primary_model /path/to/primary --consultant_model /path/to/consultant
"""

import argparse
import json
import multiprocessing
import traceback
import os
from pathlib import Path


class PromptTemplate:
    """简单的提示词模板类（替代 langchain.prompts.PromptTemplate）"""

    def __init__(self, input_variables, template):
        self.input_variables = input_variables
        self.template = template

    def format(self, **kwargs):
        result = self.template
        for var in self.input_variables:
            if var in kwargs:
                result = result.replace(f"{{{var}}}", str(kwargs[var]))
        return result


# 导入通用 LLM 客户端（用于 Client Agent）
from llm_client import create_client_from_env

# 导入三联动模型 Agent
from triple_model_agent import TripleModelAgent, get_triple_model_preset_prompt


# ============================================
# 配置
# ============================================
DATA_FILE = "../dataset/data_cn.json"
PROMPTS_DIR = "../prompts/cn/"

# 用于 Client Agent 的 LLM 客户端（全局变量，延迟加载）
client_llm = None


def load_env_file():
    """加载 .env 文件"""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        print(f"加载环境变量: {env_file}")
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
        print("环境变量加载完成")
    else:
        print("警告: 未找到 .env 文件，Client Agent 将无法工作")


def generate_with_api(prompt: str) -> str:
    """使用 API 生成响应（用于 Client Agent）"""
    global client_llm
    if client_llm is None:
        try:
            client_llm = create_client_from_env()
        except Exception as e:
            print(f"错误: 无法创建 LLM 客户端: {e}")
            print("请确保 .env 文件配置正确")
            raise
    response = client_llm.completion(prompt=prompt)
    return response.choices[0].message.content


# ============================================
# 原有 Client Agent（保持不变）
# ============================================
class ClientAgent:
    """来访者智能体（使用框架原有逻辑和提示词）"""

    def __init__(self, example):
        self.example = example
        self._load_prompt()

    def _load_prompt(self):
        """加载 Client Agent 提示词"""
        prompt_path = Path(PROMPTS_DIR) / "agent_client.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"找不到 Client Agent 提示词文件: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()

        self.attitude = (
            f"{self.example['AI_client']['attitude']}: "
            f"{self.example['AI_client']['attitude_instruction']}"
        )
        self.prompt_template = PromptTemplate(
            input_variables=["intake_form", "attitude", "history"],
            template=prompt_text
        )

    def generate(self, history):
        """生成来访者响应"""
        history_text = '\n'.join([
            f"{message['role'].capitalize()}: {message['message']}"
            for message in history
        ])

        prompt = self.prompt_template.format(
            intake_form=self.example,
            attitude=self.attitude,
            history=history_text
        )

        return generate_with_api(prompt)


# ============================================
# 三联动模型咨询会话
# ============================================
class TripleModelTherapySession:
    """使用三联动模型的咨询会话"""

    def __init__(
        self,
        example,
        max_turns: int,
        sentiment_model_path: str,
        primary_model_path: str,
        consultant_model_path: str,
        system_prompt: str = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.5,
        n_sentiment_classes: int = 2
    ):
        """
        初始化会话

        Args:
            example: 数据样本
            max_turns: 最大对话轮数
            sentiment_model_path: 情感分类模型路径
            primary_model_path: 主模型路径
            consultant_model_path: 顾问模型路径
            system_prompt: 系统提示词
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            n_sentiment_classes: 情感分类数量 (2: 消极/积极, 3: 消极/中性/积极)
        """
        self.example = example
        self.max_turns = max_turns
        self.history = []

        # 初始化智能体
        self.client_agent = ClientAgent(example=example)
        self.counselor_agent = TripleModelAgent(
            example=example,
            sentiment_model_path=sentiment_model_path,
            primary_model_path=primary_model_path,
            consultant_model_path=consultant_model_path,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            n_sentiment_classes=n_sentiment_classes
        )

    def _add_to_history(self, role: str, message: str):
        """添加消息到历史"""
        self.history.append({"role": role, "message": message})

    def _initialize_session(self):
        """初始化会话"""
        example_cbt = self.example['AI_counselor']['CBT']
        self._add_to_history("counselor", example_cbt['init_history_counselor'])
        self._add_to_history("client", example_cbt['init_history_client'])

    def _exchange_statements(self):
        """交替生成对话"""
        for turn in range(self.max_turns):
            print(f"\n    轮次 {turn + 1}/{self.max_turns}")
            print("=" * 60)

            # 咨询师回应（使用三联动模型）
            counselor_response = self.counselor_agent.generate(self.history)

            # 打印完整的咨询师回复（包含思考过程）
            print(f"    📋 咨询师:\n{counselor_response}")

            # 添加到历史（完整内容）
            self._add_to_history("counselor", counselor_response)

            # 来访者回应（使用框架原有 Client Agent）
            client_response = self.client_agent.generate(self.history)
            client_response = client_response.replace('Client: ', '')

            # 前10轮：移除 [/END] 标记，不中断对话
            # 10轮后：检测 [/END] 标记，如果存在则结束对话
            if turn < 10:
                # 移除 [/END] 标记
                client_response = client_response.replace('[/END]', '')
            else:
                # 检测是否有结束标记
                if '[/END]' in client_response:
                    # 移除标记并添加到历史
                    client_response = client_response.replace('[/END]', '')
                    self._add_to_history("client", client_response)
                    print(f"\n    📋 来访者:\n{client_response}")
                    print("\n    ✓ 检测到结束标记，会话结束")
                    print("=" * 60)
                    break

            self._add_to_history("client", client_response)
            print(f"\n    📋 来访者:\n{client_response}")

    def run_session(self):
        """运行完整会话"""
        self._initialize_session()
        self._exchange_statements()

        return {
            "example": self.example,
            "cbt_technique": "Triple Model (Sentiment + Consultant + Primary)",
            "cbt_plan": f"三联动模型: {self.counselor_agent.primary_interactor.model}",
            "cost": 0,
            "history": self.history  # 保留完整的对话历史（包含 <thinking> 标签）
        }


def run_therapy_session(
    index: int,
    example: dict,
    output_dir: Path,
    total: int,
    max_turns: int,
    sentiment_model_path: str,
    primary_model_path: str,
    consultant_model_path: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    n_sentiment_classes: int
):
    """运行单个咨询会话（用于多进程）"""
    file_number = index + 1

    try:
        print(f"\n[{file_number}/{total}] 开始生成会话")

        session = TripleModelTherapySession(
            example=example,
            max_turns=max_turns,
            sentiment_model_path=sentiment_model_path,
            primary_model_path=primary_model_path,
            consultant_model_path=consultant_model_path,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            n_sentiment_classes=n_sentiment_classes
        )

        session_data = session.run_session()

        # 保存结果
        file_name = f"session_{file_number}.json"
        file_path = output_dir / file_name

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=4)

        # 清理 GPU 缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"[{file_number}/{total}] GPU 缓存已清理")
        except Exception as e:
            print(f"[{file_number}/{total}] 清理 GPU 缓存时出现警告: {e}")

        print(f"[{file_number}/{total}] 完成，保存到 {file_name}")

    except Exception as e:
        error_file_name = f"error_{file_number}.txt"
        error_file_path = output_dir / error_file_name

        with open(error_file_path, "w", encoding="utf-8") as f:
            f.write(f"Error: {e}\n\n")
            f.write(traceback.format_exc())

        print(f"[{file_number}/{total}] 失败，错误已保存到 {error_file_name}")


def main():
    parser = argparse.ArgumentParser(
        description="使用三联动模型生成心理咨询对话",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 使用配置文件
  python inference-triple-custom.py --config ../config_triple.json

  # 使用命令行参数
  python inference-triple-custom.py --sentiment_model /path/to/sentiment.bin --primary_model /path/to/primary --consultant_model /path/to/consultant

  # 使用预设系统提示词
  python inference-triple-custom.py --sentiment_model /path/to/sentiment.bin --primary_model /path/to/primary --consultant_model /path/to/consultant --preset_prompt cbt
        """
    )
    parser.add_argument("--config", type=str, help="配置文件路径 (JSON格式)")
    parser.add_argument("--sentiment_model", type=str, help="情感分类模型路径 (.bin文件)")
    parser.add_argument("--primary_model", type=str, help="主模型路径")
    parser.add_argument("--consultant_model", type=str, help="顾问模型路径")
    parser.add_argument("--system_prompt", type=str, help="系统提示词（直接输入）")
    parser.add_argument("--preset_prompt", type=str, choices=["cbt", "person_centered", "brief"],
                        help="使用预设系统提示词")
    parser.add_argument("--system_prompt_file", type=str, help="从文件读取系统提示词")
    parser.add_argument("--output_dir", type=str, default="../output-triple-custom", help="输出目录")
    parser.add_argument("--max_turns", type=int, default=20, help="最大对话轮数")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.5, help="采样温度")
    parser.add_argument("--num_processes", type=int, default=1, help="并行进程数")
    parser.add_argument("--num_samples", type=int, help="处理样本数量（默认处理全部）")
    parser.add_argument("--n_sentiment_classes", type=int, default=2, choices=[2, 3],
                        help="情感分类数量: 2=消极/积极, 3=消极/中性/积极")

    args = parser.parse_args()

    # 加载环境变量
    load_env_file()

    # 加载配置文件
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"加载配置文件: {args.config}")

    # 合并配置（命令行参数优先）
    sentiment_model_path = args.sentiment_model or config.get("sentiment_model_path")
    primary_model_path = args.primary_model or config.get("primary_model_path")
    consultant_model_path = args.consultant_model or config.get("consultant_model_path")
    max_turns = args.max_turns or config.get("max_turns", 20)
    max_new_tokens = args.max_new_tokens or config.get("max_new_tokens", 2048)
    temperature = args.temperature if args.temperature != 0.5 else config.get("temperature", 0.5)
    n_sentiment_classes = args.n_sentiment_classes or config.get("n_sentiment_classes", 2)

    # 系统提示词处理
    system_prompt = None
    if args.system_prompt:
        system_prompt = args.system_prompt
    elif args.preset_prompt:
        system_prompt = get_triple_model_preset_prompt(args.preset_prompt)
        print(f"使用预设提示词: {args.preset_prompt}")
    elif args.system_prompt_file:
        prompt_file = Path(args.system_prompt_file)
        if prompt_file.exists():
            with open(prompt_file, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        else:
            print(f"警告: 提示词文件不存在: {args.system_prompt_file}")
    elif config.get("system_prompt"):
        system_prompt = config["system_prompt"]

    # 验证必需参数
    if not sentiment_model_path:
        print("错误: 请指定情感分类模型路径")
        print("  方式1: --sentiment_model /path/to/sentiment.bin")
        print("  方式2: 在配置文件中设置 sentiment_model_path")
        return

    if not primary_model_path:
        print("错误: 请指定主模型路径")
        print("  方式1: --primary_model /path/to/primary")
        print("  方式2: 在配置文件中设置 primary_model_path")
        return

    if not consultant_model_path:
        print("错误: 请指定顾问模型路径")
        print("  方式1: --consultant_model /path/to/consultant")
        print("  方式2: 在配置文件中设置 consultant_model_path")
        return

    # 加载数据
    data_file = Path(DATA_FILE)
    if not data_file.exists():
        print(f"错误: 数据文件不存在: {data_file}")
        print(f"请确保 {DATA_FILE} 文件存在")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 限制样本数量
    if args.num_samples:
        data = data[:args.num_samples]
        print(f"处理前 {args.num_samples} 个样本")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打印配置信息
    print("\n" + "=" * 60)
    print("配置信息:")
    print("=" * 60)
    print(f"  情感分类模型: {sentiment_model_path}")
    print(f"  情感分类数量: {n_sentiment_classes} ({'消极/积极' if n_sentiment_classes == 2 else '消极/中性/积极'})")
    print(f"  主模型: {primary_model_path}")
    print(f"  顾问模型: {consultant_model_path}")
    print(f"  系统提示词: {system_prompt[:80] if system_prompt else '默认'}{'...' if system_prompt and len(system_prompt) > 80 else ''}")
    print(f"  最大轮数: {max_turns}")
    print(f"  最大tokens: {max_new_tokens}")
    print(f"  采样温度: {temperature}")
    print(f"  输出目录: {output_dir}")
    print(f"  数据样本数: {len(data)}")
    print(f"  并行进程数: {args.num_processes}")
    print("=" * 60 + "\n")

    # 准备参数列表
    args_list = [
        (index, example, output_dir, len(data), max_turns, sentiment_model_path,
         primary_model_path, consultant_model_path, system_prompt, max_new_tokens, temperature,
         n_sentiment_classes)
        for index, example in enumerate(data)
    ]

    # 运行会话
    if args.num_processes > 1:
        print(f"使用 {args.num_processes} 个并行进程\n")
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            pool.starmap(run_therapy_session, args_list)
    else:
        for args_tuple in args_list:
            run_therapy_session(*args_tuple)

    print("\n" + "=" * 60)
    print(f"完成！共处理 {len(data)} 个会话")
    print(f"结果保存在: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
