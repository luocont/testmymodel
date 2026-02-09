import torch
import torch.nn as nn
import gradio as gr
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BertTokenizer, BertModel
import markdown
import re
import json
import datetime
import os

# ===================================================================
# Part 1 & 2: 模型定义与加载
# 所有必要的模型都在下面加载。
# ===================================================================

# --- 情感分类模型配置 ---
SENTIMENT_MODEL_PATH = '/root/autodl-tmp/xinliyisheng/model3/best_model.bin'
BERT_MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 情感分类模型定义 ---
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.act = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.drop(out.pooler_output)
        x = self.act(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)

# --- 情感分类模型加载函数 ---
def load_sentiment_model():
    print(f"正在从 '{SENTIMENT_MODEL_PATH}' 加载情感分类模型...")
    model = SentimentClassifier(n_classes=3)
    try:
        # 兼容不同方式保存的模型
        ckpt = torch.load(SENTIMENT_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt.get('model_state', ckpt))
        model.to(DEVICE)
        model.eval()
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        print(f"✅ 情感分类模型 '{SENTIMENT_MODEL_PATH}' 加载成功！")
        return model, tokenizer
    except Exception as e:
        print(f"加载情感分类模型时出错: {e}")
        raise

# --- 情感分类预测函数 ---
def predict_sentiment(text, model, tokenizer):
    encoding = tokenizer(
        text, add_special_tokens=True, max_length=MAX_LEN,
        padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
    label_map = ['消极', '中性', '积极']
    return label_map[pred.item()], conf.item()

# --- 大语言模型加载器定义 ---
class LLMInteractor:
    def __init__(self, model_path: str):
        print(f"正在从 '{model_path}' 加载大语言模型...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            print(f"✅ 大语言模型 '{model_path}' 加载成功！")
        except Exception as e:
            print(f"加载大语言模型 '{model_path}' 时出错: {e}")
            raise

# --- 核心工具函数 ---
def extract_thinking_process(raw_text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    print("警告: 在顾问模型的输出中未找到 <think>...</think> 标签。")
    return ""

# ===================================================================
# Part 3: 程序启动与模型加载
# ===================================================================
try:
    SENTIMENT_MODEL, SENTIMENT_TOKENIZER = load_sentiment_model()
    MODEL_PATH_PRIMARY = "/root/autodl-tmp/xinliyisheng/model1-zhu-GRPO"
    MODEL_PATH_CONSULTANT = "/root/autodl-tmp/xinliyisheng/model2-COT"
    INTERACTOR_PRIMARY = LLMInteractor(model_path=MODEL_PATH_PRIMARY)
    INTERACTOR_CONSULTANT = LLMInteractor(model_path=MODEL_PATH_CONSULTANT)
except Exception as e:
    print(f"程序启动失败，因为模型加载失败: {e}")
    # 在无法加载模型时，可以考虑退出程序或在UI上显示错误
    exit()

# ===================================================================
# Part 4: Gradio 核心交互逻辑 (最终优化版)
# ===================================================================

def predict(user_input, chatbot_history, model_history_state, primary_system_prompt, temperature):
    """
    集成了情感分析的智能路由交互函数。
    为顾问模型使用独立的低温和指令注入式提示词，以保证分析质量。
    顾问模型的思考过程现在是流式输出。
    """
    # --- 步骤 1: 进行情感分析 ---
    sentiment_label, sentiment_conf = predict_sentiment(user_input, SENTIMENT_MODEL, SENTIMENT_TOKENIZER)
    print(f"\n--- [情感分析结果]: {sentiment_label} (置信度: {sentiment_conf:.2%}) ---")
    sentiment_emoji_map = {'消极': '😔', '中性': '😐', '积极': '😊'}
    sentiment_display_text = f"**{sentiment_emoji_map.get(sentiment_label, '🤔')} {sentiment_label}** (置信度: {sentiment_conf:.2%})"
    
    # --- 准备聊天历史 ---
    current_model_history = model_history_state.copy()
    chatbot_history.append([user_input, ""])
    current_model_history.append({"role": "user", "content": user_input})

    display_string_stage1 = ""
    final_user_prompt_for_primary = user_input
    consultant_thinking = ""
    assistant_full_response_for_history = ""

    if sentiment_label == '消极':
        print("--- [决策]: 检测到消极情绪，启动顾问模型深度分析... ---")
        
        try:
            # ✨ 核心优化 1: 为顾问模型构建“指令注入式”提示词
            history_text = INTERACTOR_CONSULTANT.tokenizer.apply_chat_template(
                current_model_history,
                tokenize=False,
                add_generation_prompt=False
            )

            prompt_consultant = f"""【分析任务指令】
作为一名专业的心理分析顾问，你的任务是基于下方提供的完整对话历史，进行深入、结构化的思考，思考时不要关注时间定位，而应该更加关注事件本身，思考过程必须和之前是不一样的。
---
【完整对话历史】
{history_text}
<|im_start|>assistant
"""
            model_inputs_consultant = INTERACTOR_CONSULTANT.tokenizer([prompt_consultant], return_tensors="pt").to(DEVICE)
            
            # ✨ 核心优化 2: 流式输出顾问模型的思考过程
            streamer_consultant = TextIteratorStreamer(INTERACTOR_CONSULTANT.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
            
            gen_kwargs_consultant = {
                "input_ids": model_inputs_consultant.input_ids,
                "streamer": streamer_consultant,
                "max_new_tokens": 1024,
                "do_sample": True,
                "temperature": 0.1, # 使用低温以确保分析的稳定性和逻辑性
                "top_p": 0.9
            }
            
            generation_thread_consultant = Thread(target=INTERACTOR_CONSULTANT.model.generate, kwargs=gen_kwargs_consultant)
            generation_thread_consultant.start()
            
            # --- 修改部分：只流式输出 <think> 标签内的内容 ---
            consultant_full_response = ""
            last_displayed_thinking = ""
            chatbot_history[-1][1] = "🤔 **顾问模型正在思考...**"
            yield chatbot_history, model_history_state, sentiment_display_text

            for new_token in streamer_consultant:
                consultant_full_response += new_token
                
                # 尝试从当前完整响应中提取思考内容
                # 这个正则表达式会匹配 <think> 和 </think> 之间的所有内容，或者从 <think> 到字符串末尾的所有内容
                match = re.search(r"<think>(.*)", consultant_full_response, re.DOTALL)
                if match:
                    # 我们只关心 <think> 标签内部的内容
                    current_thinking_content = match.group(1)
                    
                    # 为了避免在流式输出中显示 </think>，我们在这里把它去掉
                    current_thinking_content = current_thinking_content.split("</think>")[0]

                    # 只有当思考内容更新时才更新UI，避免不必要的刷新
                    if current_thinking_content != last_displayed_thinking:
                        last_displayed_thinking = current_thinking_content
                        display_string_stage1 = f"🤔 **顾问模型思考过程:**\n\n{current_thinking_content}"
                        chatbot_history[-1][1] = display_string_stage1
                        yield chatbot_history, model_history_state, sentiment_display_text
            # --- 修改结束 ---

            # 流式输出结束后，提取最终的、完整的思考过程并进行格式化
            consultant_thinking = extract_thinking_process(consultant_full_response)
            
            if consultant_thinking:
                thinking_html = markdown.markdown(consultant_thinking)
                # 使用最终的、干净的HTML版本替换原始流
                display_string_stage1 = f"🤔 **顾问模型思考过程:**<div class='thinking-process'>{thinking_html}</div>\n\n---\n\n"
                chatbot_history[-1][1] = display_string_stage1
                yield chatbot_history, model_history_state, sentiment_display_text
                
                final_user_prompt_for_primary = f'这是来访者的问题:\n"{user_input}"\n\n这是心理顾问模型的分析和思考过程，请你参考这些思路每次只问一个问题，当遇到极端危机处理时，应立即让来访者转接专业心理咨询。不需要参考时间而是更多的关注问题的根源，然后直接以友善、专业，富有同情心，富含共情的口吻询问或者回答来访者。当问了5到7轮后应当及时给出建议不要再问问题不必再听取顾问的思路，然后直接以友善、专业，富有同情心，富含共情的口吻询问或者回答来访者:\n\n--- 顾问思路 ---\n{consultant_thinking}\n--- 思路结束 ---'
            else:
                print("--- [降级处理]: 顾问模型未提供有效思路，主模型将直接回答。---")
                display_string_stage1 = "" # 清空思考过程的显示
                chatbot_history[-1][1] = "" # 确保清除“正在思考”的消息
                yield chatbot_history, model_history_state, sentiment_display_text


        except Exception as e:
            print(f"错误：调用顾问模型时发生异常: {e}")
            chatbot_history[-1][1] = f"❌ 调用顾问模型时出错: {e}"
            yield chatbot_history, model_history_state, sentiment_display_text
    else:
        print("--- [决策]: 情绪为积极或中性，主模型将直接回复... ---")

    # --- 步骤 3: 统一调用主模型进行回复 ---
    primary_messages = current_model_history.copy()
    primary_messages[-1]['content'] = final_user_prompt_for_primary
    if primary_system_prompt:
        primary_messages.insert(0, {"role": "system", "content": primary_system_prompt})

    try:
        prompt_primary = INTERACTOR_PRIMARY.tokenizer.apply_chat_template(primary_messages, tokenize=False, add_generation_prompt=True)
        model_inputs_primary = INTERACTOR_PRIMARY.tokenizer([prompt_primary], return_tensors="pt").to(INTERACTOR_PRIMARY.model.device)
        streamer = TextIteratorStreamer(INTERACTOR_PRIMARY.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        
        # 主模型使用从UI传入的、适合对话的温度
        gen_kwargs = {"input_ids": model_inputs_primary["input_ids"], "streamer": streamer, "max_new_tokens": 2048, "do_sample": True, "temperature": temperature, "top_p": 0.9}
        
        generation_thread = Thread(target=INTERACTOR_PRIMARY.model.generate, kwargs=gen_kwargs)
        generation_thread.start()
        
        primary_full_response = ""
        for new_token in streamer:
            primary_full_response += new_token
            # 将主模型的回复追加到（现已最终确定的）第一阶段显示字符串之后
            display_string_stage2 = f"{display_string_stage1}🤖 **主模型回复:**\n\n{primary_full_response}"
            chatbot_history[-1][1] = display_string_stage2
            yield chatbot_history, model_history_state, sentiment_display_text
        
        if consultant_thinking:
            assistant_full_response_for_history = f"<think>\n{consultant_thinking}\n</think>\n{primary_full_response}"
        else:
            assistant_full_response_for_history = primary_full_response
        
        current_model_history.append({"role": "assistant", "content": assistant_full_response_for_history})
        yield chatbot_history, current_model_history, sentiment_display_text

    except Exception as e:
        print(f"错误：调用主模型时发生异常: {e}")
        chatbot_history[-1][1] = f"{display_string_stage1}\n\n❌ 调用主模型时出错: {e}"
        yield chatbot_history, model_history_state, sentiment_display_text

# --- 清空对话函数 ---
def clear_chat_and_sentiment():
    """清空聊天记录、输入框、模型历史状态和情感状态显示"""
    return None, None, [], "*等待用户输入...*", gr.update(visible=False) # 同时隐藏下载链接

# --- 新增功能：导出聊天记录 ---
def export_chat_history(history):
    """将模型历史记录导出为带时间戳的 JSON 文件。"""
    if not history:
        gr.Info("聊天记录为空，无法导出。")
        return None

    # 创建一个目录来存放导出的文件（如果不存在）
    export_dir = "chat_exports"
    os.makedirs(export_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(export_dir, f"chat_history_{timestamp}.json")
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)
    
    gr.Info(f"聊天记录已成功导出到: {filename}")
    return gr.File(value=filename, visible=True)


# ===================================================================
# Part 5: Gradio 界面定义
# ===================================================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;700&display=swap');
body, .gradio-container { font-family: 'Noto Sans SC', sans-serif !important; }
#chatbot { min-height: 600px; }
.thinking-process {
  color: #475569; background-color: #f8fafc; border-left: 4px solid #93c5fd;
  padding: 12px; margin-top: 8px; margin-bottom: 12px; border-radius: 4px;
}
.thinking-process p { margin: 0 0 8px 0; }
.thinking-process p:last-child { margin-bottom: 0; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), css=custom_css) as demo:
    # `gr.State` 用于在多次函数调用间存储干净的、符合模型格式的对话历史
    model_history = gr.State([])

    gr.Markdown("# 🤖 智能心理健康助手 (情感感知版)")
    gr.Markdown("系统会首先分析您的情绪。如果检测到**消极情绪**，将启动“顾问模型”进行深度思考；否则，“主模型”将直接、快速地进行回复。")

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="聊天窗口", bubble_full_width=False, elem_id="chatbot", render_markdown=True)
            user_input_box = gr.Textbox(show_label=False, placeholder="在这里输入您的问题，然后按回车键发送...", container=False)

        with gr.Column(scale=1):
            gr.Markdown("### 当前用户的情感")
            sentiment_display = gr.Markdown(label="用户当前情绪", value="*等待用户输入...*")
            temperature_slider = gr.Slider(minimum=0.01, maximum=1.99, value=0.5, step=0.01, label="主模型温度 (Temperature)")
            primary_system_prompt_box = gr.Textbox(
                label="主模型系统提示词",
                value="你是一位精通理情行为疗法（Rational Emotive Behavior Therapy，简称REBT）的心理咨询师，能够合理地采用理情行为疗法给来访者提供专业地指导和支持，缓解来访者的负面情绪和行为反应，帮助他们实现个人成长和心理健康。理情行为治疗主要包括以下几个阶段，下面是对话阶段列表，并简要描述了各个阶段的重点。\n（1）**检查非理性信念和自我挫败式思维**：理情行为疗法把认知干预视为治疗的“生命”，因此，几乎从治疗一开始，在问题探索阶段，咨询师就以积极的、说服教导式的态度帮助来访者探查隐藏在情绪困扰后面的原因，包括来访者理解事件的思维逻辑，产生情绪的前因后果，借此来明确问题的所在。咨询师坚定地激励来访者去反省自己在遭遇刺激事件后，在感到焦虑、抑郁或愤怒前对自己“说”了些什么。\n（2）**与非理性信念辩论**：咨询师运用多种技术（主要是认知技术）帮助来访者向非理性信念和思维质疑发难，证明它们的不现实、不合理之处，认识它们的危害进而产生放弃这些不合理信念的愿望和行为。\n（3）**得出合理信念，学会理性思维**：在识别并驳倒非理性信念的基础上，咨询师进一步诱导、帮助来访者找出对于刺激情境和事件的适宜的、理性的反应，找出理性的信念和实事求是的、指向问题解决的思维陈述，以此来替代非理性信念和自我挫bail式思维。为了巩固理性信念，咨询师要向来访者反复教导，证明为什么理性信念是合情合理的，它与非理性信念有什么不同，为什么非理性信念导致情绪失调，而理性信念导致较积极、健康的结果。\n（4）**迁移应用治疗收获**：积极鼓励来访者把在治疗中所学到的客观现实的态度，科学合理的思维方式内化成个人的生活态度，并在以后的生活中坚持不懈地按理情行为疗法的教导来解决新的问题。    你需要一步一步来，你一次最多问一个问题。需要富有同情心的回复用户的问题，并且当交流一段过程了解用户的具体情况后应该不要再问问题而是及时给出建议。", # 此处默认为空
                placeholder="请在此处输入主模型的系统提示词...",
                lines=10
            )
            with gr.Row():
                clear_button = gr.Button("🗑️ 清空对话", variant="stop", scale=1)
                export_button = gr.Button("📤 导出JSON", variant="secondary", scale=1) # 新增的导出按钮
            
            # 新增的文件下载组件，默认不可见
            download_file = gr.File(label="下载聊天记录", visible=False)


    # 定义Gradio事件处理
    user_input_box.submit(
        predict,
        inputs=[user_input_box, chatbot, model_history, primary_system_prompt_box, temperature_slider],
        outputs=[chatbot, model_history, sentiment_display]
    ).then(
        lambda: gr.update(value=""), outputs=[user_input_box] # 发送后清空输入框
    )

    clear_button.click(
        clear_chat_and_sentiment,
        inputs=[],
        outputs=[user_input_box, chatbot, model_history, sentiment_display, download_file] # 清空时也隐藏下载链接
    )

    # 新增：为导出按钮绑定事件
    export_button.click(
        fn=export_chat_history,
        inputs=[model_history],
        outputs=[download_file]
    )

# --- 启动App ---
if __name__ == "__main__":
    # 使用share=True会生成一个公网链接，方便分享
    demo.launch(share=True)
这段代码在干嘛