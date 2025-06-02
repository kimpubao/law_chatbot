import gradio as gr
import os

model_choices = [
    "KoAlpaca-7B-LoRA",
    "KoAlpaca-12.8B",
    "Llama-3-Open-Ko-8B",
    "Mistral-7B-Instruct"
]

chat_history = []

def model_response(user_input, model, file):
    response = f"[{model}] 응답: '{user_input}'에 대한 답변입니다."
    if file:
        response += f"\n📎 첨부된 파일: {file.name}"
    chat_history.append(("사용자", user_input))
    chat_history.append((f"챗봇 [{model}]", response))
    return format_history(chat_history), "", None

def format_history(history):
    return "\n\n".join([f"**{sender}**: {msg}" for sender, msg in history])

def build_interface():
    with gr.Blocks(title="세무톡") as demo:
        with gr.Row():
            gr.Markdown("<h1 style='text-align: left;'><span style='font-size: 24px;'>💼 TAX Helper</span></h1>", elem_id="title")
            with gr.Column():
                gr.Markdown("<label style='font-weight: bold;'>모델 선택:</label>")
                model_dropdown = gr.Dropdown(choices=model_choices, value=model_choices[0], label=None)

        chatbot_output = gr.Markdown("안녕하세요! 세무 상담 도우미입니다. 무엇을 도와드릴까요?", elem_id="chatbox")

        with gr.Row():
            user_input = gr.Textbox(placeholder="세무 관련 질문을 입력하세요...", lines=2, label=None)
            send_btn = gr.Button("▶", elem_id="send-btn")

        file_upload = gr.File(label="＋", file_types=[".pdf", ".xls", ".xlsx", ".csv", ".doc", ".docx", ".hwp", ".ppt", ".pptx", ".jpg", ".jpeg", ".png"], interactive=True)

        send_btn.click(fn=model_response, inputs=[user_input, model_dropdown, file_upload], outputs=[chatbot_output, user_input, file_upload])

    return demo

demo = build_interface()
demo.launch(server_name="0.0.0.0", server_port=7861)