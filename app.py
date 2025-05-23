import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, DynamicCache
import logging
import tempfile
import os

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Device Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# --- Model and Processor ---
model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path, torch_dtype=torch.float16
).to(device)

# --- Conversation State Management ---
class ConversationState:
    def __init__(self):
        self.history = []
        self.video_path = None
        self.video_type = None
        self.temp_dir = None
        self.cache = DynamicCache()

    def reset(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                if self.video_path and os.path.exists(self.video_path):
                    os.remove(self.video_path)
                os.rmdir(self.temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up: {e}")
        self.history, self.video_path, self.video_type, self.temp_dir = [], None, None, None
        self.cache = DynamicCache()

    def set_video(self, path, video_type, temp_dir=None):
        self.reset()
        self.video_path = path
        self.video_type = video_type
        self.temp_dir = temp_dir

conversation = ConversationState()

# --- Chat Processor ---
def process_video_chat(video_path, messages, use_cache=True):
    try:
        formatted_messages = [{
            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": messages[0]["content"]}
            ]
        }] + [
            {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
            for m in messages[1:]
        ]
        inputs = processor.apply_chat_template(
            formatted_messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.float16)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=256,
                past_key_values=conversation.cache if use_cache else None,
                return_dict_in_generate=True,
                use_cache=True
            )
        text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        return text.split("Assistant:")[-1].strip() if "Assistant:" in text else text.strip()

    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        return f"Error: {e}"


# --- Video Setup Handler ---
def setup_video(video_file):
    if video_file:
        conversation.set_video(video_file.name, "file")
        return "Video uploaded successfully!"
    return "Please provide a valid video file."

# --- Chat Handler ---
def chat(message, history):
    if not conversation.video_path:
        return "Please set up a video first."
    if not message.strip():
        return "Please enter a question."
    
    history_msgs = []
    for user, bot in history:
        history_msgs.extend([{"role": "user", "content": user}, {"role": "assistant", "content": bot}])
    history_msgs.append({"role": "user", "content": message})

    response = process_video_chat(conversation.video_path, history_msgs)
    history.append((message, response))
    return history
    

# --- UI Layout ---
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 Video Chat with SmolVLM2-2.2B-Instruct")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Video Setup")
            video_file = gr.File(file_types=[".mp4", ".webm", ".mkv", ".avi"], label="Upload Video File")
            setup_btn = gr.Button("Load Video")
            setup_status = gr.Textbox(interactive=False)
            reset_btn = gr.Button("Reset Chat & Video")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat History", height=500, bubble_full_width=False)
            with gr.Row():
                msg = gr.Textbox(placeholder="Ask something about the video...", scale=9)
                submit_btn = gr.Button("Send", scale=1)

    # Logic Wiring
    setup_btn.click(setup_video, inputs=[video_file], outputs=setup_status)
    reset_btn.click(lambda: (None, "", []), outputs=[video_file, setup_status, chatbot])
    msg.submit(chat, [msg, chatbot], [chatbot]).then(lambda: "", None, msg)
    submit_btn.click(chat, [msg, chatbot], [chatbot]).then(lambda: "", None, msg)

    gr.Markdown("""
    ### Instructions
    1. Upload a video file using the file uploader.
    2. Click 'Load Video' to load it.
    3. Ask questions in the chat.
    4. The model will answer and remember context.

    ### Model: SmolVLM2-2.2B-Instruct
    """)

demo.queue()
demo.launch(share=True)