import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, DynamicCache
import logging
import tempfile
import os
import yt_dlp

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
        self.youtube_url = None
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
        self.history, self.video_path, self.video_type, self.youtube_url, self.temp_dir = [], None, None, None, None
        self.cache = DynamicCache()

    def set_video(self, path, video_type, youtube_url=None, temp_dir=None):
        self.reset()
        self.video_path = path
        self.video_type = video_type
        self.youtube_url = youtube_url
        self.temp_dir = temp_dir

conversation = ConversationState()

# --- YouTube Video Downloader ---
def download_youtube_video(url):
    temp_dir = tempfile.mkdtemp()
    filepath = os.path.join(temp_dir, "video.mp4")
    ydl_opts = {
        'format': 'best[height<=720]',
        'outtmpl': filepath,
        'quiet': False
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        if not os.path.exists(filepath):
            files = [f for f in os.listdir(temp_dir) if f.endswith(('.mp4', '.webm', '.mkv'))]
            if files:
                filepath = os.path.join(temp_dir, files[0])
            else:
                raise FileNotFoundError("No video file found after download.")
        return filepath, temp_dir
    except Exception as e:
        logger.error(f"Download error: {e}")
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        raise e

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
def setup_video(input_type, video_file, youtube_url):
    if input_type == "Upload a Video File" and video_file:
        conversation.set_video(video_file.name, "file")
        return "Video uploaded successfully!"
    elif input_type == "Paste a YouTube Link" and youtube_url.strip():
        try:
            filepath, temp_dir = download_youtube_video(youtube_url)
            conversation.set_video(filepath, "youtube", youtube_url, temp_dir)
            return "YouTube video loaded successfully!"
        except Exception as e:
            return f"Error: {e}"
    return "Please provide a valid video file or YouTube URL."

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
            input_type = gr.Radio(["Upload a Video File", "Paste a YouTube Link"], value="Upload a Video File")
            with gr.Row():
                video_file = gr.File(file_types=[".mp4", ".webm", ".mkv", ".avi"])
                youtube_url = gr.Textbox(visible=False)
            setup_btn = gr.Button("Set Video")
            setup_status = gr.Textbox(interactive=False)
            reset_btn = gr.Button("Reset Chat & Video")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat History", height=500, bubble_full_width=False)
            with gr.Row():
                msg = gr.Textbox(placeholder="Ask something about the video...", scale=9)
                submit_btn = gr.Button("Send", scale=1)

    # Logic Wiring
    input_type.change(
        lambda t: (gr.update(visible=t == "Upload a Video File"), gr.update(visible=t == "Paste a YouTube Link")),
        inputs=input_type,
        outputs=[video_file, youtube_url]
    )
    setup_btn.click(setup_video, inputs=[input_type, video_file, youtube_url], outputs=setup_status)
    reset_btn.click(lambda: (None, None, "", []), outputs=[video_file, youtube_url, setup_status, chatbot])
    msg.submit(chat, [msg, chatbot], [chatbot]).then(lambda: "", None, msg)
    submit_btn.click(chat, [msg, chatbot], [chatbot]).then(lambda: "", None, msg)

    gr.Markdown("""
    ### Instructions
    1. Choose to upload a video or paste a YouTube URL.
    2. Click 'Set Video' to load it.
    3. Ask questions in the chat.
    4. The model will answer and remember context.

    ### Model: SmolVLM2-2.2B-Instruct
    """)

demo.queue()
demo.launch(debug=True)

