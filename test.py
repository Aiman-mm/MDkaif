import os
import logging
import uuid
import subprocess
import threading
import tempfile
import re
import json
from typing import Optional, Dict
from PIL import Image

from gtts import gTTS
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool

from dotenv import load_dotenv
import google.generativeai as genai
from faster_whisper import WhisperModel
from diffusers import StableDiffusionPipeline
import torch

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
FRONT_HTML = os.path.join(STATIC_DIR, "new2.html")

# --- Utility Functions ---
def unique_name(prefix: str, ext: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}.{ext}"

def cache_bust(url_path: str) -> str:
    return f"{url_path}?v={uuid.uuid4().hex}"

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

def format_structured(text: str, bullet_symbol: str = "⭐") -> str:
    if not text:
        return "- No response available."
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    formatted = []
    for para in paragraphs:
        if para.startswith("##") or para.startswith("*") or len(para.split()) < 12:
            formatted.append(para)
            continue
        sentences = SENTENCE_SPLIT.split(para)
        for s in sentences:
            s = s.strip()
            if s:
                formatted.append(f"{bullet_symbol} {s}")
    return "\n".join(formatted)

def generate_tts_audio(text: str, output_path: str):
    try:
        cleaned = str(text or "").strip() or "Sorry, I have no response."
        chunks = [cleaned[i:i+150] for i in range(0, len(cleaned), 150)]
        temp_files = []
        for chunk in chunks:
            temp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_mp3.close()
            gTTS(text=chunk, lang="en", slow=False).save(temp_mp3.name)
            temp_files.append(temp_mp3.name)

        list_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        try:
            with open(list_file.name, "w") as f:
                for tf in temp_files:
                    f.write(f"file '{os.path.abspath(tf)}'\n")
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_file.name, "-c:a", "libmp3lame", "-b:a", "128k", output_path],
                check=True, capture_output=True
            )
        finally:
            list_file.close()
            for tf in temp_files:
                try: os.remove(tf)
                except Exception: pass
            try: os.remove(list_file.name)
            except Exception: pass
    except Exception as e:
        logging.error(f"TTS error: {e}")

# --- Gemini setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")

def get_ai_response(prompt: str) -> str:
    try:
        response = GEMINI_MODEL.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        if response and hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts and parts[0].text:
                return parts[0].text.strip()
        return f"I understood your question: '{prompt}', but Gemini did not return a clear answer."
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return f"Sorry, I couldn't generate a response for: '{prompt}'."

# --- Whisper setup ---
logging.info("Loading Whisper model: tiny")
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# --- Stable Diffusion setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_pipeline_turbo():
    logging.info("Loading diffusion pipeline: stabilityai/sd-turbo")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float32 if device == "cpu" else torch.float16
    ).to(device)
    pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
    pipe.enable_attention_slicing()
    return pipe

pipe = load_pipeline_turbo()



def generate_image(prompt: str) -> Optional[Dict[str, str]]:
    if not prompt:
        logging.error("Image generation error: Prompt is missing.")
        return None
    try:
        image = pipe(
            prompt,
            guidance_scale=7.5,
            num_inference_steps=20,  # CPU-friendly (~30–60s)
            height=384,
            width=384
        ).images[0]
        filename = unique_name("generated", "png")
        img_path = os.path.join(STATIC_DIR, filename)
        image.save(img_path)
        return {"image_url": cache_bust(f"/static/{filename}")}
    except Exception as e:
        logging.error(f"Image generation error: {e}")
        return None

# --- FastAPI setup ---
app = FastAPI(title="Voice + Image Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Intent detection ---
TEXT_INTENT = {"explain", "describe", "illusion", "clarify", "meaning", "definition"}
IMAGE_INTENT = {"create", "generate", "draw", "illustrate", "design", "render", "sketch",
                "image", "picture", "diagram", "anatomy", "medical", "science", "logo",
                "poster", "fantasy", "cat", "dog", "animal", "landscape", "car", "house"}

def detect_intent(q: str):
    ql = q.lower()
    q_words = set(re.findall(r'\b\w+\b', ql))
    wants_text = bool(q_words.intersection(TEXT_INTENT))
    wants_image = bool(q_words.intersection(IMAGE_INTENT))
    if wants_image and any(word in ql for word in {"create", "generate", "draw", "illustrate"}):
        wants_text = False
    return wants_text, wants_image

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def root():
    if not os.path.exists(FRONT_HTML):
        raise HTTPException(status_code=404, detail="Frontend file 'new2.html' not found.")
    return FileResponse(FRONT_HTML)

@app.get("/audio-status/{filename}")
def audio_status(filename: str):
    path = os.path.join(STATIC_DIR, filename)
    return {"ready": os.path.exists(path)}





@app.post("/chat")
async def chat(question: str = Form(...)):
    try:
        q = (question or "").strip()
        if not q:
            return JSONResponse({"reply": "Please provide a question or prompt.", "audio": None})

        wants_text, wants_image = detect_intent(q)
        reply_text = None
        image_url = None

        if wants_image:
            image_result = await run_in_threadpool(generate_image, q)
            if image_result:
                image_url = image_result["image_url"]
                reply_text = "Here is the image you requested!"
            else:
                reply_text = "Image generation failed, providing text instead."
                wants_text = True
                wants_image = False

        if wants_text or (not wants_image and not image_url):
            answer = await run_in_threadpool(get_ai_response, q)
            reply_text = await run_in_threadpool(format_structured, answer)

        response = {"reply": reply_text}
        if image_url:
            response["image_url"] = image_url

        # Always generate audio for text replies
        if reply_text:
            filename = unique_name("answer", "mp3")
            output_mp3 = os.path.join(STATIC_DIR, filename)
            threading.Thread(
                target=generate_tts_audio,
                args=(reply_text, output_mp3),
                daemon=True
            ).start()
            response["audio"] = cache_bust(f"/static/{filename}")

        return JSONResponse(response)

    except Exception as e:
        logging.error(f"Chat route error: {e}")
        return JSONResponse({"reply": "Critical error occurred.", "audio": None}, status_code=500)
    



@app.post("/chat-audio")
async def chat_audio(file: UploadFile = File(...)):
    """Handles voice messages: transcribes audio, processes the question, and returns the response."""
    logging.info("Received audio file for transcription.")
    temp_audio_file = None
    try:
        # Save the uploaded audio file temporarily
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}")
        with open(temp_audio_file.name, "wb") as buffer:
            buffer.write(await file.read())

        # Transcribe audio using Whisper
        segments, _ = whisper_model.transcribe(temp_audio_file.name)
        transcript = " ".join(segment.text for segment in segments).strip()

        if not transcript:
            return JSONResponse({"reply": "I couldn't understand the audio. Please try speaking clearly.", "audio": None})

        logging.info(f"Transcription: {transcript}")

        # Process the transcribed text using the /chat logic
        response = await chat(question=transcript)

        # Add the transcript to the response before sending it back to the frontend
        response_data = response.body.decode()
        response_json = json.loads(response_data)
        response_json['transcript'] = transcript

        return JSONResponse(response_json)

    except Exception as e:
        logging.error(f"Audio transcription/chat error: {e}")
        return JSONResponse(
            {"reply": "An error occurred while processing your voice message.", "audio": None},
            status_code=500
        )
    finally:
        if temp_audio_file and os.path.exists(temp_audio_file.name):
            os.remove(temp_audio_file.name)
