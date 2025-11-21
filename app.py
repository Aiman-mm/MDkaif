import os
import logging
import time
import base64
import requests
import subprocess
import threading
from gtts import gTTS

from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import google.generativeai as genai
from faster_whisper import WhisperModel

# Allow duplicate OpenMP runtime (workaround for Whisper on Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

FRONT_HTML = os.path.join(STATIC_DIR, "new.html")

# Gemini setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_TEXT_MODEL = "gemini-2.5-flash"

# Whisper setup (use "tiny" for speed, "small" for accuracy)
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# Hugging Face setup
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")

# FastAPI setup
app = FastAPI(title="Voice + Image Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Utility Functions ---
def get_ai_response(prompt: str) -> str:
    """Get text response from Gemini API."""
    try:
        model = genai.GenerativeModel(GEMINI_TEXT_MODEL)
        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts and hasattr(parts[0], "text"):
                return parts[0].text.strip()
        return "No response text found."
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "Sorry, I couldn't generate a response."


def format_structured(text: str, bullet_symbol: str = "⭐\n") -> str:
    """Format Gemini text into structured Markdown with optional bullet symbol."""
    if not text:
        return "\n\n- No response available."

    paragraphs = text.split("\n")
    formatted = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if para.startswith("##"):
            formatted.append(para)
            continue
        if len(para.split()) < 12:
            formatted.append(para)
            continue
        sentences = para.split(". ")
        for s in sentences:
            s = s.strip()
            if s:
                formatted.append(f"{bullet_symbol} {s}")

    return "\n".join(formatted)


def generate_tts_audio(text: str, output_path: str):
    """Generate MP3 audio using gTTS, splitting long text into chunks."""
    try:
        cleaned = (text or "").strip()
        if not cleaned:
            cleaned = "Sorry, I have no response."

        # Split into chunks (~250 chars each)
        chunks = [cleaned[i:i+250] for i in range(0, len(cleaned), 250)]
        temp_files = []

        for idx, chunk in enumerate(chunks):
            tts = gTTS(text=chunk, lang="en", slow=False)
            temp_file = f"{output_path}_{idx}.mp3"
            tts.save(temp_file)
            temp_files.append(temp_file)

        # Concatenate with ffmpeg
        list_file = f"{output_path}_list.txt"
        with open(list_file, "w") as f:
            for tf in temp_files:
                f.write(f"file '{os.path.abspath(tf)}'\n")

        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file, "-c", "copy", output_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Cleanup
        for tf in temp_files:
            os.remove(tf)
        os.remove(list_file)

        logging.info(f"TTS saved: {output_path}")
    except Exception as e:
        logging.error(f"TTS error: {e}")


def generate_image_hf(prompt: str):
    """Generate image using Hugging Face Router Inference API."""
    if not HF_API_TOKEN:
        logging.error("HF_API_TOKEN not set. Cannot generate image.")
        return None
    try:
        url = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Accept": "image/png",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "guidance_scale": 7.5,
                "num_inference_steps": 20
            }
        }
        response = requests.post(url, headers=headers, json=payload, timeout=120)

        if response.status_code != 200:
            logging.error(f"Hugging Face error {response.status_code}: {response.text}")
            return None

        img_bytes = response.content
        filename = f"generated_{int(time.time())}.png"
        img_path = os.path.join(STATIC_DIR, filename)
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return {
            "reply": f"## 🎨 Generated Image\n\n⭐ Prompt: {prompt}",
            "image_url": f"/static/{filename}",
            "image_base64": img_b64
        }

    except Exception as e:
        logging.error(f"Hugging Face image error: {e}")
        return None


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse(FRONT_HTML)


@app.get("/audio-status/{filename}")
def audio_status(filename: str):
    """Check if the generated audio file is ready."""
    path = os.path.join(STATIC_DIR, filename)
    return {"ready": os.path.exists(path)}


@app.post("/chat")
async def chat(question: str = Form(...), need_audio: bool = Form(False)):
    """Text input: image generation or Gemini structured reply."""
    try:
        q_lower = question.lower()
        image_keywords = ["generate", "create", "draw", "illustrate", "design", "picture", "image", "art"]

        if any(k in q_lower for k in image_keywords):
            image_result = generate_image_hf(question)
            if image_result:
                return JSONResponse(image_result)
            else:
                return JSONResponse({"reply": "## ❌ Error\n\n⭐ Couldn't generate an image."})

        answer = get_ai_response(question)
        structured = format_structured(answer)

        if need_audio:
            filename = f"answer_{int(time.time())}.mp3"
            output_mp3 = os.path.join(STATIC_DIR, filename)
            threading.Thread(target=generate_tts_audio, args=(answer, output_mp3), daemon=True).start()
            return JSONResponse({"reply": structured, "audio": f"/static/{filename}"})
        else:
            return JSONResponse({"reply": structured})

    except Exception as e:
        logging.error(f"Chat route error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/chat-audio")
async def chat_audio(file: UploadFile = File(...), need_audio: bool = Form(False)):
    try:
        # Save uploaded file
        input_path = os.path.join(STATIC_DIR, "live_input.webm")
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # Convert to WAV for Whisper
        wav_path = os.path.join(STATIC_DIR, "live_input.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
        ], check=True)

        # Transcribe with Whisper
        segments, _ = whisper_model.transcribe(wav_path)
        transcript = " ".join([seg.text for seg in segments]).strip()
        if not transcript:
            transcript = "hi"

        q_lower = transcript.lower()
        image_keywords = ["generate", "create", "draw", "illustrate", "design", "picture", "image", "art"]

        # If audio request is for image
        if any(k in q_lower for k in image_keywords):
            image_result = generate_image_hf(transcript)
            if image_result:
                return JSONResponse(image_result)
            else:
                                return JSONResponse({"reply": "## ❌ Error\n\n⭐ Couldn't generate an image."})

        # Otherwise, Gemini text reply
        answer = get_ai_response(transcript)
        structured = format_structured(answer)

        if need_audio:
            filename = f"answer_{int(time.time())}.mp3"
            output_mp3 = os.path.join(STATIC_DIR, filename)
            threading.Thread(
                target=generate_tts_audio,
                args=(answer, output_mp3),
                daemon=True
            ).start()
            return JSONResponse({
                "transcript": transcript,
                "reply": structured,
                "audio": f"/static/{filename}"
            })
        else:
            return JSONResponse({
                "transcript": transcript,
                "reply": structured
            })

    except Exception as e:
        logging.error(f"Chat-audio route error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")







"""import os
import logging
import time
import base64
import requests
import subprocess
from gtts import gTTS

from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import google.generativeai as genai
from faster_whisper import WhisperModel

# Allow duplicate OpenMP runtime (workaround for Whisper on Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

FRONT_HTML = os.path.join(STATIC_DIR, "new.html")

# Gemini setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_TEXT_MODEL = "gemini-2.5-flash"

# Whisper setup
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# Hugging Face setup
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")

# FastAPI setup
app = FastAPI(title="Voice + Image Chatbot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Utility Functions ---
def get_ai_response(prompt: str) -> str:
 
    try:
        model = genai.GenerativeModel(GEMINI_TEXT_MODEL)
        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts and hasattr(parts[0], "text"):
                return parts[0].text.strip()
        return "No response text found."
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "Sorry, I couldn't generate a response."


def format_structured(text: str) -> str:

    if not text:
        return "\n\n- No response available."

    bullet_symbol = "⭐"
    paragraphs = text.split("\n")
    formatted = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if para.startswith("##"):
            formatted.append(para)
            continue
        if len(para.split()) < 12:
            formatted.append(para)
            continue
        sentences = para.split(". ")
        for s in sentences:
            s = s.strip()
            if s:
                formatted.append(f"{bullet_symbol} {s}")

    return "\n".join(formatted)


def generate_tts_audio(text: str, output_path: str):

    try:
        cleaned = (text or "").strip()
        if not cleaned:
            cleaned = "Sorry, I have no response."
        tts = gTTS(text=cleaned, lang="en", slow=False)
        tts.save(output_path)
    except Exception as e:
        logging.error(f"TTS error: {e}")
        raise


def generate_image_hf(prompt: str):

    if not HF_API_TOKEN:
        logging.error("HF_API_TOKEN not set. Cannot generate image.")
        return None
    try:
        url = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Accept": "image/png",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "guidance_scale": 7.5,
                "num_inference_steps": 50
            }
        }
        response = requests.post(url, headers=headers, json=payload, timeout=120)

        if response.status_code != 200:
            logging.error(f"Hugging Face error {response.status_code}: {response.text}")
            return None

        img_bytes = response.content
        filename = f"generated_{int(time.time())}.png"
        img_path = os.path.join(STATIC_DIR, filename)
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return {
            "reply": f"## 🎨 Generated Image\n\n- Prompt: {prompt}",
            "image_url": f"/static/{filename}",
            "image_base64": img_b64
        }

    except Exception as e:
        logging.error(f"Hugging Face image error: {e}")
        return None


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse(FRONT_HTML)


@app.post("/chat-audio")
async def chat_audio(file: UploadFile = File(...)):


    try:
        # Save uploaded file
        input_path = os.path.join(STATIC_DIR, "live_input.webm")
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # Convert to WAV for Whisper
        wav_path = os.path.join(STATIC_DIR, "live_input.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
        ], check=True)

        # Transcribe with Whisper
        segments, _ = whisper_model.transcribe(wav_path)
        transcript = " ".join([seg.text for seg in segments]).strip()

        # Get Gemini reply
        answer = get_ai_response(transcript)
        structured = format_structured(answer)

        # Generate unique MP3 filename
        filename = f"answer_{int(time.time())}.mp3"
        output_mp3 = os.path.join(STATIC_DIR, filename)
        generate_tts_audio(answer, output_mp3)

        return JSONResponse({
            "transcript": transcript,
            "reply": structured,
            "audio": f"/static/{filename}"
        })

    except Exception as e:
        logging.error(f"Chat-audio route error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/favicon.ico")
def favicon():
    path = os.path.join(STATIC_DIR, "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path)
    return HTMLResponse(content="", status_code=204)
""""""
"""
"""import os
import logging
import time
import base64
import requests
from gtts import gTTS

from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import google.generativeai as genai
from faster_whisper import WhisperModel

# Allow duplicate OpenMP runtime (workaround for Whisper on Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

OUTPUT_MP3 = os.path.join(STATIC_DIR, "live_answer.mp3")
FRONT_HTML = os.path.join(STATIC_DIR, "new.html")

# Gemini setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_TEXT_MODEL = "gemini-2.5-flash"

# Whisper setup
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# Hugging Face setup
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")

# FastAPI setup
app = FastAPI(title="Voice + Image Chatbot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Utility Functions ---
def get_ai_response(prompt: str) -> str:

    try:
        model = genai.GenerativeModel(GEMINI_TEXT_MODEL)
        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts and hasattr(parts[0], "text"):
                return parts[0].text.strip()
        return "No response text found."
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "Sorry, I couldn't generate a response."

def format_structured(text: str) -> str:

    if not text:
        return "\n\n- No response available."

    # Define custom bullet symbol
    bullet_symbol = "⭐"   # you can change to "✔️", "•", "→", etc.

    paragraphs = text.split("\n")
    formatted = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Preserve Gemini's headings
        if para.startswith("##"):
            formatted.append(para)
            continue

        # Keep short greetings plain
        if len(para.split()) < 12:
            formatted.append(para)
            continue

        # Otherwise bulletize sentences with custom symbol
        sentences = para.split(". ")
        for s in sentences:
            s = s.strip()
            if s:
                formatted.append(f"{bullet_symbol} {s}")

    return "\n".join(formatted)





def generate_tts_audio(text: str, output_path: str):

    try:
        cleaned = (text or "").strip()
        if not cleaned:
            cleaned = "Sorry, I have no response."
        tts = gTTS(text=cleaned, lang="en", slow=False)
        tts.save(output_path)
    except Exception as e:
        logging.error(f"TTS error: {e}")
        raise


def generate_image_hf(prompt: str):

    if not HF_API_TOKEN:
        logging.error("HF_API_TOKEN not set. Cannot generate image.")
        return None
    try:
        url = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Accept": "image/png",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "guidance_scale": 7.5,
                "num_inference_steps": 50
            }
        }
        response = requests.post(url, headers=headers, json=payload, timeout=120)

        if response.status_code != 200:
            logging.error(f"Hugging Face error {response.status_code}: {response.text}")
            return None

        img_bytes = response.content
        filename = f"generated_{int(time.time())}.png"
        img_path = os.path.join(STATIC_DIR, filename)
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return {
            "reply": f"## 🎨 Generated Image\n\n- Prompt: {prompt}",
            "image_url": f"/static/{filename}",
            "image_base64": img_b64
        }

    except Exception as e:
        logging.error(f"Hugging Face image error: {e}")
        return None


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse(FRONT_HTML)
import subprocess

@app.post("/chat-audio")
async def chat_audio(file: UploadFile = File(...)):

    try:
        # Save uploaded file
        input_path = os.path.join(STATIC_DIR, "live_input.webm")
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # Convert to WAV for Whisper
        wav_path = os.path.join(STATIC_DIR, "live_input.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm16", wav_path
        ], check=True)

        # Transcribe with Whisper
        segments, _ = whisper_model.transcribe(wav_path)
        transcript = " ".join([seg.text for seg in segments]).strip()

        # Get Gemini reply
        answer = get_ai_response(transcript)
        structured = format_structured(answer)

        # Generate unique MP3 filename
        filename = f"answer_{int(time.time())}.mp3"
        output_mp3 = os.path.join(STATIC_DIR, filename)
        generate_tts_audio(answer, output_mp3)

        return JSONResponse({
            "transcript": transcript,
            "reply": structured,
            "audio": f"/static/{filename}"
        })

    except Exception as e:
        logging.error(f"Chat-audio route error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/chat")
async def chat(question: str = Form(...)):

    try:
        q_lower = question.lower()
        image_keywords = ["generate", "create", "draw", "illustrate", "design", "picture", "image", "art"]

        if any(k in q_lower for k in image_keywords):
            image_result = generate_image_hf(question)
            if image_result:
                return JSONResponse(image_result)
            else:
                return JSONResponse({"reply": "## ❌ Error\n\n- Couldn't generate an image."})

        # Otherwise, use Gemini for text reply
        start = time.time()
        answer = get_ai_response(question)
        structured = format_structured(answer)
        return JSONResponse({
            "reply": structured,
            "llm_time": round(time.time() - start, 2)
        })

    except Exception as e:
        logging.error(f"Chat route error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/chat-audio")
async def chat_audio(file: UploadFile = File(...)):

    try:
        input_path = os.path.join(STATIC_DIR, "live_input.webm")
        with open(input_path, "wb") as f:
            f.write(await file.read())

        segments, _ = whisper_model.transcribe(input_path)
        transcript = " ".join([seg.text for seg in segments]).strip()

        answer = get_ai_response(transcript)
        structured = format_structured(answer)

        generate_tts_audio(answer, OUTPUT_MP3)

        return JSONResponse({
            "transcript": transcript,
            "reply": structured,
            "audio": "/static/live_answer.mp3"
        })

    except Exception as e:
        logging.error(f"Chat-audio route error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/static/live_answer.mp3")
async def get_audio():
    if not os.path.exists(OUTPUT_MP3):
        raise HTTPException(status_code=404, detail="Audio not found.")
    return FileResponse(OUTPUT_MP3, media_type="audio/mpeg")


@app.get("/favicon.ico")
def favicon():
    path = os.path.join(STATIC_DIR, "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path)
    return HTMLResponse(content="", status_code=204)
"""