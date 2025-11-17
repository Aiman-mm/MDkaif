import os
import logging
import subprocess
import asyncio
import base64
import json
import requests
from typing import Optional, Tuple
from fastapi import FastAPI, HTTPException, WebSocket, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
import google.generativeai as genai
import pyttsx3
from dotenv import load_dotenv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

INPUT_WEBM = "live_input.webm"
INPUT_WAV = "live_input.wav"
OUTPUT_MP3 = "live_answer.mp3"
PLACEHOLDER_PATH = "static/placeholder.png"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

genai.configure(api_key=GEMINI_API_KEY)

# Text model
GEMINI_TEXT_MODEL = "gemini-2.5-flash"

# Whisper
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

app = FastAPI(title="Gemini Voice Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Ensure placeholder exists ---
from PIL import Image, ImageDraw
if not os.path.exists(PLACEHOLDER_PATH):
    os.makedirs("static", exist_ok=True)
    img = Image.new("RGB", (512, 512), color="white")
    d = ImageDraw.Draw(img)
    d.text((50, 250), "Diagram not available", fill="black")
    img.save(PLACEHOLDER_PATH)

# --- Utility Functions ---
def clean_files():
    for f in [INPUT_WEBM, INPUT_WAV, OUTPUT_MP3]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception as e:
                logging.warning(f"Failed to remove {f}: {e}")

def get_ai_response(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(GEMINI_TEXT_MODEL)
        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text:
            return response.text
        return "No response text found."
    except Exception as e:
        logging.error(f"Gemini text error: {e}")
        return "Sorry, I couldn't generate a response."

def url_to_base64(image_url: str) -> Optional[Tuple[str, str]]:
    """Fetch an image from a URL and return base64 string + MIME type."""
    try:
        resp = requests.get(image_url)
        resp.raise_for_status()
        mime_type = resp.headers.get("Content-Type", "image/png")  # default
        img_b64 = base64.b64encode(resp.content).decode("utf-8")
        return img_b64, mime_type
    except Exception as e:
        logging.error(f"Image fetch error: {e}")
        return None

def maybe_generate_image(prompt: str):

    if "solar system" in prompt.lower():
        image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c3/Solar_sys.jpg"
    elif "digestive system" in prompt.lower():
        image_url = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Digestive_system_diagram_en.svg"
    elif "water cycle" in prompt.lower():
        image_url = "https://upload.wikimedia.org/wikipedia/commons/1/15/Water_cycle.png"
    else:
        image_url = "https://upload.wikimedia.org/wikipedia/commons/5/5c/Water_cycle_diagram.png"


    img_data = url_to_base64(image_url)
    if img_data:
        img_b64, mime_type = img_data
        return {"image_base64": img_b64, "mime_type": mime_type, "image_url": None}
    else:
        return {"image_base64": None, "mime_type": None, "image_url": None}

def generate_tts_audio(text: str, output_path: str):
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, output_path)
        engine.runAndWait()
    except Exception as e:
        logging.error(f"TTS error: {e}")

# --- WebSocket Endpoint ---
@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_data = bytearray()

    try:
        clean_files()
        while True:
            message = await asyncio.wait_for(websocket.receive(), timeout=10.0)
            if message.get("bytes"):
                audio_data.extend(message["bytes"])
            elif message.get("text"):
                data = json.loads(message["text"])
                if data.get("done"):
                    break
            if len(audio_data) > 2_000_000:
                break
    except Exception as e:
        logging.info(f"WebSocket closed or timeout: {e}")

    if not audio_data:
        await websocket.send_json({"error": "No audio received."})
        await websocket.close()
        return

    with open(INPUT_WEBM, "wb") as f:
        f.write(audio_data)

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-probesize", "5000000",
            "-fflags", "+genpts",
            "-i", INPUT_WEBM,
            "-c:a", "pcm_s16le",
            "-ar", "48000",
            "-ac", "1",
            INPUT_WAV
        ], check=True)
    except subprocess.CalledProcessError as e:
        await websocket.send_json({"error": f"FFmpeg failed: {e}"})
        await websocket.close()
        return

    segments, _ = whisper_model.transcribe(INPUT_WAV)
    transcription = " ".join([seg.text.strip() for seg in segments if seg.text]) or "No clear speech detected."

    answer = await asyncio.to_thread(get_ai_response, transcription)
    await asyncio.to_thread(generate_tts_audio, answer, OUTPUT_MP3)
    image_result = await asyncio.to_thread(maybe_generate_image, transcription)

    await websocket.send_json({
        "transcription": transcription,
        "answer": answer,
        "audio_url": "/live_answer.mp3",
        "image_base64": image_result["image_base64"],
        "mime_type": image_result["mime_type"],
        "image_url": image_result["image_url"]
    })
    await websocket.close()

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse("static/index.html")

@app.get("/favicon.ico")
def favicon():
    if not os.path.exists("static/favicon.ico"):
        return HTMLResponse(content="", status_code=204)
    return FileResponse("static/favicon.ico")

@app.get("/live_answer.mp3")
async def get_audio():
    if not os.path.exists(OUTPUT_MP3):
        raise HTTPException(status_code=404, detail="Audio not found.")
    return FileResponse(OUTPUT_MP3, media_type="audio/mpeg")

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    try:
        answer = await asyncio.to_thread(get_ai_response, question)
        await asyncio.to_thread(generate_tts_audio, answer, OUTPUT_MP3)
        image_result = await asyncio.to_thread(maybe_generate_image, question)

        return {
            "question": question,
            "answer": answer,
            "audio_url": "/live_answer.mp3",
            "image_base64": image_result["image_base64"],
            "mime_type": image_result["mime_type"],
            "image_url": image_result["image_url"]
        }
    except Exception as e:
        logging.error(f"Ask route error: {e}")
        return {"error": "Internal Server Error", "details": str(e)}

def url_to_base64(image_url: str) -> Optional[tuple[str, str]]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        resp = requests.get(image_url, headers=headers)
        resp.raise_for_status()
        mime_type = resp.headers.get("Content-Type", "image/png")
        img_b64 = base64.b64encode(resp.content).decode("utf-8")
        return img_b64, mime_type
    except Exception as e:
        logging.error(f"Image fetch error: {e}")
        return None


"""import os
import logging
import subprocess
import asyncio
import base64
import json
from typing import Optional
from fastapi import FastAPI, HTTPException, WebSocket, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
import google.generativeai as genai
import pyttsx3
from dotenv import load_dotenv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

INPUT_WEBM = "live_input.webm"
INPUT_WAV = "live_input.wav"
OUTPUT_MP3 = "live_answer.mp3"
PLACEHOLDER_PATH = "static/placeholder.png"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

genai.configure(api_key=GEMINI_API_KEY)

# Text model
GEMINI_TEXT_MODEL = "gemini-2.5-flash"
# Image-capable model
GEMINI_IMAGE_MODEL = "models/gemini-2.5-flash-image"

# Whisper
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

app = FastAPI(title="Gemini Voice Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Ensure placeholder exists ---
from PIL import Image, ImageDraw
if not os.path.exists(PLACEHOLDER_PATH):
    os.makedirs("static", exist_ok=True)
    img = Image.new("RGB", (512, 512), color="white")
    d = ImageDraw.Draw(img)
    d.text((50, 250), "Diagram not available", fill="black")
    img.save(PLACEHOLDER_PATH)

# --- Utility Functions ---
def clean_files():
    for f in [INPUT_WEBM, INPUT_WAV, OUTPUT_MP3]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception as e:
                logging.warning(f"Failed to remove {f}: {e}")

def get_ai_response(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(GEMINI_TEXT_MODEL)
        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text:
            return response.text
        return "No response text found."
    except Exception as e:
        logging.error(f"Gemini text error: {e}")
        return "Sorry, I couldn't generate a response."

def read_local_placeholder_b64() -> str:
    try:
        with open(PLACEHOLDER_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        # Transparent 1x1 PNG fallback
        return (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
            "ASsJTYQAAAAASUVORK5CYII="
        )

def generate_image(prompt: str) -> Optional[str]:

    try:
        model = genai.GenerativeModel(GEMINI_IMAGE_MODEL)
        response = model.generate_content(f"Create a clear diagram for: {prompt}")

        # Extract inline image data
        for cand in getattr(response, "candidates", []):
            content = getattr(cand, "content", None)
            if content and hasattr(content, "parts"):
                for part in content.parts:
                    inline = getattr(part, "inline_data", None)
                    if inline and inline.mime_type.startswith("image/"):
                        return inline.data  # base64 string
        logging.warning("No image found in response.")
        return None
    except Exception as e:
        logging.error(f"Image generation error: {e}")
        return None

def maybe_generate_image(prompt: str):


    trigger_words = ["explain", "diagram", "illustrate", "show", "draw", "visualize", "flowchart"]
    if any(word in prompt.lower() for word in trigger_words):
        img_b64 = generate_image(prompt)
        if img_b64:
            return {"image_base64": img_b64, "image_url": None}
        else:
            # Fallback: Google Images search link
            search_url = f"https://www.google.com/search?tbm=isch&q={prompt}+diagram"
            return {"image_base64": None, "image_url": search_url}
    return {"image_base64": None, "image_url": None}

def generate_tts_audio(text: str, output_path: str):
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, output_path)
        engine.runAndWait()
    except Exception as e:
        logging.error(f"TTS error: {e}")

# --- WebSocket Endpoint ---
@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_data = bytearray()

    try:
        clean_files()
        while True:
            message = await asyncio.wait_for(websocket.receive(), timeout=10.0)
            if message.get("bytes"):
                audio_data.extend(message["bytes"])
            elif message.get("text"):
                data = json.loads(message["text"])
                if data.get("done"):
                    break
            if len(audio_data) > 2_000_000:
                break
    except Exception as e:
        logging.info(f"WebSocket closed or timeout: {e}")

    if not audio_data:
        await websocket.send_json({"error": "No audio received."})
        await websocket.close()
        return

    with open(INPUT_WEBM, "wb") as f:
        f.write(audio_data)

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-probesize", "5000000",
            "-fflags", "+genpts",
            "-i", INPUT_WEBM,
            "-c:a", "pcm_s16le",
            "-ar", "48000",
            "-ac", "1",
            INPUT_WAV
        ], check=True)
    except subprocess.CalledProcessError as e:
        await websocket.send_json({"error": f"FFmpeg failed: {e}"})
        await websocket.close()
        return

    segments, _ = whisper_model.transcribe(INPUT_WAV)
    transcription = " ".join([seg.text.strip() for seg in segments if seg.text]) or "No clear speech detected."

    answer = await asyncio.to_thread(get_ai_response, transcription)
    await asyncio.to_thread(generate_tts_audio, answer, OUTPUT_MP3)
    image_result = await asyncio.to_thread(maybe_generate_image, transcription)

    await websocket.send_json({
        "transcription": transcription,
        "answer": answer,
        "audio_url": "/live_answer.mp3",
        "image_base64": image_result["image_base64"],
        "image_url": image_result["image_url"]
    })
    await websocket.close()

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse("static/index.html")

@app.get("/favicon.ico")
def favicon():
    if not os.path.exists("static/favicon.ico"):
        return HTMLResponse(content="", status_code=204)
    return FileResponse("static/favicon.ico")

@app.get("/live_answer.mp3")
async def get_audio():
    if not os.path.exists(OUTPUT_MP3):
        raise HTTPException(status_code=404, detail="Audio not found.")
    return FileResponse(OUTPUT_MP3, media_type="audio/mpeg")

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    try:
        answer = await asyncio.to_thread(get_ai_response, question)
        await asyncio.to_thread(generate_tts_audio, answer, OUTPUT_MP3)
        image_result = await asyncio.to_thread(maybe_generate_image, question)

        return {
            "question": question,
            "answer": answer,
            "audio_url": "/live_answer.mp3",
            "image_base64": image_result["image_base64"],
            "image_url": image_result["image_url"]
        }
    except Exception as e:
        logging.error(f"Ask route error: {e}")
        return {"error": "Internal Server Error", "details": str(e)}




"""





"""
import google.generativeai as genai
import os
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

for m in genai.list_models():
    print(m.name, m.supported_generation_methods)
"""



"""import os
import logging
import subprocess
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
import google.generativeai as genai
import pyttsx3
from dotenv import load_dotenv
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from faster_whisper import WhisperModel

from fastapi import WebSocket



app = FastAPI()

# Mount static files to a specific path
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

INPUT_WEBM = "live_input.webm"
INPUT_WAV = "live_input.wav"
OUTPUT_MP3 = "live_answer.mp3"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-2.5-flash"
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

app = FastAPI(title="Gemini Voice Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_data = bytearray()

    try:
        clean_files()
        while True:
            chunk = await asyncio.wait_for(websocket.receive_bytes(), timeout=10.0)
            audio_data.extend(chunk)
            if len(audio_data) > 2_000_000:
                break
    except Exception:
        pass

    if not audio_data:
        await websocket.send_json({"error": "No audio received."})
        await websocket.close()
        return

    # Save received audio to WebM file
    with open(INPUT_WEBM, "wb") as f:
        f.write(audio_data)

    # 🔧 FFmpeg conversion with robust flags
    subprocess.run([
        "ffmpeg",
        "-probesize", "5000000",
        "-fflags", "+genpts",
        "-i", INPUT_WEBM,
        "-c:a", "pcm_s16le",
        "-ar", "48000",
        "-ac", "1",
        INPUT_WAV
    ], check=True)

    # Transcribe with FasterWhisper
    segments, _ = whisper_model.transcribe(INPUT_WAV)
    transcription = " ".join([seg.text.strip() for seg in segments if seg.text])
    if not transcription:
        transcription = "No clear speech detected."

    # Generate AI response
    answer = await asyncio.to_thread(get_ai_response, transcription)

    # Generate TTS audio
    await asyncio.to_thread(generate_tts_audio, answer, OUTPUT_MP3)

    # Send response
    await websocket.send_json({
        "transcription": transcription,
        "answer": answer
    })
    await websocket.close()



@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse("static/index.html")

@app.get("/favicon.ico")
def favicon():
    return FileResponse("static/favicon.ico")

# --- Utility Functions ---
def clean_files():
    for f in [INPUT_WEBM, INPUT_WAV, OUTPUT_MP3]:
        if os.path.exists(f):
            os.remove(f)

def get_ai_response(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text if hasattr(response, "text") else "No response text found."
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "Sorry, I couldn't generate a response."


def generate_tts_audio(text: str, output_path: str):
    engine = pyttsx3.init()
    engine.save_to_file(text, output_path)
    engine.runAndWait()

# --- WebSocket Endpoint ---
@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_data = bytearray()

    try:
        clean_files()
        while True:
            chunk = await asyncio.wait_for(websocket.receive_bytes(), timeout=10.0)
            audio_data.extend(chunk)
            if len(audio_data) > 2_000_000:
                break
    except Exception:
        pass

    if not audio_data:
        await websocket.send_json({"error": "No audio received."})
        await websocket.close()
        return

    with open(INPUT_WEBM, "wb") as f:
        f.write(audio_data)

    subprocess.run(["ffmpeg", "-y", "-i", INPUT_WEBM, INPUT_WAV], check=True)

    segments, _ = whisper_model.transcribe(INPUT_WAV)
    transcription = " ".join([seg.text.strip() for seg in segments if seg.text])
    if not transcription:
        transcription = "No clear speech detected."

    answer = await asyncio.to_thread(get_ai_response, transcription)
    await asyncio.to_thread(generate_tts_audio, answer, OUTPUT_MP3)

    await websocket.send_json({
        "transcription": transcription,
        "answer": answer
    })
    await websocket.close()

# --- Serve MP3 ---
@app.get("/live_answer.mp3")
async def get_audio():
    if not os.path.exists(OUTPUT_MP3):
        raise HTTPException(status_code=404, detail="Audio not found.")
    return FileResponse(OUTPUT_MP3, media_type="audio/mpeg")
"""