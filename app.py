import os
import logging
import subprocess
import asyncio
import base64
import json

import requests
from fastapi import FastAPI, HTTPException, WebSocket, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
import google.generativeai as genai
import pyttsx3
from dotenv import load_dotenv
from PIL import Image, ImageDraw

# --- Configuration ---
# Set logging level
logging.basicConfig(level=logging.INFO)

# Load environment variables (GEMINI_API_KEY, HF_API_TOKEN, HF_MODEL)
load_dotenv()

# Define paths
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

INPUT_WEBM = "live_input.webm"
INPUT_WAV = "live_input.wav"
OUTPUT_MP3 = os.path.join(STATIC_DIR, "live_answer.mp3")
PLACEHOLDER_PATH = os.path.join(STATIC_DIR, "placeholder.png")
FETCHED_PATH = os.path.join(STATIC_DIR, "fetched.png")
FRONT_HTML = os.path.join(STATIC_DIR, "front.html")

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_TEXT_MODEL = "gemini-2.5-flash"

# Hugging Face Configuration
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
# Using SDXL base model as the default
HF_MODEL = os.getenv("HF_MODEL", "stabilityai/stable-diffusion-xl-base-1.0") 
if not HF_API_TOKEN:
    # Changed to logging.error to allow the script to potentially run without image features
    # if the user wants to test only voice, but raised if the image func is called
    logging.error("HF_API_TOKEN not set in .env. Image generation will fail.")

# Whisper Configuration
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# FastAPI Setup
app = FastAPI(title="Voice + Image Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Ensure placeholder exists ---
if not os.path.exists(PLACEHOLDER_PATH):
    img = Image.new("RGB", (512, 512), color="white")
    d = ImageDraw.Draw(img)
    d.text((50, 240), "No image available", fill="black")
    img.save(PLACEHOLDER_PATH)

# --- Utility Functions ---

def clean_files():
    """Removes temporary files."""
    for f in [INPUT_WEBM, INPUT_WAV, OUTPUT_MP3, FETCHED_PATH]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception as e:
                logging.warning(f"Failed to remove {f}: {e}")

def get_ai_response(prompt: str) -> str:
    """Gets a text response from the Gemini API."""
    try:
        model = genai.GenerativeModel(GEMINI_TEXT_MODEL)
        response = model.generate_content(prompt)
        if getattr(response, "text", None):
            return response.text.strip()
        return "No response text found."
    except Exception as e:
        logging.error(f"Gemini text error: {e}")
        return "Sorry, I couldn't generate a response."

def generate_tts_audio(text: str, output_path: str):
    """Generates an MP3 file using pyttsx3."""
    try:
        engine = pyttsx3.init()
        # Ensure the rate is set for a natural sound
        engine.setProperty('rate', 150) 
        engine.save_to_file(text, output_path)
        engine.runAndWait()
    except Exception as e:
        logging.error(f"TTS error: {e}")

def generate_image_hf(prompt: str):
    """Generate image from Hugging Face Inference API (Fixed URL)"""
    if not HF_API_TOKEN:
         logging.error("HF_API_TOKEN is missing. Cannot generate image.")
         return None
         
    try:
        # --- FIXED URL: Added /models/ after /hf-inference/ ---
        url = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
        
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        
        # Added parameters for better SDXL generation and to prevent model loading issues
        payload = {
            "inputs": f"highly detailed diagram or illustration of: {prompt}",
            "parameters": {
                "wait_for_model": True, # Wait if the model is loading
                "height": 512,
                "width": 512,
            }
        }

        # Set a 60-second timeout for image generation
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            logging.error(f"Hugging Face error {response.status_code}: {response.text}")
            return None

        content_type = response.headers.get("content-type", "application/octet-stream")
        if not content_type.startswith("image/"):
            logging.error(f"Hugging Face returned non-image content: {content_type}")
            return None

        img_bytes = response.content
        
        # Save the image bytes to a local file
        with open(FETCHED_PATH, "wb") as f:
            f.write(img_bytes)

        # Convert to Base64 for the WebSocket/API response
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        return {
            "image_base64": img_b64,
            "mime_type": content_type,
            "image_url": "/static/fetched.png"
        }
    except requests.exceptions.Timeout:
        logging.error("Hugging Face request timed out after 60 seconds.")
        return None
    except Exception as e:
        logging.error(f"Hugging Face image error: {e}")
        return None

def maybe_generate_image(prompt: str):
    """Attempts to generate an image and falls back to a placeholder."""
    result = generate_image_hf(prompt)
    if result:
        return result
        
    # Fallback to placeholder
    try:
        with open(PLACEHOLDER_PATH, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        return {"image_base64": img_b64, "mime_type": "image/png", "image_url": "/static/placeholder.png"}
    except Exception:
        # Final fallback if placeholder itself can't be read
        return {"image_base64": None, "mime_type": "image/png", "image_url": "/static/placeholder.png"}

# --- WebSocket Endpoint ---
@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_data = bytearray()
    clean_files() # Clean up before starting

    try:
        while True:
            # Removed the 10.0 timeout from wait_for to let the client control the connection
            message = await websocket.receive()
            
            if message.get("bytes"):
                audio_data.extend(message["bytes"])
            elif message.get("text"):
                try:
                    data = json.loads(message["text"])
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    pass
            
            if len(audio_data) > 5_000_000:
                logging.warning("Received too much audio data. Stopping.")
                break
    except Exception as e:
        logging.info(f"WebSocket closed by client or connection error: {e}")

    if not audio_data:
        await websocket.send_json({"error": "No audio received."})
        await websocket.close()
        return

    # Process audio
    with open(INPUT_WEBM, "wb") as f:
        f.write(audio_data)

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", INPUT_WEBM,
            "-c:a", "pcm_s16le",
            "-ar", "48000",
            "-ac", "1",
            INPUT_WAV
        ], check=True, capture_output=True) # Added capture_output=True for cleaner runs
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed: {e.stderr.decode()}")
        await websocket.send_json({"error": f"FFmpeg failed: {e.stderr.decode()}"})
        await websocket.close()
        return

    # Transcription, AI Response, TTS, and Image Generation are moved to threads
    segments, _ = whisper_model.transcribe(INPUT_WAV)
    transcription = " ".join([seg.text.strip() for seg in segments if seg.text]) or "No clear speech detected."

    answer = await asyncio.to_thread(get_ai_response, transcription)
    await asyncio.to_thread(generate_tts_audio, answer, OUTPUT_MP3)
    image_result = await asyncio.to_thread(maybe_generate_image, transcription)

    # Send final result
    await websocket.send_json({
        "transcription": transcription,
        "answer": answer,
        "audio_url": "/static/live_answer.mp3",
        "image_base64": image_result.get("image_base64"),
        "mime_type": image_result.get("mime_type"),
        "image_url": image_result.get("image_url")
    })
    await websocket.close()
    
    # Final cleanup (optional, but good practice)
    # clean_files() 

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse(FRONT_HTML)

@app.get("/favicon.ico")
def favicon():
    path = os.path.join(STATIC_DIR, "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path)
    return HTMLResponse(content="", status_code=204)

@app.get("/static/live_answer.mp3")
async def get_audio():
    if not os.path.exists(OUTPUT_MP3):
        raise HTTPException(status_code=404, detail="Audio not found.")
    return FileResponse(OUTPUT_MP3, media_type="audio/mpeg")

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    """Handles text-based questions (without streaming audio)."""
    try:
        # Generate AI answer
        answer = await asyncio.to_thread(get_ai_response, question)

        # Generate TTS audio file
        await asyncio.to_thread(generate_tts_audio, answer, OUTPUT_MP3)

        # Generate or fetch diagram image
        image_result = await asyncio.to_thread(maybe_generate_image, question)

        # Return JSON response
        return {
            "question": question,
            "answer": answer,
            "audio_url": "/static/live_answer.mp3",
            "image_base64": image_result.get("image_base64"),
            "mime_type": image_result.get("mime_type"),
            "image_url": image_result.get("image_url")
        }
    except Exception as e:
        logging.error(f"Ask route error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")




"""
import os
import logging
import subprocess
import asyncio
import base64
import json

import requests
from fastapi import FastAPI, HTTPException, WebSocket, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
import google.generativeai as genai
import pyttsx3
from dotenv import load_dotenv
from PIL import Image, ImageDraw

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

INPUT_WEBM = "live_input.webm"
INPUT_WAV = "live_input.wav"
OUTPUT_MP3 = os.path.join(STATIC_DIR, "live_answer.mp3")
PLACEHOLDER_PATH = os.path.join(STATIC_DIR, "placeholder.png")
FETCHED_PATH = os.path.join(STATIC_DIR, "fetched.png")
FRONT_HTML = os.path.join(STATIC_DIR, "front.html")

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_TEXT_MODEL = "gemini-2.5-flash"

# Hugging Face
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set in .env")

# Whisper
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# FastAPI
app = FastAPI(title="Voice + Image Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Ensure placeholder exists ---
if not os.path.exists(PLACEHOLDER_PATH):
    img = Image.new("RGB", (512, 512), color="white")
    d = ImageDraw.Draw(img)
    d.text((50, 240), "No image available", fill="black")
    img.save(PLACEHOLDER_PATH)

# --- Utility Functions ---
def clean_files():
    for f in [INPUT_WEBM, INPUT_WAV, OUTPUT_MP3, FETCHED_PATH]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception as e:
                logging.warning(f"Failed to remove {f}: {e}")

def get_ai_response(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(GEMINI_TEXT_MODEL)
        response = model.generate_content(prompt)
        if getattr(response, "text", None):
            return response.text.strip()
        return "No response text found."
    except Exception as e:
        logging.error(f"Gemini text error: {e}")
        return "Sorry, I couldn't generate a response."

def generate_tts_audio(text: str, output_path: str):
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, output_path)
        engine.runAndWait()
    except Exception as e:
        logging.error(f"TTS error: {e}")

def generate_image_hf(prompt: str):

    try:
        url = f"https://router.huggingface.co/hf-inference/{HF_MODEL}"
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {"inputs": prompt}

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            logging.error(f"Hugging Face error {response.status_code}: {response.text}")
            return None

        if "application/json" in response.headers.get("content-type", ""):
            logging.error(f"Hugging Face returned JSON instead of image: {response.text}")
            return None

        img_bytes = response.content
        with open(FETCHED_PATH, "wb") as f:
            f.write(img_bytes)

        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return {
            "image_base64": img_b64,
            "mime_type": "image/png",
            "image_url": "/static/fetched.png"
        }
    except Exception as e:
        logging.error(f"Hugging Face image error: {e}")
        return None

def maybe_generate_image(prompt: str):
    result = generate_image_hf(prompt)
    if result:
        return result
    try:
        with open(PLACEHOLDER_PATH, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        return {"image_base64": img_b64, "mime_type": "image/png", "image_url": "/static/placeholder.png"}
    except Exception:
        return {"image_base64": None, "mime_type": "image/png", "image_url": "/static/placeholder.png"}

# --- WebSocket Endpoint ---
@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_data = bytearray()
    clean_files()

    try:
        while True:
            message = await websocket.receive()
            if message.get("bytes"):
                audio_data.extend(message["bytes"])
            elif message.get("text"):
                try:
                    data = json.loads(message["text"])
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    pass
            if len(audio_data) > 5_000_000:
                break
    except Exception as e:
        logging.info(f"WebSocket closed: {e}")

    if not audio_data:
        await websocket.send_json({"error": "No audio received."})
        await websocket.close()
        return

    with open(INPUT_WEBM, "wb") as f:
        f.write(audio_data)

    try:
        subprocess.run([
            "ffmpeg", "-y",
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
        "audio_url": "/static/live_answer.mp3",
        "image_base64": image_result.get("image_base64"),
        "mime_type": image_result.get("mime_type"),
        "image_url": image_result.get("image_url")
    })
    await websocket.close()

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse(FRONT_HTML)

@app.get("/favicon.ico")
def favicon():
    path = os.path.join(STATIC_DIR, "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path)
    return HTMLResponse(content="", status_code=204)

@app.get("/static/live_answer.mp3")
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
            "audio_url": "/static/live_answer.mp3",
            "image_base64": image_result.get("image_base64"),
            "mime_type": image_result.get("mime_type"),
            "image_url": image_result.get("image_url")
        }
    except Exception as e:
        logging.error(f"Ask route error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
"""