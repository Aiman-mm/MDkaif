import os
import logging
import uuid
import subprocess
import threading
import tempfile
import re
import json
import time
from typing import Optional, Dict
from functions import search_images
from fastapi.responses import RedirectResponse

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
import requests
from io import BytesIO

from fastapi.responses import RedirectResponse, JSONResponse
from database import verify_user   

from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, RedirectResponse
from database import register_user, verify_user, init_db

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()
init_db()
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
#FRONT_HTML = os.path.join(STATIC_DIR, "new.html")
FRONT_HTML = os.path.join(STATIC_DIR, "new3.html")

# --- Utility Functions ---
def unique_name(prefix: str, ext: str) -> str:
    safe_prefix = re.sub(r'[^a-zA-Z0-9_-]', '_', prefix)
    return f"{safe_prefix}_{uuid.uuid4().hex}.{ext}"

def cache_bust(url_path: str) -> str:
    return f"{url_path}?v={uuid.uuid4().hex}"

def cleanup_static(max_age_hours: int = 24):
    now = time.time()
    cutoff = now - (max_age_hours * 3600)
    for fname in os.listdir(STATIC_DIR):
        fpath = os.path.join(STATIC_DIR, fname)
        try:
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
                logging.info(f"Cleaned up old file: {fname}")
        except Exception as e:
            logging.warning(f"Failed to cleanup {fname}: {e}")

# ✅ Regex fixed (double-escaped to silence warnings)
SENTENCE_SPLIT = re.compile("(?<=[.!?])\\s+")

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
    """Generates TTS audio using gTTS and concatenates chunks with ffmpeg."""
    try:
        cleaned = str(text or "").strip() or "Sorry, I have no response."
        chunks = [cleaned[i:i+100] for i in range(0, len(cleaned), 100)]
        temp_files = []
        for i, chunk in enumerate(chunks):
            fd, temp_mp3_path = tempfile.mkstemp(suffix=f"_{i}.mp3")
            os.close(fd)
            gTTS(text=chunk, lang="en", slow=False).save(temp_mp3_path)
            temp_files.append(temp_mp3_path)

        fd, list_file_path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            with open(list_file_path, "w") as f:
                for tf in temp_files:
                    f.write(f"file '{os.path.abspath(tf)}'\n")

            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", list_file_path, "-c:a", "libmp3lame", "-b:a", "128k", output_path],
                check=True, capture_output=True
            )
            logging.info(f"TTS audio saved to {output_path}")
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg error: {e.stderr.decode()}")
            if temp_files:
                os.rename(temp_files[0], output_path)
        finally:
            for tf in temp_files:
                try: os.remove(tf)
                except Exception: pass
            if os.path.exists(list_file_path):
                os.remove(list_file_path)
    except Exception as e:
        logging.error(f"TTS error: {e}")

# --- Gemini setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")
else:
    GEMINI_MODEL = None

def get_ai_response(prompt: str) -> str:
    if not GEMINI_MODEL:
        return f"(AI text generation disabled) No GEMINI_API_KEY found. You asked: {prompt}"
    try:
        response = GEMINI_MODEL.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        return f"I understood your question: '{prompt}', but Gemini did not return a clear answer."
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return f"Sorry, I couldn't generate a response for: '{prompt}'."

# --- Whisper setup ---
logging.info("Loading Whisper model: small")
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

# --- Stable Diffusion setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_pipeline_turbo():
    logging.info("Loading diffusion pipeline: stabilityai/sd-turbo")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)
    pipe.safety_checker = None
    pipe.enable_attention_slicing()
    return pipe

pipe = load_pipeline_turbo()

# --- Intent detection ---
TEXT_INTENT = {"explain", "describe", "illusion", "clarify", "meaning", "definition"}
IMAGE_INTENT = {"create", "generate", "draw", "illustrate", "design", "render", "sketch",
                "image", "picture", "diagram", "science", "logo", "poster", "fantasy",
                "cat", "dog", "animal", "landscape", "car", "house"}
ANATOMY_INTENT = {"anatomy", "medical", "brain", "heart", "lung", "skeleton", "kidney"}

def detect_intent(q: str):
    ql = q.lower()
    q_words = set(re.findall(r'\b\w+\b', ql))
    wants_text = bool(q_words.intersection(TEXT_INTENT))
    wants_image = bool(q_words.intersection(IMAGE_INTENT))
    wants_anatomy = bool(q_words.intersection(ANATOMY_INTENT))
    if wants_image and any(word in ql for word in {"create", "generate", "draw", "illustrate"}):
        wants_text = False
    return wants_text, wants_image, wants_anatomy

# --- Image generation ---
def generate_image(prompt: str) -> Optional[Dict[str, str]]:
    if not prompt:
        logging.error("Image generation error: Prompt is missing.")
        return None
    try:
        image = pipe(
            prompt,
            guidance_scale=7.5,
            num_inference_steps=20,
            height=384,
            width=384
        ).images[0]
        filename = unique_name("generated", "png")
        img_path = os.path.join(STATIC_DIR, filename)
        image.save(img_path)
        logging.info(f"Generated image saved to {img_path}")
        return {"image_url": cache_bust(f"/static/{filename}")}
    except Exception as e:
        logging.error(f"Image generation error: {e}")
        return None

# --- Anatomy diagram fetch ---
def fetch_real_diagram(prompt: str) -> Optional[Dict[str, str]]:
    try:
        results = search_images({"query": prompt, "page": 0})
        if results and "images" in results and results["images"]:
            image_url = results["images"][0]["url"]
            resp = requests.get(image_url, timeout=10)
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                ext = "jpg"
                if "png" in content_type:
                    ext = "png"
                elif "svg" in content_type:
                    # Save SVG directly
                    filename = unique_name("diagram", "svg")
                    filepath = os.path.join(STATIC_DIR, filename)
                    with open(filepath, "wb") as f:
                        f.write(resp.content)
                    return {"image_url": cache_bust(f"/static/{filename}")}

                filename = unique_name("diagram", ext)
                filepath = os.path.join(STATIC_DIR, filename)
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                return {"image_url": cache_bust(f"/static/{filename}")}

            # If request failed, return external URL
            return {"image_url": image_url}
        return None
    except Exception as e:
        logging.error(f"Diagram fetch error: {e}")
        return None


# --- FastAPI setup ---
app = FastAPI(title="Voice + Image Chatbot")

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")



# --- Routes ---

@app.get("/", response_class=HTMLResponse)
def root():
    # Show login page first
    path = os.path.join(STATIC_DIR, "login.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Login page not found.")
    return FileResponse(path)

@app.get("/login", response_class=HTMLResponse)
def login_page():
    # Serve the same login.html
    path = os.path.join(STATIC_DIR, "login.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Login page not found.")
    return FileResponse(path)


@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    if register_user(username, password):
        return JSONResponse({"success": True, "message": "User registered successfully!"})
    else:
        return JSONResponse({"success": False, "message": "Username already exists."})


@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    # ✅ Check credentials against database
    if verify_user(username, password):
        # Redirect to chatbot page if valid
        return RedirectResponse(url="/chatbot", status_code=303)
    else:
        # Invalid credentials
        return JSONResponse({"success": False, "message": "Invalid credentials."})


@app.get("/chatbot", response_class=HTMLResponse)
def chatbot_page():
    if not os.path.exists(FRONT_HTML):
        #raise HTTPException(status_code=404, detail="Frontend file 'new.html' not found.")
        raise HTTPException(status_code=404, detail="Frontend file 'new3.html' not found.")
    return FileResponse(FRONT_HTML)


@app.get("/favicon.ico")
def favicon():
    path = os.path.join(STATIC_DIR, "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path)
    return HTMLResponse(content="", status_code=204)

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

        wants_text, wants_image, wants_anatomy = detect_intent(q)
        reply_text = None
        image_url = None

        # 🧠 Anatomy override
        if wants_anatomy:
            image_result = await run_in_threadpool(fetch_real_diagram, q)
            if image_result:
                image_url = image_result["image_url"]
                reply_text = "Here is a labeled anatomical diagram from trusted sources."
            else:
                reply_text = "I couldn't find a proper diagram, but here's an explanation instead."
                wants_text = True
                wants_image = False

        # 🎨 Image generation (non-anatomy)
        elif wants_image:
            image_result = await run_in_threadpool(generate_image, q)
            if image_result:
                image_url = image_result["image_url"]
                reply_text = "Here is the image you requested!"
            else:
                reply_text = "Image generation failed, providing text instead."
                wants_text = True
                wants_image = False

        # 📝 Text response
        if wants_text or (not wants_image and not image_url):
            answer = await run_in_threadpool(get_ai_response, q)
            reply_text = await run_in_threadpool(format_structured, answer)

        # 🧾 Build response
        response = {"reply": reply_text}
        if image_url:
            response["image_url"] = image_url

        # 🔊 Audio generation
        if reply_text:
            filename = unique_name("answer", "mp3")
            output_mp3 = os.path.join(STATIC_DIR, filename)
            audio_url_path = f"/static/{filename}"
            threading.Thread(
                target=generate_tts_audio,
                args=(reply_text, output_mp3),
                daemon=True
            ).start()
            response["audio"] = cache_bust(audio_url_path)

        # Cleanup old files periodically
        cleanup_static(max_age_hours=24)

        return JSONResponse(response)

    except Exception as e:
        logging.error(f"Chat route error: {e}", exc_info=True)
        return JSONResponse({"reply": "Critical error occurred.", "audio": None}, status_code=500)


@app.post("/chat-audio")
async def chat_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded audio temporarily
        fd, temp_path = tempfile.mkstemp(suffix=".webm")
        os.close(fd)
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # 🎤 Transcribe with Whisper
        segments, info = whisper_model.transcribe(temp_path)
        transcript = " ".join([seg.text for seg in segments]).strip()

        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass

        if not transcript:
            return JSONResponse({"reply": "I couldn't understand the audio.", "audio": None})

        # Run intent detection
        wants_text, wants_image, wants_anatomy = detect_intent(transcript)
        reply_text = None
        image_url = None

        # 🧠 Anatomy override
        if wants_anatomy:
            image_result = await run_in_threadpool(fetch_real_diagram, transcript)
            if image_result:
                image_url = image_result["image_url"]
                reply_text = "Here is a labeled anatomical diagram from trusted sources."
            else:
                reply_text = "I couldn't find a proper diagram, but here's an explanation instead."
                wants_text = True
                wants_image = False

        # 🎨 Image generation
        elif wants_image:
            image_result = await run_in_threadpool(generate_image, transcript)
            if image_result:
                image_url = image_result["image_url"]
                reply_text = "Here is the image you requested!"
            else:
                reply_text = "Image generation failed, providing text instead."
                wants_text = True
                wants_image = False

        # 📝 Text response
        if wants_text or (not wants_image and not image_url):
            answer = await run_in_threadpool(get_ai_response, transcript)
            reply_text = await run_in_threadpool(format_structured, answer)

        # 🧾 Build response
        response = {"reply": reply_text, "transcript": transcript}
        if image_url:
            response["image_url"] = image_url

        # 🔊 Audio generation
        if reply_text:
            filename = unique_name("answer", "mp3")
            output_mp3 = os.path.join(STATIC_DIR, filename)
            audio_url_path = f"/static/{filename}"
            threading.Thread(
                target=generate_tts_audio,
                args=(reply_text, output_mp3),
                daemon=True
            ).start()
            response["audio"] = cache_bust(audio_url_path)

        # Cleanup old files periodically
        cleanup_static(max_age_hours=24)

        return JSONResponse(response)

    except Exception as e:
        logging.error(f"Chat-audio route error: {e}", exc_info=True)
        return JSONResponse({"reply": "Critical error occurred.", "audio": None}, status_code=500)


# --- Entry point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "run:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )


"""

# --- Load both pipelines ---
from diffusers import StableDiffusionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# ⚡ Fast pipeline (SD-Turbo)
pipe_turbo = StableDiffusionPipeline.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(device)
pipe_turbo.safety_checker = None
pipe_turbo.enable_attention_slicing()

# 🎨 High-quality pipeline (SD v1.5)
pipe_quality = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(device)
pipe_quality.safety_checker = None
pipe_quality.enable_attention_slicing()


# --- Improved image generation ---
def generate_image(prompt: str, high_quality: bool = False) -> Optional[Dict[str, str]]:
    if not prompt:
        logging.error("Image generation error: Prompt is missing.")
        return None
    try:
        # Pick pipeline
        pipe = pipe_quality if high_quality else pipe_turbo

        # Adjust parameters
        steps = 40 if high_quality else 20
        size = 512 if high_quality else 384

        image = pipe(
            prompt,
            guidance_scale=7.5,
            num_inference_steps=steps,
            height=size,
            width=size
        ).images[0]

        filename = unique_name("generated", "png")
        img_path = os.path.join(STATIC_DIR, filename)
        image.save(img_path)
        logging.info(f"Generated image saved to {img_path}")

        return {"image_url": cache_bust(f"/static/{filename}")}

    except Exception as e:
        logging.error(f"Image generation error: {e}")
        return None
"""





"""
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
from functions import search_images 

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
import requests

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
FRONT_HTML = os.path.join(STATIC_DIR, "new3.html")

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
        chunks = [cleaned[i:i+100] for i in range(0, len(cleaned), 100)] 
        
        temp_files = []
        for i, chunk in enumerate(chunks):
            fd, temp_mp3_path = tempfile.mkstemp(suffix=f"_{i}.mp3")
            os.close(fd) 
            
            gTTS(text=chunk, lang="en", slow=False).save(temp_mp3_path)
            temp_files.append(temp_mp3_path)

        list_file_path = None
        try:
            fd, list_file_path = tempfile.mkstemp(suffix=".txt")
            os.close(fd)

            with open(list_file_path, "w") as f:
                for tf in temp_files:
                    # Use os.path.abspath to ensure ffmpeg gets a safe, full path
                    f.write(f"file '{os.path.abspath(tf)}'\n")

            # Concatenate audio chunks using ffmpeg
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", list_file_path, "-c:a", "libmp3lame", "-b:a", "128k", output_path],
                check=True,  
                capture_output=True
            )
            logging.info(f"Successfully concatenated TTS audio to {output_path}")

        except subprocess.CalledProcessError as e:
            logging.error(f"TTS FFmpeg error (stderr): {e.stderr.decode()}")
            if temp_files and os.path.exists(temp_files[0]):
                 # Fallback: copy the first audio file if concatenation fails
                 os.rename(temp_files[0], output_path)
                 logging.warning("FFmpeg failed, falling back to first audio chunk.")
            else:
                 raise # Re-raise if even fallback fails
        finally:
            for tf in temp_files:
                try: os.remove(tf)
                except Exception as e: logging.warning(f"Failed to remove temp audio file {tf}: {e}")
            
            if list_file_path and os.path.exists(list_file_path):
                try: os.remove(list_file_path)
                except Exception as e: logging.warning(f"Failed to remove concat list file: {e}")
                
    except Exception as e:
        logging.error(f"TTS error: {e}")

# --- Gemini setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")
else:
    GEMINI_MODEL = None

def get_ai_response(prompt: str) -> str:
    if not GEMINI_MODEL:
        return f"(Gemini disabled) You asked: {prompt}"
    try:
        response = GEMINI_MODEL.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
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
    dtype = torch.float32 if device == "cpu" else torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=dtype
    ).to(device)
    pipe.safety_checker = None 
    pipe.enable_attention_slicing()
    return pipe

pipe = load_pipeline_turbo()

# --- Intent detection ---
TEXT_INTENT = {"explain", "describe", "illusion", "clarify", "meaning", "definition"}
IMAGE_INTENT = {"create", "generate", "draw", "illustrate", "design", "render", "sketch",
                "image", "picture","science", "logo","poster", "fantasy", "cat", "dog", "animal", "landscape", "car", "house"}
ANATOMY_INTENT = {"anatomy", "medical", "brain", "heart", "lung", "skeleton", "kidney"}


def detect_intent(q: str):
    ql = q.lower()
    q_words = set(re.findall(r'\b\w+\b', ql))
    wants_text = bool(q_words.intersection(TEXT_INTENT))
    wants_image = bool(q_words.intersection(IMAGE_INTENT))
    wants_anatomy = bool(q_words.intersection(ANATOMY_INTENT))
    if wants_image and any(word in ql for word in {"create", "generate", "draw", "illustrate"}):
        wants_text = False
    return wants_text, wants_image, wants_anatomy

# --- Image generation ---
def generate_image(prompt: str) -> Optional[Dict[str, str]]:
    if not prompt:
        logging.error("Image generation error: Prompt is missing.")
        return None
    try:
        image = pipe(
            prompt,
            guidance_scale=7.5,
            num_inference_steps=20,
            height=384,
            width=384
        ).images[0]

        filename = unique_name("generated", "png")
        img_path = os.path.join(STATIC_DIR, filename)
        image.save(img_path)

        logging.info(f"Generated image saved to {img_path}")
        return {"image_url": cache_bust(f"/static/{filename}")}
    except Exception as e:
        logging.error(f"Image generation error: {e}")
        return None

# --- Anatomy diagram fetch ---
def fetch_real_diagram(prompt: str) -> Optional[Dict[str, str]]:
    try:
        results = search_images({"query": prompt, "page": 0})
        if results and "images" in results and results["images"]:
            image_url = results["images"][0]["url"]
            logging.info(f"Diagram search returned URL: {image_url}")

            resp = requests.get(image_url, timeout=10)
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                ext = "jpg"
                if "png" in content_type: ext = "png"
                elif "svg" in content_type: ext = "svg"

                filename = unique_name("diagram", ext)
                filepath = os.path.join(STATIC_DIR, filename)

                with open(filepath, "wb") as f:
                    f.write(resp.content)

                logging.info(f"Diagram saved to {filepath}")
                return {"image_url": cache_bust(f"/static/{filename}")}

            logging.warning("Failed to save diagram locally, returning external URL")
            return {"image_url": image_url}

        logging.warning("No images found in search results")
        return None
    except Exception as e:
        logging.error(f"Diagram fetch error: {e}")
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

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def root():
    if not os.path.exists(FRONT_HTML):
        raise HTTPException(status_code=404, detail="Frontend file 'new3.html' not found.")
    return FileResponse(FRONT_HTML)

@app.get("/favicon.ico")
def favicon():
    path = os.path.join(STATIC_DIR, "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path)
    return HTMLResponse(content="", status_code=204)

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

        wants_text, wants_image, wants_anatomy = detect_intent(q)
        reply_text = None
        image_url = None

        # 🧠 Anatomy override
        if wants_anatomy:
            image_result = await run_in_threadpool(fetch_real_diagram, q)
            if image_result:
                image_url = image_result["image_url"]
                reply_text = "Here is a labeled anatomical diagram from trusted sources."
            else:
                reply_text = "I couldn't find a proper diagram, but here's an explanation instead."
                wants_text = True
                wants_image = False

        # 🎨 Image generation (non-anatomy)
        elif wants_image:
            image_result = await run_in_threadpool(generate_image, q)
            if image_result:
                image_url = image_result["image_url"]
                reply_text = "Here is the image you requested!"
            else:
                reply_text = "Image generation failed, providing text instead."
                wants_text = True
                wants_image = False

        # 📝 Text response
        if wants_text or (not wants_image and not image_url):
            answer = await run_in_threadpool(get_ai_response, q)
            reply_text = await run_in_threadpool(format_structured, answer)

        # 🧾 Build response
        response = {"reply": reply_text}
        if image_url:
            response["image_url"] = image_url

        # 🔊 Audio generation
        if reply_text:
            filename = unique_name("answer", "mp3")
            
            # 1. output_mp3: OS-specific path for SAVING the file (for ffmpeg).
            output_mp3 = os.path.join(STATIC_DIR, filename) 
            
            # 2. audio_url_path: URL convention (forward slashes) for RETRIEVING the file.
            audio_url_path = f"/static/{filename}" 
            
            threading.Thread(
                target=generate_tts_audio,
                args=(reply_text, output_mp3), 
                daemon=True
            ).start()
            
            # Use the URL path for the client response
            response["audio"] = cache_bust(audio_url_path)

        return JSONResponse(response)

    except Exception as e:
        logging.error(f"Chat route error: {e}")
        return JSONResponse({"reply": "Critical error occurred.", "audio": None}, status_code=500)


@app.post("/chat-audio")
async def chat_audio(file: UploadFile = File(...)):

    logging.info(f"Received audio file for transcription: {file.filename}")
    temp_audio_file_path = None
    
    try:
        _, ext = os.path.splitext(file.filename)
        if not ext: ext = ".wav" 
        
        fd, temp_audio_file_path = tempfile.mkstemp(suffix=ext)
        os.close(fd) 

        # Write the uploaded file content to the temp file
        with open(temp_audio_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        def transcribe_audio(path):
            segments, _ = whisper_model.transcribe(path)
            return " ".join(segment.text for segment in segments).strip()

        transcript = await run_in_threadpool(transcribe_audio, temp_audio_file_path)

        if not transcript:
            return JSONResponse({"reply": "I couldn't understand the audio. Please try speaking clearly.", "audio": None, "transcript": ""})

        logging.info(f"Transcription: {transcript}")

        response_data = await chat(question=transcript)

        if isinstance(response_data, JSONResponse):
            response_json = json.loads(response_data.body.decode())
        else:
            response_json = {"reply": "Error in chat processing.", "audio": None}
            
        response_json['transcript'] = transcript

        return JSONResponse(response_json)

    except Exception as e:
        logging.error(f"Audio transcription/chat error: {e}", exc_info=True)
        return JSONResponse(
            {"reply": "An error occurred while processing your voice message.", "audio": None, "transcript": "Error"},
            status_code=500
        )
    finally:
        if temp_audio_file_path and os.path.exists(temp_audio_file_path):
            try:
                os.remove(temp_audio_file_path)
            except Exception as e:
                logging.warning(f"Failed to remove temp audio file: {e}")
                
                
def load_pipeline_turbo():
    logging.info("Loading diffusion pipeline: stabilityai/sd-turbo")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.safety_checker = None
    pipe.enable_attention_slicing()
    return pipe
"""