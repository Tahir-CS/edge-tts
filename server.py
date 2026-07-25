from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import uvicorn
import onnxruntime as ort
from ttstokenizer import IPATokenizer
import numpy as np
import requests
import os
import io
import wave
import asyncio

app = FastAPI()

# Model download URLs (thewh1teagle GitHub releases)
ONNX_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

ONNX_FILE = "kokoro-v1.0.int8.onnx"
VOICES_FILE = "voices-v1.0.bin"

session = None
tokenizer = None
voices = None
tts_lock = asyncio.Lock()

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename} (~100MB)... This may take a minute.")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename}")

@app.on_event("startup")
async def startup_event():
    global session, tokenizer, voices
    # Download models at boot to keep Git repo small
    download_file(ONNX_URL, ONNX_FILE)
    download_file(VOICES_URL, VOICES_FILE)
    
    print("Initializing ONNX CPU session...")
    # Strict CPU optimization to minimize RAM on Render 512MB
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(ONNX_FILE, sess_options=options, providers=["CPUExecutionProvider"])
    
    print("Loading tokenizer and voices...")
    tokenizer = IPATokenizer()
    with open(VOICES_FILE, "rb") as f:
        voices = np.load(f, allow_pickle=True).item()
    print("Kokoro-INT8 TTS Ready!")

def generate_wav_buffer(audio_array, sample_rate=24000):
    # Convert float32 audio array to 16-bit PCM WAV in memory
    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_array = (audio_array * 32767).astype(np.int16)
    
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_array.tobytes())
    buf.seek(0)
    return buf.read()

@app.get("/ping")
def ping():
    # Dedicated lightweight endpoint for Cron Jobs
    return Response(content="pong", media_type="text/plain")

@app.get("/")
def root():
    return {"status": "ready", "message": "InkFolio Kokoro-INT8 TTS Server (Render)"}

@app.post("/tts")
async def tts_endpoint(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        voice = data.get("voice", "af_heart")
        speed = float(data.get("speed", 1.0))
        
        if not text:
            return JSONResponse(status_code=400, content={"error": "Text is required"})
            
        if voice not in voices:
            voice = "af_heart"
            
        # Ensure strict sequential generation to protect 512MB RAM
        async with tts_lock:
            tokens = tokenizer(text)
            
            # Format inputs exactly as the INT8 model expects
            style = voices[voice]
            
            inputs = {
                "tokens": [[0, *tokens, 0]],
                "style": style if len(style.shape) == 2 else [style],
                "speed": np.ones(1, dtype=np.float32) * speed
            }
            
            # Run inference
            audio_array = session.run(None, inputs)[0]
            
            # Convert to WAV
            wav_bytes = generate_wav_buffer(audio_array[0] if len(audio_array.shape) > 1 else audio_array)
            
            return Response(content=wav_bytes, media_type="audio/wav", headers={
                "Cache-Control": "public, max-age=86400"
            })
            
    except Exception as e:
        print(f"TTS Generation Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
