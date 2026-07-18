from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import edge_tts

app = FastAPI(title="Edge TTS Proxy API for Folio Reader")

# Allow Folio Reader to communicate with this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, you can change this to your Cloudflare domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-JennyNeural"

@app.post("/tts")
async def generate_tts(req: TTSRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # edge-tts generates the audio asynchronously. 
    # We capture the stream and return it immediately so the user doesn't wait
    async def audio_stream():
        communicate = edge_tts.Communicate(req.text, req.voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    return StreamingResponse(audio_stream(), media_type="audio/mpeg")

@app.get("/")
def read_root():
    return {"status": "Edge TTS Server is running perfectly on Render!"}