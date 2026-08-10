import os
import json
import io
import wave
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import numpy as np
import soundfile as sf

PIPER_BIN = "./piper/piper"

MODELS = {
    "en_US-lessac-high": "models/lessac-high.onnx",
    "en_US-ryan-high": "models/ryan-high.onnx",
    "en_US-amy-low": "models/amy-low.onnx",
    "en_GB-alba-medium": "models/alba-medium.onnx"
}

# Mutex to ensure strictly one generation at a time
tts_lock = threading.Lock()

class TTSHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Piper TTS CLI Server Ready')
        
    def do_POST(self):
        if self.path != '/tts':
            self.send_response(404)
            self.end_headers()
            return
            
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            text = data.get("text", "")
            voice = data.get("voice", "en_US-lessac-high")
            
            if not text:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "Text required"}')
                return

            model_path = MODELS.get(voice, MODELS["en_US-lessac-high"])

            with tts_lock:
                # Run the Piper C++ standalone binary, feed text via stdin, get raw PCM via stdout
                process = subprocess.Popen(
                    [PIPER_BIN, "-m", model_path, "--output_raw"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                raw_pcm, err = process.communicate(input=text.encode('utf-8'))
                
                if process.returncode != 0:
                    raise Exception(f"Piper Error: {err.decode('utf-8')}")

                # Piper outputs 16-bit PCM raw data.
                # Load it into a NumPy array
                audio_data = np.frombuffer(raw_pcm, dtype=np.int16)

                # Compress directly to OGG Vorbis
                buf = io.BytesIO()
                sf.write(buf, audio_data, 22050, format='OGG')
                
                buf.seek(0)
                ogg_bytes = buf.read()
                
                self.send_response(200)
                self.send_header('Content-type', 'audio/ogg')
                self.end_headers()
                self.wfile.write(ogg_bytes)
                
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f'{{"error": "{str(e)}" }}'.encode())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    server = ThreadingHTTPServer(('0.0.0.0', port), TTSHandler)
    print(f"Piper TTS CLI server running on port {port}")
    server.serve_forever()
