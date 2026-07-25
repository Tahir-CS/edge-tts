import os
import json
import io
import wave
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

# Model paths
MODEL_PATH = "model.onnx"
PIPER_BIN = "./piper/piper"

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
            
            if not text:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "Text required"}')
                return

            with tts_lock:
                # Run the Piper C++ standalone binary, feed text via stdin, get raw PCM via stdout
                process = subprocess.Popen(
                    [PIPER_BIN, "-m", MODEL_PATH, "--output_raw"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                raw_pcm, err = process.communicate(input=text.encode('utf-8'))
                
                if process.returncode != 0:
                    raise Exception(f"Piper Error: {err.decode('utf-8')}")
                
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(22050) # Piper High quality sample rate
                    wav_file.writeframes(raw_pcm)
                
                buf.seek(0)
                wav_bytes = buf.read()
                
                self.send_response(200)
                self.send_header('Content-type', 'audio/wav')
                self.end_headers()
                self.wfile.write(wav_bytes)
                
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f'{{"error": "{str(e)}" }}'.encode())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    server = ThreadingHTTPServer(('0.0.0.0', port), TTSHandler)
    print(f"Piper TTS CLI server running on port {port}")
    server.serve_forever()
