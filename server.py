import os
import json
import io
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from piper.voice import PiperVoice
import threading

# Model paths
MODEL_PATH = "model.onnx"
CONFIG_PATH = "model.onnx.json"

print("Loading Piper TTS (High Quality) Voice...")
voice = PiperVoice.load(MODEL_PATH, config_path=CONFIG_PATH)

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
        self.wfile.write(b'Piper TTS Raw Server Ready')
        
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
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(voice.config.sample_rate)
                    # Use streaming raw API to explicitly write frames
                    for audio_bytes in voice.synthesize_stream_raw(text):
                        wav_file.writeframes(audio_bytes)
                
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
    print(f"Piper TTS bare metal server running on port {port}")
    server.serve_forever()
