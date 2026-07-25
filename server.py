import os
import json
import io
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import onnxruntime as ort
from ttstokenizer import IPATokenizer
import numpy as np
import threading
import nltk

print("Downloading NLTK data into runtime environment...")
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('cmudict', quiet=True)

# Model paths
ONNX_FILE = "kokoro-v1.0.int8.onnx"
VOICES_FILE = "voices-v1.0.bin"

print("Initializing ONNX CPU session...")
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False

session = ort.InferenceSession(ONNX_FILE, sess_options=options, providers=["CPUExecutionProvider"])

print("Loading tokenizer...")
tokenizer = IPATokenizer()

# Mutex to ensure strictly one generation at a time
tts_lock = threading.Lock()

def generate_wav_buffer(audio_array, sample_rate=24000):
    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_array = (audio_array * 32767).astype(np.int16)
    
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_array.tobytes())
    buf.seek(0)
    return buf.read()

class TTSHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Kokoro TTS Raw Server Ready')
        
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
            voice = data.get("voice", "af_heart")
            speed = float(data.get("speed", 1.0))
            
            if not text:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "Text required"}')
                return

            with tts_lock:
                tokens = tokenizer(text)
                token_len = min(len(tokens), 509)
                
                # Extreme RAM starvation: lazy load voice
                voices_data = np.load(VOICES_FILE, allow_pickle=True)
                if voice not in voices_data.files:
                    voice = "af_heart"
                raw_style = voices_data[voice][token_len]
                voices_data.close()
                del voices_data
                
                import gc
                gc.collect()
                
                style_array = np.atleast_2d(np.squeeze(raw_style)).astype(np.float32)
                tokens_array = np.array([[0, *tokens, 0]], dtype=np.int64)
                speed_array = np.array([speed], dtype=np.float32)
                
                inputs = {
                    "tokens": tokens_array,
                    "style": style_array,
                    "speed": speed_array
                }
                
                audio_array = session.run(None, inputs)[0]
                wav_bytes = generate_wav_buffer(audio_array[0] if len(audio_array.shape) > 1 else audio_array)
                
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
    print(f"Bare metal server running on port {port}")
    server.serve_forever()
