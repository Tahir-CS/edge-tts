import os
import urllib.request
import tarfile

MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx"
CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx.json"
PIPER_BINARY_URL = "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_linux_x86_64.tar.gz"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")

if __name__ == "__main__":
    download_file(MODEL_URL, "model.onnx")
    download_file(CONFIG_URL, "model.onnx.json")
    download_file(PIPER_BINARY_URL, "piper.tar.gz")
    
    if not os.path.exists("piper/piper"):
        print("Extracting Piper standalone binary...")
        with tarfile.open("piper.tar.gz", "r:gz") as tar:
            tar.extractall()
        # Ensure executable permissions
        os.system("chmod +x piper/piper")
        
    print("Piper TTS Setup complete!")
