import os
import urllib.request
import tarfile

MODELS = {
    "lessac-high": {
        "model": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx",
        "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/high/en_US-lessac-high.onnx.json"
    },
    "ryan-high": {
        "model": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx",
        "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx.json"
    },
    "amy-low": {
        "model": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx",
        "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx.json"
    },
    "alba-medium": {
        "model": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx",
        "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx.json"
    }
}

PIPER_BINARY_URL = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    for voice, urls in MODELS.items():
        download_file(urls["model"], f"models/{voice}.onnx")
        download_file(urls["config"], f"models/{voice}.onnx.json")

    download_file(PIPER_BINARY_URL, "piper.tar.gz")
    
    if not os.path.exists("piper/piper"):
        print("Extracting Piper standalone binary...")
        with tarfile.open("piper.tar.gz", "r:gz") as tar:
            tar.extractall()
        # Ensure executable permissions
        os.system("chmod +x piper/piper")
        
    print("Piper TTS Setup complete!")
