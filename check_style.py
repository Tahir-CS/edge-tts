import numpy as np
import requests
import os

if not os.path.exists("voices-v1.0.bin"):
    print("Downloading voices...")
    r = requests.get("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin")
    with open("voices-v1.0.bin", "wb") as f:
        f.write(r.content)

try:
    voices_data = np.load("voices-v1.0.bin", allow_pickle=True)
    style = voices_data["af_heart"]
    print(f"Shape of style array: {style.shape}")
    print(f"Rank of style array: {len(style.shape)}")
except Exception as e:
    print(f"Error: {e}")
