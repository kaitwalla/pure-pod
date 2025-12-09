#!/usr/bin/env python3
"""Download ML models required by the worker.

Due to macOS Local Network permissions, this must be run from Script Editor:
1. Open Script Editor
2. Run: do shell script "cd /Users/kait/Dev/PurePod/worker && HF_TOKEN=your_token .venv/bin/python scripts/download_models.py"
3. Wait for models to download (~5GB total)

Get your token at: https://huggingface.co/settings/tokens
"""
import os
import sys

token = os.environ.get("HF_TOKEN")
if not token:
    print("ERROR: Set HF_TOKEN environment variable")
    print("Get your token at: https://huggingface.co/settings/tokens")
    sys.exit(1)

from huggingface_hub import login
login(token=token)

print("Downloading LLM model (Llama-3-8B-Instruct-4bit)...")
sys.stdout.flush()
from mlx_lm import load
model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
print("LLM model ready")

print("Downloading Whisper model (distil-whisper-large-v3)...")
sys.stdout.flush()
from huggingface_hub import snapshot_download
snapshot_download("mlx-community/distil-whisper-large-v3")
print("Whisper model ready")

print("All models downloaded!")
