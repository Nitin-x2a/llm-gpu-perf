# colab_setup.py
# Central setup script for "Why Is My LLM Slow?"
# Save this file to Google Drive at:  MyDrive/llm-perf-book-labs/colab_setup.py
#
# Each chapter notebook runs it with two cells:
#
#   Cell 1 — mount Drive (once per session):
#     from google.colab import drive
#     drive.mount('/content/drive')
#
#   Cell 2 — run this script:
#     %run /content/drive/MyDrive/llm-perf-book-labs/colab_setup.py
#
# That's it. All installs and verification are handled here.

import subprocess, sys

def _pip(packages):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q"] + packages
    )

# ── Versions ────────────────────────────────────────────────────────────────
TORCH_VERSION  = "2.10.0"
TORCH_INDEX    = "https://download.pytorch.org/whl/cu126"
BNB_VERSION    = "bitsandbytes>=0.45.0"
HF_VERSION     = "transformers>=4.47.0"

# ── Install ──────────────────────────────────────────────────────────────────
print("[ 1/3 ] Installing PyTorch, Triton ...")
_pip([f"torch=={TORCH_VERSION}", "torchvision", "torchaudio",
      "--index-url", TORCH_INDEX])
_pip(["triton"])

print("[ 2/3 ] Installing bitsandbytes, transformers, accelerate ...")
_pip([BNB_VERSION, HF_VERSION, "accelerate", "huggingface_hub"])

# ── Verify ───────────────────────────────────────────────────────────────────
print("[ 3/3 ] Verifying ...")

import torch
assert torch.cuda.is_available(), (
    "\n✗  No GPU found. Runtime → Change runtime type → T4 GPU, then re-run."
)
sm = torch.cuda.get_device_capability(0)
print(f"  GPU     : {torch.cuda.get_device_name(0)}")
print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  SM cap  : sm_{sm[0]}{sm[1]}")
print(f"  PyTorch : {torch.__version__}")

import triton
print(f"  Triton  : {triton.__version__}")

import bitsandbytes as bnb
_ = bnb.nn.Linear8bitLt(64, 64)   # raises if CUDA backend missing
print(f"  bnb     : {bnb.__version__}")

print("\n✓  Environment ready.\n")
