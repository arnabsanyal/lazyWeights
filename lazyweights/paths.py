"""Where things live. Everything defaults to folders inside the repository; override with environment variables.

  LW_MODELS     model checkpoints, GGUFs and runtime bundles   (default: ./models)
  LW_LLAMA_CPP  llama.cpp checkout, built (build/bin/...)      (default: ./third_party/llama.cpp)
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(REPO, "results")
MODELS = os.path.abspath(os.environ.get("LW_MODELS", os.path.join(REPO, "models")))
LLAMA = os.path.abspath(os.environ.get("LW_LLAMA_CPP", os.path.join(REPO, "third_party", "llama.cpp")))


def gguf_py():
    """Put llama.cpp's gguf-py (the version pinned with llama.cpp) first on the path and return the module."""
    p = os.path.join(LLAMA, "gguf-py")
    if p not in sys.path:
        sys.path.insert(1, p)
    import gguf
    return gguf
