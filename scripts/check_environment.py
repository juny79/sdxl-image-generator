"""Check Python, PyTorch, CUDA, and available GPU memory."""

import sys


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    try:
        import torch
    except ImportError:
        print("PyTorch: not installed")
        return 1

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return 2

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name}")
    print(f"VRAM GiB: {props.total_memory / 1024**3:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
