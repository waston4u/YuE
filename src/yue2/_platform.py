"""Platform gate — YuE2 runs on Apple Silicon Macs only."""
import platform
import sys


def is_apple_silicon():
    """True on arm64 macOS (native, not Rosetta — platform.machine() is x86_64 there)."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def require_apple_silicon(what="YuE2"):
    """Refuse to start anywhere but an Apple Silicon Mac."""
    if is_apple_silicon():
        return
    raise RuntimeError(
        f"{what} runs only on Apple Silicon Macs (arm64 macOS); "
        f"detected {sys.platform}/{platform.machine() or 'unknown'}")
