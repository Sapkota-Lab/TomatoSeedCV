"""
Desktop entry point for TomatoSeedCV.

Runs the existing Shiny `app` (from app.py) with a local uvicorn server and
opens it in the user's default browser, so the whole app can be packaged into
a single Windows .exe with PyInstaller and shared with other people.
"""
import os
import sys
import threading
import webbrowser

import uvicorn
from dotenv import load_dotenv

HOST = "127.0.0.1"
PORT = 8231


def _base_dir() -> str:
    """Directory of the .exe (frozen) or this script (dev)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = _base_dir()

# Let end users drop their own ROBOFLOW_API_KEY next to the .exe without rebuilding.
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

sys.path.insert(0, BASE_DIR)
from app import app  # noqa: E402  (must import after .env is loaded)


def _open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}/")


def main():
    if not os.getenv("ROBOFLOW_API_KEY"):
        print(
            "WARNING: ROBOFLOW_API_KEY is not set. Create a .env file next to "
            "this executable containing:\n  ROBOFLOW_API_KEY=your_key_here"
        )
    threading.Timer(1.0, _open_browser).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
