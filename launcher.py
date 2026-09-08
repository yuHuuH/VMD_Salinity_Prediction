from __future__ import annotations

import os
import socket
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

from streamlit.web import cli as stcli


def resource_path(relative_path: str) -> str:
    """Return a path that works in normal Python and PyInstaller builds."""
    if hasattr(sys, "_MEIPASS"):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parent
    return str(base_path / relative_path)


def get_free_port(preferred: int = 8501) -> int:
    """Use 8501 when free; otherwise ask Windows for a free local port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def open_browser(port: int) -> None:
    time.sleep(2.5)
    webbrowser.open(f"http://localhost:{port}")


if __name__ == "__main__":
    # SHAP imports Numba/llvmlite. In a bundled EXE, give Numba a writable
    # cache location so import-time checks do not fail in read-only bundle folders.
    numba_cache = Path(tempfile.gettempdir()) / "mekong_salinity_numba_cache"
    numba_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache))

    # Keep relative paths predictable when running from an .exe.
    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).resolve().parent)
    else:
        os.chdir(Path(__file__).resolve().parent)

    app_path = resource_path("app.py")
    port = get_free_port(8501)

    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(port),
        "--server.address",
        "localhost",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--global.developmentMode",
        "false",
    ]

    sys.exit(stcli.main())
