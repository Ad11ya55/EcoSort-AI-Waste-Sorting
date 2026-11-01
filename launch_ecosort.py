"""Desktop launcher for the packaged EcoSort Streamlit application."""

from __future__ import annotations

import sys
import os
import socket
import threading
import time
import webbrowser
from urllib.request import urlopen
from pathlib import Path
from urllib.error import URLError


# ============================================================================
# 1. CHECK WHETHER AN ECOSORT STREAMLIT SERVER IS ALREADY AVAILABLE
# Reusing it makes a second double-click reopen the page instead of failing.
# ============================================================================
def healthy_streamlit(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
            return response.status == 200 and response.read(16).strip() == b"ok"
    except (OSError, URLError):
        return False


# ============================================================================
# 2. FIND A FREE LOCAL PORT
# Keep the server on this computer and avoid collisions with other applications.
# ============================================================================
def available_port(start: int = 8501, stop: int = 8510) -> int:
    for port in range(start, stop + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            try:
                server_socket.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No available EcoSort port found ({start}-{stop}). Close another local server and try again.")


# ============================================================================
# 3. START STREAMLIT AND OPEN THE ECOSORT PAGE
# PyInstaller places app.py and its model files beside this launcher in the bundle.
# ============================================================================
def main() -> None:
    # PyInstaller extracts bundled files beside this entrypoint in one-folder mode.
    root = Path(__file__).resolve().parent
    os.chdir(root)
    app_path = root / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"EcoSort app.py was not found beside the launcher: {root}")

    # A second double-click should bring the existing EcoSort page back instead
    # of starting a duplicate server and closing with a port-in-use error.
    if healthy_streamlit(8501):
        webbrowser.open("http://127.0.0.1:8501")
        return

    port = available_port()

    def open_browser() -> None:
        for _ in range(60):
            try:
                with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1):
                    webbrowser.open(f"http://127.0.0.1:{port}")
                    return
            except Exception:
                time.sleep(1)

    if os.environ.get("ECOSORT_NO_BROWSER") != "1":
        threading.Thread(target=open_browser, daemon=True).start()
    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", str(app_path),
        "--server.headless=true",
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false",
    ]
    stcli.main()


# ============================================================================
# 4. SHOW A USEFUL MESSAGE IF THE PACKAGED APP CANNOT START
# Save the details in the user's local EcoSort folder for troubleshooting.
# ============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Packaged builds run without a console; retain details for support and
        # show a visible error instead of silently closing on startup failures.
        log_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "EcoSort"
        message = f"EcoSort could not start.\n\n{exc}"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "launcher.log").write_text(message, encoding="utf-8")
        except OSError:
            pass
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, "EcoSort startup error", 0x10)
        except Exception:
            print(message, file=sys.stderr)
        raise SystemExit(1)

