#!/usr/bin/env python3
"""Helper script for launching the Streamlit web UI."""

import argparse
import json
import os
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def resolve_host_name(host: str) -> str:
    try:
        return socket.gethostbyaddr(host)[0]
    except Exception:
        return host


def http_get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read()
            try:
                text = body.decode("utf-8")
            except UnicodeDecodeError:
                text = body.decode("utf-8", errors="ignore")
            return response.getcode(), text
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


def show_remote_status() -> None:
    host = os.getenv("RAG_SERVICE_HOST", "10.61.11.54")
    vm_user = os.getenv("VM_USER", "user")
    rag_port = int(os.getenv("RAG_SERVICE_PORT", "8000"))
    health_endpoint = os.getenv("RAG_HEALTH_ENDPOINT", "/health") or "/health"
    if not health_endpoint.startswith("/"):
        health_endpoint = f"/{health_endpoint}"
    rag_health_url = f"http://{host}:{rag_port}{health_endpoint}"

    qdrant_host = os.getenv("QDRANT_HOST", host)
    if qdrant_host in {"localhost", "127.0.0.1"}:
        qdrant_host = host
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}"

    vm_name = resolve_host_name(host)
    print("+------------------------------+")
    print("|   Remote service status      |")
    print("+------------------------------+")
    print(f"[INFO] VM: {vm_user}@{host} ({vm_name})")

    def print_status(label: str, url: str, expect_json: bool = True) -> None:
        try:
            status_code, body = http_get(url, timeout=5)
            if status_code == 200:
                if expect_json:
                    payload = None
                    try:
                        payload = json.loads(body)
                    except json.JSONDecodeError:
                        payload = None

                    if isinstance(payload, dict):
                        status_value = payload.get('status') or payload.get('result') or 'ok'
                        if isinstance(status_value, dict):
                            status_value = status_value.get('status', 'ok')
                        print(f"[OK]   {label}: {status_value}")
                    else:
                        snippet = body.strip().replace("\n", " ").replace("\r", " ")
                        print(f"[OK]   {label}: {snippet[:80]}")
                else:
                    print(f"[OK]   {label}: HTTP {status_code}")
            else:
                snippet = body.strip().replace("\n", " ").replace("\r", " ")
                print(f"[WARN] {label}: HTTP {status_code} ({snippet[:80]})")
        except Exception as exc:
            print(f"[FAIL] {label}: {exc}")

    print_status('RAG service', rag_health_url, expect_json=True)
    print_status('Qdrant', qdrant_url, expect_json=True)
    print('')


def check_streamlit_installed() -> bool:
    try:
        import streamlit  # noqa: F401
        return True
    except ImportError:
        return False


def install_requirements() -> bool:
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print("[FAIL] requirements.txt not found")
        return False

    print("[INFO] Installing dependencies...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Streamlit web UI")
    parser.add_argument("--port", type=int, help="Port for the web server (defaults to PORT env or 8501)")
    args = parser.parse_args()

    port = args.port if args.port is not None else int(os.getenv("PORT", 8501))

    print("[INFO] Preparing web interface...")

    if not check_streamlit_installed():
        print("[WARN] Streamlit not found. Installing dependencies...")
        if not install_requirements():
            print("[FAIL] Could not install dependencies")
            return

    web_ui_file = Path(__file__).parent / "web_ui.py"
    if not web_ui_file.exists():
        print("[FAIL] web_ui.py not found")
        return

    show_remote_status()

    print("[INFO] Starting Streamlit server...")
    print(f"[INFO] Open http://localhost:{port} in your browser")
    print("[INFO] Press Ctrl+C to stop")
    print("-" * 50)

    try:
        # Force UTF-8 mode for the Streamlit subprocess to avoid Windows charmap issues
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(web_ui_file),
            "--server.address",
            "localhost",
            "--server.port",
            str(port),
            "--browser.gatherUsageStats",
            "false",
        ], check=False, env=env)
    except KeyboardInterrupt:
        print("[INFO] Web interface stopped")
    except Exception as exc:
        print(f"[FAIL] Streamlit error: {exc}")


if __name__ == "__main__":
    main()
