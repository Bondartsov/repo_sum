#!/usr/bin/env python3
"""
Скрипт запуска веб-интерфейса анализатора репозиториев.
"""

import subprocess
import sys
import os
import argparse
import socket
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path


def resolve_host_name(host: str) -> str:
    try:
        return socket.gethostbyaddr(host)[0]
    except Exception:
        return host


def http_get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status_code = response.getcode()
            body = response.read()
            try:
                body_text = body.decode('utf-8')
            except UnicodeDecodeError:
                body_text = body.decode('utf-8', errors='ignore')
            return status_code, body_text
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc))
    except Exception as exc:
        raise RuntimeError(str(exc))


def show_remote_status():
    host = os.getenv("RAG_SERVICE_HOST", "10.61.11.54")
    vm_user = os.getenv("VM_USER", "user")
    rag_port = int(os.getenv("RAG_SERVICE_PORT", "8000"))
    health_endpoint = os.getenv("RAG_HEALTH_ENDPOINT", "/health")
    rag_health_url = f"http://{host}:{rag_port}{health_endpoint if health_endpoint.startswith('/') else '/' + health_endpoint}"

    qdrant_host = os.getenv("QDRANT_HOST", host)
    if qdrant_host in {"localhost", "127.0.0.1"}:
        qdrant_host = host
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}"

    vm_name = resolve_host_name(host)
    print("┌────────────────────────────┐")
    print("│  Проверка удалённых сервисов  │")
    print("└────────────────────────────┘")
    print(f"ℹ️  Удалённая ВМ: {vm_user}@{host} ({vm_name})")

    def print_status(label: str, url: str, expect_json: bool = True):
        try:
            status_code, body = http_get(url, timeout=5)
            if status_code == 200:
                if expect_json:
                    try:
                        data = json.loads(body)
                        status_value = data.get('status') or data.get('result') or 'ok'
                        info = status_value
                        if isinstance(status_value, dict):
                            info = status_value.get('status', 'ok')
                        print(f"✅ {label}: {info}")
                    except Exception:
                        snippet = body.strip().replace('
', ' ')
                        print(f"✅ {label}: {snippet[:80]}")
                else:
                    print(f"✅ {label}: HTTP {status_code}")
            else:
                snippet = body.strip().replace('
', ' ')
                print(f"⚠️ {label}: HTTP {status_code} ({snippet[:80]})")
        except Exception as exc:
            print(f"❌ {label}: {exc}")

    print_status("RAG сервис", rag_health_url, expect_json=True)
    print_status("Qdrant", qdrant_url, expect_json=True)
    print("")

def check_streamlit_installed():
    """Проверяет установлен ли Streamlit"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_requirements():
    """Устанавливает зависимости"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        print("📦 Устанавливаю зависимости...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
    else:
        print("❌ Файл requirements.txt не найден")
        return False
    return True

def main():
    """Основная функция запуска"""
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Запуск веб-интерфейса анализатора репозиториев")
    parser.add_argument("--port", type=int, help="Порт для веб-сервера (по умолчанию из .env или 8501)")
    args = parser.parse_args()
    
    # Определяем порт: CLI > env > default
    port = args.port
    if port is None:
        port = int(os.getenv("PORT", 8501))
    
    print("🚀 Запуск веб-интерфейса анализатора репозиториев...")
    
    # Проверяем установлен ли Streamlit
    if not check_streamlit_installed():
        print("⚠️  Streamlit не установлен. Устанавливаю зависимости...")
        if not install_requirements():
            print("❌ Ошибка установки зависимостей")
            return
    
    # Проверяем наличие веб-интерфейса
    web_ui_file = Path(__file__).parent / "web_ui.py"
    if not web_ui_file.exists():
        print("❌ Файл web_ui.py не найден")
        return

    show_remote_status()

    # Запускаем Streamlit
    print("🌐 Запускаю веб-интерфейс...")
    print(f"📱 Откройте браузер и перейдите по адресу: http://localhost:{port}")
    print("🛑 Для остановки нажмите Ctrl+C")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(web_ui_file),
            "--server.address", "localhost",
            "--server.port", str(port),
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Веб-интерфейс остановлен")
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")

if __name__ == "__main__":
    main()