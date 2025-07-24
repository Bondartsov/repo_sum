#!/usr/bin/env python3
"""
Скрипт запуска веб-интерфейса анализатора репозиториев.
"""

import subprocess
import sys
import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

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
    
    # Запускаем Streamlit
    print("🌐 Запускаю веб-интерфейс...")
    print("📱 Откройте браузер и перейдите по адресу: http://localhost:8501")
    print("🛑 Для остановки нажмите Ctrl+C")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(web_ui_file),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Веб-интерфейс остановлен")
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")

if __name__ == "__main__":
    main()