#!/usr/bin/env python3
"""
Диагностический скрипт для инициализации OpenAI API в веб-интерфейсе.
"""
import os

def main():
    # Загрузка .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("load_dotenv: OK")
    except ImportError:
        print("load_dotenv: dotenv не установлен")
    except Exception as e:
        print(f"load_dotenv error: {e}")

    from config import get_config, reload_config

    print(f"os.getenv('OPENAI_API_KEY'): {os.getenv('OPENAI_API_KEY')}")

    cfg = get_config(require_api_key=False)
    print(f"get_config(require_api_key=False).openai.api_key: {cfg.openai.api_key}")

    cfg2 = reload_config(require_api_key=False)
    print(f"reload_config(require_api_key=False).openai.api_key: {cfg2.openai.api_key}")

    try:
        cfg3 = reload_config(require_api_key=True)
        print(f"reload_config(require_api_key=True).openai.api_key: {cfg3.openai.api_key}")
    except Exception as e:
        print(f"reload_config(require_api_key=True) error: {e}")

    # Тест инстанцирования менеджера
    try:
        from openai_integration import OpenAIManager
        manager = OpenAIManager()
        print("OpenAIManager: instantiated successfully")
    except Exception as e:
        print(f"OpenAIManager Error: {e}")

if __name__ == "__main__":
    main()