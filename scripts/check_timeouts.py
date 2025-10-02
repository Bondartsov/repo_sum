"""
Скрипт диагностики всех timeout и retry параметров в системе.

Проверяет:
- Config файлы (config.py, settings.json)
- Environment variables
- Runtime значения в памяти
- Hardcoded значения в коде
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import Config


def check_environment_variables() -> Dict[str, Any]:
    """Проверка environment variables связанных с timeout."""
    env_vars = {
        'RAG_TIMEOUT_SECONDS': os.getenv('RAG_TIMEOUT_SECONDS'),
        'RAG_MAX_RETRIES': os.getenv('RAG_MAX_RETRIES'),
        'RAG_RETRY_DELAY': os.getenv('RAG_RETRY_DELAY'),
        'OPENAI_TIMEOUT': os.getenv('OPENAI_TIMEOUT'),
        'OPENAI_RETRY_ATTEMPTS': os.getenv('OPENAI_RETRY_ATTEMPTS'),
        'OPENAI_RETRY_DELAY': os.getenv('OPENAI_RETRY_DELAY'),
    }

    return {k: v for k, v in env_vars.items() if v is not None}


def check_config_py() -> Dict[str, Any]:
    """Проверка config.py (runtime значения)."""
    try:
        config = Config()

        return {
            'RemoteServiceConfig': {
                'timeout_seconds': config.rag.remote_service.timeout_seconds,
                'max_retries': config.rag.remote_service.max_retries,
                'retry_delay': config.rag.remote_service.retry_delay,
            },
            'OpenAIConfig': {
                'retry_attempts': config.openai.retry_attempts,
                'retry_delay': config.openai.retry_delay,
            }
        }
    except Exception as e:
        return {'error': str(e)}


def check_settings_json() -> Dict[str, Any]:
    """Проверка settings.json."""
    settings_path = project_root / 'settings.json'

    if not settings_path.exists():
        return {'error': 'settings.json не найден'}

    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        result = {}

        # RAG settings
        if 'rag' in data and 'remote_service' in data['rag']:
            result['remote_service'] = data['rag']['remote_service']

        # OpenAI settings
        if 'openai' in data:
            result['openai'] = {
                k: v for k, v in data['openai'].items()
                if 'retry' in k.lower() or 'timeout' in k.lower()
            }

        return result
    except Exception as e:
        return {'error': str(e)}


def check_runtime_objects() -> Dict[str, Any]:
    """Проверка runtime объектов (требует импорта модулей)."""
    result = {}

    try:
        # RemoteVMEmbedder
        from rag.remote_embedder import RemoteVMEmbedder
        from config import Config

        config = Config()
        embedder = RemoteVMEmbedder(config.rag.embedding, config.rag.remote_service)

        result['RemoteVMEmbedder'] = {
            'timeout_seconds': embedder.timeout_seconds,
            'max_retries': embedder.max_retries,
            'retry_delay': embedder.retry_delay,
            'retry_policy.config.timeout_seconds': embedder.retry_policy.config.timeout_seconds,
            'retry_policy.config.max_attempts': embedder.retry_policy.config.max_attempts,
            'retry_policy.config.base_delay': embedder.retry_policy.config.base_delay,
            'retry_policy.config.max_delay': embedder.retry_policy.config.max_delay,
            'circuit_breaker.config.timeout_seconds': embedder.circuit_breaker.config.timeout_seconds,
            'circuit_breaker.config.failure_threshold': embedder.circuit_breaker.config.failure_threshold,
        }
    except Exception as e:
        result['RemoteVMEmbedder'] = {'error': str(e)}

    try:
        # RetryPolicy defaults
        from rag.retry_policy import RetryConfig

        retry_config = RetryConfig()
        result['RetryConfig (defaults)'] = {
            'max_attempts': retry_config.max_attempts,
            'base_delay': retry_config.base_delay,
            'max_delay': retry_config.max_delay,
            'timeout_seconds': retry_config.timeout_seconds,
        }
    except Exception as e:
        result['RetryConfig'] = {'error': str(e)}

    try:
        # CircuitBreaker defaults
        from rag.circuit_breaker import CircuitBreakerConfig

        cb_config = CircuitBreakerConfig()
        result['CircuitBreakerConfig (defaults)'] = {
            'failure_threshold': cb_config.failure_threshold,
            'timeout_seconds': cb_config.timeout_seconds,
            'success_threshold': cb_config.success_threshold,
        }
    except Exception as e:
        result['CircuitBreakerConfig'] = {'error': str(e)}

    return result


def check_hardcoded_values() -> Dict[str, List[str]]:
    """Поиск hardcoded timeout значений в коде."""
    patterns = [
        ('timeout=', 'timeout parameter'),
        ('timeout_seconds=', 'timeout_seconds parameter'),
        ('ClientTimeout(', 'aiohttp.ClientTimeout'),
        ('asyncio.wait_for(', 'asyncio.wait_for timeout'),
    ]

    files_to_check = [
        'rag/remote_embedder.py',
        'rag/remote_vector_store.py',
        'rag/event_loop_manager.py',
        'rag/retry_policy.py',
        'rag/circuit_breaker.py',
        'config.py',
    ]

    result = {}

    for file_path in files_to_check:
        full_path = project_root / file_path
        if not full_path.exists():
            continue

        matches = []
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    for pattern, desc in patterns:
                        if pattern in line and not line.strip().startswith('#'):
                            matches.append(f"Line {line_num}: {line.strip()}")
        except Exception as e:
            matches.append(f"Error: {e}")

        if matches:
            result[file_path] = matches

    return result


def format_section(title: str, data: Dict[str, Any], indent: int = 0) -> str:
    """Форматирует секцию для вывода."""
    lines = []
    prefix = "  " * indent

    lines.append(f"\n{prefix}{'=' * 60}")
    lines.append(f"{prefix}{title}")
    lines.append(f"{prefix}{'=' * 60}")

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"\n{prefix}{key}:")
                for k, v in value.items():
                    lines.append(f"{prefix}  {k}: {v}")
            elif isinstance(value, list):
                lines.append(f"\n{prefix}{key}:")
                for item in value:
                    lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")

    return "\n".join(lines)


def main():
    """Главная функция."""
    print("\n" + "=" * 80)
    print("🔍 ДИАГНОСТИКА TIMEOUT И RETRY ПАРАМЕТРОВ")
    print("=" * 80)

    # 1. Environment Variables
    env_vars = check_environment_variables()
    if env_vars:
        print(format_section("1. Environment Variables", env_vars))
    else:
        print("\n" + "=" * 60)
        print("1. Environment Variables")
        print("=" * 60)
        print("❌ Нет установленных environment variables для timeout/retry")

    # 2. config.py (runtime)
    config_data = check_config_py()
    print(format_section("2. config.py (Runtime Values)", config_data))

    # 3. settings.json
    settings_data = check_settings_json()
    print(format_section("3. settings.json", settings_data))

    # 4. Runtime Objects
    print("\n" + "=" * 60)
    print("4. Runtime Objects (Actual Instances)")
    print("=" * 60)
    print("⚠️  Инициализация объектов...")

    runtime_data = check_runtime_objects()
    for obj_name, obj_data in runtime_data.items():
        print(f"\n{obj_name}:")
        if isinstance(obj_data, dict):
            for key, value in obj_data.items():
                print(f"  {key}: {value}")

    # 5. Hardcoded Values
    hardcoded = check_hardcoded_values()
    if hardcoded:
        print("\n" + "=" * 60)
        print("5. Hardcoded Timeout Values in Code")
        print("=" * 60)
        for file_path, matches in hardcoded.items():
            print(f"\n📄 {file_path}:")
            for match in matches[:10]:  # Показываем первые 10
                print(f"  {match}")
            if len(matches) > 10:
                print(f"  ... и ещё {len(matches) - 10} совпадений")

    # 6. Рекомендации
    print("\n" + "=" * 60)
    print("6. Рекомендации")
    print("=" * 60)

    # Проверяем несоответствия
    config_timeout = config_data.get('RemoteServiceConfig', {}).get('timeout_seconds', 'N/A')

    print(f"\n✅ Config timeout_seconds: {config_timeout}")

    if config_timeout == 600:
        print("✅ HOTFIX применён корректно (timeout_seconds = 600)")
    elif config_timeout == 60:
        print("❌ HOTFIX НЕ применён! Используются старые значения (60s)")
        print("   Рекомендация: Перезапустите приложение")
    else:
        print(f"⚠️  Неожиданное значение: {config_timeout}")

    # Проверяем environment variables
    if not env_vars:
        print("\n⚠️  Environment variables не установлены")
        print("   Можно установить для переопределения config:")
        print("   export RAG_TIMEOUT_SECONDS=600")
        print("   export RAG_MAX_RETRIES=5")
        print("   export RAG_RETRY_DELAY=10.0")

    print("\n" + "=" * 80)
    print("Диагностика завершена!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
