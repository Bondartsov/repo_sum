#!/usr/bin/env python3
"""
Автономная утилита очистки коллекций Qdrant (HTTP/gRPC), не зависящая от остального кода проекта.

Функции:
- Показать найденные коллекции (имя, точки, размерность, distance)
- Удалить конкретную коллекцию
- Удалить все коллекции (с подтверждением)

Примеры:
  python scripts/cleanup_old_collections.py --host 10.61.11.54 --port 6333
  QDRANT_HOST=localhost QDRANT_PORT=6333 python scripts/cleanup_old_collections.py --all --yes
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any

try:
    from qdrant_client import QdrantClient
except ImportError:
    print("[FAIL] Требуется пакет qdrant-client. Установите: pip install qdrant-client")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("cleanup")


def list_collections(client: QdrantClient) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    colls = client.get_collections()
    for c in colls.collections:
        try:
            info = client.get_collection(c.name)
            vectors = getattr(getattr(getattr(info, 'config', None), 'params', None), 'vectors', None)
            if vectors and hasattr(vectors, 'size'):
                dim = vectors.size
                distance = str(vectors.distance)
            else:
                dim = 'unknown'
                distance = 'unknown'
            points = getattr(info, 'points_count', 0)
            result[c.name] = { 'points': points, 'vector_size': dim, 'distance': distance }
        except Exception as e:
            result[c.name] = { 'error': str(e) }
    return result


def delete_collection(client: QdrantClient, name: str) -> bool:
    try:
        client.delete_collection(name)
        log.info(f"Удалена коллекция: {name}")
        return True
    except Exception as e:
        log.error(f"Ошибка удаления {name}: {e}")
        return False


def _parse_host_port(host: str, port: int) -> tuple[str, int]:
    """Разбирает строку хоста, если в ней указан порт (host:port)."""
    h = host.strip()
    if ':' in h and not h.startswith(('http://', 'https://')):
        parts = h.split(':')
        # IPv6 в квадратных скобках [::1]:6333 пока не поддерживаем явно
        try:
            phost, pport = ':'.join(parts[:-1]), int(parts[-1])
            return phost or 'localhost', pport
        except ValueError:
            pass
    return h, int(port)


def _load_env_from_file() -> dict:
    """Простая загрузка переменных из файла .env в корне репозитория.
    Не тянем зависимости, парсим базово: KEY=VALUE, игнорируем комментарии.
    """
    env: dict[str, str] = {}
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        env_path = os.path.join(repo_root, '.env')
        if not os.path.exists(env_path):
            return env
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip()
    except Exception:
        pass
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description="Очистка коллекций Qdrant")
    parser.add_argument("--host", default=os.getenv("QDRANT_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    parser.add_argument("--grpc", action="store_true", help="Использовать gRPC (по умолчанию HTTP)")
    parser.add_argument("--all", action="store_true", help="Удалить все коллекции")
    parser.add_argument("--yes", action="store_true", help="Не спрашивать подтверждение")
    args = parser.parse_args()

    # Никаких вопросов по умолчанию: берём из ENV/флагов,
    # если ENV не задан — пытаемся прочитать ../.env
    if len(sys.argv) == 1:
        env_host = os.getenv('QDRANT_HOST')
        env_port = os.getenv('QDRANT_PORT')
        if not env_host or not env_port:
            file_env = _load_env_from_file()
            env_host = env_host or file_env.get('QDRANT_HOST')
            env_port = env_port or file_env.get('QDRANT_PORT')
        if env_host:
            args.host = env_host
        if env_port:
            try:
                args.port = int(env_port)
            except ValueError:
                pass
        # Если пользователь положил host в формате host:port — разберём
        args.host, args.port = _parse_host_port(args.host, args.port)

    # Поддержка случая, когда в host пришло "host:port"
    args.host, args.port = _parse_host_port(args.host, args.port)

    client = QdrantClient(host=args.host, port=args.port, prefer_grpc=args.grpc, timeout=30)

    # 1) Список коллекций
    log.info(f"Подключение к Qdrant: {args.host}:{args.port} ({'gRPC' if args.grpc else 'HTTP'})")
    collections = list_collections(client)
    names = list(collections.keys())
    print("\nНайдены коллекции:")
    if not names:
        print("  (нет коллекций)")
        return 0
    for i, name in enumerate(names, 1):
        info = collections[name]
        if 'error' in info:
            print(f"  {i}. {name}: error={info['error']}")
        else:
            print(f"  {i}. {name}: points={info['points']}, dim={info['vector_size']}, distance={info['distance']}")

    # 2) Удаление всех (без вопросов, если задано через ENV/флаги)
    env_delete_all = os.getenv('CLEANUP_DELETE_ALL', '').lower() in ('1','true','yes','on')
    if args.all or env_delete_all:
        if not (args.yes or os.getenv('CLEANUP_YES', '').lower() in ('1','true','yes','on')):
            confirm = input("\nУдалить ВСЕ коллекции? (y/N): ").strip().lower()
            if confirm not in ("y", "yes", "д", "да"):
                print("Отменено.")
                return 0
        ok_all = True
        for name in names:
            ok_all &= delete_collection(client, name)
        return 0 if ok_all else 2

    # 3) Удаление конкретной
    # 3) Удаление конкретной: можно задать через ENV, иначе спросим
    env_choice = os.getenv('CLEANUP_DELETE_NAME', '').strip()
    prompt = "\nУдалить конкретную коллекцию? Введите имя (или пусто для выхода, 'all' — удалить все): "
    choice = env_choice or input(prompt).strip()

    # Поддержка 'кнопки' удалить все — пользователь вводит 'all'/'все'
    if choice.lower() in ("all", "*", "все", "всё"):
        if not (args.yes or os.getenv('CLEANUP_YES', '').lower() in ('1','true','yes','on')):
            confirm = input("\nТочно удалить ВСЕ коллекции? (y/N): ").strip().lower()
            if confirm not in ("y", "yes", "д", "да"):
                print("Отменено.")
                return 0
        ok_all = True
        for name in names:
            ok_all &= delete_collection(client, name)
        return 0 if ok_all else 2
    if not choice:
        print("Выход без изменений.")
        return 0
    if choice not in names:
        print(f"Коллекция '{choice}' не найдена.")
        return 1
    if not args.yes:
        confirm = input(f"Удалить '{choice}'? (y/N): ").strip().lower()
        if confirm not in ("y", "yes", "д", "да"):
            print("Отменено.")
            return 0
    return 0 if delete_collection(client, choice) else 2


if __name__ == "__main__":
    raise SystemExit(main())
