#!/usr/bin/env python3
"""
Проверка корректности .env на ВМ.

Запуск на ВМ из каталога репозитория:
  source venv/bin/activate && python scripts/validate_vm_env.py

Проверяет, что:
- RAG_SERVICE_HOST=127.0.0.1
- RAG_SERVICE_PORT задан
- RAG_*_ENDPOINT указывают на 127.0.0.1:<port>
- QDRANT_HOST=localhost, QDRANT_PORT задан
- EMBEDDING_PROVIDER=fastembed, VECTOR_STORE_PROVIDER=local
- EMBEDDING_DIMENSION == EMB_TRUNCATE_DIM (если указано)
"""

import os
from pathlib import Path

def bool_ok(cond: bool) -> str:
    return "OK" if cond else "FAIL"

def main() -> int:
    env_file = Path('.env')
    if not env_file.exists():
        print("[FAIL] .env не найден в текущей директории")
        return 1

    # Загружаем через простой парсер, чтобы не зависеть от python-dotenv на ВМ
    values = {}
    for line in env_file.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        values[k.strip()] = v.strip()

    host = values.get('RAG_SERVICE_HOST', '')
    port = values.get('RAG_SERVICE_PORT', '')
    emb = values.get('RAG_EMBEDDINGS_ENDPOINT', '')
    srch = values.get('RAG_SEARCH_ENDPOINT', '')
    idx = values.get('RAG_INDEX_ENDPOINT', '')
    qh = values.get('QDRANT_HOST', '')
    qp = values.get('QDRANT_PORT', '')
    prov_e = values.get('EMBEDDING_PROVIDER', '')
    prov_v = values.get('VECTOR_STORE_PROVIDER', '')
    dim = values.get('EMBEDDING_DIMENSION', '')
    trunc = values.get('EMB_TRUNCATE_DIM', '')

    checks = [
        ("RAG_SERVICE_HOST", bool_ok(host == '127.0.0.1'), host),
        ("RAG_SERVICE_PORT", bool_ok(bool(port)), port),
        ("RAG_EMBEDDINGS_ENDPOINT", bool_ok(emb.startswith(f"http://127.0.0.1:{port}/")), emb),
        ("RAG_SEARCH_ENDPOINT", bool_ok(srch.startswith(f"http://127.0.0.1:{port}/")), srch),
        ("RAG_INDEX_ENDPOINT", bool_ok(idx.startswith(f"http://127.0.0.1:{port}/")), idx),
        ("QDRANT_HOST", bool_ok(qh == 'localhost'), qh),
        ("QDRANT_PORT", bool_ok(bool(qp)), qp),
        ("EMBEDDING_PROVIDER", bool_ok(prov_e.lower() == 'fastembed'), prov_e),
        ("VECTOR_STORE_PROVIDER", bool_ok(prov_v.lower() == 'local'), prov_v),
    ]

    print("\n=== VM .env validation ===")
    bad = 0
    for k, status, val in checks:
        ok = status == 'OK'
        print(f"{k:26} {status:4}  {val}")
        if not ok:
            bad += 1

    if dim and trunc:
        same = (dim == trunc)
        print(f"{'EMBEDDING_DIMENSION vs EMB_TRUNCATE_DIM':26} {bool_ok(same):4}  {dim} vs {trunc}")
        if not same:
            bad += 1

    print("==========================\n")
    if bad:
        print(f"[FAIL] Найдено проблем: {bad}")
        return 2
    print("[OK] .env на ВМ выглядит корректно")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

