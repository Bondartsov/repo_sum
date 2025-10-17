#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Скрипт: dense-only smoke-тест поиска против VM без внешних зависимостей.
# Запуск:
#   python -X utf8 scripts/rag_dense_smoke_test.py
#   # Переопределение базового URL (опционально):
#   # POSIX: RAG_BASE_URL=http://HOST:PORT python -X utf8 scripts/rag_dense_smoke_test.py
#   # PowerShell: $env:RAG_BASE_URL="http://HOST:PORT"; python -X utf8 scripts/rag_dense_smoke_test.py
#
# Действия:
#  1) Устанавливает переменные окружения RAG_SEARCH_ENDPOINT и RAG_TEXT_SEARCH_ENDPOINT.
#  2) Получает эмбеддинг через /v1/embeddings с truncate_dim=768.
#  3) Проверяет размерность эмбеддинга.
#  4) Делает dense-only запрос на /v1/search_v2 и печатает JSON-ответ.

import json
import os
import sys
import time
from urllib import request, error

BASE_FALLBACK = "http://10.61.11.54:8000"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Contract": "v1.0.0",
}

def _read_error_body(e: Exception, limit: int = 2048) -> str:
    """Читает тело ошибки, если доступно, и ограничивает длину."""
    try:
        if hasattr(e, "read"):
            raw = e.read()  # type: ignore[call-arg]
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="replace")[:limit]
            return str(raw)[:limit]
    except Exception:
        return ""
    return ""

def post_json(url: str, payload: dict, timeout: float = 60.0) -> dict:
    """POST JSON без внешних зависимостей. Возвращает распарсенный JSON.

    При ошибке печатает статус/сообщение и до 2К символов тела ответа, затем завершает процесс.
    """
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url=url, data=data, headers=HEADERS, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            body = resp.read()
            text = body.decode(charset, errors="replace")
            try:
                return json.loads(text)
            except json.JSONDecodeError as je:
                print(f"[ОШИБКА] Некорректный JSON в ответе {url}: {je}", file=sys.stderr)
                print(text[:2048], file=sys.stderr)
                sys.exit(1)
    except error.HTTPError as he:
        snippet = _read_error_body(he, 2048)
        status = getattr(he, "code", "unknown")
        print(f"[HTTP {status}] Ошибка запроса к {url}", file=sys.stderr)
        if snippet:
            print(snippet, file=sys.stderr)
        sys.exit(1)
    except error.URLError as ue:
        reason = getattr(ue, "reason", ue)
        snippet = _read_error_body(ue, 2048)
        print(f"[URL ERROR] Не удалось обратиться к {url}: {reason}", file=sys.stderr)
        if snippet:
            print(snippet, file=sys.stderr)
        sys.exit(1)

def main() -> None:
    # Форсируем UTF‑8 в stdout для корректной печати русского текста/JSON
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    base = os.environ.get("RAG_BASE_URL", BASE_FALLBACK).rstrip("/")

    # Временная перенастройка эндпоинтов в текущем процессе
    os.environ["RAG_SEARCH_ENDPOINT"] = f"{base}/v1/search_v2"
    os.environ["RAG_TEXT_SEARCH_ENDPOINT"] = f"{base}/v1/search"
    print(
        "RAG endpoints set: "
        f"RAG_SEARCH_ENDPOINT={os.environ['RAG_SEARCH_ENDPOINT']}, "
        f"RAG_TEXT_SEARCH_ENDPOINT={os.environ['RAG_TEXT_SEARCH_ENDPOINT']}"
    )

    # 1) Получаем эмбеддинг с усечением до 768
    emb_url = f"{base}/v1/embeddings"
    emb_payload = {
        "texts": ["для каких языков программирования у меня есть парсеры?"],
        "task": "retrieval.query",
        "normalize": True,
        "truncate_dim": 768,
    }
    emb_start = time.time()
    emb_resp = post_json(emb_url, emb_payload, timeout=60.0)
    emb_elapsed = time.time() - emb_start

    # Извлекаем первый вектор
    vec = None
    try:
        embs = emb_resp.get("embeddings")  # type: ignore[assignment]
        if isinstance(embs, (list, tuple)) and embs:
            vec = embs[0]
    except Exception:
        vec = None

    if not isinstance(vec, (list, tuple)):
        print("[ПРЕДУПРЕЖДЕНИЕ] Ответ /v1/embeddings не содержит корректного списка векторов.", file=sys.stderr)
        # Пытаемся продолжить, но запрос к search_v2 вероятно упадёт
        vec = []

    dim = len(vec)
    if dim != 768:
        print(f"[ПРЕДУПРЕЖДЕНИЕ] Ожидалась размерность 768, фактическая: {dim}. Продолжаю...", file=sys.stderr)
    print(f"Embedding dim={dim} (получено за {emb_elapsed:.2f} с)")

    # 2) Dense-only поиск по /v1/search_v2
    search_url = f"{base}/v1/search_v2"
    search_payload = {
        "protocol": "dense",
        "dense_vector": list(vec),  # на случай tuple
        "top_k": 10,
        "filters": {},
    }
    search_start = time.time()
    search_resp = post_json(search_url, search_payload, timeout=60.0)
    search_elapsed = time.time() - search_start

    print(f"Поиск выполнен за {search_elapsed:.2f} с. Ответ:")
    print(json.dumps(search_resp, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()