import os
import time
import math
import pytest

from rag.remote_vector_store import RemoteVMVectorStore


def _compute_percentiles(values):
    if not values:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan")}
    s = sorted(values)
    n = len(s)

    def interp(p):
        if n == 1:
            return s[0]
        k = (p / 100.0) * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return s[int(k)]
        return s[f] * (c - k) + s[c] * (k - f)

    return {"p50": interp(50), "p95": interp(95), "p99": interp(99)}


def test_search_smoke_perf():
    if os.getenv("RUN_SEARCH_SMOKE") != "1":
        pytest.skip("RUN_SEARCH_SMOKE != '1' — smoke/perf тест поиска пропущен по умолчанию")

    rvs = RemoteVMVectorStore()  # дефолтный транспорт
    iterations_per_topk = 10
    top_ks = (10, 50, 100)
    perf_p95_threshold = float(os.getenv("PERF_P95_SEC", "10.0"))

    for top_k in top_ks:
        durations = []
        for _ in range(iterations_per_topk):
            t0 = time.perf_counter()
            results = rvs.search_by_text("smoke test", top_k=top_k)
            t1 = time.perf_counter()
            durations.append(t1 - t0)
            assert isinstance(results, list)  # не логируем содержимое

        pct = _compute_percentiles(durations)
        p50, p95, p99 = pct["p50"], pct["p95"], pct["p99"]

        # Краткий отчёт без контента запросов/ответов
        print(f"[SMOKE]/search top_k={top_k} count={len(durations)} "
              f"p50={p50:.3f}s p95={p95:.3f}s p99={p99:.3f}s")

        # Основной ассерт по p95
        assert p95 <= perf_p95_threshold, (
            f"p95 {p95:.3f}s > PERF_P95_SEC={perf_p95_threshold:.3f}s при top_k={top_k}"
        )