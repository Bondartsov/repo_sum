import argparse, os, json, pathlib, requests, time, sys, re

def iter_needles_from_synth(repo_root: pathlib.Path):
    # ищем маркеры NEEDLE_ в синтетических файлах
    for p in repo_root.rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"NEEDLE_[\w_]+", text):
            phrase = m.group(0)
            yield {"file": str(p), "phrase": phrase}

def embed(vm_base: str, texts, task="retrieval.query"):
    url = vm_base.rstrip("/") + "/embeddings"
    payload = {"texts": texts, "task": task}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    # ожидаем формат {"embeddings": [[...], ...]}
    return data.get("embeddings", data)

def qdrant_search(qdrant_url: str, collection: str, vector, limit=5):
    url = qdrant_url.rstrip("/") + f"/collections/{collection}/points/search"
    payload = {"vector": vector, "limit": limit, "with_payload": True}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries-from-synth", required=True, help="Путь к синтетическому репозиторию")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", default="code_chunks")
    ap.add_argument("--vm", required=True, help="База URL сервиса эмбеддингов, напр. http://10.61.11.54:8000")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--max-queries", type=int, default=50)
    args = ap.parse_args()

    repo = pathlib.Path(args.queries_from_synth)
    queries = list(iter_needles_from_synth(repo))
    if not queries:
        print("Нет маркеров NEEDLE_ в синтетическом репо", file=sys.stderr)
        sys.exit(2)
    queries = queries[: args.max_queries]

    hits = 0
    total = 0
    lat_embed = []
    lat_search = []

    for q in queries:
        phrase = q["phrase"]
        t0 = time.time()
        vecs = embed(args.vm, [phrase], task="retrieval.query")
        lat_embed.append(time.time() - t0)
        if not vecs:
            continue
        t1 = time.time()
        res = qdrant_search(args.qdrant_url, args.collection, vecs[0], limit=args.limit)
        lat_search.append(time.time() - t1)
        # простая эваль: считаем попадание, если хотя бы один payload содержит фразу
        points = res.get("result", [])
        ok = any(phrase in (pt.get("payload", {}).get("content", "") or "") for pt in points)
        hits += 1 if ok else 0
        total += 1

    recall_at_k = hits / total if total else 0.0
    print(json.dumps({
        "tested": total,
        "hits": hits,
        "recall_at_k": round(recall_at_k, 3),
        "latency_ms": {
            "embed_avg": round(1000 * sum(lat_embed)/max(len(lat_embed),1), 1),
            "search_avg": round(1000 * sum(lat_search)/max(len(lat_search),1), 1)
        }
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
