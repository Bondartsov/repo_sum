#!/usr/bin/env bash
set -euo pipefail
INTERVAL=1
OUT="./uvicorn_mem.csv"
PATTERN="uvicorn"
PORT="8000"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) INTERVAL="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --pattern) PATTERN="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

pid=""
# try find by port first
if command -v lsof >/dev/null 2>&1; then
  pid=$(lsof -iTCP -sTCP:LISTEN -nP | awk -v p=":${PORT}" '$9 ~ p {print $2; exit}')
fi
if [[ -z "$pid" ]]; then
  pid=$(pgrep -f "$PATTERN" | head -n1 || true)
fi
if [[ -z "$pid" ]]; then
  echo "PID not found for pattern=$PATTERN port=$PORT" >&2
  exit 2
fi

echo "Monitoring PID=$pid every ${INTERVAL}s -> $OUT"
echo "t_seconds,rss_kb,cpu_percent" > "$OUT"
t0=$(date +%s)
# prime CPU
if command -v ps >/dev/null 2>&1; then
  ps -p "$pid" -o %cpu= >/dev/null 2>&1 || true
fi

while kill -0 "$pid" >/dev/null 2>&1; do
  now=$(date +%s)
  # VmRSS from /proc
  rss_kb=$(grep -i '^VmRSS:' /proc/$pid/status | awk '{print $2}')
  # CPU via ps (approx)
  cpu=$(ps -p "$pid" -o %cpu= | awk '{print $1}')
  echo "$((now - t0)),$(echo $rss_kb | tr -d ' '),${cpu:-0}" >> "$OUT"
  sleep "$INTERVAL"
done

echo "Process $pid exited."
