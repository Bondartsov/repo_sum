#!/usr/bin/env bash
set -euo pipefail
echo "=== OOM entries (dmesg) ==="
dmesg -T | grep -i 'Out of memory' || echo "No OOM found."
