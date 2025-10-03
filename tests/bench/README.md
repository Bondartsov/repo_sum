# Тестовый стенд для проверки оптимизаций RAM

Этот набор скриптов позволяет воспроизводимо тестировать изменения из `rules/rerfactor_oom.md`:
- генерировать синтетический репозиторий кода,
- запускать индексацию в двух профилях (baseline/candidate) под разными переменными окружения,
- сэмплировать память и CPU процесса индексации,
- собирать сводные метрики (время, пик RAM),
- (опционально) прогонять A/B-проверку качества поиска в Qdrant.

## Структура
- `generate_synthetic_repo.py` — генератор тестового репозитория с крупными и мелкими блоками кода.
- `run_ab.py` — запуск базового и кандидатного прогона индексации с замером метрик.
- `memory_sampler.py` — утилита для автономного мониторинга RSS процесса (если нужно отдельно).
- `monitor_mem.sh` — мониторинг памяти uvicorn на ВМ (Linux).
- `check_oom_linux.sh` — быстрый детектор OOM по `dmesg` на ВМ (Linux).
- `retrieval_ab.py` — A/B проверка качества поиска через Qdrant (опционально).
- `run_index_benchmark.ps1` — оболочка для Windows для запусков индексации с метриками.
- `requirements.txt` — зависимости (только для тестового стенда).

## Быстрый старт (Windows ПК + Linux ВМ)
1. Установить зависимости (на ПК):
   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Сгенерировать синтетический репозиторий (на ПК):
   ```powershell
   python .\generate_synthetic_repo.py --out-dir ..\..\..\synthetic_repo --files 200 --large-classes 10 --large-class-lines 2000 --small-func-files 50 --seed 42
   ```

3. Прогнать baseline и candidate (на ПК):
   - Пример команды индексации: `python .\web_ui.py --index ..\..\..\synthetic_repo`
   - Запуск:
     ```powershell
     python .\run_ab.py        --index-cmd "python .\web_ui.py --index ..\..\..\synthetic_repo"        --metrics-dir ..\..\..\metrics        --profile baseline --env "INDEX_BATCH_SIZE=512" --env "EMBED_BATCH_MIN=8" --env "EMBED_BATCH_MAX=128"
     python .\run_ab.py        --index-cmd "python .\web_ui.py --index ..\..\..\synthetic_repo"        --metrics-dir ..\..\..\metrics        --profile candidate --env "INDEX_BATCH_SIZE=128" --env "EMBED_BATCH_MIN=1" --env "EMBED_BATCH_MAX=8" --env "CHUNK_MAX_TOKENS=768" --env "TRUNCATE_DIM=512"
     ```

4. На ВМ параллельно запустить монитор памяти uvicorn (Linux):
   ```bash
   ./monitor_mem.sh --interval 1 --out /tmp/uvicorn_mem.csv
   ./check_oom_linux.sh
   ```

5. Сравнить метрики:
   - Пики RAM в CSV/JSON внутри папки `metrics`.
   - Проверить `dmesg` на ВМ: `Out of memory` — должно быть **0** записей.

6. (Опционально) Проверить качество поиска:
   ```powershell
   python .\retrieval_ab.py --queries-from-synth ..\..\..\synthetic_repo --qdrant-url http://localhost:6333 --collection code_chunks --vm http://10.61.11.54:8000
   ```


**FYI**
# Как тестировать (пошаговая методика)

## A. Подготовка (ПК, Windows)

1. **Установить зависимости тест-стенда** (не трогает прод):
   ```powershell
   cd tests\bench
   python -m pip install -r requirements.txt

2. **Сгенерировать синтетический репозиторий** (крупные классы + мелкие функции):

   ```powershell
   python .\generate_synthetic_repo.py --out-dir ..\..\..\synthetic_repo --files 200 --large-classes 10 --large-class-lines 2000 --small-func-files 50 --seed 42
   

## B. Мониторинг на ВМ (Linux)

На ВМ в отдельной SSH‑сессии:

```bash
chmod +x tests/bench/monitor_mem.sh tests/bench/check_oom_linux.sh
./tests/bench/monitor_mem.sh --interval 1 --out /tmp/uvicorn_mem.csv

# Периодически:
./tests/bench/check_oom_linux.sh
```

## C. Прогон baseline (ПК)

1. **Быстрый запуск через PowerShell‑оболочку**:

   ```powershell
   # Пример команды индексации (адаптируйте путь к вашему index-скрипту/CLI)
   $cmd = 'python .\web_ui.py --index ..\..\..\synthetic_repo'

   # Baseline: «как есть» (большие батчи и без ограничений)
   .\tests\bench\run_index_benchmark.ps1 `
     -IndexCmd $cmd `
     -MetricsDir ..\..\..\metrics `
     -Profile baseline `
     -Env "INDEX_BATCH_SIZE=512" `
     -Env "EMBED_BATCH_MIN=8" `
     -Env "EMBED_BATCH_MAX=128"
   ```

2. **Результат**: в `metrics\` появятся:

   * `*_baseline_mem.csv` — кривая RAM/CPU,
   * `*_baseline_summary.json` — сводка (пик RSS, время, код возврата),
   * `*_baseline_stdout.log`, `*_baseline_stderr.log` — логи индексации.

---

## D. Прогон candidate (ПК)

**Кандидат** с настройками из `rerfactor_oom.md` (мягкие ограничения):

```powershell
$cmd = 'python .\web_ui.py --index ..\..\..\synthetic_repo'

.\tests\bench\run_index_benchmark.ps1 `
  -IndexCmd $cmd `
  -MetricsDir ..\..\..\metrics `
  -Profile candidate `
  -Env "INDEX_BATCH_SIZE=128" `
  -Env "EMBED_BATCH_MIN=1" `
  -Env "EMBED_BATCH_MAX=8" `
  -Env "CHUNK_MAX_TOKENS=768" `
  -Env "TRUNCATE_DIM=512"
```

> **Примечание.** Если параметры читаются из `.env`/YAML — отразите связку. Иначе читайте их в коде (через `os.getenv(...)`).

---

## E. Анализ результатов

1. **ПК**: сравнить `*_baseline_summary.json` vs `*_candidate_summary.json`:

   * `peak_rss_mb` у кандидата **существенно ниже**,
   * `return_code` — **0**,
   * `elapsed_sec` — допустимый рост (см. KPI).

2. **ВМ**: проверить `/tmp/uvicorn_mem.csv`:

   * RSS uvicorn **не должен** расти до десятков гигабайт,
   * `./tests/bench/check_oom_linux.sh` — записей `Out of memory` быть **не должно**.

---

## F. (Опционально) A/B‑проверка качества поиска

Если проиндексирован `synthetic_repo` в Qdrant:

```powershell
python .\tests\bench\retrieval_ab.py `
  --queries-from-synth ..\..\..\synthetic_repo `
  --qdrant-url http://localhost:6333 `
  --collection code_chunks `
  --vm http://10.61.11.54:8000 `
  --limit 5 --max-queries 50
```

Скрипт берёт маркеры `NEEDLE_...` из синтетики, эмбеддит фразы через VM `/embeddings`, делает `search_points` в Qdrant и считает **recall@k**.

---

## Технические детали по файлам

### 1) `tests/bench/run_ab.py`

* Запускает **любой** индексатор командой (`--index-cmd`), например:
  `python .\web_ui.py --index D:\Scripts_Python\repo_sum`
* Применяет env‑переменные (`--env KEY=VALUE`) для батчей/лимитов.
* Сэмплирует **память/CPU дочернего процесса** (psutil), пишет CSV и JSON‑сводку.
* Возвращает код 0/1 (можно цеплять в CI).

### 2) `tests/bench/run_index_benchmark.ps1`

* Обёртка над `run_ab.py` для Windows, удобная прокладка `-Env`.
* Логи и метрики в `-MetricsDir`.

### 3) `tests/bench/monitor_mem.sh` (ВМ)

* Находит PID uvicorn (по порту 8000 или по имени процесса) и пишет `uvicorn_mem.csv` (t, RSS, CPU).

### 4) `tests/bench/check_oom_linux.sh` (ВМ)

* Выводит все записи OOM из `dmesg -T`.

### 5) `tests/bench/generate_synthetic_repo.py`

* Создаёт Python‑файлы с **крупными классами** и **пачками мелких функций**, помеченных `NEEDLE_...`.
* Позволяет быстро воспроизводить тяжёлые кейсы и проверять нарезку по блокам.

### 6) `tests/bench/retrieval_ab.py` (опционально)

* Эмбеддит фразы через ваш VM `/embeddings`,
* Ищет в `Qdrant /collections/<name>/points/search`,
* Выводит `recall@k` и средние задержки embed/search.

---

## Что считать «пройденным тестом» (в связке с KPI)

* **Ноль OOM** в `dmesg` на ВМ за весь прогон.
* **Пик RSS** (по `*_candidate_summary.json`) **ниже** baseline минимум на 30–60%.
* **Время индексации** — в пределах оговорённого порога.
* **Recall@k** по синтетическим маркерам — **не хуже** baseline (±2%).

---

## В какие параметры «крутить ручки»

**На ПК:**

* `INDEX_BATCH_SIZE` — 64 → 128 → 256 (баланс скорость/память),
* `EMBED_BATCH_MIN/MAX` — 1–8 (снижение пиков RAM при эмбеддинге),
* `CHUNK_MAX_TOKENS` — 512 → 768 → 1024 (контроль длины входа),
* `TRUNCATE_DIM` — 512/768 вместо 1024 (меньше footprint индекса, быстрее поиск).

**На ВМ:**

* Ограничить конкуренцию uvicorn (1–2 одновременных запроса),
* OMP/MKL потоки = 1,
* (временно) swap 32–64 ГБ для защиты от внезапного kill во время тестов.

