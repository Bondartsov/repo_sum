# 🧪 СТРАТЕГИЯ ТЕСТИРОВАНИЯ RAG СИСТЕМЫ

**Дата:** 02 октября 2025
**Версия:** 1.7.0
**Статус:** Production-Ready с устойчивой CI-матрицей и контрактной поверхностью для тестов

> Обновлено на базе предыдущей версии стратегии и текущих договорённостей по рефакторингу тестового контракта. 

---

## 🎯 ОБЗОР И КЛЮЧЕВЫЕ ПРИНЦИПЫ

Стратегия строится на принципах «контрактов поверх реализаций», строгой изоляции тестов и разделения окружений. Цель — снизить хрупкость, устранить доступ к приватным деталям и гарантировать воспроизводимость.

**Основные принципы:**

* **Контракты через `Protocol`**: проверяем публичное поведение и наблюдаемость, а не приватные методы/поля.
* **Пирамида тестирования**: максимум unit, минимум E2E.
* **Изоляция режимов**: `real` и `mock` реализации переключаются маркерами/CLI-флагами; без глобальных патчей.
* **Наблюдаемость**: метрики и состояния доступны через публичный API (`get_stats()`), не через внутренние атрибуты.
* **Детерминизм async**: строгий режим `pytest-asyncio`, стабильные дедлайны, единая политика event loop на Windows.
* **Чистое окружение**: офлайн-профиль включается целевыми фикстурами, а не по умолчанию.
* **Документированные предусловия**: интеграционные/VM-тесты выполняются только при выполненных pre-checks.

---

## 🏷️ КАТЕГОРИЗАЦИЯ И МАРКЕРЫ PYTEST

В `pytest.ini` регистрируются маркеры:

```ini
[pytest]
markers =
    unit: Тесты без внешних зависимостей (по умолчанию — без маркера)
    integration: Интеграционные тесты (файлы/БД/API)
    functional: CLI/subprocess сценарии
    e2e: Сквозные сценарии с реальными сервисами
    vm: Тесты, требующие доступности VM сервиса
    real_embedder: Тесты, где нужен реальный RemoteVMEmbedder
    mock_embedder: Тесты, которые должны использовать мок-реализацию
    slow: >5s
    stress: Нагрузочные
    benchmark: Бенчмарки
    offline: Тесты, требующие офлайн-профиля
asyncio_mode = strict
```

**Классификация:**

* **Unit** — тесты **без** явных маркеров (или с `@pytest.mark.unit`), изолированные, работают с `--disable-socket`.
* **Integration** — `@pytest.mark.integration`, используют внешние зависимости.
* **Functional** — `@pytest.mark.functional`, CLI/subprocess.
* **VM** — `@pytest.mark.vm`, требуют живой сервис на удалённой VM.
* **Real/Mock** — тонкая настройка используемой реализации эмбеддера.

---

## ⚙️ ПОЛИТИКА `conftest.py` И ИЗОЛЯЦИЯ МОКОВ

**Запрещено:**

* Глобальная подмена классов в `pytest_configure` (никаких тотальных `patch('rag.remote_embedder.RemoteVMEmbedder', ...)`).

**Разрешено/обязательно:**

* Scoped-фикстуры (`session`/`module`/`function`) для условного мокинга.
* CLI-флаг `--use-mock-embedder` и маркеры `@pytest.mark.real_embedder` / `@pytest.mark.mock_embedder` — маршрутизация реализаций на этапе создания инстансов.
* Отсутствие `autouse=True` для офлайн-фикстур; офлайн-вариант включается **явно** (`@pytest.mark.offline` или явным использованием фикстуры).
* Пре-чек для `@pytest.mark.vm`: быстрый `socket.create_connection((host,port), timeout=0.5)`; при недоступности — `pytest.skip(...)`.

**Windows/async нюансы:**

```python
# conftest.py — единоразово, до тестов
import sys, asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

---

## 📐 КОНТРАКТЫ (TYPES) И НАБЛЮДАЕМОСТЬ

### `EmbedderProtocol`

Формализует публичную поверхность embedder-компонентов (CPU/Remote/Mock). Проверка соответствия — **в тестах**, а не в прод-коде (избегаем `assert isinstance(self, Protocol)` в `__init__`).

**Публичный API (обязательное):**

* `embed_texts(texts: List[str], task: Optional[str], deadline_ms: int) -> ArrayLike`
* `get_stats() -> Dict[str, Any]`
* `reset_stats() -> None`
* `warmup() -> None` (может быть noop в моке)
* `check_health() -> Dict[str, Any]`

**Структура `get_stats()` (стабильно версионированная):**

```json
{
  "schema_version": 1,
  "requests": {"total": int, "errors": int, "texts": int},
  "retry": {"total_retries": int, "attempts": int},
  "latency": {"avg_ms": float},
  "cb": {"state": "closed|open|half_open", "failure_count": int},
  "total_requests": int,
  "total_texts": int,
  "error_count": int,
  "retry_count": int,           // дублирует retry.total_retries
  "is_warmed_up": bool,
  "provider": str,
  "model_name": str
}
```

> Все тесты читают **только публичную статистику** и контрактные исключения; никаких обращений к приватным полям/методам.

---

## 🔧 УРОВЕНЬ 1: UNIT-ТЕСТЫ (OFFLINE-READY)

**Запуск:**

```bash
pytest -m "not integration and not functional and not e2e and not vm" --disable-socket -v
```

**Цели:**

* Проверка конфигураций, импортов, базовой инициализации.
* Локальная логика без внешних сервисов.
* Проверка контрактов реализаций (remote/mock) через `EmbedderProtocol`.

**Best Practices:**

* Не патчить приватные методы (`_make_single_request` и т.п.). Если нужен «spy» низкоуровневого вызова — инжектируйте тонкий транспортный интерфейс (`TransportClientProtocol`) и подменяйте **его**.
* Для асинхронных сценариев используйте `pytest.mark.asyncio` и детерминированные таймауты.
* Валидируйте метрики через `get_stats()`; не читайте внутренние счётчики объектов retry/CB напрямую.

**Типовое покрытие:**

* Конфигурации (EmbeddingConfig/VectorStoreConfig/QueryEngineConfig).
* Базовые импорты и `__all__`.
* Инициализация CPU-реализаций.
* Валидация векторного хранилища и генерации параметров коллекций.
* Кэш/health/метрики на уровне движка.

---

## 🔗 УРОВЕНЬ 2: ИНТЕГРАЦИОННЫЕ ТЕСТЫ

**Запуск:**

```bash
pytest -m "integration" -v
```

**Цели:**

* Взаимодействие компонентов (сканирование → чанкинг → эмбеддинги → индексация → поиск).
* Реалистичные данные и сценарии.
* Грациозные фоллбэки и обработка ошибок.

**Практики:**

* Конфиги и адреса — **только** из ENV (никакого хардкода).
* Проверки на качество результатов, метрики p50/p95/p99 где уместно.
* Асинхронные вызовы — через строгий режим, без смешения sync/async.

---

## 🎭 УРОВЕНЬ 3: FUNCTIONAL (CLI/SUBPROCESS)

**Запуск:**

```bash
pytest -m "functional" -v
```

**Цели:**

* Проверка CLI-команд и пользовательских сценариев.
* Гибкая валидация выводов (локализация, разные формулировки ошибок).

**Практики:**

* `subprocess.run(..., text=True, encoding="utf-8")` — корректная обработка вывода.
* Для async-зависимостей в CLI — корректное мокирование async-функций.
* Исключаем хрупкие предположения по тексту ошибок (проверяем набор допустимых сообщений).

---

## 🌐 УРОВЕНЬ 4: WEB UI (BACKEND-НАСТРОЙ)

**Запуск (backend-only):**

```bash
pytest tests/test_web_ui_vm_rag.py -v
```

**Цели:**

* Тестируем логику backend без AppTest/фронтовых фреймворков.
* Мокаем VM-сервис, сценарии поиска/индексации, ошибки и фоллбэки.

**Практики:**

* Без `AppTest`/контекст-менеджеров, только прямые вызовы backend-функций.
* Метрики UI-операций (время/успехи/ошибки) — простым классом-коллектором.

---

## ⚡ УРОВЕНЬ 5: ПРОИЗВОДИТЕЛЬНОСТЬ И СТРЕСС

**Запуск:**

```bash
pytest -m "benchmark or stress" -v
```

**Цели:**

* Throughput, латентность, профилирование памяти/CPU.
* Деградация под конкурентной нагрузкой.

**Практики:**

* Размеры батчей: 8/32/128/512 — сравнение p50/p95/p99.
* Нагрузочные сценарии (≥20 параллельных пользователей) — детект деградации.
* Отдельные окружения в CI (не смешивать с unit).

---

## 🧰 ФИКСТУРЫ, ДАННЫЕ И OFFLINE-ПРОФИЛЬ

**Данные:**

* Реалистичные образцы кода в `tests/fixtures/test_repo/`.

**Фикстуры:**

* `embedder_factory` — фабрика real/mock по маркерам/CLI/env.
* `offline_env` — **не-autouse** фикстура для офлайн-режима.
* Транспортные клиенты — инжектируемые интерфейсы для подмены в тестах (вместо патча приватных методов).

**Изоляция:**

* `reset_stats()` — обязателен перед/после сценариев, чтобы не протекало состояние.
* Параллельный запуск — `pytest -n auto --dist worksteal`.

---

## 🖧 VM-ТЕСТЫ (ИНФРАСТРУКТУРНЫЕ)

**Запуск:**

```bash
pytest -m "vm" -v
```

**Предусловия:**

* Доступность `VM_HOST`/`VM_PORT` (например, `:8000`) проверяется пре-чеком; иначе `pytest.skip`.
* UFW/iptables состояние фиксируется в логах, но отсутствие LISTEN — **операционная** проблема, не тестовая.

**Критерии:**

* `External Connectivity Test` — PASS только при доступности сервиса.
* Отчёт формирует рекомендации (запустить сервис/проверить правила).

---

## 🧪 КОНТРАКТНЫЕ ТЕСТЫ ДЛЯ РЕАЛИЗАЦИЙ

**Цель:** удостовериться, что и `RemoteVMEmbedder`, и `MockRemoteEmbedder` реализуют `EmbedderProtocol` и выдают согласованные метрики/исключения.

**Проверки:**

* `isinstance(obj, EmbedderProtocol)` с `@runtime_checkable`.
* Наличие публичных методов.
* Cогласованность `get_stats()` (включая `schema_version`).
* Доменные исключения (`VMTimeoutError`, `VMConnectionError`) при имитации ошибок — одинаковая семантика.

---

## 🧯 АНТИПАТТЕРНЫ (ЗАПРЕЩЕНО)

* Доступ к приватным методам/полям (`obj._private`, `_make_request_with_retry`, `obj.circuit_breaker.failure_count`, и т.п.).
* Глобальные патчи реализаций в `pytest_configure`.
* Жёсткие проверки конкретных текстов ошибок (с учётом локализации и контекста).
* Смешение sync/async без `pytest-asyncio` и строгих дедлайнов.

> Рекомендация: добавить линтер-правило (ruff/flake8) для запрета `._private` в `tests/**`.

---

## 🚀 CI/CD: МАТРИЦА И ПРОФИЛИ ПРОГОНА

**Матрица GitHub Actions (пример):**

```yaml
- name: Unit (real)
  run: pytest -m "not integration and not functional and not e2e and not vm" --disable-socket -v

- name: Unit (mock)
  run: pytest -m "not integration and not functional and not e2e and not vm" --disable-socket -v --use-mock-embedder

- name: Integration
  run: pytest -m "integration" -v

- name: Functional (CLI)
  run: pytest -m "functional" -v

- name: VM
  run: pytest -m "vm" -v
```

**Критерии стабильности:**

* Один и тот же поведенческий тест даёт одинаковый вердикт в `unit-real` и `unit-mock` (если сценарий не требует реального транспорта).
* Изменение реализаций не требует переписывать тесты, если контракт не менялся.
* Изменение `conftest.py` не влияет на смысл проверок (только на выбор реализаций).

---

## 📊 ОТЧЁТНОСТЬ И МОНИТОРИНГ

* HTML-репорты (`--html=...`), coverage (`--cov=...`), профилировщики (`--profile-svg`).
* Метрики: success rate, coverage, perf-тренды, flaky detection.
* Аналитика падений: корневые причины (код/тест/среда), ретроспективы.

---

## 📝 CHANGELOG 1.7.0 (02.10.2025)

* Введён контракт `EmbedderProtocol` и версионированная схема `get_stats(schema_version=1)`.
* Убраны глобальные патчи из `conftest.py`; добавлены scoped-фикстуры и CLI `--use-mock-embedder`.
* Добавлены маркеры `real_embedder`, `mock_embedder`, `vm`, `offline`; `asyncio_mode = strict`.
* Запрещён доступ к приватным методам/полям в тестах; рекомендовано линтер-правило.
* Добавлен пре-чек доступности VM для `@pytest.mark.vm`; падения по подключению трактуются как невыполненные предусловия.
* Обновлена CI-матрица: `unit-real`, `unit-mock`, `integration`, `functional`, `vm`.

---

## КОМАНДЫ ЗАПУСКА (ШПАРГАЛКА)

```bash
# Unit (real)
pytest -m "not integration and not functional and not e2e and not vm" --disable-socket -v

# Unit (mock)
pytest -m "not integration and not functional and not e2e and not vm" --disable-socket -v --use-mock-embedder

# Integration
pytest -m "integration" -v

# Functional (CLI)
pytest -m "functional" -v

# VM
pytest -m "vm" -v
```

---
