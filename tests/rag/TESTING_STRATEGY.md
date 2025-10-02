# 🧪 СТРАТЕГИЯ ТЕСТИРОВАНИЯ RAG СИСТЕМЫ

**Дата:** 02 октября 2025
**Версия:** 2.0.0
**Статус:** Production-Ready с тремя режимами тестирования и контрактной поверхностью

> Полностью обновлённая стратегия после Test Contract Refactoring с поддержкой offline/mock режима.

---

## 🎯 ОБЗОР И КЛЮЧЕВЫЕ ПРИНЦИПЫ

Стратегия строится на принципах «контрактов поверх реализаций», строгой изоляции тестов и **трёх режимах тестирования** для гибкого управления зависимостями от VM инфраструктуры.

**Основные принципы:**

* **Контракты через `Protocol`**: проверяем публичное поведение и наблюдаемость, а не приватные методы/поля.
* **Три режима тестирования**: дефолт (real), mock (offline), VM-only — для разных сценариев разработки.
* **Пирамида тестирования**: максимум unit, минимум E2E.
* **Изоляция режимов**: `real` и `mock` реализации переключаются маркерами/CLI-флагами; без глобальных патчей.
* **Наблюдаемость**: метрики и состояния доступны через публичный API (`get_stats()`), не через внутренние атрибуты.
* **Детерминизм async**: строгий режим `pytest-asyncio`, стабильные дедлайны, единая политика event loop на Windows.
* **Документированные предусловия**: интеграционные/VM-тесты выполняются только при выполненных pre-checks.

---

## 🚀 ТРИ РЕЖИМА ТЕСТИРОВАНИЯ

### 1️⃣ Дефолтный режим (Real Embedder)

```bash
pytest tests/
```

**Что запускается:**
- ✅ Unit тесты (без внешних зависимостей)
- ✅ Integration тесты с **реальным RemoteVMEmbedder**
- ✅ VM тесты (требуют доступную VM)

**Требования:**
- ⚠️ Доступная VM на `10.61.11.54:8000`
- ⚠️ Запущенные сервисы: FastAPI, Qdrant, Jina v3

**Когда использовать:**
- CI/CD pipeline на master с доступом к VM
- Финальная валидация перед деплоем
- Performance тестирование

**Метрики:** 48+ минут, 100% покрытие (unit + integration + VM)

---

### 2️⃣ Mock режим (Offline Testing)

```bash
pytest tests/ --use-mock-embedder
```

**Что запускается:**
- ✅ Unit тесты с **MockRemoteEmbedder**
- ✅ Contract тесты (Protocol validation)
- ❌ VM/Integration тесты **автоматически пропускаются**

**Требования:**
- ✅ Работает **полностью offline**
- ✅ Не требует никаких внешних сервисов

**Когда использовать:**
- 🚀 Локальная разработка без VM
- 🚀 Быстрая проверка логики (16-20 сек вместо 48 мин)
- 🚀 CI для feature branches без VM доступа

**Метрики:** 16-20 секунд, ~70% покрытие (unit + contracts)

---

### 3️⃣ VM-only режим

```bash
pytest tests/ -m vm
```

**Что запускается:**
- ✅ Только VM integration тесты
- ❌ Unit тесты пропускаются

**Требования:**
- ⚠️ Доступная VM на `10.61.11.54:8000`
- ⚠️ Все сервисы запущены

**Когда использовать:**
- Проверка VM инфраструктуры после изменений
- Отладка VM connectivity issues
- Performance benchmarks

**Метрики:** 10-15 минут, только VM integration

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
    vm: Тесты, требующие доступности VM сервиса (10.61.11.54:8000)
    real_embedder: Тесты, где нужен реальный RemoteVMEmbedder
    mock_embedder: Тесты, которые должны использовать мок-реализацию
    slow: >5s
    stress: Нагрузочные
    benchmark: Бенчмарки
    offline: Тесты, требующие офлайн-профиля
asyncio_mode = strict
```

### Маркер `@pytest.mark.vm`

**Назначение:** Тесты, требующие доступную VM с запущенными сервисами.

**Обязательный комментарий:**
```python
@pytest.mark.vm  # Требует доступную VM (10.61.11.54:8000) с запущенными FastAPI, Qdrant, Jina v3 сервисами
class TestVMBackendIntegration:
    ...
```

**Файлы с этим маркером:**
- `tests/rag/test_rag_integration.py`
- `tests/rag/test_rag_performance.py`
- `tests/rag/test_vm_backend_integration.py`
- `tests/rag/test_rag_e2e_cli.py`
- `tests/rag/test_jina_v3_vs_bge_benchmarking.py`

**Поведение:**
- ✅ **Дефолт**: запускаются с реальным embedder
- ❌ **Mock режим (`--use-mock-embedder`)**: **автоматически пропускаются**
- ✅ **VM-only (`-m vm`)**: только эти тесты

### Другие маркеры

**`@pytest.mark.integration`**
- Integration тесты с внешними зависимостями (OpenAI, Qdrant, filesystem)
- Запускаются во всех режимах
- Могут использовать как real, так и mock embedder

**`@pytest.mark.real_embedder`**
- Тесты, которые **обязательно** используют RemoteVMEmbedder
- Примеры: `tests/test_remote_embedder_fixes.py`, тесты контрактов для production кода

**`@pytest.mark.mock_embedder`**
- Тесты, которые **обязательно** используют MockRemoteEmbedder
- Примеры: contract validation, isolation тесты

**`@pytest.mark.asyncio`**
- **Обязателен** для всех async test функций в режиме `asyncio_mode=strict`
- Async фикстуры должны использовать `@pytest_asyncio.fixture` вместо `@pytest.fixture`

---

## ⚙️ ПОЛИТИКА `conftest.py` И ИЗОЛЯЦИЯ МОКОВ

**Запрещено:**

* Глобальная подмена классов в `pytest_configure` (никаких тотальных `patch('rag.remote_embedder.RemoteVMEmbedder', ...)`).

**Разрешено/обязательно:**

* Scoped-фикстуры (`session`/`module`/`function`) для условного мокинга.
* CLI-флаг `--use-mock-embedder` и маркеры `@pytest.mark.real_embedder` / `@pytest.mark.mock_embedder` — маршрутизация реализаций на этапе создания инстансов.
* Отсутствие `autouse=True` для офлайн-фикстур; офлайн-вариант включается **явно** (`@pytest.mark.offline` или явным использованием фикстуры).
* Пре-чек для `@pytest.mark.vm`: быстрый `socket.create_connection((host,port), timeout=0.5)`; при недоступности — `pytest.skip(...)`.

### Ключевая фикстура: `embedder_factory`

Создаёт правильный embedder на основе:
1. CLI флага `--use-mock-embedder`
2. Pytest маркера теста (`@pytest.mark.real_embedder` / `@pytest.mark.mock_embedder`)
3. Environment variable `USE_MOCK_EMBEDDER`

```python
@pytest.fixture(scope="session")
def embedder_factory(request):
    """
    Фабрика для создания embedder (real или mock).

    Приоритет:
    1. CLI флаг --use-mock-embedder
    2. Маркер теста @pytest.mark.real_embedder / @pytest.mark.mock_embedder
    3. Environment variable USE_MOCK_EMBEDDER (default: "0" - real)
    """
```

### Auto-skip VM тестов

```python
def pytest_collection_modifyitems(config, items):
    """
    Автоматически пропускает VM тесты если:
    1. Указан флаг --use-mock-embedder
    2. VM недоступна (connectivity check)
    """
    use_mock = config.getoption("--use-mock-embedder", False)
    vm_available = check_vm_availability(vm_host, vm_port)

    for item in items:
        if "vm" in item.keywords:
            if use_mock or not vm_available:
                item.add_marker(pytest.mark.skip(reason="VM not available or mock mode"))
```

**Windows/async нюансы:**

```python
# conftest.py — единоразово, до тестов
@pytest.fixture(scope="session", autouse=True)
def setup_event_loop_policy():
    """Windows async stability via WindowsSelectorEventLoopPolicy"""
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
  "latency": {"avg_ms": float, "total_time": float},
  "cb": {"state": "closed|open|half_open", "failure_count": int},
  "total_requests": int,
  "total_texts": int,
  "error_count": int,
  "retry_count": int,
  "is_warmed_up": bool,
  "provider": str,
  "model_name": str
}
```

> Все тесты читают **только публичную статистику** и контрактные исключения; никаких обращений к приватным полям/методам.

### Transport Injection Pattern

Для тестирования HTTP взаимодействий используется `TransportClientProtocol`:

```python
@runtime_checkable
class TransportClientProtocol(Protocol):
    async def post_json(self, url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        ...
```

**Production:** `AiohttpTransportClient` (real HTTP)
**Testing:** `MockTransportClient` (spy/stub без сети)

---

## 🔧 УРОВЕНЬ 1: UNIT-ТЕСТЫ (OFFLINE-READY)

**Запуск (real embedder):**
```bash
pytest -m "not integration and not functional and not e2e and not vm" --disable-socket -v
```

**Запуск (mock embedder):**
```bash
pytest -m "not integration and not functional and not e2e and not vm" --disable-socket -v --use-mock-embedder
```

**Цели:**

* Проверка конфигураций, импортов, базовой инициализации.
* Локальная логика без внешних сервисов.
* Проверка контрактов реализаций (remote/mock) через `EmbedderProtocol`.

**Best Practices:**

* Не патчить приватные методы (`_make_single_request` и т.п.). Если нужен «spy» низкоуровневого вызова — инжектируйте `TransportClientProtocol` и подменяйте **его**.
* Для асинхронных сценариев используйте `@pytest.mark.asyncio` и детерминированные таймауты.
* Валидируйте метрики через `get_stats()`; не читайте внутренние счётчики объектов retry/CB напрямую.
* Async фикстуры: используйте `@pytest_asyncio.fixture` вместо `@pytest.fixture`.

**Типовое покрытие:**

* Конфигурации (EmbeddingConfig/VectorStoreConfig/QueryEngineConfig).
* Базовые импорты и `__all__`.
* Инициализация CPU-реализаций.
* Валидация векторного хранилища и генерации параметров коллекций.
* Кэш/health/метрики на уровне движка.

---

## 🔗 УРОВЕНЬ 2: ИНТЕГРАЦИОННЫЕ ТЕСТЫ

**Запуск (real embedder):**
```bash
pytest -m "integration" -v
```

**Запуск (mock embedder):**
```bash
pytest -m "integration" -v --use-mock-embedder
```

**Цели:**

* Взаимодействие компонентов (сканирование → чанкинг → эмбеддинги → индексация → поиск).
* Реалистичные данные и сценарии.
* Грациозные фоллбэки и обработка ошибок.

**Практики:**

* Конфиги и адреса — **только** из ENV (никакого хардкода).
* Проверки на качество результатов, метрики p50/p95/p99 где уместно.
* Асинхронные вызовы — через строгий режим, без смешения sync/async.
* Тесты с `@pytest.mark.vm` автоматически пропускаются в mock режиме.

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

## 🖧 VM-ТЕСТЫ (ИНФРАСТРУКТУРНЫЕ)

**Запуск:**
```bash
pytest -m "vm" -v
```

**Предусловия:**

* Доступность `VM_HOST:VM_PORT` (10.61.11.54:8000) проверяется пре-чеком; иначе `pytest.skip`.
* UFW/iptables состояние фиксируется в логах, но отсутствие LISTEN — **операционная** проблема, не тестовая.

**Критерии:**

* `External Connectivity Test` — PASS только при доступности сервиса.
* Отчёт формирует рекомендации (запустить сервис/проверить правила).

**Автоматический skip:**

* VM тесты **автоматически пропускаются** при использовании флага `--use-mock-embedder`
* VM тесты **автоматически пропускаются** если VM недоступна (connectivity check)

---

## 🧪 КОНТРАКТНЫЕ ТЕСТЫ ДЛЯ РЕАЛИЗАЦИЙ

**Файл:** `tests/test_embedder_contract.py`

**Цель:** удостовериться, что и `RemoteVMEmbedder`, и `MockRemoteEmbedder` реализуют `EmbedderProtocol` и выдают согласованные метрики/исключения.

**Проверки:**

* `isinstance(obj, EmbedderProtocol)` с `@runtime_checkable`.
* Наличие публичных методов.
* Cогласованность `get_stats()` (включая `schema_version`).
* Доменные исключения (`VMTimeoutError`, `VMConnectionError`) при имитации ошибок — одинаковая семантика.

---

## 🧰 ФИКСТУРЫ, ДАННЫЕ И OFFLINE-ПРОФИЛЬ

**Данные:**

* Реалистичные образцы кода в `tests/fixtures/test_repo/`.

**Ключевые фикстуры:**

* `embedder_factory` — фабрика real/mock по маркерам/CLI/env.
* `offline_env` — **не-autouse** фикстура для офлайн-режима.
* `mock_embedder_session` — session-scoped mock embedder для изоляции.
* `setup_event_loop_policy` — Windows async stability (autouse=True).
* Транспортные клиенты — инжектируемые интерфейсы для подмены в тестах (вместо патча приватных методов).

**Изоляция:**

* `reset_stats()` — обязателен перед/после сценариев, чтобы не протекало состояние.
* Параллельный запуск — `pytest -n auto --dist worksteal`.

---

## 🧯 АНТИПАТТЕРНЫ (ЗАПРЕЩЕНО)

* ❌ Доступ к приватным методам/полям (`obj._private`, `_make_request_with_retry`, `obj.circuit_breaker.failure_count`, и т.п.).
* ❌ Глобальные патчи реализаций в `pytest_configure`.
* ❌ Жёсткие проверки конкретных текстов ошибок (с учётом локализации и контекста).
* ❌ Смешение sync/async без `pytest-asyncio` и строгих дедлайнов.
* ❌ Async фикстуры с `@pytest.fixture` вместо `@pytest_asyncio.fixture` в strict mode.
* ❌ Отсутствие `@pytest.mark.asyncio` на async test функциях в strict mode.

> Рекомендация: добавлен линтер-правило `.ruff.toml` для запрета `._private` в `tests/**` (кроме `tests/mocks/`).

---

## 🚀 CI/CD: МАТРИЦА И ПРОФИЛИ ПРОГОНА

**Матрица GitHub Actions:**

```yaml
jobs:
  unit-real:
    name: Unit Tests (Real Embedder)
    run: pytest -m "not integration and not functional and not e2e and not vm" --disable-socket -v

  unit-mock:
    name: Unit Tests (Mock Embedder)
    run: pytest -m "not integration and not functional and not e2e and not vm" --disable-socket -v --use-mock-embedder

  integration-mock:
    name: Integration Tests (Mock, Fast)
    run: pytest -m "integration" -v --use-mock-embedder

  integration-real:
    name: Integration Tests (Real VM)
    run: pytest -m "integration" -v
    # Требует доступ к VM

  functional:
    name: Functional Tests (CLI)
    run: pytest -m "functional" -v

  vm:
    name: VM Backend Tests
    run: pytest -m "vm" -v
    # Требует доступ к VM
```

**Критерии стабильности:**

* Один и тот же поведенческий тест даёт одинаковый вердикт в `unit-real` и `unit-mock` (если сценарий не требует реального транспорта).
* Изменение реализаций не требует переписывать тесты, если контракт не менялся.
* Изменение `conftest.py` не влияет на смысл проверок (только на выбор реализаций).

---

## 📊 МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ

| Режим | Команда | Время | Покрытие | Когда использовать |
|-------|---------|-------|----------|-------------------|
| **Дефолт (Real)** | `pytest tests/` | 48+ минут | 100% (unit + integration + VM) | CI master, production validation |
| **Mock (Offline)** | `pytest tests/ --use-mock-embedder` | 16-20 секунд | ~70% (unit + contracts) | Локальная разработка, feature CI |
| **VM-only** | `pytest tests/ -m vm` | 10-15 минут | VM integration только | VM infrastructure check |

**Ускорение разработки:**
- Mock режим даёт **~150x ускорение** (48 минут → 18 секунд)
- Работает полностью offline без VM инфраструктуры
- Идеально для TDD цикла разработки

---

## 🔧 TROUBLESHOOTING

### Тесты падают в mock режиме

**Причина:** Тест не помечен `@pytest.mark.vm` но требует VM.

**Решение:**
```python
@pytest.mark.integration
@pytest.mark.vm  # Требует доступную VM (10.61.11.54:8000) с запущенными FastAPI, Qdrant, Jina v3 сервисами
class TestRAGIntegration:
    ...
```

### VM тесты не пропускаются в mock режиме

**Причина:** Маркер `@pytest.mark.vm` не установлен на класс или функцию.

**Проверка:**
```bash
pytest tests/rag/test_rag_integration.py --collect-only
```

### Async тесты падают с "coroutine object is not callable"

**Причина:** Отсутствует декоратор `@pytest.mark.asyncio` или async fixture использует `@pytest.fixture` вместо `@pytest_asyncio.fixture`.

**Решение:**
```python
import pytest
import pytest_asyncio

@pytest_asyncio.fixture
async def async_fixture():
    ...

@pytest.mark.asyncio
async def test_async_function():
    ...
```

### "async def functions are not natively supported"

**Причина:** В `pytest.ini` установлен `asyncio_mode = strict`, но не добавлен декоратор.

**Решение:**
```python
@pytest.mark.asyncio
async def test_my_async_code():
    ...
```

---

## 📊 ОТЧЁТНОСТЬ И МОНИТОРИНГ

* HTML-репорты (`--html=...`), coverage (`--cov=...`), профилировщики (`--profile-svg`).
* Метрики: success rate, coverage, perf-тренды, flaky detection.
* Аналитика падений: корневые причины (код/тест/среда), ретроспективы.

---

## 📝 CHANGELOG

### 2.0.0 (02.10.2025) - MAJOR UPDATE

**Три режима тестирования:**
* ✅ Дефолт (real embedder) - 48+ минут, требует VM
* ✅ Mock (offline) - 16-20 секунд, не требует VM
* ✅ VM-only - только VM integration тесты

**Async fixes:**
* ✅ Исправлены 11 async тестов (circuit_breaker, cpu_query_engine)
* ✅ Async фикстуры переведены на `@pytest_asyncio.fixture`
* ✅ Добавлены `@pytest.mark.asyncio` декораторы

**VM маркеры:**
* ✅ Добавлен `@pytest.mark.vm` на 5 файлов с VM тестами
* ✅ Автоматический skip VM тестов в mock режиме
* ✅ Обязательный комментарий с требованиями VM инфраструктуры

**Результаты:**
* ✅ `pytest tests/ --use-mock-embedder`: 47 passed, 1 skipped за 18 сек
* ✅ VM тесты: 16 skipped автоматически (было 15 failed)

### 1.7.0 (02.10.2025)

* Введён контракт `EmbedderProtocol` и версионированная схема `get_stats(schema_version=1)`.
* Убраны глобальные патчи из `conftest.py`; добавлены scoped-фикстуры и CLI `--use-mock-embedder`.
* Добавлены маркеры `real_embedder`, `mock_embedder`, `vm`, `offline`; `asyncio_mode = strict`.
* Запрещён доступ к приватным методам/полям в тестах; добавлено линтер-правило `.ruff.toml`.
* Добавлен пре-чек доступности VM для `@pytest.mark.vm`; падения по подключению трактуются как невыполненные предусловия.
* Обновлена CI-матрица: `unit-real`, `unit-mock`, `integration`, `functional`, `vm`.

---

## 💡 ПРАКТИЧЕСКИЕ ПРИМЕРЫ

### Локальная разработка (без VM)

```bash
# Быстрая проверка логики
pytest tests/ --use-mock-embedder -v

# Проверка конкретного модуля
pytest tests/rag/test_circuit_breaker.py --use-mock-embedder -v

# С покрытием
pytest tests/ --use-mock-embedder --cov=. --cov-report=html
```

### CI/CD Pipeline

```bash
# Feature branch (без VM) - быстро!
pytest tests/ --use-mock-embedder --cov=. --cov-report=html

# Master branch (с VM) - полная проверка
pytest tests/ --cov=. --cov-report=html

# VM health check
pytest tests/ -m vm -v
```

### Отладка VM проблем

```bash
# Только VM тесты
pytest tests/ -m vm -v

# Проверка connectivity
pytest tests/test_vm_availability.py -v

# Отладка конкретного VM теста
pytest tests/rag/test_vm_backend_integration.py::TestVMBackendIntegration::test_full_rag_workflow -v -s
```

---

## 📚 КОМАНДЫ ЗАПУСКА (ШПАРГАЛКА)

```bash
# ============================================================
# РЕЖИМ 1: ДЕФОЛТ (Real Embedder)
# ============================================================
pytest tests/                                      # Все тесты
pytest tests/ --cov=.                             # С покрытием

# ============================================================
# РЕЖИМ 2: MOCK (Offline, Fast)
# ============================================================
pytest tests/ --use-mock-embedder                 # Все тесты (offline)
pytest tests/ --use-mock-embedder -v              # Verbose
pytest tests/ --use-mock-embedder --cov=.         # С покрытием

# ============================================================
# РЕЖИМ 3: VM-ONLY
# ============================================================
pytest tests/ -m vm                               # Только VM тесты
pytest tests/ -m vm -v                            # Verbose

# ============================================================
# КАТЕГОРИИ ТЕСТОВ
# ============================================================
# Unit (без сети)
pytest -m "not integration and not functional and not e2e and not vm" --disable-socket -v

# Unit (mock)
pytest -m "not integration and not functional and not e2e and not vm" --disable-socket -v --use-mock-embedder

# Integration
pytest -m "integration" -v

# Integration (mock, быстро)
pytest -m "integration" -v --use-mock-embedder

# Functional (CLI)
pytest -m "functional" -v

# Performance/Stress
pytest -m "benchmark or stress" -v

# ============================================================
# ОТЛАДКА
# ============================================================
pytest tests/rag/test_circuit_breaker.py -v -s    # Один файл
pytest -k "test_async" -v                         # По имени
pytest --lf                                       # Last failed
pytest --ff                                       # Failed first
pytest -x                                         # Stop on first fail

# ============================================================
# ПАРАЛЛЕЛИЗМ
# ============================================================
pytest -n auto --dist worksteal                   # Параллельно
```

---

## 📝 КОНТРИБЬЮЦИЯ

При добавлении новых тестов:

1. **VM тесты** → добавить `@pytest.mark.vm` + комментарий
2. **Integration тесты** → добавить `@pytest.mark.integration`
3. **Async тесты** → добавить `@pytest.mark.asyncio`
4. **Async фикстуры** → использовать `@pytest_asyncio.fixture`

Проверить:
```bash
# Mock режим работает
pytest tests/ --use-mock-embedder -v

# VM тесты пропускаются
pytest tests/rag/test_rag_integration.py --use-mock-embedder -v  # Должны быть skipped
```

---

## 🔗 ССЫЛКИ

### Основная документация
- [README.md](../../README.md) - главная документация
- [rules/AGENTS.md](../../rules/AGENTS.md) - правила работы агентов
- [rules/Development Roadmap.md](../../rules/Development Roadmap.md) - дорожная карта

### Техническая документация
- [rules/Technical Architecture.md](../../rules/Technical Architecture.md) - архитектура
- [rules/Technical Debt.md](../../rules/Technical Debt.md) - технический долг
- [test_contract_implementation_plan.md](../../test_contract_implementation_plan.md) - план рефакторинга тестов

### Конфигурация
- [conftest.py](../conftest.py) - конфигурация pytest
- [pytest.ini](../../pytest.ini) - настройки pytest
- [.ruff.toml](../../.ruff.toml) - линтер правила
- [.claude/CLAUDE.md](../../.claude/CLAUDE.md) - правила разработки

---

**Автор:** Test Infrastructure Team
**Последнее обновление:** 2 октября 2025
**Статус:** Production-Ready ✅
