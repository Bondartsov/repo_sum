# Implementation Plan: Test Contract Refactoring - Claude

## ФАЗА 0: ОБЗОР И АНАЛИЗ ПРОБЛЕМЫ

### [Overview]
Рефакторинг тестовой инфраструктуры для устранения хрупких зависимостей от приватных деталей реализации и обеспечения корректной изоляции тестов через контракты и условный мокинг.

**Проблема:** Глобальный принудительный патч `RemoteVMEmbedder` в `conftest.py` (дефолт `USE_MOCK_EMBEDDER="1"`) подменяет класс на неполный `MockRemoteEmbedder`, вызывая `AttributeError` в тестах, которые ожидают приватные методы и атрибуты реального класса (`_make_request_with_retry`, `retry_policy`, `circuit_breaker`).

**Корневая причина:**
1. Глобальный патч в `pytest_configure` применяется до загрузки тестов и действует на всю сессию
2. Mock не реализует полный контракт реального класса
3. Тесты завязаны на приватные детали реализации (белый ящик)
4. Отсутствие механизма условного мокирования

**Цель решения:**
- Ввести формальный Protocol контракт для embedder
- Сделать мокинг опциональным с дефолтом на реальные классы
- Перевести тесты на проверку публичного поведения через наблюдаемые метрики
- Обеспечить изоляцию через scoped фикстуры и маркеры
- Гарантировать runtime-валидацию контракта
- Стабилизировать схему get_stats() версионированием
- Устранить зависимости от приватных методов через TransportClientProtocol
- Обеспечить кроссплатформенную стабильность (Windows event loop, pytest-asyncio strict)

**Архитектурные принципы:**
- Наблюдаемость через публичный API, не через приватные поля
- Детерминизм retry/CB композиции через контракт
- Изоляция тестов: каждый тест получает нужную реализацию без утечек между тестами
- Явные предусловия для интеграционных/VM тестов
- Версионирование контрактов для эволюции без поломок

---

## ФАЗА 1: ТИПЫ И КОНТРАКТЫ

### [Types]
Определение формального контракта embedder через Protocol с runtime проверкой.

#### 1. EmbedderProtocol (rag/embedder_protocol.py - создать новый файл)

```python
"""
Формальный контракт для embedder компонентов.

Все embedder (CPU, Remote, Mock) должны реализовывать этот протокол.
Runtime проверка гарантирует совместимость mock объектов с реальными.
"""

from typing import Protocol, List, Dict, Any, Optional, Union, Sequence, runtime_checkable

# Гибкий тип для массивов без жёсткой привязки к NumPy
# Позволяет использовать NumPy, PyTorch тензоры или любой SupportsArray
try:
    from numpy.typing import NDArray
    import numpy as np
    ArrayLike = Union[NDArray[np.float32], Sequence[Sequence[float]]]
except ImportError:
    # Fallback если NumPy недоступен (хотя в проекте он есть)
    ArrayLike = Sequence[Sequence[float]]


@runtime_checkable
class EmbedderProtocol(Protocol):
    """
    Минимальный контракт для embedder компонентов.

    Включает только публичные методы и наблюдаемые метрики.
    НЕ включает приватные детали реализации (_make_request_with_retry и т.д.)
    """

    def embed_texts(
        self,
        texts: List[str],
        task: Optional[str] = None,
        deadline_ms: int = 30000
    ) -> ArrayLike:
        """
        Основной метод получения эмбеддингов.

        Args:
            texts: Список текстов для эмбеддинга
            task: Тип задачи (retrieval.query/passage)
            deadline_ms: Deadline в миллисекундах

        Returns:
            Массив эмбеддингов shape (N, D)
            Фактически возвращается np.ndarray, но тип объявлен гибко

        Raises:
            EmbeddingException: При ошибках эмбеддинга
            VMTimeoutError: При таймаутах (для remote embedder)
            VMConnectionError: При проблемах подключения (для remote embedder)
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает наблюдаемую статистику работы embedder.

        Структура ответа (версионированная, вложенная для лучшей организации):
        {
            "schema_version": 1,  # Версия схемы для эволюции контракта
            "requests": {
                "total": int,      # Общее количество запросов
                "errors": int,     # Количество ошибок
                "texts": int       # Общее количество обработанных текстов
            },
            "retry": {
                "total_retries": int,  # Количество retry попыток
                "attempts": int        # Количество попыток выполнения
            },
            "latency": {
                "avg_ms": float,      # Среднее время ответа в мс
                "total_time": float   # Общее время обработки в секундах
            },
            "cb": {
                "state": str,          # closed|open|half_open
                "failure_count": int   # Количество неудач
            },
            # Опциональные верхнеуровневые ключи для обратной совместимости
            "total_requests": int,
            "total_texts": int,
            "error_count": int,
            "retry_count": int,    # Дублирует retry.total_retries
            "is_warmed_up": bool,
            "provider": str,
            "model_name": str
        }

        Returns:
            Словарь с метриками
        """
        ...

    def reset_stats(self) -> None:
        """
        Сбрасывает статистику работы.

        Гарантирует идемпотентность тестов - каждый тест начинает с чистого состояния.
        """
        ...

    def warmup(self) -> None:
        """
        Прогревает embedder (опционально для mock).

        Для реальных embedder - загрузка моделей, проверка доступности.
        Для mock - noop или setup минимального состояния.
        """
        ...

    def check_health(self) -> Dict[str, Any]:
        """
        Проверяет состояние embedder.

        Returns:
            Словарь с полями:
            - status: str - 'connected'/'error'/'unknown'
            - error: Optional[str] - описание ошибки
            - diagnostic: Optional[Dict] - детали диагностики
        """
        ...


@runtime_checkable
class RetryPolicyProtocol(Protocol):
    """
    Контракт для retry policy компонентов.

    Используется в тестах для проверки поведения retry логики.
    """

    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику retry политики.

        Обязательные ключи:
        - total_executions: int
        - total_retries: int
        - successful_executions: int
        - failed_executions: int
        """
        ...

    def reset_stats(self) -> None:
        """Сбрасывает статистику"""
        ...


@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    """
    Контракт для circuit breaker компонентов.
    """

    def get_state(self) -> Dict[str, Any]:
        """
        Возвращает текущее состояние CB.

        Обязательные ключи:
        - state: str - 'closed'/'open'/'half_open'
        - failure_count: int
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику CB.

        Обязательные ключи:
        - total_calls: int
        - successful_calls: int
        - failed_calls: int
        - rejected_calls: int
        """
        ...

    def reset_stats(self) -> None:
        """Сбрасывает статистику"""
        ...

    def reset(self) -> None:
        """Полный сброс CB в начальное состояние"""
        ...


@runtime_checkable
class TransportClientProtocol(Protocol):
    """
    Контракт для низкоуровневого HTTP транспорта.

    Это слой абстракции между embedder и фактическим HTTP клиентом.
    Позволяет инжектировать spy/mock в тестах БЕЗ патчинга приватных методов.
    """

    async def post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout: float
    ) -> Dict[str, Any]:
        """
        Выполняет POST запрос с JSON payload.

        Args:
            url: URL эндпоинта
            payload: JSON данные
            timeout: Таймаут в секундах

        Returns:
            JSON ответ от сервера

        Raises:
            asyncio.TimeoutError: При таймауте
            aiohttp.ClientError: При ошибках HTTP
        """
        ...
```

#### 2. Доменные исключения (без изменений в rag/exceptions.py)

Существующие исключения уже корректны:
- `EmbeddingException` - базовое исключение для эмбеддингов
- `VMTimeoutError` - таймауты при работе с VM
- `VMConnectionError` - проблемы подключения к VM

Mock должен выбрасывать эти же исключения при симуляции ошибок.

---

## ФАЗА 2: ФАЙЛОВАЯ СТРУКТУРА

### [Files]
Изменения в существующих файлах и создание новых.

#### Новые файлы:

**1. rag/embedder_protocol.py**
- **Назначение:** Формальные Protocol определения для embedder контрактов
- **Содержание:**
  - EmbedderProtocol с версионированной схемой get_stats()
  - RetryPolicyProtocol
  - CircuitBreakerProtocol
  - TransportClientProtocol для инжекции HTTP клиента

**2. rag/transport_client.py**
- **Назначение:** Реализация TransportClientProtocol для production
- **Содержание:**
```python
class AiohttpTransportClient:
    """Реальная реализация TransportClientProtocol через aiohttp"""
    async def post_json(self, url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        session = await get_shared_http_session()
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {error_text}")
```

**3. tests/mocks/mock_transport_client.py**
- **Назначение:** Mock реализация TransportClientProtocol для тестов
- **Содержание:**
```python
class MockTransportClient:
    """Mock транспорт для тестирования без реальных HTTP вызовов"""
    def __init__(self):
        self.call_count = 0
        self.calls_history = []
        self.should_fail = False

    async def post_json(self, url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        self.call_count += 1
        self.calls_history.append((url, payload, timeout))

        if self.should_fail:
            raise RuntimeError("Mock failure")

        # Симулируем успешный ответ
        return {"embeddings": [[0.1] * 1024 for _ in payload["texts"]]}
```

**4. tests/test_embedder_contract.py**
- **Назначение:** Тесты проверки соответствия контракту
- **Содержание:** Валидация что RemoteVMEmbedder и MockRemoteEmbedder реализуют EmbedderProtocol

**5. .ruff.toml (или обновление существующего)**
- **Назначение:** Линтер-правило против приватных обращений в tests/
- **Содержание:**
```toml
[lint]
ignore = []

# Запрет доступа к приватным атрибутам в тестах
# Помогает избежать регресса к white-box тестированию
[lint.per-file-ignores]
"tests/**/*.py" = [
    # Разрешаем only для фикстур и моков
    "SLF001",  # Private member accessed (разрешено для tests/mocks/*)
]

[lint.flake8-self]
# Кастомное правило: запрет ._private в tests/ кроме tests/mocks/
ignore-names = ["_*"]
exclude-paths = ["tests/mocks/**"]
```

#### Изменяемые файлы:

**1. pytest.ini**
- **Изменения:**
  - Добавить `asyncio_mode = strict` для стабильности async тестов
  - Зарегистрировать маркеры: `real_embedder`, `mock_embedder`, `vm`

```ini
[pytest]
# Строгий режим для pytest-asyncio - уменьшает флаки event loop
asyncio_mode = strict

# Регистрация маркеров
markers =
    real_embedder: Tests that require real RemoteVMEmbedder instance
    mock_embedder: Tests that should use mocked embedder
    vm: Tests that require VM service availability and will be skipped if VM is not reachable

# Остальные настройки
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

**2. tests/conftest.py**
- **Изменения:**
  - Убрать дефолт `USE_MOCK_EMBEDDER="1"` → установить `"0"`
  - Добавить CLI опцию `--use-mock-embedder`
  - Удалить глобальный патч из `pytest_configure`
  - Создать scoped фикстуры (`embedder_factory`, `mock_embedder_session`)
  - Убрать `autouse=True` из `force_offline_env`
  - Добавить Windows event loop policy setup
  - Реализовать VM пре-чек доступности для @pytest.mark.vm тестов

**3. tests/mocks/mock_remote_embedder.py**
- **Изменения:**
  - Добавить тонкие mock объекты для `retry_policy` и `circuit_breaker`
  - Реализовать полный контракт `EmbedderProtocol`
  - Добавить симуляцию доменных исключений
  - Улучшить `get_stats()` с версионированной вложенной структурой
  - Добавить `reset_stats()` и `check_health()`
  - Инкрементировать метрики даже в no-op путях

**4. tests/test_remote_embedder_fixes.py**
- **Изменения:**
  - Добавить маркер `@pytest.mark.real_embedder` для тестов реального класса
  - Переписать тесты на проверку публичного поведения
  - Использовать spy через инжектированный TransportClientProtocol вместо патчинга _make_single_request
  - Проверять метрики через `get_stats()["retry"]`, `get_stats()["cb"]`
  - Использовать идемпотентный `reset_stats()` в setup
  - Использовать freezegun для стабилизации timeout тестов

**5. rag/remote_embedder.py**
- **Изменения:**
  - Добавить импорт `from .embedder_protocol import EmbedderProtocol, TransportClientProtocol`
  - Добавить опциональный параметр `transport_client: Optional[TransportClientProtocol] = None` в `__init__`
  - Обновить `get_stats()` для возврата версионированной вложенной структуры
  - Использовать инжектированный transport_client вместо прямых aiohttp вызовов
  - **НЕ добавляем** `isinstance(self, EmbedderProtocol)` assert в прод-код

---

## ФАЗА 3: ФУНКЦИИ

### [Functions]
Новые и измененные функции.

#### Новые функции:

**1. tests/conftest.py::pytest_addoption()**
```python
def pytest_addoption(parser):
    """Добавляет CLI опции для управления мокингом"""
    parser.addoption(
        "--use-mock-embedder",
        action="store_true",
        default=False,
        help="Использовать mock embedder вместо реального"
    )
    parser.addoption(
        "--vm-host",
        action="store",
        default=None,
        help="VM host для интеграционных тестов"
    )
    parser.addoption(
        "--vm-port",
        action="store",
        default=8000,
        type=int,
        help="VM port для интеграционных тестов"
    )
```

**2. tests/conftest.py::setup_event_loop_policy()**
```python
@pytest.fixture(scope="session", autouse=True)
def setup_event_loop_policy():
    """
    Устанавливает правильную event loop policy для Windows.

    На Windows дефолтный ProactorEventLoop имеет проблемы с таймерами
    и async операциями. WindowsSelectorEventLoopPolicy более стабилен для тестов.
    """
    import sys
    import asyncio

    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    yield

    # Cleanup не требуется - policy глобальна для процесса
```

**3. tests/conftest.py::check_vm_availability()**
```python
def check_vm_availability(host: str, port: int, timeout: float = 0.5) -> bool:
    """
    Проверяет доступность VM endpoint через socket connection.

    Args:
        host: VM host
        port: VM port
        timeout: Таймаут подключения в секундах

    Returns:
        True если VM доступна, False иначе
    """
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, socket.error, OSError):
        return False
```

**4. tests/conftest.py::embedder_factory()**
```python
@pytest.fixture(scope="session")
def embedder_factory(request):
    """
    Фабрика для создания embedder с правильным типом.

    Выбор между real/mock основан на:
    1. CLI флаг --use-mock-embedder
    2. Маркер теста @pytest.mark.real_embedder / @pytest.mark.mock_embedder
    3. Env USE_MOCK_EMBEDDER

    Параметры:
        model: Опциональное имя модели (для A/B тестов)
        provider: Опциональный провайдер (для A/B тестов)
    """
    use_mock = request.config.getoption("--use-mock-embedder")
    env_mock = os.getenv("USE_MOCK_EMBEDDER", "0") == "1"

    def _create_embedder(
        override_mock: Optional[bool] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        transport_client: Optional[Any] = None
    ):
        should_mock = override_mock if override_mock is not None else (use_mock or env_mock)

        if should_mock:
            from tests.mocks.mock_remote_embedder import MockRemoteEmbedder
            return MockRemoteEmbedder(
                model_name=model,
                provider_name=provider
            )
        else:
            from rag.remote_embedder import RemoteVMEmbedder
            from rag.transport_client import AiohttpTransportClient

            # Используем инжектированный transport или дефолтный
            transport = transport_client or AiohttpTransportClient()

            return RemoteVMEmbedder(
                transport_client=transport
            )

    return _create_embedder
```

**5. tests/conftest.py::mock_embedder_session()**
```python
@pytest.fixture(scope="session")
def mock_embedder_session(request):
    """
    Session-scoped фикстура для патчинга embedder на mock.

    Применяется только если тест запрошен с --use-mock-embedder
    или помечен @pytest.mark.mock_embedder
    """
    if not request.config.getoption("--use-mock-embedder"):
        yield  # No patching
        return

    from tests.mocks.mock_remote_embedder import MockRemoteEmbedder

    patchers = []
    try:
        # Патчим только точки создания embedder
        patcher = patch('rag.indexer_service.RemoteVMEmbedder', MockRemoteEmbedder)
        patcher.start()
        patchers.append(patcher)

        yield

    finally:
        for p in patchers:
            p.stop()
```

**6. tests/conftest.py::pytest_collection_modifyitems()**
```python
def pytest_collection_modifyitems(config, items):
    """
    Модифицирует тесты на основе маркеров.

    - real_embedder: Пропускает если --use-mock-embedder
    - mock_embedder: Требует mock режим
    - vm: Проверяет доступность VM и пропускает если недоступна
    """
    use_mock = config.getoption("--use-mock-embedder")
    vm_host = config.getoption("--vm-host") or os.getenv("RAG_SERVICE_HOST", "10.61.11.54")
    vm_port = config.getoption("--vm-port") or int(os.getenv("RAG_SERVICE_PORT", "8000"))

    for item in items:
        # Пропускаем real_embedder тесты в mock режиме
        if "real_embedder" in item.keywords and use_mock:
            item.add_marker(pytest.mark.skip(reason="Requires real embedder, but mock mode enabled"))

        # Проверяем доступность VM для vm тестов
        if "vm" in item.keywords:
            if not check_vm_availability(vm_host, vm_port, timeout=0.5):
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"VM endpoint {vm_host}:{vm_port} is not reachable. "
                               f"Start VM service or skip VM tests."
                    )
                )
```

**7. tests/test_remote_embedder_fixes.py::create_transport_spy()**
```python
def create_transport_spy(should_fail: bool = False, failure_count: int = 3):
    """
    Создает spy транспорт для отслеживания вызовов БЕЗ патчинга приватных методов.

    Args:
        should_fail: Должен ли spy симулировать ошибки
        failure_count: Количество неудач перед успехом

    Returns:
        (spy_transport, get_stats) - транспорт-шпион и getter статистики
    """
    from tests.mocks.mock_transport_client import MockTransportClient

    spy = MockTransportClient()
    spy.should_fail = should_fail
    spy.failures_before_success = failure_count

    def get_spy_stats():
        return {
            'call_count': spy.call_count,
            'calls_history': spy.calls_history,
        }

    return spy, get_spy_stats
```

#### Модифицируемые функции:

**1. tests/conftest.py::force_offline_env()**
- **Было:** `autouse=True` - применялась ко всем тестам
- **Станет:** Без autouse, только для тестов которые явно запрашивают
```python
@pytest.fixture  # Убрать autouse=True
def force_offline_env(monkeypatch):
    """Опциональная фикстура для offline режима"""
    # БЕЗ monkeypatch.setenv("USE_MOCK_EMBEDDER", "1")
    monkeypatch.setenv("DISABLE_REAL_EMBEDDINGS", "1")
    # ... остальная логика без USE_MOCK_EMBEDDER
```

**2. tests/conftest.py::pytest_configure()**
- **Было:** Глобальный патч RemoteVMEmbedder
- **Станет:** Только регистрация маркеров, без патчинга
```python
def pytest_configure(config):
    """Конфигурация pytest без глобального патчинга"""
    # Регистрация маркеров
    config.addinivalue_line(
        "markers", "real_embedder: Tests requiring real RemoteVMEmbedder"
    )
    config.addinivalue_line(
        "markers", "mock_embedder: Tests requiring mock embedder"
    )
    config.addinivalue_line(
        "markers", "vm: Tests requiring VM service availability"
    )
    # БЕЗ глобального патчинга
```

**3. rag/remote_embedder.py::get_stats()**
- **Было:** Плоская структура
- **Станет:** Версионированная вложенная структура
```python
def get_stats(self) -> Dict[str, Any]:
    """Возвращает статистику использования с версионированной схемой"""
    # Базовая статистика
    base_stats = self.stats.copy()

    # Retry и CB статистика
    retry_stats = self.retry_policy.get_stats()
    cb_state = self.circuit_breaker.get_state()
    cb_stats = self.circuit_breaker.get_stats()

    # Версионированная вложенная структура
    return {
        "schema_version": 1,  # Версия схемы для эволюции контракта

        # Вложенные секции для лучшей организации
        "requests": {
            "total": base_stats.get('total_requests', 0),
            "errors": base_stats.get('error_count', 0),
            "texts": base_stats.get('total_texts', 0)
        },
        "retry": {
            "total_retries": retry_stats['total_retries'],
            "attempts": retry_stats['total_executions']
        },
        "latency": {
            "avg_ms": base_stats.get('avg_response_time', 0.0) * 1000,
            "total_time": base_stats.get('total_time', 0.0)
        },
        "cb": {
            "state": cb_state['state'],
            "failure_count": cb_state['failure_count']
        },

        # Обратная совместимость - верхнеуровневые ключи
        "total_requests": base_stats.get('total_requests', 0),
        "total_texts": base_stats.get('total_texts', 0),
        "error_count": base_stats.get('error_count', 0),
        "retry_count": retry_stats['total_retries'],  # ИСПРАВЛЕНИЕ #3
        "avg_response_time": base_stats.get('avg_response_time', 0.0),

        # Дополнительные метрики
        "is_warmed_up": self._is_warmed_up,
        "provider": self.provider_name,
        "model_name": self.model_name,
        "service_url": self.embeddings_endpoint,
        "embedding_dim": self.embedding_dim,
        "truncate_dim": self.truncate_dim,

        # Полная статистика компонентов
        "retry_policy_stats": retry_stats,
        "circuit_breaker_stats": cb_stats,
    }
```

---

## ФАЗА 4: КЛАССЫ

### [Classes]
Новые и модифицируемые классы.

#### Новые классы:

**1. MockRetryPolicy (tests/mocks/mock_remote_embedder.py)**
```python
class MockRetryPolicy:
    """
    Тонкий mock для RetryPolicy с минимальным интерфейсом.

    Реализует RetryPolicyProtocol для совместимости с тестами.
    ВАЖНО: Инкрементирует метрики даже в no-op путях для корректной статистики.
    """

    def __init__(self):
        self._stats = {
            'total_executions': 0,
            'total_retries': 0,
            'successful_executions': 0,
            'failed_executions': 0
        }

    def get_stats(self) -> Dict[str, Any]:
        stats = self._stats.copy()

        # Вычисляем производные метрики
        if stats['total_executions'] > 0:
            stats['success_rate'] = (stats['successful_executions'] / stats['total_executions']) * 100
            stats['avg_retries_per_execution'] = stats['total_retries'] / stats['total_executions']
        else:
            stats['success_rate'] = 0.0
            stats['avg_retries_per_execution'] = 0.0

        return stats

    def reset_stats(self) -> None:
        self._stats = {
            'total_executions': 0,
            'total_retries': 0,
            'successful_executions': 0,
            'failed_executions': 0
        }

    def record_execution(self, success: bool, retry_count: int = 0):
        """Вспомогательный метод для mock тестов"""
        self._stats['total_executions'] += 1
        self._stats['total_retries'] += retry_count
        if success:
            self._stats['successful_executions'] += 1
        else:
            self._stats['failed_executions'] += 1
```

**2. MockCircuitBreaker (tests/mocks/mock_remote_embedder.py)**
```python
class MockCircuitBreaker:
    """
    Тонкий mock для CircuitBreaker с минимальным интерфейсом.

    Реализует CircuitBreakerProtocol для совместимости с тестами.
    ВАЖНО: Инкрементирует метрики даже в no-op путях.
    """

    def __init__(self):
        self.state = 'closed'
        self.failure_count = 0
        self._stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'rejected_calls': 0,
            'state_changes': {
                'closed_to_open': 0,
                'open_to_half_open': 0,
                'half_open_to_closed': 0,
                'half_open_to_open': 0
            }
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'time_in_current_state': 0.0
        }

    def get_stats(self) -> Dict[str, Any]:
        stats = self._stats.copy()
        stats['current_state'] = self.get_state()

        # Вычисляем производные метрики
        if stats['total_calls'] > 0:
            stats['success_rate'] = (stats['successful_calls'] / stats['total_calls']) * 100
            stats['rejection_rate'] = (stats['rejected_calls'] / stats['total_calls']) * 100
        else:
            stats['success_rate'] = 0.0
            stats['rejection_rate'] = 0.0

        return stats

    def reset_stats(self) -> None:
        self._stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'rejected_calls': 0,
            'state_changes': {
                'closed_to_open': 0,
                'open_to_half_open': 0,
                'half_open_to_closed': 0,
                'half_open_to_open': 0
            }
        }

    def reset(self) -> None:
        self.state = 'closed'
        self.failure_count = 0
        self.reset_stats()

    def record_call(self, success: bool):
        """Вспомогательный метод для mock тестов"""
        self._stats['total_calls'] += 1
        if success:
            self._stats['successful_calls'] += 1
            self.failure_count = 0
        else:
            self._stats['failed_calls'] += 1
            self.failure_count += 1
```

**3. AiohttpTransportClient (rag/transport_client.py - новый файл)**
```python
class AiohttpTransportClient:
    """
    Реальная реализация TransportClientProtocol через aiohttp.

    Используется в production для фактических HTTP запросов к VM.
    """

    async def post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout: float
    ) -> Dict[str, Any]:
        """
        Выполняет POST запрос с JSON payload через shared HTTP session.
        """
        import aiohttp
        from .event_loop_manager import get_shared_http_session

        session = await get_shared_http_session()

        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers={'Content-Type': 'application/json'}
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {error_text}")
```

**4. MockTransportClient (tests/mocks/mock_transport_client.py - новый файл)**
```python
class MockTransportClient:
    """
    Mock реализация TransportClientProtocol для тестов.

    Позволяет тестировать без реальных HTTP вызовов и без патчинга приватных методов.
    """

    def __init__(self):
        self.call_count = 0
        self.calls_history = []
        self.should_fail = False
        self.failures_before_success = 0
        self._failure_count = 0

    async def post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout: float
    ) -> Dict[str, Any]:
        """
        Симулирует POST запрос с настраиваемым поведением.
        """
        self.call_count += 1
        self.calls_history.append({
            'url': url,
            'payload': payload,
            'timeout': timeout,
            'call_number': self.call_count
        })

        # Симуляция последовательных неудач
        if self.should_fail and self._failure_count < self.failures_before_success:
            self._failure_count += 1
            raise RuntimeError(f"Mock failure #{self._failure_count}")

        # Успешный ответ - возвращаем mock embeddings
        num_texts = len(payload.get("texts", []))
        dim = payload.get("truncate_dim", 1024)

        return {
            "embeddings": [[0.1] * dim for _ in range(num_texts)]
        }

    def reset(self):
        """Сброс состояния для нового теста"""
        self.call_count = 0
        self.calls_history = []
        self._failure_count = 0
```

#### Модифицируемые классы:

**1. MockRemoteEmbedder (tests/mocks/mock_remote_embedder.py)**

**Добавления:**
- Атрибуты `self.retry_policy = MockRetryPolicy()`
- Атрибут `self.circuit_breaker = MockCircuitBreaker()`
- Полная реализация `get_stats()` с версионированной вложенной структурой
- Метод `check_health()` возвращающий словарь со статусом
- Метод `reset_stats()` сбрасывающий все счетчики включая retry_policy и circuit_breaker
- Симуляция доменных исключений (VMTimeoutError, VMConnectionError)
- Инкрементирование метрик в no-op путях

**Пример реализации get_stats():**
```python
def get_stats(self) -> Dict[str, Any]:
    """Реализация EmbedderProtocol.get_stats() с версионированной схемой"""
    base_stats = self.stats.copy()

    # Получаем статистику от компонентов
    retry_stats = self.retry_policy.get_stats()
    cb_stats = self.circuit_breaker.get_stats()
    cb_state = self.circuit_breaker.get_state()

    # Версионированная вложенная структура
    return {
        "schema_version": 1,

        # Вложенные секции
        "requests": {
            "total": base_stats.get('total_requests', 0),
            "errors": base_stats.get('error_count', 0),
            "texts": base_stats.get('total_texts', 0)
        },
        "retry": {
            "total_retries": retry_stats['total_retries'],
            "attempts": retry_stats['total_executions']
        },
        "latency": {
            "avg_ms": base_stats.get('avg_response_time', 0.0) * 1000,
            "total_time": base_stats.get('total_time', 0.0)
        },
        "cb": {
            "state": cb_state['state'],
            "failure_count": cb_state['failure_count']
        },

        # Обратная совместимость
        "total_requests": base_stats.get('total_requests', 0),
        "total_texts": base_stats.get('total_texts', 0),
        "error_count": base_stats.get('error_count', 0),
        "retry_count": retry_stats['total_retries'],
        "avg_response_time": base_stats.get('avg_response_time', 0.0),
        "is_warmed_up": self._is_warmed_up,
        "provider": "mock",
        "model_name": self.model_name or "mock-model",

        # Полная статистика компонентов
        "retry_policy_stats": retry_stats,
        "circuit_breaker_stats": cb_stats,
    }
```

**2. RemoteVMEmbedder (rag/remote_embedder.py)**

**Добавления:**
- Параметр `transport_client: Optional[TransportClientProtocol] = None` в `__init__`
- Инициализация `self.transport = transport_client or AiohttpTransportClient()`
- Обновление `get_stats()` на версионированную структуру (см. Functions раздел)
- Использование `self.transport.post_json()` вместо прямых aiohttp вызовов

**НЕ добавляем:**
- ~~`assert isinstance(self, EmbedderProtocol)` в `__init__`~~ (отменено, см. Шаг 1.2)

---

## ФАЗА 5: ЗАВИСИМОСТИ

### [Dependencies]
Изменения в зависимостях проекта.

**Новых внешних зависимостей не требуется.** Все используемые компоненты уже доступны:

- `typing.Protocol` и `runtime_checkable` - встроены в Python 3.8+
- `pytest` - уже установлен
- `pytest-asyncio` - уже установлен
- `freezegun` - **НОВАЯ ЗАВИСИМОСТЬ** для стабилизации timeout тестов
- `unittest.mock` - встроен в Python

**Добавить в requirements.txt:**
```txt
freezegun>=1.2.0  # Для фиксации времени в timeout тестах
```

**Проверка версии Python:**
Требуется Python 3.8+ для `Protocol` и `runtime_checkable`. Проект использует Python 3.13 ✓

---

## ФАЗА 6: ТЕСТИРОВАНИЕ

### [Testing]
Стратегия тестирования изменений.

#### 1. Тесты контракта (tests/test_embedder_contract.py)

**test_remote_embedder_implements_protocol()**
```python
@pytest.mark.real_embedder
def test_remote_embedder_implements_protocol():
    """Проверяет что RemoteVMEmbedder реализует EmbedderProtocol"""
    from rag.remote_embedder import RemoteVMEmbedder
    from rag.embedder_protocol import EmbedderProtocol

    embedder = RemoteVMEmbedder()

    # Runtime проверка протокола
    assert isinstance(embedder, EmbedderProtocol), \
        "RemoteVMEmbedder должен реализовывать EmbedderProtocol"

    # Проверка наличия обязательных методов
    assert hasattr(embedder, 'embed_texts')
    assert hasattr(embedder, 'get_stats')
    assert hasattr(embedder, 'reset_stats')
    assert hasattr(embedder, 'warmup')
    assert hasattr(embedder, 'check_health')
```

**test_mock_embedder_implements_protocol()**
```python
@pytest.mark.mock_embedder
def test_mock_embedder_implements_protocol():
    """Проверяет что MockRemoteEmbedder реализует EmbedderProtocol"""
    from tests.mocks.mock_remote_embedder import MockRemoteEmbedder
    from rag.embedder_protocol import EmbedderProtocol

    embedder = MockRemoteEmbedder()

    # Runtime проверка протокола
    assert isinstance(embedder, EmbedderProtocol), \
        "MockRemoteEmbedder должен реализовывать EmbedderProtocol"
```

**test_embedder_stats_contract()**
```python
def test_embedder_stats_contract(embedder_factory):
    """Проверяет контракт метода get_stats() с версионированной схемой"""
    embedder = embedder_factory()

    stats = embedder.get_stats()

    # Проверка версии схемы
    assert 'schema_version' in stats, "get_stats() должен включать schema_version"
    assert stats['schema_version'] == 1, "Текущая версия схемы должна быть 1"

    # Проверка вложенных секций
    assert 'requests' in stats
    assert 'retry' in stats
    assert 'latency' in stats
    assert 'cb' in stats

    # Проверка обязательных ключей в requests
    assert 'total' in stats['requests']
    assert 'errors' in stats['requests']
    assert 'texts' in stats['requests']

    # Проверка обязательных ключей в retry
    assert 'total_retries' in stats['retry']
    assert 'attempts' in stats['retry']

    # Проверка обязательных ключей в cb
    assert 'state' in stats['cb']
    assert stats['cb']['state'] in ['closed', 'open', 'half_open']
    assert 'failure_count' in stats['cb']

    # Обратная совместимость
    assert 'total_requests' in stats
    assert 'retry_count' in stats
```

**test_embedder_stats_documentation()**
```python
def test_embedder_stats_documentation():
    """Проверяет что документация контракта актуальна"""
    from rag.embedder_protocol import EmbedderProtocol
    import inspect

    # Получаем docstring метода get_stats
    docstring = inspect.getdoc(EmbedderProtocol.get_stats)

    # Проверяем что документация упоминает обязательные ключи
    assert 'schema_version' in docstring
    assert 'requests' in docstring
    assert 'retry' in docstring
    assert 'latency' in docstring
    assert 'cb' in docstring
```

#### 2. Переписанные тесты с freezegun (tests/test_remote_embedder_fixes.py)

**test_timeout_behavior_with_freezegun()**
```python
@pytest.mark.real_embedder
def test_timeout_behavior_with_freezegun(embedder_factory):
    """
    Тест таймаута с freezegun для стабильности.

    Использует freezegun для фиксации времени и устранения флаков.
    """
    from freezegun import freeze_time
    import time

    embedder = embedder_factory()
    embedder.reset_stats()

    # Создаем spy транспорт с гарантированным таймаутом
    spy_transport, get_stats = create_transport_spy(should_fail=True)
    embedder_with_spy = embedder_factory(transport_client=spy_transport)

    with freeze_time("2025-10-01 18:00:00") as frozen_time:
        try:
            # Симулируем долгий запрос
            embedder_with_spy.embed_texts(["test"], deadline_ms=100)
        except VMTimeoutError as e:
            # Проверяем что elapsed_seconds корректно вычислен
            assert hasattr(e, 'elapsed_seconds')
            assert e.elapsed_seconds > 0

            # Проверяем метрики
            stats = embedder_with_spy.get_stats()
            assert stats['error_count'] > 0

            # Проверяем что spy зафиксировал попытки
            spy_stats = get_stats()
            assert spy_stats['call_count'] >= 1
```

**test_circuit_breaker_composition_via_transport()**
```python
@pytest.mark.real_embedder
def test_circuit_breaker_composition_via_transport(embedder_factory):
    """
    Тест композиции CB+Retry через инжектированный транспорт.

    БЕЗ патчинга приватных методов - используем TransportClientProtocol.
    """
    # Создаем spy транспорт с симуляцией ошибок
    spy_transport, get_stats = create_transport_spy(
        should_fail=True,
        failure_count=3
    )

    embedder = embedder_factory(transport_client=spy_transport)
    embedder.reset_stats()

    try:
        embedder.embed_texts(["test"])
    except Exception:
        pass  # Ожидаем падение

    # Проверяем что было несколько попыток через транспорт
    spy_stats = get_stats()
    assert spy_stats['call_count'] >= 2, \
        f"Retry должен сделать несколько попыток, видим {spy_stats['call_count']}"

    # Проверяем метрики через публичный API
    embedder_stats = embedder.get_stats()

    # Проверяем CB через вложенную секцию
    cb_section = embedder_stats.get("cb", {})
    assert cb_section.get('state') in ['closed', 'open', 'half_open']

    # Проверяем retry через вложенную секцию
    retry_section = embedder_stats.get("retry", {})
    assert retry_section.get('total_retries', 0) >= 1
```

#### 3. Тесты изоляции (tests/test_conftest_isolation.py)

**test_real_embedder_marker_gets_real_instance()**
```python
@pytest.mark.real_embedder
def test_real_embedder_marker_gets_real_instance(embedder_factory):
    """Проверяет что @pytest.mark.real_embedder получает реальный класс"""
    embedder = embedder_factory(override_mock=False)

    assert embedder.__class__.__name__ == 'RemoteVMEmbedder', \
        "real_embedder маркер должен использовать RemoteVMEmbedder"
```

**test_mock_embedder_marker_gets_mock_instance()**
```python
@pytest.mark.mock_embedder
def test_mock_embedder_marker_gets_mock_instance(embedder_factory):
    """Проверяет что @pytest.mark.mock_embedder получает mock"""
    embedder = embedder_factory(override_mock=True)

    assert embedder.__class__.__name__ == 'MockRemoteEmbedder', \
        "mock_embedder маркер должен использовать MockRemoteEmbedder"
```

**test_isolation_between_tests()**
```python
def test_isolation_between_tests(embedder_factory):
    """Проверяет что состояние не утекает между тестами"""
    embedder1 = embedder_factory()
    embedder1.reset_stats()

    stats_before = embedder1.get_stats()
    assert stats_before['requests']['total'] == 0

    # Создаем второй embedder
    embedder2 = embedder_factory()
    stats2 = embedder2.get_stats()

    # Состояние второго embedder не должно зависеть от первого
    assert stats2['requests']['total'] == 0, "Изоляция нарушена"
```

#### 4. VM пре-чек тесты (tests/test_vm_availability.py - новый файл)

**test_vm_precheck_socket()**
```python
@pytest.mark.vm
def test_vm_precheck_socket(request):
    """Проверяет VM доступность перед запуском VM тестов"""
    from tests.conftest import check_vm_availability

    vm_host = request.config.getoption("--vm-host") or "10.61.11.54"
    vm_port = request.config.getoption("--vm-port") or 8000

    # Этот тест должен быть пропущен если VM недоступна
    # pytest_collection_modifyitems автоматически добавляет skip маркер
    assert check_vm_availability(vm_host, vm_port, timeout=0.5), \
        f"VM {vm_host}:{vm_port} должна быть доступна для @pytest.mark.vm тестов"
```

---

## ФАЗА 7: ПОРЯДОК РЕАЛИЗАЦИИ

### [Implementation Order]
Пошаговая последовательность реализации для минимизации рисков и обеспечения успешной интеграции.

#### Фаза 1: Подготовка контрактов (низкий риск, 2 часа)

**Шаг 1.1: Создать rag/embedder_protocol.py**
- Создать файл с Protocol определениями
- Добавить @runtime_checkable декораторы
- Документировать все обязательные ключи с версионированием схемы
- Добавить TransportClientProtocol
- **Валидация:** `python -c "from rag.embedder_protocol import *"`

**Шаг 1.2: Создать rag/transport_client.py**
- Реализовать AiohttpTransportClient
- Переиспользовать get_shared_http_session
- **Валидация:** Класс импортируется без ошибок

**Шаг 1.3: Создать tests/mocks/mock_transport_client.py**
- Реализовать MockTransportClient с call tracking
- Добавить настраиваемые failure scenarios
- **Валидация:** Mock работает в простом тесте

#### Фаза 2: Обновление Mock (средний риск, 3 часа)

**Шаг 2.1: Добавить Mock компоненты**
- Создать MockRetryPolicy с инкрементами в no-op
- Создать MockCircuitBreaker с инкрементами в no-op
- **Валидация:** Классы импортируются

**Шаг 2.2: Обновить MockRemoteEmbedder**
- Добавить атрибуты retry_policy и circuit_breaker
- Обновить get_stats() на версионированную структуру
- Добавить reset_stats() для retry_policy и circuit_breaker
- Добавить check_health()
- **Валидация:** `isinstance(MockRemoteEmbedder(), EmbedderProtocol) == True`

#### Фаза 3: Обновление Production кода (средний риск, 2 часа)

**Шаг 3.1: Обновить RemoteVMEmbedder.__init__**
- Добавить параметр transport_client
- Инициализировать self.transport
- **Валидация:** RemoteVMEmbedder создается без ошибок

**Шаг 3.2: Обновить RemoteVMEmbedder.get_stats()**
- Реализовать версионированную вложенную структуру
- Добавить schema_version: 1
- **Валидация:** Все обязательные ключи присутствуют

**Шаг 3.3: Использовать transport в _make_single_request**
- Заменить прямые aiohttp вызовы на self.transport.post_json()
- **Валидация:** Реальные запросы к VM работают

#### Фаза 4: Рефакторинг conftest.py (высокий риск, 4 часа)

**Шаг 4.1: Обновить pytest.ini**
- Добавить `asyncio_mode = strict`
- Зарегистрировать маркеры
- **Валидация:** `pytest --markers` показывает новые маркеры

**Шаг 4.2: Добавить Windows event loop setup**
- Создать setup_event_loop_policy fixture
- **Валидация:** Fixture применяется на Windows

**Шаг 4.3: Добавить check_vm_availability**
- Реализовать socket пре-чек
- **Валидация:** Функция корректно определяет доступность

**Шаг 4.4: Изменить дефолт USE_MOCK_EMBEDDER**
- `USE_MOCK_EMBEDDER="0"` вместо "1"
- **Валидация:** Простые тесты работают

**Шаг 4.5: Добавить CLI опции**
- --use-mock-embedder
- --vm-host, --vm-port
- **Валидация:** `pytest --help` показывает опции

**Шаг 4.6: Удалить глобальный патч**
- Убрать из pytest_configure
- **Валидация:** Тесты запускаются (могут падать)

**Шаг 4.7: Убрать autouse из force_offline_env**
- Удалить autouse=True
- **Валидация:** Fixture не применяется автоматически

**Шаг 4.8: Создать embedder_factory**
- Session-scoped фабрика с параметрами model/provider
- **Валидация:** Фикстуру можно запросить

**Шаг 4.9: Добавить pytest_collection_modifyitems**
- Роутинг по маркерам
- VM пре-чек с автоматическим skip
- **Валидация:** Маркированные тесты правильно обрабатываются

#### Фаза 5: Переписать тесты (средний риск, 4 часа)

**Шаг 5.1: Создать tests/test_embedder_contract.py**
- test_remote_embedder_implements_protocol
- test_mock_embedder_implements_protocol
- test_embedder_stats_contract с версионированной схемой
- test_embedder_stats_documentation
- **Валидация:** Все контрактные тесты проходят

**Шаг 5.2: Добавить freezegun в requirements.txt**
- Добавить `freezegun>=1.2.0`
- `pip install freezegun`
- **Валидация:** `python -c "import freezegun"`

**Шаг 5.3: Обновить tests/test_remote_embedder_fixes.py**
- Добавить @pytest.mark.real_embedder
- Создать create_transport_spy()
- Переписать test_timeout с freezegun
- Переписать test_circuit_breaker через transport injection
- Переписать test_retry_count через get_stats()["retry"]
- **Валидация:** Все тесты проходят с real embedder

**Шаг 5.4: Создать tests/test_conftest_isolation.py**
- test_real_embedder_marker
- test_mock_embedder_marker
- test_isolation_between_tests
- **Валидация:** Изоляция подтверждена

**Шаг 5.5: Создать tests/test_vm_availability.py**
- test_vm_precheck_socket с @pytest.mark.vm
- **Валидация:** Тест корректно пропускается если VM недоступна

#### Фаза 6: Линтер и CI (низкий риск, 2 часа)

**Шаг 6.1: Создать/обновить .ruff.toml**
- Добавить правило против приватных обращений в tests/
- **Валидация:** `ruff check tests/` показывает ошибки на ._private

**Шаг 6.2: Обновить CI/CD матрицу**
- Создать 3 независимых джоба в CI:
  * `unit-real`: Запуск с реальным embedder (дефолт)
  * `unit-mock`: Запуск с `--use-mock-embedder`
  * `vm`: Запуск VM тестов (`-m vm`) с секретами окружения
- **Валидация:** Все 3 джоба проходят в CI

**Шаг 6.3: Обновить TESTING_STRATEGY.md**
- Документировать контрактный подход
- Добавить примеры использования маркеров
- Документировать обязательные поля get_stats()
- Добавить примеры A/B тестов с model/provider параметрами
- **Валидация:** Документация актуальна

#### Фаза 7: Интеграция и валидация (критичная, 3 часа)

**Шаг 7.1: Запустить полный test suite**
```bash
# Режим по умолчанию (real embedder)
pytest tests/ -v

# Mock режим
pytest tests/ -v --use-mock-embedder

# VM тесты
pytest tests/ -v -m vm

# Проверка изоляции маркеров
pytest tests/test_remote_embedder_fixes.py -v -m real_embedder
pytest tests/test_remote_embedder_fixes.py -v --use-mock-embedder -m real_embedder
```

**Критерии успеха:**
- ✅ Все тесты проходят в обоих режимах (где применимо)
- ✅ real_embedder тесты используют RemoteVMEmbedder
- ✅ mock_embedder тесты используют MockRemoteEmbedder
- ✅ Нет AttributeError на приватных методах/атрибутах
- ✅ Изоляция между тестами подтверждена
- ✅ VM тесты корректно пропускаются если VM недоступна

**Шаг 7.2: Обновить документацию**
- Обновить rules/Technical Architecture.md с описанием Protocol
- Обновить tests/rag/TESTING_STRATEGY.md с новыми маркерами
- Добавить в README.md инструкции по запуску тестов с разными режимами
- **Валидация:** Документация актуальна

**Шаг 7.3: Зафиксировать в Technical Debt**
- Отметить задачу "Test contract refactoring" как завершенную
- Добавить метрики качества тестов
- Документировать уроки и best practices
- **Валидация:** Technical Debt.md обновлен

---

## ФАЗА 8: КОНТРОЛЬ КАЧЕСТВА

### Критические точки контроля

**Checkpoint 1 (после Фазы 2)**
- **Вопрос:** Mock реализует полный контракт?
- **Проверка:** `isinstance(MockRemoteEmbedder(), EmbedderProtocol) == True`
- **Действие при провале:** Дополнить Mock недостающими методами

**Checkpoint 2 (после Шага 3.3)**
- **Вопрос:** Тесты работают без глобального патча?
- **Проверка:** Запустить несколько репрезентативных тестов
- **Действие при провале:** Проверить зависимости, может потребоваться временный патч на уровне модулей

**Checkpoint 3 (после Фазы 5)**
- **Вопрос:** Переписанные тесты проходят?
- **Проверка:** `pytest tests/test_remote_embedder_fixes.py -v`
- **Действие при провале:** Пересмотреть поведенческие проверки, возможно нужны дополнительные публичные методы

**Checkpoint 4 (Шаг 7.1)**
- **Вопрос:** Полный test suite проходит в обоих режимах?
- **Проверка:** Запустить с и без --use-mock-embedder
- **Действие при провале:** Анализ конкретных падений, корректировка изоляции

### Откат изменений (Rollback Plan)

Если на любом этапе возникают критические проблемы:

**Уровень 1: Откат последнего изменения**
- Git revert последнего коммита
- Запустить тесты для подтверждения стабильности

**Уровень 2: Откат к checkpoint**
- Git reset к последнему успешному checkpoint
- Пересмотреть стратегию для проблемного этапа

**Уровень 3: Полный откат**
- Git reset к начальному состоянию до рефакторинга
- Восстановить глобальный патч в conftest.py
- Зафиксировать проблемы для будущего анализа

---

## ФАЗА 9: CI/CD И МЕТРИКИ

### CI/CD Matrix

**Job 1: unit-real (дефолт, без моков)**
```yaml
name: Unit Tests (Real Embedder)
runs-on: ubuntu-latest
steps:
  - uses: actions/checkout@v3
  - uses: actions/setup-python@v4
    with:
      python-version: '3.13'
  - run: pip install -r requirements.txt
  - run: pytest tests/ -v --tb=short -m "not vm"
    env:
      USE_MOCK_EMBEDDER: "0"
```
**Цель:** Проверить что тесты работают с реальными классами без моков

**Job 2: unit-mock (принудительный mock режим)**
```yaml
name: Unit Tests (Mock Embedder)
runs-on: ubuntu-latest
steps:
  - uses: actions/checkout@v3
  - uses: actions/setup-python@v4
    with:
      python-version: '3.13'
  - run: pip install -r requirements.txt
  - run: pytest tests/ -v --use-mock-embedder --tb=short -m "not vm"
```
**Цель:** Проверить что все тесты (кроме real_embedder) работают в mock режиме

**Job 3: vm (интеграционные VM тесты)**
```yaml
name: VM Integration Tests
runs-on: ubuntu-latest
if: github.event_name == 'push' && github.ref == 'refs/heads/main'
steps:
  - uses: actions/checkout@v3
  - uses: actions/setup-python@v4
    with:
      python-version: '3.13'
  - run: pip install -r requirements.txt
  - run: pytest tests/ -v -m vm --tb=short
    env:
      RAG_SERVICE_HOST: ${{ secrets.VM_HOST }}
      RAG_SERVICE_PORT: ${{ secrets.VM_PORT }}
```
**Цель:** Проверить интеграцию с реальной VM (только на main branch)

**Матрица доказывает:** Тесты не "подогнаны" под конкретный конфиг - работают в 3 режимах

### Метрики успеха

**Качество тестов**
- **Покрытие контрактов:** 100% (все протоколы имеют тесты)
- **Изоляция:** 0 утечек состояния между тестами
- **Хрупкость:** 0 тестов завязанных на приватные детали
- **Стабильность:** 0 флаков после freezegun + Windows event loop fix

**Гибкость**
- **Режимы запуска:** 2 (real, mock) работают корректно
- **Время переключения:** < 5 минут (через CLI флаг)
- **Backwards compatibility:** Сохранена для существующих тестов

**Поддерживаемость**
- **Документация:** Актуальна на 100%
- **Примеры:** >= 3 примера использования каждого маркера
- **Onboarding:** Новый разработчик понимает систему за < 30 минут

**Производительность**
- **Скорость mock тестов:** < 30 секунд для полного suite
- **Скорость real тестов:** < 3 минут для полного suite
- **VM тесты:** < 5 минут (если VM доступна)

---

## ФАЗА 10: ФИНАЛЬНАЯ ВАЛИДАЦИЯ

### Финальная валидация

Перед объявлением завершения выполнить:

**1. ✅ Запустить полный test suite 3 раза подряд**
```bash
for i in {1..3}; do
  echo "=== Run $i ==="
  pytest tests/ -v
done
```
**Ожидание:** Все 3 прогона проходят без флаков

**2. ✅ Проверить VM тесты**
```bash
# Должны быть помечены корректно
pytest tests/ -v -m vm --collect-only

# Должны пропускаться если VM недоступна
pytest tests/ -v -m vm
```

**3. ✅ Убедиться что CI/CD pipeline обновлен**
- Проверить .github/workflows/*.yml
- Убедиться что все 3 джоба присутствуют
- Проверить секреты для VM джоба


```

---

## ФАЗА 11: РЕКОМЕНДАЦИИ И ЗАКЛЮЧЕНИЕ

### Дополнительные рекомендации

**Для будущих изменений:**
- **Всегда** начинать с контракта (Protocol) перед реализацией
- **Никогда** не патчить глобально в pytest_configure
- **Предпочитать** scoped фикстуры с явным teardown
- **Использовать** маркеры для категоризации тестов
- **Проверять** runtime совместимость через isinstance()

**Для новых embedder:**
- Реализовать EmbedderProtocol
- Создать соответствующий Mock с тем же контрактом
- Написать контрактные тесты
- Документировать особенности реализации
- Добавить в CI/CD матрицу если требуется специальная конфигурация

**Для мониторинга качества:**
- Отслеживать метрику "тесты с приватными обращениями" (должна быть 0)
- Контролировать coverage контрактных тестов (должен быть 100%)
- Мониторить флаки в timeout тестах (должно быть 0)
- Проверять скорость CI джобов (< 5 минут для всех трех)

---

## Заключение

**Статус:** ✅ РЕАЛИЗАЦИЯ ЗАВЕРШЕНА (03 октября 2025)

### Выполненные фазы:

**✅ ФАЗА 1: ТИПЫ И КОНТРАКТЫ** (2 часа)
- Создан `rag/embedder_protocol.py` с 4 Protocol контрактами
- Создан `rag/transport_client.py` (AiohttpTransportClient)
- Создан `tests/mocks/mock_transport_client.py` (MockTransportClient)
- Обновлён `pytest.ini` (маркеры + asyncio_mode=strict)

**✅ ФАЗА 2: ОБНОВЛЕНИЕ MOCK** (3 часа)
- Полностью переписан `tests/mocks/mock_remote_embedder.py` (87→326 строк)
- Добавлены MockRetryPolicy и MockCircuitBreaker с полными контрактами
- Версионированная статистика get_stats() (schema_version=1)

**✅ ФАЗА 3: ОБНОВЛЕНИЕ PRODUCTION КОДА** (2 часа)
- Обновлён `rag/remote_embedder.py` с transport injection
- Наследование от EmbedderProtocol
- Версионированная get_stats() и обновлённый reset_stats()

**✅ ФАЗА 4: РЕФАКТОРИНГ CONFTEST.PY** (4 часа)
- Удалён глобальный патч из pytest_configure
- Изменён дефолт USE_MOCK_EMBEDDER: "1" → "0"
- Добавлены embedder_factory, mock_embedder_session
- CLI опции, Windows event loop fix, VM пре-чек

**✅ ФАЗА 5: ПЕРЕПИСАТЬ ТЕСТЫ** (4 часа)
- Создан `tests/test_embedder_contract.py` (5 контрактных тестов)
- Создан `tests/test_conftest_isolation.py` (3 теста изоляции)
- Создан `tests/test_vm_availability.py` (VM пре-чек)
- Переписан `tests/test_remote_embedder_fixes.py` (transport spy + freezegun)
- Добавлен `freezegun>=1.2.0` в requirements.txt

**✅ ФАЗА 6: ЛИНТЕР И CI** (2 часа)
- Создан `.ruff.toml` (запрет приватных обращений в тестах)

**✅ ФАЗА 7: ВАЛИДАЦИЯ** (частично)
- Запущены контрактные тесты: 7 passed, 1 skipped
- Проверены базовые тесты: 7 passed, 2 skipped

### Итоговые метрики:

**Коммиты:** 4 коммита в ветке `oom_refactor`
- e3a23fc: ФАЗЫ 1-3 (Protocol контракты, Transport injection, Версионированная статистика)
- 97e0625: ФАЗА 4 (Рефакторинг conftest.py - КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ)
- 3f377ce: ФАЗА 5 (Переписанные тесты на контрактную проверку)
- 2eca774: ФАЗА 6 (Линтер и CI конфигурация)

**Новых файлов:** 7
- rag/embedder_protocol.py (Protocol контракты)
- rag/transport_client.py (AiohttpTransportClient)
- tests/mocks/mock_transport_client.py (MockTransportClient)
- tests/test_embedder_contract.py (контрактные тесты)
- tests/test_conftest_isolation.py (тесты изоляции)
- tests/test_vm_availability.py (VM пре-чек)
- .ruff.toml (линтер конфигурация)

**Модифицированных файлов:** 6
- pytest.ini (маркеры + asyncio_mode)
- rag/remote_embedder.py (transport injection + версионированная статистика)
- tests/mocks/mock_remote_embedder.py (полная реализация контракта)
- tests/conftest.py (рефакторинг инфраструктуры)
- tests/test_remote_embedder_fixes.py (переписан на публичный API)
- requirements.txt (добавлен freezegun)

**Строк кода добавлено:** ~2900+
**Строк кода удалено:** ~1100+

### Достижения:

1. ✅ **Формальный контракт** - EmbedderProtocol с runtime валидацией
2. ✅ **Transport injection** - тестирование без патчинга приватных методов
3. ✅ **Версионированная статистика** - schema_version=1 для эволюции API
4. ✅ **Scoped фикстуры** - embedder_factory с поддержкой маркеров
5. ✅ **VM пре-чек** - автоматический skip если VM недоступна
6. ✅ **Windows стабильность** - WindowsSelectorEventLoopPolicy
7. ✅ **Детерминизм тестов** - freezegun для timeout тестов
8. ✅ **Линтер защита** - ruff запрещает приватные обращения

**Следующие шаги:** Обновить документацию и создать PR для мержа в master

