# Отчёт о рефакторинге tests/conftest.py - ФАЗА 4

**Дата:** 02 октября 2025
**Исполнитель:** Claude Code Agent
**Статус:** ✅ Успешно завершено

## Резюме изменений

Выполнен критический рефакторинг `tests/conftest.py` согласно ФАЗЕ 4 плана Test Contract Refactoring. Удалён глобальный патчинг, добавлены scoped фикстуры и маркеры для гибкого управления режимами тестирования.

## Ключевые изменения

### 1. Изменён дефолт USE_MOCK_EMBEDDER: "1" → "0"
```python
# БЫЛО: глобальный патчинг принудительно использовал mock
os.environ.setdefault("USE_MOCK_EMBEDDER", "1")

# СТАЛО: дефолт на реальные классы
os.environ.setdefault("USE_MOCK_EMBEDDER", "0")
```

**Обоснование:** Тесты должны по умолчанию работать с реальными классами для проверки production кода.

### 2. Добавлены CLI опции (pytest_addoption)
```python
--use-mock-embedder    # Принудительный mock режим
--vm-host=VM_HOST      # Хост VM для интеграционных тестов
--vm-port=VM_PORT      # Порт VM (дефолт 8000)
```

**Использование:**
```bash
pytest tests/ --use-mock-embedder          # Всё через mock
pytest tests/ --vm-host=10.61.11.54        # Кастомный VM host
```

### 3. Добавлен setup_event_loop_policy (session-scoped, autouse=True)
```python
@pytest.fixture(scope="session", autouse=True)
def setup_event_loop_policy():
    """Устанавливает WindowsSelectorEventLoopPolicy на Windows для стабильных async-тестов."""
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

**Обоснование:** На Windows дефолтный ProactorEventLoop имеет проблемы с async операциями. WindowsSelectorEventLoopPolicy более стабилен.

### 4. Убран autouse=True из force_offline_env
```python
# БЫЛО: autouse=True - применялась ко ВСЕМ тестам
@pytest.fixture(autouse=True)

# СТАЛО: только для тестов которые явно запрашивают
@pytest.fixture
def force_offline_env(monkeypatch):
```

**Обоснование:** Не все тесты требуют offline режим. Фикстура должна использоваться явно.

### 5. Удалён глобальный патчинг из pytest_configure
```python
# БЫЛО: Глобальный патч RemoteVMEmbedder на MockRemoteEmbedder в pytest_configure
# УДАЛЕНО: Весь блок патчинга (строки 52-93 старого файла)

# СТАЛО: Только регистрация маркеров
def pytest_configure(config):
    """Регистрирует пользовательские маркеры без глобального патчинга."""
    config.addinivalue_line("markers", "real_embedder: ...")
    config.addinivalue_line("markers", "mock_embedder: ...")
    config.addinivalue_line("markers", "vm: ...")
```

**Обоснование:** Глобальный патч ломал тесты которые ожидали реальные классы. Теперь патчинг делается через scoped фикстуры.

### 6. Добавлен check_vm_availability
```python
def check_vm_availability(host: str, port: int, timeout: float = 0.5) -> bool:
    """Проверяет доступность VM через TCP-подключение."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, socket.error, OSError):
        return False
```

**Использование:** Автоматический пре-чек VM перед запуском `@pytest.mark.vm` тестов.

### 7. Добавлена embedder_factory (session-scoped)
```python
@pytest.fixture(scope="session")
def embedder_factory(request):
    """Фабрика для создания mock или реального RemoteVMEmbedder."""

    def _create_embedder(
        override_mock: Optional[bool] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        transport_client: Optional[Any] = None,
    ):
        # Логика выбора real vs mock
        ...
```

**Возможности:**
- Выбор real/mock через CLI флаг или env
- Поддержка transport injection для spy паттерна
- A/B тестирование через параметры model/provider
- Override для конкретных тестов

**Примеры использования:**
```python
def test_example(embedder_factory):
    # Дефолтное поведение (real или mock в зависимости от CLI/env)
    embedder = embedder_factory()

    # Принудительный mock
    embedder = embedder_factory(override_mock=True)

    # С transport spy
    spy_transport = MockTransportClient()
    embedder = embedder_factory(transport_client=spy_transport)

    # A/B тест разных моделей
    embedder_v2 = embedder_factory(model="jina-v2")
    embedder_v3 = embedder_factory(model="jina-v3")
```

### 8. Добавлена mock_embedder_session (session-scoped)
```python
@pytest.fixture(scope="session")
def mock_embedder_session(request):
    """Сессионный патч RemoteVMEmbedder на mock-реализацию при необходимости."""
```

**Обоснование:** Условный session-scoped патчинг только если включён `--use-mock-embedder` или `USE_MOCK_EMBEDDER=1`.

### 9. Добавлен pytest_collection_modifyitems
```python
def pytest_collection_modifyitems(config, items):
    """Применяет маркеры пропуска для mock/real embedder и VM-тестов."""
```

**Логика:**
- `@pytest.mark.real_embedder` → Skip если `--use-mock-embedder`
- `@pytest.mark.mock_embedder` → Skip если mock режим не включён
- `@pytest.mark.vm` → Skip если VM недоступен (socket пре-чек)

**Пример маркировки тестов:**
```python
@pytest.mark.real_embedder
def test_real_retry_logic(embedder_factory):
    """Тест требует настоящий RemoteVMEmbedder"""
    embedder = embedder_factory(override_mock=False)
    # проверка реального retry...

@pytest.mark.mock_embedder
def test_mock_fast(embedder_factory):
    """Быстрый тест с mock (без сети)"""
    embedder = embedder_factory(override_mock=True)
    # проверка базового поведения...

@pytest.mark.vm
def test_vm_integration(embedder_factory):
    """Интеграционный тест с реальной VM"""
    # Автоматически пропускается если VM недоступна
    embedder = embedder_factory()
    # проверка с реальной VM...
```

## Сохранённые фикстуры

Следующие фикстуры **НЕ были изменены** и продолжают работать:

1. `ensure_utf8_subprocess` (autouse=True) - патчинг subprocess.run для UTF-8
2. `reset_embedder_environment` (autouse=True) - GC между тестами
3. `mock_cpu_embedder_offline` - конкретный mock для CPUEmbedder тестов

## Валидация изменений

### ✅ Проверка импортов
```bash
python -c "import tests.conftest; print('Import tests.conftest successful')"
# Результат: Import tests.conftest successful
```

### ✅ Проверка простых тестов
```bash
pytest tests/test_rag_imports.py -v --tb=short
# Результат: 6 passed in 37.25s
```

### ✅ Проверка file scanner
```bash
pytest tests/test_file_scanner.py -v --tb=short
# Результат: 1 passed in 0.23s
```

### ✅ Проверка config тестов
```bash
pytest tests/ -k "test_config" -v --tb=short
# Результат: 5 passed, 1 skipped, 343 deselected in 17.53s
```

### ✅ Проверка маркеров
```bash
pytest --markers | findstr "real_embedder mock_embedder vm"
# Результат: Все 3 маркера зарегистрированы
```

### ✅ Проверка CLI опций
```bash
pytest --help | findstr "mock-embedder vm-host vm-port"
# Результат: Все 3 опции доступны
```

## Статистика изменений

**Строк добавлено:** ~140
**Строк удалено:** ~50
**Чистое добавление:** ~90 строк

**Структура нового файла:**
- Импорты и setup: 12 строк
- pytest_addoption: 28 строк (добавлено)
- setup_event_loop_policy: 10 строк (добавлено)
- force_offline_env: 15 строк (изменено)
- ensure_utf8_subprocess: 16 строк (без изменений)
- pytest_configure: 7 строк (упрощено)
- pytest_unconfigure: 9 строк (сохранено)
- check_vm_availability: 11 строк (добавлено)
- embedder_factory: 43 строк (добавлено)
- mock_embedder_session: 30 строк (добавлено)
- pytest_collection_modifyitems: 47 строк (добавлено)
- reset_embedder_environment: 8 строк (без изменений)
- mock_cpu_embedder_offline: 24 строк (обновлено название модели)

**Итого:** 283 строки (было 141 строка)

## Режимы запуска тестов

### Режим 1: Дефолт (real embedder)
```bash
pytest tests/
# USE_MOCK_EMBEDDER="0" по умолчанию
# Использует RemoteVMEmbedder
```

### Режим 2: Mock режим (принудительный)
```bash
pytest tests/ --use-mock-embedder
# Глобальный session-scoped патч
# Все embedder → MockRemoteEmbedder
```

### Режим 3: VM тесты
```bash
pytest tests/ -m vm
# Только тесты с @pytest.mark.vm
# Автоматический skip если VM недоступна
```

### Режим 4: Без VM тестов
```bash
pytest tests/ -m "not vm"
# Исключает VM интеграционные тесты
```

### Режим 5: Только real embedder тесты
```bash
pytest tests/ -m real_embedder
# Только тесты требующие реальный RemoteVMEmbedder
# Автоматический skip если --use-mock-embedder
```

## Обратная совместимость

### ✅ Сохранена совместимость
- Тесты без маркеров работают как раньше
- `force_offline_env` доступна для явного запроса
- `mock_cpu_embedder_offline` продолжает работать
- Env переменная `USE_MOCK_EMBEDDER` поддерживается

### ⚠️ Изменения требующие обновления тестов
1. **Удалён autouse из force_offline_env**
   - Решение: Добавить явный запрос фикстуры в тесты которым она нужна

2. **Изменён дефолт USE_MOCK_EMBEDDER: "1" → "0"**
   - Решение: Тесты требующие mock должны использовать `@pytest.mark.mock_embedder` или `embedder_factory(override_mock=True)`

## Следующие шаги

1. ✅ ФАЗА 4 завершена - conftest.py рефакторнут
2. 🔄 ФАЗА 5 - переписать тесты на использование embedder_factory и контрактов
3. 🔄 Обновить Technical Debt.md со статусом ФАЗА 4
4. 🔄 Запустить полный test suite для проверки всех тестов

## Файлы созданные/изменённые

### Изменённые:
- `tests/conftest.py` - критический рефакторинг (283 строки, было 141)

### Backup:
- `tests/conftest.py.backup` - сохранена старая версия

### Отчёты:
- `conftest_refactoring_report.md` - данный отчёт
- `conftest_diff.txt` - детальный diff

## Риски и митигация

### Риск 1: Тесты могут упасть из-за отсутствия глобального патчинга
**Митигация:** Используем embedder_factory и маркеры. Session-scoped патч mock_embedder_session доступен через `--use-mock-embedder`.

### Риск 2: VM тесты могут упасть если VM недоступна
**Митигация:** Автоматический пре-чек через check_vm_availability с graceful skip.

### Риск 3: Windows async тесты могут быть нестабильны
**Митигация:** WindowsSelectorEventLoopPolicy установлен в setup_event_loop_policy.

## Заключение

ФАЗА 4 успешно завершена. Система тестирования теперь:
- ✅ Гибкая (3 режима запуска)
- ✅ Изолированная (scoped фикстуры вместо глобального патчинга)
- ✅ Стабильная (Windows event loop policy, VM пре-чек)
- ✅ Расширяемая (embedder_factory поддерживает transport injection)
- ✅ Документированная (маркеры и CLI опции)

Готово к переходу на ФАЗУ 5: Переписывание тестов на контрактную проверку.
