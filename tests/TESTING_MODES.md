# Режимы тестирования repo_sum

Эта система поддерживает **3 режима тестирования** для гибкого управления зависимостями от VM инфраструктуры.

## 🎯 Три режима тестирования

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
- CI/CD pipeline с доступом к VM
- Финальная валидация перед деплоем
- Performance тестирование

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
- 🚀 CI для feature branches

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

---

## 📊 Маркеры pytest

### `@pytest.mark.vm`
**Назначение:** Тесты, требующие доступную VM с запущенными сервисами.

**Комментарий в коде:**
```python
@pytest.mark.vm  # Требует доступную VM (10.61.11.54:8000) с запущенными FastAPI, Qdrant, Jina v3 сервисами
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

### `@pytest.mark.integration`
**Назначение:** Integration тесты с внешними зависимостями (OpenAI, Qdrant, filesystem).

**Поведение:**
- Запускаются во всех режимах
- Могут использовать как real, так и mock embedder

### `@pytest.mark.real_embedder`
**Назначение:** Тесты, которые **обязательно** используют RemoteVMEmbedder.

**Примеры:**
- `tests/test_remote_embedder_fixes.py`
- Тесты контрактов для production кода

### `@pytest.mark.mock_embedder`
**Назначение:** Тесты, которые **обязательно** используют MockRemoteEmbedder.

**Примеры:**
- Contract validation тесты
- Isolation тесты

---

## 🚀 Практические примеры

### Локальная разработка (без VM)
```bash
# Быстрая проверка логики
pytest tests/ --use-mock-embedder -v

# Проверка конкретного модуля
pytest tests/rag/test_circuit_breaker.py --use-mock-embedder -v
```

### CI/CD Pipeline
```bash
# Feature branch (без VM)
pytest tests/ --use-mock-embedder --cov=. --cov-report=html

# Master branch (с VM)
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
pytest tests/rag/test_vm_backend_integration.py::TestVMBackendIntegration::test_full_rag_workflow_index_search_results -v
```

---

## ⚙️ Конфигурация conftest.py

### Фикстура `embedder_factory`
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
    3. Environment variable USE_MOCK_EMBEDDER
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
```

---

## 📈 Метрики производительности

| Режим | Время выполнения | Покрытие | Когда использовать |
|-------|-----------------|----------|-------------------|
| **Дефолт** | 48+ минут | 100% (unit + integration + VM) | CI master, production validation |
| **Mock** | 16-20 секунд | ~70% (unit + contracts) | Локальная разработка, feature CI |
| **VM-only** | 10-15 минут | VM integration только | VM infrastructure check |

---

## 🔧 Troubleshooting

### Тесты падают в mock режиме
**Причина:** Тест не помечен `@pytest.mark.vm` но требует VM.

**Решение:**
```python
@pytest.mark.vm  # Требует доступную VM (10.61.11.54:8000) с запущенными FastAPI, Qdrant, Jina v3 сервисами
def test_full_pipeline():
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
import pytest_asyncio

@pytest_asyncio.fixture
async def async_fixture():
    ...

@pytest.mark.asyncio
async def test_async_function():
    ...
```

---

## 📝 Контрибьюция

При добавлении новых тестов:

1. **VM тесты** → добавить `@pytest.mark.vm`
2. **Integration тесты** → добавить `@pytest.mark.integration`
3. **Async тесты** → добавить `@pytest.mark.asyncio`
4. **Async фикстуры** → использовать `@pytest_asyncio.fixture`

Проверить:
```bash
# Mock режим работает
pytest tests/ --use-mock-embedder -v

# VM тесты пропускаются
pytest tests/ --use-mock-embedder --collect-only | grep -i "skipped"
```

---

## 🔗 См. также

- [tests/rag/TESTING_STRATEGY.md](rag/TESTING_STRATEGY.md) - детальная стратегия тестирования
- [tests/conftest.py](conftest.py) - конфигурация pytest
- [pytest.ini](../pytest.ini) - настройки pytest
- [CLAUDE.md](../.claude/CLAUDE.md) - правила разработки

---

**Автор:** Test Infrastructure Team
**Дата обновления:** 2 октября 2025
**Версия:** 1.0.0
