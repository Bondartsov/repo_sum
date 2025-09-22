# Active Tasks: Repository Analyzer

**Дата обновления:** 22 сентября 2025
**Статус:** M2.5 VM Migration - 80% ЗАВЕРШЕНО
**Версия:** 0.7.1 (M2.5 VM Migration SUCCESS + async/sync fixes required)

---

## 🎯 КОНСОЛИДАЦИЯ АКТИВНЫХ ЗАДАЧ

### **КРИТИЧЕСКИЙ ПУТЬ M2.5 ЗАВЕРШЕНИЯ (3-5 дней):**

---

## 🚨 КРИТИЧЕСКИЕ ЗАДАЧИ (ВЫСШИЙ ПРИОРИТЕТ)

### **1. Async/Sync Integration Fix**
**Статус:** ❌ НЕ ЗАВЕРШЕНО
**Приоритет:** КРИТИЧЕСКИЙ (блокирует весь поиск)
**Время:** 1-2 дня
**Ответственный:** Разработчик

#### **Проблема:**
```python
# В remote_embedder.py:
async def embed_texts() -> np.ndarray  # Возвращает coroutine

# В search_service.py:
embeddings = self.embedder.embed_texts(texts)  # Синхронный вызов
# Результат: RuntimeWarning: coroutine was never awaited
```

#### **Решение:**
```python
def embed_texts(self, texts: List[str], task: str = None) -> np.ndarray:
    """Синхронный wrapper для async HTTP запроса"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.create_task(self._async_embed_texts(texts, task))
        else:
            return asyncio.run(self._async_embed_texts(texts, task))
    except Exception as e:
        logger.error(f"Ошибка sync wrapper: {e}")
        return np.zeros((len(texts), self.truncate_dim))
```

#### **Файлы для изменения:**
- `rag/remote_embedder.py` - добавить sync wrapper
- `rag/remote_vector_store.py` - аналогичные исправления
- `tests/test_remote_clients.py` - обновить тесты

#### **Критерии успеха:**
- [ ] ❌ `python main.py rag search` работает без async warnings
- [ ] ❌ Нет RuntimeWarning о coroutine в логах
- [ ] ❌ SearchService возвращает корректные результаты

---

### **2. SearchService Integration Testing**
**Статус:** ❌ НЕ ЗАВЕРШЕНО
**Приоритет:** КРИТИЧЕСКИЙ (проверка функциональности)
**Время:** 1 день
**Ответственный:** Тестировщик

#### **Проблема:**
- `object of type 'coroutine' has no len()` при поиске
- Неполная интеграция remote клиентов

#### **Задачи:**
- [ ] ❌ Проверить полный workflow: index → search → результаты
- [ ] ❌ Тестирование CLI команд с VM backend
- [ ] ❌ Проверка обработки ошибок и fallback логики
- [ ] ❌ Валидация качества поиска Jina v3 vs BGE

#### **Файлы для тестирования:**
- `main.py` - CLI команды rag search
- `rag/search_service.py` - основной поиск
- `rag/query_engine.py` - движок запросов
- `tests/rag/test_rag_integration.py` - интеграционные тесты

#### **Критерии успеха:**
- [ ] ❌ Поиск возвращает релевантные результаты
- [ ] ❌ Нет ошибок типа "coroutine has no len"
- [ ] ❌ Fallback логика работает при проблемах с VM

---

### **3. Web UI Integration Testing**
**Статус:** ❌ НЕ ЗАВЕРШЕНО
**Приоритет:** КРИТИЧЕСКИЙ (пользовательский интерфейс)
**Время:** 1 день
**Ответственный:** Frontend разработчик

#### **Проблема:**
- Streamlit RAG функции не протестированы с VM backend
- Возможные проблемы с async/sync в UI

#### **Задачи:**
- [ ] ❌ Тестирование вкладки "🔍 RAG: Поиск по коду"
- [ ] ❌ Проверка Q&A интерфейса с VM RAG
- [ ] ❌ Валидация real-time поиска с Jina v3
- [ ] ❌ Тестирование параллельной индексации

#### **Файлы для тестирования:**
- `web_ui.py` - Streamlit интерфейс
- `rag/search_service.py` - RAG поиск
- `tests/test_web_ui.py` - UI тесты

#### **Критерии успеха:**
- [ ] ❌ Web UI RAG поиск функционирует корректно
- [ ] ❌ Q&A система работает с VM контекстом
- [ ] ❌ Real-time поиск показывает результаты Jina v3

---

## ⚠️ ВАЖНЫЕ ЗАДАЧИ (СРЕДНИЙ ПРИОРИТЕТ)

### **4. Performance Benchmarking**
**Статус:** ❌ НЕ ЗАВЕРШЕНО
**Приоритет:** ВЫСОКИЙ
**Время:** 1-2 дня
**Ответственный:** Performance инженер

#### **Задачи:**
- [ ] ❌ Сравнение качества Jina v3 vs BGE
- [ ] ❌ Бенчмарки latency VM поиска
- [ ] ❌ Тестирование concurrent пользователей
- [ ] ❌ Оптимизация VM request batching

#### **Метрики для проверки:**
- **Search Quality**: +40-60% improvement vs BGE
- **Latency**: <200ms cached, <500ms cold
- **Throughput**: 15-20 файлов/сек индексация
- **Concurrency**: 50+ пользователей

#### **Файлы для работы:**
- `tests/rag/test_rag_performance.py` - performance тесты
- `rag/search_service.py` - оптимизации поиска
- `scripts/benchmarks/` - benchmarking скрипты

---

### **5. Error Handling Robustness**
**Статус:** ❌ НЕ ЗАВЕРШЕНО
**Приоритет:** ВЫСОКИЙ
**Время:** 1 день
**Ответственный:** Разработчик

#### **Задачи:**
- [ ] ❌ Улучшение fallback логики в remote клиентах
- [ ] ❌ Comprehensive retry logic для VM соединений
- [ ] ❌ Graceful degradation при проблемах с VM
- [ ] ❌ Мониторинг и alerting для VM сервиса

#### **Файлы для работы:**
- `rag/remote_embedder.py` - error handling
- `rag/remote_vector_store.py` - retry логика
- `rag/exceptions.py` - система исключений

---

### **6. Documentation Updates**
**Статус:** 🔄 В ПРОЦЕССЕ
**Приоритет:** СРЕДНИЙ
**Время:** 1 день
**Ответственный:** Технический писатель

#### **Задачи:**
- [ ] ✅ Обновление SETUP.md для VM архитектуры
- [ ] ✅ Документация troubleshooting VM проблем
- [ ] ✅ Обновление README.md с VM инструкциями
- [ ] ✅ Создание FAQ по VM Migration

#### **Файлы для обновления:**
- `SETUP.md` - единая инструкция по настройке
- `README.md` - основная документация
- `scripts/vm_setup_phase1.py` - VM автоматизация

---

## 📋 ПАРАЛЛЕЛЬНЫЕ ЗАДАЧИ (НИЗКИЙ ПРИОРИТЕТ)

### **7. Monitoring & Observability**
**Статус:** ❌ НЕ НАЧАТО
**Приоритет:** НИЗКИЙ
**Время:** 2-3 дня
**Ответственный:** DevOps инженер

#### **Задачи:**
- [ ] ❌ Настройка Prometheus метрик для VM services
- [ ] ❌ Grafana дашборды для VM performance
- [ ] ❌ Health checks и auto-recovery
- [ ] ❌ Логирование VM API запросов

### **8. Unicode Logging Fix**
**Статус:** ❌ НЕ НАЧАТО
**Приоритет:** НИЗКИЙ (косметическая)
**Время:** 0.5 дня
**Ответственный:** Разработчик

#### **Проблема:**
- Windows cmd.exe не поддерживает emoji в STDERR
- Логи содержат символы замены вместо emoji

#### **Решение:**
- ASCII-only logging для Windows терминалов
- Conditional emoji support detection

---

## 📊 СТАТУС ЗАДАЧ ПО КАТЕГОРИЯМ

### **ПО ПРИОРИТЕТАМ:**
- **КРИТИЧЕСКИЕ (блокируют релиз):** 3 задачи (1-2 дня)
- **ВАЖНЫЕ (влияют на качество):** 3 задачи (2-3 дня)
- **ПАРАЛЛЕЛЬНЫЕ (улучшения):** 2 задачи (2.5 дня)

### **ПО СТАТУСУ:**
- **НЕ ЗАВЕРШЕНО:** 6 задач
- **В ПРОЦЕССЕ:** 1 задача
- **НЕ НАЧАТО:** 2 задачи

### **ОБЩИЙ ПРОГРЕСС:**
```
КРИТИЧЕСКИЕ: █░░░░░░░ 15% (3/20 задач)
ВАЖНЫЕ:      ██░░░░░░ 20% (4/20 задач)
ВСЕГО:       ███░░░░░ 35% (7/20 задач)
```

---

## 🎯 КРИТЕРИИ УСПЕХА M2.5

### **ТЕХНИЧЕСКИЕ КРИТЕРИИ:**
- [ ] ❌ Remote клиенты работают без async warnings
- [ ] ❌ Поиск возвращает релевантные результаты
- [ ] ❌ Web UI RAG функции работают корректно
- [ ] ❌ Performance benchmarks показывают улучшения

### **ПОЛЬЗОВАТЕЛЬСКИЕ КРИТЕРИИ:**
- [ ] ❌ `python main.py rag search "query"` работает без ошибок
- [ ] ❌ Streamlit RAG поиск функционирует
- [ ] ❌ Качество ответов значительно улучшено vs BGE
- [ ] ✅ Setup инструкции позволяют воспроизвести результат

### **PRODUCTION КРИТЕРИИ:**
- [ ] ❌ VM сервис работает стабильно 24/7
- [ ] ❌ Automatic failover при проблемах
- [ ] ✅ Comprehensive documentation обновлена
- [ ] ❌ Monitoring и alerting настроены

---

## 📅 КРАТКОСРОЧНЫЙ ПЛАН (1-2 НЕДЕЛИ)

### **НЕДЕЛЯ 1: Критические исправления**
1. **Дни 1-2**: Async/sync fix в remote_embedder.py и remote_vector_store.py
2. **День 3**: Интеграционное тестирование поиска
3. **День 4**: Web UI integration testing
4. **День 5**: Performance benchmarking против BGE

### **НЕДЕЛЯ 2: Финализация M2.5**
1. **Дни 1-2**: Production testing и оптимизация
2. **День 3**: Documentation updates
3. **Дни 4-5**: M3 планирование и архитектура

---

## 🔗 СВЯЗАННАЯ ИНФОРМАЦИЯ

**Статус проекта:** [project_status.md](project_status.md)
**Текущий фокус:** [activeContext.md](activeContext.md)
**История задач:** [progress.md](progress.md)
**Технический контекст:** [techContext.md](techContext.md)

---

## 📝 ПРИМЕЧАНИЯ

### **БЛОКИРУЮЩИЕ ПРОБЛЕМЫ:**
- Async/Sync проблемы блокируют весь RAG функционал
- Требуют немедленного внимания
- ETA решения: 1-2 дня

### **ЗАВИСИМОСТИ:**
- Все задачи 1-3 зависят от решения async/sync проблем
- Задачи 4-5 могут выполняться параллельно после исправления 1-3
- Задачи 6-8 не критичны для M2.5 completion

### **РЕСУРСЫ:**
- VM доступна: 10.61.11.54:8000 (health check: healthy)
- Jina v3 модель загружена и работает
- Qdrant настроен и функционирует

---

**Дата создания:** 22 сентября 2025
**Статус:** Active tasks consolidation
**Следующее обновление:** При изменении статуса задач или появлении новых