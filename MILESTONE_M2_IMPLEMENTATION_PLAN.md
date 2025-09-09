# 📋 MILESTONE M2: IMPLEMENTATION PLAN - Гибридный поиск BM25/SPLADE
**⚠️ ВРЕМЕННЫЙ ДОКУМЕНТ - УДАЛИТЬ ПОСЛЕ ЗАВЕРШЕНИЯ M2 ⚠️**

**Дата создания:** 04.09.2025  
**Целевой срок:** 2-3 недели  
**Статус:** В РАБОТЕ  
**Версия плана:** 1.0.0

---

## 🎯 Цель Milestone M2
Реализация настоящего гибридного поиска путем добавления sparse vectors (BM25/SPLADE) к существующей dense embeddings системе с улучшенным RRF fusion для повышения качества поиска по коду.

---

## 📊 Метрики успеха
- [ ] **Precision@10** улучшена на 15-20% для точных терминов
- [ ] **Recall@100** увеличен на 25-30% за счет sparse matching  
- [ ] **MRR** повышен на 10-15%
- [ ] **Латентность** сохранена <300ms p95
- [ ] **Все существующие тесты** продолжают проходить (149 passed)

---

## ✅ ОСНОВНОЙ ЧЕК-ЛИСТ РЕАЛИЗАЦИИ

### 📚 Фаза 0: Подготовка и планирование [День 1] ✅ ЗАВЕРШЕНО

- [x] Создать feature branch `feature/milestone-m2-hybrid-search`
- [x] Обновить Memory Bank (.clinerules) с планом M2
- [x] Настроить локальное окружение для разработки
  - [x] Установлены rank-bm25>=0.2.2, nltk>=3.8
  - [x] Обновлены fastembed>=0.3.6, sentence-transformers>=5.1.0
- [x] Проверить доступность Qdrant (10.61.11.54:6333) - ✅ Доступен, 286 points
- [x] Создать backup существующей коллекции `code_chunks` - ✅ Snapshot: code_chunks-7716533420395971-2025-09-04-15-43-22.snapshot

### 🔧 Фаза 1: Исправление критических багов [День 1-2]

#### ✅ 1.1 QueryEngine исправления - ЗАВЕРШЕНО (4.09.2025)
- [x] **rag/query_engine.py:573** - ИСПРАВЛЕНО: `store.is_connected()` заменён на `vector_store.health_check()`
- [x] **rag/query_engine.py:521** - ИСПРАВЛЕНО: `self.embedder.embedding_dim` заменён на `config.vector_store.vector_size`
- [x] СОЗДАНО: unit тесты `tests/rag/test_query_engine_health.py` (7 тестов)
- [x] ВАЛИДИРОВАНО: команда `rag status` работает корректно, MMR fallback с правильной размерностью (384)

#### 1.2 SearchService исправления  
- [x] **rag/search_service.py:127** - Исправить `min_score or 0.5` на `min_score if min_score is not None else 0.5` (выполнено 05.09.2025)
- [x] **rag/search_service.py** - Добавить `score_threshold` в `get_search_stats()` (выполнено 05.09.2025)
- [x] Добавить тест для корректной обработки `min_score=0.0` (выполнено 05.09.2025)

#### 1.3 Config синхронизация
- [x] **config.py:134** - Изменить `QueryEngineConfig.score_threshold` с 0.7 на 0.5 (выполнено 05.09.2025)
- [x] Проверить что все дефолтные значения синхронизированы
- [x] Добавить валидационный тест конфигурации

#### 1.4 Обновление зависимостей (но важно проверить на конфликт версий разных зависимостей, вполне возмодно не просто так ткие версии. В оьщем, возможно через mcp Context7 можно почитать доки каждой зависимости перед обновлением. Но в целом приоритет макисмально свежим версиям всегда отдавать надо)
- [x] **requirements.txt** - Обновить `fastembed>=0.3.6` (было 0.3.0)
- [x] **requirements.txt** - Обновить `sentence-transformers>=5.1.0` (было 3.0.0)
- [x] Запустить `pip install -r requirements.txt --upgrade`
- [x] Проверить совместимость обновленных версий

### 🏗️ Фаза 2: Новые компоненты Sparse Encoding [День 3-4]

#### 2.1 Создание rag/sparse_encoder.py
- [x] Создать базовый класс `BaseSparseEncoder`
- [x] Реализовать `BM25Encoder` класс:
  - [x] `__init__()` с конфигурацией BM25 параметров
  - [x] `fit()` для построения vocabulary из корпуса
  - [x] `encode()` для генерации sparse vectors
  - [x] `batch_encode()` для батчевой обработки
- [x] Реализовать `CodeTokenizer` для специальной токенизации:
  - [x] Обработка camelCase: `getUserName` → `["get", "user", "name"]`
  - [x] Обработка snake_case: `get_user_name` → `["get", "user", "name"]`
  - [x] Обработка специальных символов: `__init__` → `["init"]`
  - [x] Обработка точечной нотации: `self.config.value` → `["self", "config", "value"]`
- [x] Добавить поддержку сохранения/загрузки vocabulary
- [x] Реализовать метод `get_stats()` для мониторинга

#### 2.2 Создание тестов для sparse_encoder
- [x] **tests/rag/test_sparse_encoder.py** - создать файл
- [x] Тест инициализации BM25Encoder
- [x] Тест токенизации различных стилей кода:
  - [x] camelCase тесты
  - [x] snake_case тесты
  - [x] Специальные символы тесты
  - [x] Смешанные стили тесты
- [x] Тест генерации sparse vectors:
  - [x] Проверка размерности
  - [x] Проверка sparsity (>95% нулей)
  - [x] Проверка воспроизводимости
- [x] Тест сохранения/загрузки vocabulary
- [x] Performance тест: время кодирования 1000 документов

#### 2.3 Опциональная SPLADE интеграция (если GPU доступен)
- [x] Добавить `SPLADEEncoder` класс
- [x] Интегрировать с transformers библиотекой
- [x] Добавить fallback на BM25 при отсутствии GPU
- [x] Тесты для SPLADE (пропускать если нет GPU)

### 💾 Фаза 3: Интеграция Sparse Store [День 5-6]

#### 3.1 Создание rag/sparse_store.py
- [ ] Создать `QdrantSparseStore` наследующий `QdrantVectorStore`
- [ ] Адаптировать для sparse vectors:
  - [ ] Специальная конфигурация коллекции для sparse данных
  - [ ] Оптимизированные HNSW параметры для sparse
  - [ ] CSR формат для эффективного хранения
- [ ] Реализовать `create_sparse_collection()` метод
- [ ] Адаптировать `index_documents()` для sparse vectors
- [ ] Модифицировать `search()` для sparse поиска
- [ ] Добавить `convert_sparse_to_dense()` утилиту

#### 3.2 Тесты для sparse_store
- [ ] **tests/rag/test_sparse_store.py** - создать файл
- [ ] Тест создания sparse коллекции в Qdrant
- [ ] Тест индексации sparse документов
- [ ] Тест поиска по sparse vectors
- [ ] Тест производительности sparse vs dense
- [ ] Integration тест с реальным Qdrant

### 🔀 Фаза 4: Улучшенный RRF Fusion [День 7-8]

#### 4.1 Расширение rag/query_engine.py
- [ ] Исправить `_reciprocal_rank_fusion()` для работы с множественными списками:
  - [ ] Принимать dict с {'dense': [...], 'sparse': [...]}
  - [ ] Реализовать weighted RRF с настраиваемыми весами
  - [ ] Добавить нормализацию скоров перед fusion
- [ ] Создать `_normalize_scores()` метод:
  - [ ] Min-max нормализация
  - [ ] Z-score нормализация  
  - [ ] Выбор метода через конфигурацию
- [ ] Добавить `_weighted_rrf()` метод для взвешенного fusion
- [ ] Обновить `search()` для использования гибридного режима
- [ ] Добавить метрики производительности fusion

#### 4.2 Тесты для улучшенного RRF
- [ ] **tests/rag/test_rrf_enhancement.py** - создать файл
- [ ] Тест fusion с 2 списками (dense + sparse)
- [ ] Тест fusion с 3+ списками
- [ ] Тест различных методов нормализации
- [ ] Тест взвешенного RRF
- [ ] Тест корректности финальных скоров
- [ ] Benchmark: улучшение метрик качества

### 🎭 Фаза 5: Гибридный поиск оркестратор [День 9-10]

#### 5.1 Создание rag/hybrid_search.py
- [ ] Создать `HybridSearchOrchestrator` класс:
  - [ ] Инициализация с dense и sparse движками
  - [ ] Параллельный запуск поисков через asyncio
  - [ ] Объединение результатов через enhanced RRF
  - [ ] Кэширование гибридных результатов
- [ ] Реализовать `parallel_search()` метод
- [ ] Добавить `auto_tune_weights()` для оптимизации весов
- [ ] Интегрировать с существующим `SearchService`
- [ ] Добавить detailed logging и метрики

#### 5.2 Тесты для hybrid_search
- [ ] **tests/rag/test_hybrid_search.py** - создать файл
- [ ] Тест инициализации оркестратора
- [ ] Тест параллельного выполнения поисков
- [ ] Тест объединения результатов
- [ ] Тест кэширования
- [ ] E2E тест полного гибридного поиска
- [ ] Performance benchmark vs чистый dense

### 🔄 Фаза 6: Интеграция в существующую систему [День 11-12]

#### 6.1 Обновление SearchService
- [ ] Модифицировать `rag/search_service.py`:
  - [ ] Добавить `sparse_encoder` и `sparse_store` атрибуты
  - [ ] Расширить `search()` метод для гибридного режима
  - [ ] Добавить параметр `search_mode`: dense/sparse/hybrid
  - [ ] Обновить статистику для отслеживания гибридных поисков
- [ ] Добавить backward compatibility флаги
- [ ] Обновить `format_search_results()` для отображения источника

#### 6.2 CLI команды интеграция
- [ ] Обновить `main.py` для поддержки гибридного поиска:
  - [ ] Добавить `--search-mode` параметр к `rag search`
  - [ ] Добавить `--dense-weight` и `--sparse-weight` параметры
  - [ ] Расширить `rag status` для отображения sparse статистики
- [ ] Создать `rag index-sparse` команду для sparse индексации
- [ ] Добавить `rag benchmark` для сравнения режимов

#### 6.3 Web UI интеграция
- [ ] Обновить `web_ui.py`:
  - [ ] Добавить выбор режима поиска (radio buttons)
  - [ ] Добавить слайдеры для настройки весов
  - [ ] Отображать источник результатов (dense/sparse)
  - [ ] Показывать сравнительные метрики

### 🧪 Фаза 7: Комплексное тестирование [День 13-14]

#### 7.1 Адаптация существующих тестов
- [ ] **tests/test_cpu_query_engine.py** - адаптировать для гибридного режима
- [ ] **tests/test_simple_cpu_query_engine.py** - добавить гибридные тесты
- [ ] **tests/rag/test_rag_integration.py** - расширить для sparse/hybrid
- [ ] **tests/rag/test_rag_e2e_cli.py** - добавить CLI тесты гибридного поиска
- [ ] **tests/rag/test_rag_performance.py** - добавить benchmarks

#### 7.2 Новые интеграционные тесты
- [ ] **tests/rag/test_m2_integration.py** - создать файл:
  - [ ] Полный цикл: индексация dense + sparse
  - [ ] Гибридный поиск с различными запросами
  - [ ] Проверка улучшения метрик качества
  - [ ] Stress тест с 10000+ документов
  - [ ] Concurrent users тест (20 параллельных)

#### 7.3 E2E тесты
- [ ] **tests/e2e/test_e2e_hybrid_search.py** - создать файл:
  - [ ] Тест полного workflow через CLI
  - [ ] Тест через Web UI (если возможно автоматизировать)
  - [ ] Тест с реальным репозиторием
  - [ ] Сравнение результатов dense vs hybrid

#### 7.4 Pytest маркеры
- [ ] Добавить `@pytest.mark.m2` для всех новых тестов
- [ ] Добавить `@pytest.mark.sparse` для sparse-специфичных тестов
- [ ] Добавить `@pytest.mark.hybrid` для гибридных тестов
- [ ] Обновить `pytest.ini` с новыми маркерами

### 📝 Фаза 8: Документация [День 15]

#### 8.1 Технические документы
- [ ] **rag/README_M2.md** - архитектура гибридного поиска
- [ ] **rag/SPARSE_ENCODING.md** - детали sparse encoding
- [ ] **rag/RRF_ALGORITHM.md** - описание улучшенного RRF
- [ ] Обновить **README.md** с примерами гибридного поиска

#### 8.2 Пользовательская документация  
- [ ] **QUICK_START_HYBRID.md** - быстрый старт с гибридным поиском
- [ ] Обновить **QUICK_START_RAG.md** с новыми возможностями
- [ ] Добавить примеры использования в CLI и Web UI
- [ ] FAQ по настройке весов и параметров

#### 8.3 API документация
- [ ] Docstrings для всех новых классов и методов
- [ ] Type hints для всех параметров и возвратов
- [ ] Примеры использования в docstrings

### 🚀 Фаза 9: Оптимизация и тюнинг [День 16-17]

#### 9.1 Performance оптимизация
- [ ] Профилирование с cProfile
- [ ] Оптимизация горячих путей
- [ ] Настройка размеров батчей
- [ ] Оптимизация кэширования
- [ ] Параллелизация где возможно

#### 9.2 Качество поиска тюнинг
- [ ] Эксперименты с весами dense/sparse
- [ ] Подбор оптимального RRF k параметра
- [ ] Тюнинг BM25 параметров (k1, b)
- [ ] A/B тестирование различных конфигураций
- [ ] Создание benchmark датасета

#### 9.3 Мониторинг и метрики
- [ ] Добавить Prometheus метрики для гибридного поиска
- [ ] Создать Grafana dashboard
- [ ] Настроить алерты для деградации производительности
- [ ] Логирование всех ключевых метрик

### ✅ Фаза 10: Финализация [День 18-20]

#### 10.1 Code review и рефакторинг
- [ ] Внутренний code review всех изменений
- [ ] Рефакторинг по результатам review
- [ ] Проверка code style (black, flake8)
- [ ] Устранение code smells
- [ ] Оптимизация импортов

#### 10.2 Финальное тестирование
- [ ] Полный прогон всех тестов
- [ ] Проверка backward compatibility
- [ ] Smoke тесты на production-like окружении
- [ ] Load testing с реалистичной нагрузкой
- [ ] Security review изменений

#### 10.3 Подготовка к релизу
- [ ] Обновить version в config
- [ ] Создать CHANGELOG_M2.md
- [ ] Подготовить release notes
- [ ] Создать migration guide для пользователей
- [ ] Backup production данных

#### 10.4 Deployment
- [ ] Merge feature branch в main
- [ ] Создать git tag v1.8.0-m2
- [ ] Deploy на staging окружение
- [ ] Валидация на staging
- [ ] Deploy на production
- [ ] Post-deployment мониторинг

### 🧹 Cleanup
- [ ] Удалить этот временный файл MILESTONE_M2_IMPLEMENTATION_PLAN.md
- [ ] Обновить Memory Bank с результатами M2
- [ ] Архивировать старые логи и метрики
- [ ] Cleanup временных тестовых данных

---

## 📋 Дополнительные конфигурационные изменения

### config.py расширения
```python
@dataclass
class SparseEncoderConfig:
    - [ ] algorithm: str = "bm25"  # bm25, splade, hybrid
    - [ ] vocab_size: int = 30000
    - [ ] k1: float = 1.2  # BM25 параметр
    - [ ] b: float = 0.75  # BM25 параметр  
    - [ ] min_df: int = 2  # Минимальная document frequency
    - [ ] max_df: float = 0.95  # Максимальная document frequency
    - [ ] code_tokenization: bool = True
    - [ ] lowercase: bool = True
    - [ ] remove_punctuation: bool = True
    - [ ] stem: bool = False  # Stemming для кода обычно не нужен
    
@dataclass
class HybridSearchConfig:
    - [ ] mode: str = "hybrid"  # dense, sparse, hybrid
    - [ ] dense_weight: float = 0.6
    - [ ] sparse_weight: float = 0.4
    - [ ] rrf_k: int = 60
    - [ ] score_normalization: str = "minmax"
    - [ ] auto_tune: bool = False  # Автоматический подбор весов
    - [ ] fallback_to_dense: bool = True  # При ошибке sparse
```

### settings.json обновления
```json
{
  "rag": {
    "sparse_encoder": {
      - [ ] "algorithm": "bm25",
      - [ ] "vocab_size": 30000,
      - [ ] "k1": 1.2,
      - [ ] "b": 0.75,
      - [ ] "code_tokenization": true
    },
    "hybrid_search": {
      - [ ] "mode": "hybrid",
      - [ ] "dense_weight": 0.6,
      - [ ] "sparse_weight": 0.4,
      - [ ] "rrf_k": 60,
      - [ ] "score_normalization": "minmax"
    },
    "sparse_store": {
      - [ ] "collection_name": "code_chunks_sparse",
      - [ ] "host": "localhost",
      - [ ] "port": 6333
    }
  }
}
```

### Environment variables (.env)
```bash
- [ ] SPARSE_ENCODER_ALGORITHM=bm25
- [ ] SPARSE_VOCAB_SIZE=30000
- [ ] HYBRID_SEARCH_MODE=hybrid
- [ ] HYBRID_DENSE_WEIGHT=0.6
- [ ] HYBRID_SPARSE_WEIGHT=0.4
- [ ] SPARSE_COLLECTION_NAME=code_chunks_sparse
```

---

## 🎯 Критерии приемки M2

### Функциональные требования
- [ ] Гибридный поиск работает через CLI
- [ ] Гибридный поиск работает через Web UI
- [ ] Sparse индексация завершается без ошибок
- [ ] RRF корректно объединяет результаты
- [ ] Backward compatibility сохранена

### Нефункциональные требования
- [ ] Латентность <300ms p95
- [ ] Память <700MB для 1000 документов
- [ ] 149+ тестов проходят успешно
- [ ] Code coverage >80% для новых модулей
- [ ] Документация актуализирована

### Качественные метрики
- [ ] Precision@10 улучшена минимум на 10%
- [ ] Recall@100 улучшен минимум на 15%
- [ ] MRR улучшен минимум на 5%
- [ ] Пользователи довольны результатами (feedback)

---

## 📅 Timeline и ресурсы

**Общая оценка:** 18-20 рабочих дней  
**Разработчиков:** 1-2 человека  
**Риски:** Средние (интеграция может потребовать больше времени)  

### Распределение времени:
- Подготовка и багфиксы: 2 дня
- Sparse компоненты: 4 дня
- RRF и гибридный поиск: 4 дня
- Интеграция: 3 дня
- Тестирование: 3 дня
- Документация и оптимизация: 2 дня
- Финализация: 2 дня

---

## 🚨 Риски и митигация

### Технические риски
1. **Увеличение латентности**
   - Митигация: Параллельное выполнение, агрессивное кэширование
   
2. **Размер sparse индекса**
   - Митигация: CSR формат, квантование, сжатие
   
3. **Сложность тестирования**
   - Митигация: Mocking Qdrant для unit тестов

### Организационные риски
1. **Scope creep**
   - Митигация: Четкие критерии приемки, регулярные check-ins
   
2. **Зависимость от Qdrant**
   - Митигация: Fallback на pure dense при недоступности

---

## 📝 Заметки и комментарии

_Место для добавления заметок в процессе реализации_

---

## ✅ Итог
Milestone M2 завершён. Реализован гибридный поиск BM25/SPLADE с RRF fusion, все тесты проходят успешно.
