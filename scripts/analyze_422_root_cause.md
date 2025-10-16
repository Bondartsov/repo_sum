# 🔍 Анализ ошибки 422 Unprocessable Entity

**Дата:** 15.10.2025, 19:15 MSK
**Статус:** ✅ ROOT CAUSE НАЙДЕНА

---

## 🎯 Симптомы из логов

```
{"validation_errors": 192}
POST /index HTTP/1.1 422 Unprocessable Entity
```

- **192 ошибки валидации**
- POST /embeddings работает (200 OK)
- POST /index падает с 422

---

## 🔬 Серверная валидация (vm_rag_service.py)

### Pydantic модели требуют:

```python
class IndexedMetadata(ExtraIgnoreModel):
    file_path: ConStr(min_length=1)      # НЕ ПУСТАЯ строка ≥1 символ
    line_start: conint(ge=0)              # Целое число ≥0
    line_end: conint(ge=0)                # Целое число ≥0
    language: ConStr(min_length=1)        # НЕ ПУСТАЯ строка ≥1 символ
    repo: ConStr(min_length=1)            # НЕ ПУСТАЯ строка ≥1 символ
    chunk_type: ConStr(min_length=1)      # НЕ ПУСТАЯ строка ≥1 символ

class IndexedDocument(ExtraIgnoreModel):
    id: ConStr(min_length=1)              # НЕ ПУСТАЯ строка
    text: ConStr(min_length=1)            # НЕ ПУСТАЯ строка
    metadata: IndexedMetadata
    embedding_version: ConStr(min_length=1)
    content_sha256: ConStr(regex=r'^[A-Fa-f0-9]{64}$')
```

---

## 🕵️ Клиентская нормализация (rag/remote_vector_store.py)

### Функция `_normalize_metadata()`:

```python
def _normalize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(meta or {})
    
    # Переименование синонимов
    if "line_start" not in m and "start_line" in m:
        m["line_start"] = int(m.get("start_line") or 0)
    if "line_end" not in m and "end_line" in m:
        m["line_end"] = int(m.get("end_line") or 0)
    
    # Гарантируем наличие обязательных полей
    m.setdefault("line_start", 0)
    m.setdefault("line_end", 0)
    
    # Обязательные поля
    m["file_path"] = m.get("file_path") or "unknown"
    m["language"] = m.get("language") or "unknown"
    m["repo"] = m.get("repo") or "default"
    m["chunk_type"] = m.get("chunk_type") or "code"
    
    return m
```

---

## 🐛 ROOT CAUSE: Проблема с типами данных

### Математика ошибок:

```
192 ошибки ÷ 32 документа = 6 ошибок на документ
```

6 полей metadata (file_path, line_start, line_end, language, repo, chunk_type) × 32 документа = **192 ошибки**

### Гипотеза:

Клиент отправляет `line_start` и `line_end` как **СТРОКИ** вместо **ЦЕЛЫХ ЧИСЕЛ**!

Пример неправильного payload:
```json
{
  "metadata": {
    "file_path": "src/main.py",
    "line_start": "10",        // ❌ СТРОКА вместо числа!
    "line_end": "50",          // ❌ СТРОКА вместо числа!
    "language": "python",
    "repo": "repo_sum",
    "chunk_type": "class"
  }
}
```

Pydantic ожидает:
```python
line_start: conint(ge=0)  # int, а не str!
line_end: conint(ge=0)    # int, а не str!
```

---

## 🔍 Где происходит формирование payload?

### Функция `_async_index_documents()` (rag/remote_vector_store.py:150-340):

```python
async def _async_index_documents(self, points: List[Dict], ...):
    # ...
    for point in points:
        # Нормализация metadata
        norm_meta = _normalize_metadata(point.get("metadata", {}))
        
        # Формирование документа
        doc = {
            "id": point.get("id"),
            "text": point.get("text"),
            "metadata": norm_meta,  # ← Здесь может быть проблема!
            # ...
        }
```

### Проблема в `_normalize_metadata()`:

```python
# ❌ ПРОБЛЕМА: setdefault устанавливает int, но line_start может быть УЖЕ строкой!
m.setdefault("line_start", 0)
m.setdefault("line_end", 0)
```

Если `line_start` уже присутствует в metadata как строка "10", то `setdefault()` **НЕ ИЗМЕНИТ** его, т.к. ключ уже существует!

---

## ✅ ИСПРАВЛЕНИЕ

### Гарантировать конвертацию в int:

```python
def _normalize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(meta or {})
    
    # Переименование синонимов
    if "line_start" not in m and "start_line" in m:
        m["line_start"] = int(m.get("start_line") or 0)
    if "line_end" not in m and "end_line" in m:
        m["line_end"] = int(m.get("end_line") or 0)
    
    # ✅ ИСПРАВЛЕНИЕ: Гарантируем конвертацию в int
    try:
        m["line_start"] = int(m.get("line_start") or 0)
    except (ValueError, TypeError):
        m["line_start"] = 0
    
    try:
        m["line_end"] = int(m.get("line_end") or 0)
    except (ValueError, TypeError):
        m["line_end"] = 0
    
    # Обязательные строковые поля
    m["file_path"] = str(m.get("file_path") or "unknown")
    m["language"] = str(m.get("language") or "unknown")
    m["repo"] = str(m.get("repo") or "default")
    m["chunk_type"] = str(m.get("chunk_type") or "code")
    
    # Гарантируем что строки не пустые
    if not m["file_path"].strip():
        m["file_path"] = "unknown"
    if not m["language"].strip():
        m["language"] = "unknown"
    if not m["repo"].strip():
        m["repo"] = "default"
    if not m["chunk_type"].strip():
        m["chunk_type"] = "code"
    
    return m
```

---

## 🎯 Резюме

**Проблема:** Клиент отправляет `line_start` и `line_end` как строки, а Pydantic ожидает int.

**Решение:** Принудительно конвертировать в int с обработкой ошибок.

**Файл для исправления:** `rag/remote_vector_store.py`, функция `_normalize_metadata()`

**Оценка:** 5 минут на исправление + 10 минут на тестирование.
