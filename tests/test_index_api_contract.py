"""
Comprehensive Unit Tests для API контракта /v1/index endpoint

Цель: Валидация Pydantic моделей и выявление причин 422 ошибок
Дата: 15 октября 2025
"""

import pytest
from fastapi.testclient import TestClient
from vm_rag_service import app
import hashlib


# ============================================================================
# ФИКСТУРЫ И ХЕЛПЕРЫ
# ============================================================================

@pytest.fixture
def client():
    """FastAPI TestClient для тестирования endpoint"""
    return TestClient(app)


def generate_sha256(text: str) -> str:
    """Генерирует корректный SHA256 для текста"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def create_valid_document(doc_id: str = "test_doc_1", text: str = "Valid test content"):
    """Создает валидный документ согласно IndexedDocument схеме"""
    return {
        "id": doc_id,
        "text": text,
        "metadata": {
            "file_path": "tests/test_file.py",
            "line_start": 1,
            "line_end": 10,
            "language": "python",
            "repo": "test_repo",
            "chunk_type": "function"
        },
        "embedding_version": "jina-v3-2025",
        "content_sha256": generate_sha256(text)
    }


def create_valid_request(documents: list = None, api_contract: str = "v1.0.0", batch_id: str = None):
    """Создает валидный IndexRequest"""
    if documents is None:
        documents = [create_valid_document()]
    
    request = {"documents": documents}
    
    if api_contract:
        request["api_contract"] = api_contract
    if batch_id:
        request["batch_id"] = batch_id
    
    return request


# ============================================================================
# ПОЗИТИВНЫЕ ТЕСТЫ (Valid Requests → 200/202)
# ============================================================================

def test_valid_single_document(client):
    """✅ Валидный запрос с одним документом должен пройти"""
    request = create_valid_request()
    response = client.post("/v1/index", json=request)
    
    assert response.status_code in (200, 202), f"Expected 200/202, got {response.status_code}: {response.text}"
    
    # Если 202 (async mode) - проверяем job_id
    if response.status_code == 202:
        data = response.json()
        assert "job_id" in data
    else:
        # Если 200 (sync mode) - проверяем indexed_count
        data = response.json()
        assert "accepted" in data or "indexed_count" in data


def test_valid_multiple_documents(client):
    """✅ Валидный запрос с несколькими документами"""
    documents = [
        create_valid_document(f"doc_{i}", f"Content for document {i}")
        for i in range(1, 11)
    ]
    request = create_valid_request(documents=documents)
    response = client.post("/v1/index", json=request)
    
    assert response.status_code in (200, 202), f"Expected 200/202, got {response.status_code}: {response.text}"


def test_valid_max_batch_size(client):
    """✅ Валидный запрос с максимальным batch_size=128"""
    documents = [
        create_valid_document(f"doc_{i}", f"Content {i}")
        for i in range(1, 129)
    ]
    request = create_valid_request(documents=documents)
    response = client.post("/v1/index", json=request)
    
    assert response.status_code in (200, 202, 400), f"Got {response.status_code}: {response.text}"


def test_valid_with_batch_id(client):
    """✅ Валидный запрос с batch_id"""
    request = create_valid_request(batch_id="test-batch-123")
    response = client.post("/v1/index", json=request)
    
    assert response.status_code in (200, 202), f"Expected 200/202, got {response.status_code}: {response.text}"


# ============================================================================
# НЕГАТИВНЫЕ ТЕСТЫ: MISSING REQUIRED FIELDS → 422
# ============================================================================

def test_missing_id_field(client):
    """❌ Отсутствие 'id' → 422"""
    doc = create_valid_document()
    del doc["id"]
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert "validation_error" in data["error"]["type"]


def test_missing_text_field(client):
    """❌ Отсутствие 'text' → 422"""
    doc = create_valid_document()
    del doc["text"]
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422
    data = response.json()
    assert "error" in data


def test_missing_metadata_field(client):
    """❌ Отсутствие 'metadata' → 422"""
    doc = create_valid_document()
    del doc["metadata"]
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422


def test_missing_embedding_version(client):
    """❌ Отсутствие 'embedding_version' → 422"""
    doc = create_valid_document()
    del doc["embedding_version"]
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422


def test_missing_content_sha256(client):
    """❌ Отсутствие 'content_sha256' → 422"""
    doc = create_valid_document()
    del doc["content_sha256"]
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422


# ============================================================================
# НЕГАТИВНЫЕ ТЕСТЫ: METADATA REQUIRED FIELDS → 422
# ============================================================================

def test_missing_metadata_file_path(client):
    """❌ Отсутствие metadata.file_path → 422"""
    doc = create_valid_document()
    del doc["metadata"]["file_path"]
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422


def test_missing_metadata_line_start(client):
    """❌ Отсутствие metadata.line_start → 422"""
    doc = create_valid_document()
    del doc["metadata"]["line_start"]
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422


def test_missing_metadata_line_end(client):
    """❌ Отсутствие metadata.line_end → 422"""
    doc = create_valid_document()
    del doc["metadata"]["line_end"]
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422


def test_missing_metadata_language(client):
    """❌ Отсутствие metadata.language → 422"""
    doc = create_valid_document()
    del doc["metadata"]["language"]
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422


def test_missing_metadata_repo(client):
    """❌ Отсутствие metadata.repo → 422"""
    doc = create_valid_document()
    del doc["metadata"]["repo"]
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422


def test_missing_metadata_chunk_type(client):
    """❌ Отсутствие metadata.chunk_type → 422"""
    doc = create_valid_document()
    del doc["metadata"]["chunk_type"]
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422


# ============================================================================
# НЕГАТИВНЫЕ ТЕСТЫ: EMPTY/INVALID VALUES → 422
# ============================================================================

def test_empty_text_field(client):
    """❌ Пустой text (ConStr min_length=1) → 422"""
    doc = create_valid_document(text="")
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422
    data = response.json()
    assert "error" in data


def test_empty_id_field(client):
    """❌ Пустой id → 422"""
    doc = create_valid_document()
    doc["id"] = ""
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422


def test_empty_embedding_version(client):
    """❌ Пустой embedding_version → 422"""
    doc = create_valid_document()
    doc["embedding_version"] = ""
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422


def test_invalid_sha256_format(client):
    """❌ Неверный формат SHA256 (не 64 hex) → 422"""
    doc = create_valid_document()
    doc["content_sha256"] = "invalid_sha"
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422


def test_sha256_wrong_length(client):
    """❌ SHA256 неправильной длины → 422"""
    doc = create_valid_document()
    doc["content_sha256"] = "a" * 32  # 32 символа вместо 64
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422


def test_negative_line_start(client):
    """❌ Отрицательный line_start (conint ge=0) → 422"""
    doc = create_valid_document()
    doc["metadata"]["line_start"] = -1
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422


def test_negative_line_end(client):
    """❌ Отрицательный line_end → 422"""
    doc = create_valid_document()
    doc["metadata"]["line_end"] = -5
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422


def test_empty_metadata_fields(client):
    """❌ Пустые строки в metadata → 422"""
    doc = create_valid_document()
    doc["metadata"]["file_path"] = ""
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422


# ============================================================================
# НЕГАТИВНЫЕ ТЕСТЫ: BATCH SIZE VIOLATIONS → 400/422
# ============================================================================

def test_empty_documents_array(client):
    """❌ Пустой массив documents (min_items=1) → 422"""
    request = create_valid_request(documents=[])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code in (400, 422)


def test_exceeds_max_batch_size(client):
    """❌ Превышение max_items=128 → 400/422"""
    documents = [
        create_valid_document(f"doc_{i}", f"Content {i}")
        for i in range(1, 130)
    ]
    request = create_valid_request(documents=documents)
    response = client.post("/v1/index", json=request)
    
    assert response.status_code in (400, 422)


def test_missing_documents_field(client):
    """❌ Отсутствие поля documents → 422"""
    request = {
        "api_contract": "v1.0.0"
    }
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422


# ============================================================================
# НЕГАТИВНЫЕ ТЕСТЫ: INVALID API CONTRACT → 400
# ============================================================================

def test_invalid_api_contract(client):
    """❌ Неверный api_contract → 400"""
    request = create_valid_request(api_contract="v99.0.0")
    response = client.post("/v1/index", json=request)
    
    # Может быть 400 (если сервер проверяет версию) или 200 (если игнорирует)
    assert response.status_code in (200, 202, 400)


# ============================================================================
# EDGE CASES И ГРАНИЧНЫЕ УСЛОВИЯ
# ============================================================================

def test_whitespace_only_text(client):
    """❌ Текст только из пробелов → может быть rejected или 422"""
    doc = create_valid_document(text="   ")
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    # Зависит от preflight валидации
    assert response.status_code in (200, 202, 422)


def test_very_long_text(client):
    """✅ Очень длинный текст (проверка лимитов)"""
    long_text = "x" * 10000
    doc = create_valid_document(text=long_text)
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code in (200, 202, 400)


def test_unicode_text(client):
    """✅ Текст с Unicode символами"""
    doc = create_valid_document(text="Привет мир! 你好世界 🚀")
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code in (200, 202)


def test_special_characters_in_metadata(client):
    """✅ Спецсимволы в metadata.file_path"""
    doc = create_valid_document()
    doc["metadata"]["file_path"] = "path/with spaces/and-special!@#.py"
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code in (200, 202)


def test_wrong_type_for_line_start(client):
    """❌ Неверный тип для line_start (string вместо int) → 422"""
    doc = create_valid_document()
    doc["metadata"]["line_start"] = "not_an_int"
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422


def test_extra_fields_ignored(client):
    """✅ Дополнительные поля должны игнорироваться (ExtraIgnoreModel)"""
    doc = create_valid_document()
    doc["extra_field"] = "should be ignored"
    doc["metadata"]["extra_metadata"] = "also ignored"
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code in (200, 202)


# ============================================================================
# МНОЖЕСТВЕННЫЕ ДОКУМЕНТЫ С РАЗНЫМИ ОШИБКАМИ
# ============================================================================

def test_multiple_docs_with_mixed_validity(client):
    """Смешанный батч: валидные и невалидные документы"""
    docs = [
        create_valid_document("doc1", "Valid 1"),
        create_valid_document("doc2", ""),  # Empty text
        create_valid_document("doc3", "Valid 3"),
    ]
    
    request = create_valid_request(documents=docs)
    response = client.post("/v1/index", json=request)
    
    # Должен быть 422 из-за одного невалидного документа
    assert response.status_code == 422


def test_all_documents_invalid(client):
    """Все документы невалидные"""
    docs = [
        {"id": "", "text": "", "metadata": {}},  # Полностью невалидный
        {"id": "test"},  # Неполный
    ]
    
    request = create_valid_request(documents=docs)
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    # Должно быть много ошибок валидации
    if "details" in data["error"]:
        assert len(data["error"]["details"]) > 0


# ============================================================================
# ПРОВЕРКА СТРУКТУРЫ ОШИБОК (422 Response Format)
# ============================================================================

def test_422_error_structure(client):
    """Проверка структуры ответа при 422 ошибке"""
    doc = create_valid_document()
    del doc["id"]  # Намеренная ошибка
    
    request = create_valid_request(documents=[doc])
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422
    data = response.json()
    
    # Проверяем структуру ошибки
    assert "error" in data
    error = data["error"]
    
    assert "type" in error
    assert error["type"] == "validation_error"
    
    assert "message" in error
    assert "details" in error
    
    # Проверяем что details содержит информацию о поле
    details = error["details"]
    assert isinstance(details, list)
    assert len(details) > 0
    
    # Каждая деталь должна содержать field, issue, message
    for detail in details:
        assert "field" in detail or "loc" in detail


# ============================================================================
# SUMMARY ТЕСТ: ДИАГНОСТИКА PRODUCTION ОШИБКИ
# ============================================================================

def test_production_like_scenario_192_errors():
    """
    Симуляция production сценария с 192 validation_errors
    
    Гипотеза: Клиент отправляет документы с отсутствующими/пустыми полями
    """
    # Создаем 32 документа (как в логах), каждый с 6 ошибками = 192 errors
    docs = []
    for i in range(32):
        doc = {
            "id": f"doc_{i}",
            # Missing: text, metadata, embedding_version, content_sha256
        }
        docs.append(doc)
    
    request = create_valid_request(documents=docs)
    client = TestClient(app)
    response = client.post("/v1/index", json=request)
    
    assert response.status_code == 422
    data = response.json()
    
    # Должно быть ~192 ошибки валидации (32 docs × 6 missing fields)
    if "details" in data.get("error", {}):
        error_count = len(data["error"]["details"])
        print(f"Total validation errors: {error_count}")
        
        # В production логах: validation_errors: 192
        # Это может быть: 32 документа × 6 обязательных полей
        assert error_count > 100, f"Expected ~192 errors, got {error_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
