#!/usr/bin/env python3
"""
Debug скрипт для диагностики 422 ошибок при индексации

Выполняет preflight проверки payload перед отправкой на VM,
показывая точные причины потенциального отклонения.

Usage:
    python scripts/debug_422_validation.py

Или программно:
    from scripts.debug_422_validation import validate_documents
    issues = validate_documents(documents)
"""

import sys
import hashlib
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_sha256(text: str) -> str:
    """Вычисляет SHA256 для текста."""
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_metadata(meta: Dict[str, Any], doc_index: int) -> List[str]:
    """
    Валидирует metadata согласно серверному контракту IndexedMetadata.
    
    Обязательные поля:
    - file_path: str (min_length=1)
    - line_start: int (ge=0)
    - line_end: int (ge=0)
    - language: str (min_length=1)
    - repo: str (min_length=1)
    - chunk_type: str (min_length=1)
    
    Returns:
        Список ошибок валидации
    """
    issues = []
    
    # Проверка обязательных полей
    required_fields = {
        'file_path': 'str',
        'line_start': 'int',
        'line_end': 'int',
        'language': 'str',
        'repo': 'str',
        'chunk_type': 'str'
    }
    
    for field, field_type in required_fields.items():
        if field not in meta:
            issues.append(f"[Doc {doc_index}] Отсутствует обязательное поле metadata.{field}")
            continue
            
        value = meta[field]
        
        # Проверка типов
        if field_type == 'str':
            if not isinstance(value, str):
                issues.append(f"[Doc {doc_index}] metadata.{field} должно быть строкой, получено {type(value).__name__}")
            elif len(value) == 0:
                issues.append(f"[Doc {doc_index}] metadata.{field} не может быть пустой строкой")
                
        elif field_type == 'int':
            if not isinstance(value, int):
                issues.append(f"[Doc {doc_index}] metadata.{field} должно быть int, получено {type(value).__name__}")
            elif value < 0:
                issues.append(f"[Doc {doc_index}] metadata.{field}={value} должно быть >= 0")
    
    # Предупреждения о синонимах (start_line/end_line вместо line_start/line_end)
    if 'start_line' in meta and 'line_start' not in meta:
        issues.append(f"[Doc {doc_index}] ⚠️ Найдено 'start_line' вместо 'line_start' - будет автоматически переименовано")
    
    if 'end_line' in meta and 'line_end' not in meta:
        issues.append(f"[Doc {doc_index}] ⚠️ Найдено 'end_line' вместо 'line_end' - будет автоматически переименовано")
    
    return issues


def validate_document(doc: Dict[str, Any], doc_index: int) -> Tuple[bool, List[str]]:
    """
    Валидирует один документ согласно серверному контракту IndexedDocument.
    
    Обязательные поля:
    - id: str (min_length=1)
    - text: str (min_length=1)
    - metadata: IndexedMetadata
    - embedding_version: str (min_length=1)
    - content_sha256: str (regex=r'^[A-Fa-f0-9]{64}$')
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    # 1. Проверка 'id'
    doc_id = doc.get('id')
    if not doc_id:
        issues.append(f"[Doc {doc_index}] Отсутствует обязательное поле 'id'")
    elif not isinstance(doc_id, str):
        issues.append(f"[Doc {doc_index}] 'id' должно быть строкой, получено {type(doc_id).__name__}")
    elif len(doc_id) == 0:
        issues.append(f"[Doc {doc_index}] 'id' не может быть пустой строкой")
    
    # 2. Проверка 'text' (КРИТИЧНО!)
    text = doc.get('text')
    if text is None:
        issues.append(f"[Doc {doc_index}] ❌ КРИТИЧНО: Отсутствует обязательное поле 'text'")
    elif not isinstance(text, str):
        issues.append(f"[Doc {doc_index}] ❌ КРИТИЧНО: 'text' должно быть строкой, получено {type(text).__name__}")
    elif len(text.strip()) == 0:
        issues.append(f"[Doc {doc_index}] ❌ КРИТИЧНО: 'text' не может быть пустым (причина отклонения: empty_text)")
    
    # 3. Проверка 'metadata'
    metadata = doc.get('metadata')
    if not metadata:
        issues.append(f"[Doc {doc_index}] Отсутствует обязательное поле 'metadata'")
    elif not isinstance(metadata, dict):
        issues.append(f"[Doc {doc_index}] 'metadata' должно быть объектом, получено {type(metadata).__name__}")
    else:
        # Детальная валидация metadata
        meta_issues = validate_metadata(metadata, doc_index)
        issues.extend(meta_issues)
    
    # 4. Проверка 'embedding_version'
    emb_ver = doc.get('embedding_version')
    if not emb_ver:
        issues.append(f"[Doc {doc_index}] Отсутствует обязательное поле 'embedding_version'")
    elif not isinstance(emb_ver, str):
        issues.append(f"[Doc {doc_index}] 'embedding_version' должно быть строкой")
    elif len(emb_ver) == 0:
        issues.append(f"[Doc {doc_index}] 'embedding_version' не может быть пустым")
    
    # 5. Проверка 'content_sha256'
    sha256 = doc.get('content_sha256')
    if not sha256:
        issues.append(f"[Doc {doc_index}] Отсутствует обязательное поле 'content_sha256'")
    elif not isinstance(sha256, str):
        issues.append(f"[Doc {doc_index}] 'content_sha256' должно быть строкой")
    elif len(sha256) != 64:
        issues.append(f"[Doc {doc_index}] 'content_sha256' должно быть 64 символа, получено {len(sha256)}")
    elif not all(c in '0123456789ABCDEFabcdef' for c in sha256):
        issues.append(f"[Doc {doc_index}] 'content_sha256' должно быть hex строкой (0-9, A-F)")
    else:
        # Проверка соответствия SHA256 с текстом
        if text is not None and isinstance(text, str):
            computed_sha = compute_sha256(text)
            if sha256.lower() != computed_sha.lower():
                issues.append(
                    f"[Doc {doc_index}] ❌ КРИТИЧНО: 'content_sha256' не совпадает с хешем 'text' "
                    f"(причина отклонения: sha256_mismatch)\n"
                    f"  Ожидалось: {computed_sha}\n"
                    f"  Получено:  {sha256}"
                )
    
    is_valid = len(issues) == 0
    return is_valid, issues


def validate_documents(documents: List[Dict[str, Any]], verbose: bool = True) -> Dict[str, Any]:
    """
    Валидирует список документов перед отправкой на сервер.
    
    Args:
        documents: Список документов для проверки
        verbose: Выводить детальную информацию
        
    Returns:
        Словарь с результатами валидации
    """
    total = len(documents)
    valid = 0
    invalid = 0
    all_issues = []
    
    print(f"\n{'='*80}")
    print(f"🔍 PREFLIGHT ВАЛИДАЦИЯ: {total} документов")
    print(f"{'='*80}\n")
    
    for i, doc in enumerate(documents):
        is_valid, issues = validate_document(doc, i)
        
        if is_valid:
            valid += 1
            if verbose:
                doc_id = doc.get('id', 'unknown')
                text_len = len(doc.get('text', ''))
                print(f"✅ [Doc {i}] id={doc_id}, text_len={text_len} - ВАЛИДЕН")
        else:
            invalid += 1
            all_issues.extend(issues)
            
            if verbose:
                doc_id = doc.get('id', 'unknown')
                print(f"\n❌ [Doc {i}] id={doc_id} - НЕВАЛИДЕН:")
                for issue in issues:
                    if issue.startswith(f"[Doc {i}]"):
                        print(f"   {issue}")
    
    # Итоговая статистика
    print(f"\n{'='*80}")
    print("📊 РЕЗУЛЬТАТЫ ВАЛИДАЦИИ:")
    print(f"{'='*80}")
    print(f"  Всего документов: {total}")
    print(f"  ✅ Валидных:      {valid} ({100*valid/max(1,total):.1f}%)")
    print(f"  ❌ Невалидных:    {invalid} ({100*invalid/max(1,total):.1f}%)")
    print(f"  🔧 Всего проблем: {len(all_issues)}")
    
    if invalid > 0:
        print(f"\n⚠️  ВНИМАНИЕ: {invalid} документов будут отклонены сервером с 422 ошибкой!")
        print("   Исправьте ошибки перед отправкой на индексацию.\n")
    else:
        print("\n✅ Все документы валидны! Можно отправлять на индексацию.\n")
    
    return {
        'total': total,
        'valid': valid,
        'invalid': invalid,
        'issues': all_issues,
        'success_rate': valid / max(1, total)
    }


def main():
    """Пример использования debug скрипта."""
    # Пример 1: Невалидный документ (пустой text)
    invalid_doc = {
        'id': 'doc1',
        'text': '',  # ❌ Пустой text
        'metadata': {
            'file_path': 'test.py',
            'line_start': 0,
            'line_end': 10,
            'language': 'python',
            'repo': 'test',
            'chunk_type': 'code'
        },
        'embedding_version': '2025-10-A',
        'content_sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    }
    
    # Пример 2: Невалидный документ (неправильный SHA256)
    invalid_sha = {
        'id': 'doc2',
        'text': 'def hello(): pass',
        'metadata': {
            'file_path': 'test.py',
            'line_start': 0,
            'line_end': 1,
            'language': 'python',
            'repo': 'test',
            'chunk_type': 'code'
        },
        'embedding_version': '2025-10-A',
        'content_sha256': 'wrong_hash_1234567890abcdef' + '0' * 38  # ❌ Неправильный хеш
    }
    
    # Пример 3: Валидный документ
    valid_doc = {
        'id': 'doc3',
        'text': 'def hello(): pass',
        'metadata': {
            'file_path': 'test.py',
            'line_start': 0,
            'line_end': 1,
            'language': 'python',
            'repo': 'test',
            'chunk_type': 'code'
        },
        'embedding_version': '2025-10-A',
        'content_sha256': compute_sha256('def hello(): pass')  # ✅ Правильный хеш
    }
    
    # Пример 4: Документ с устаревшими именами полей
    legacy_doc = {
        'id': 'doc4',
        'text': 'print("hello")',
        'metadata': {
            'file_path': 'test.py',
            'start_line': 5,  # ⚠️ Должно быть line_start
            'end_line': 6,    # ⚠️ Должно быть line_end
            'language': 'python',
            'repo': 'test',
            'chunk_type': 'code'
        },
        'embedding_version': '2025-10-A',
        'content_sha256': compute_sha256('print("hello")')
    }
    
    documents = [invalid_doc, invalid_sha, valid_doc, legacy_doc]
    
    result = validate_documents(documents, verbose=True)
    
    # Возвращаем exit code: 0 если все валидно, 1 если есть ошибки
    sys.exit(0 if result['invalid'] == 0 else 1)


if __name__ == '__main__':
    main()
