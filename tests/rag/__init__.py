"""
Модуль тестов RAG системы.

Содержит:
- test_rag_integration.py - интеграционные тесты компонентов
- test_rag_e2e_cli.py - End-to-End тесты CLI команд
- test_rag_performance.py - тесты производительности
- conftest.py - общие фикстуры

Для запуска всех RAG тестов:
    pytest tests/rag/ -v

Для запуска определённых типов тестов:
    pytest tests/rag/ -m "unit"                    # только unit тесты  
    pytest tests/rag/ -m "integration_rag"         # интеграционные тесты
    pytest tests/rag/ -m "e2e_rag"                 # E2E тесты
    pytest tests/rag/ -m "perf_rag"                # тесты производительности
    pytest tests/rag/ -m "not slow"                # исключить медленные тесты
    pytest tests/rag/ -m "not stress"              # исключить стресс-тесты

Для тестов с реальными компонентами (требует запущенный Qdrant):
    pytest tests/rag/ -m "real"
"""

__version__ = "1.0.0"
__author__ = "RAG System Testing Suite"

# Экспортируем основные утилиты для тестирования
from .conftest import (
    assert_valid_search_result,
    assert_performance_within_limits
)

__all__ = [
    'assert_valid_search_result',
    'assert_performance_within_limits'
]