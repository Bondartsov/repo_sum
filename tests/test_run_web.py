# tests/test_run_web.py
# Smoke-тест для run_web.py
import pytest
import importlib

def test_run_web_import():
    """
    Проверяет, что run_web.py импортируется без ошибок (основной smoke-тест).
    """
    importlib.import_module('run_web')