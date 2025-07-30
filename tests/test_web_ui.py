# tests/test_web_ui.py
# Smoke-тест для web_ui.py (Streamlit-интерфейс)
import pytest
import importlib

# Проверяем, что модуль web_ui импортируется без ошибок

def test_web_ui_import():
    """
    Проверяет, что web_ui.py импортируется без ошибок (основной smoke-тест).
    """
    importlib.import_module('web_ui')