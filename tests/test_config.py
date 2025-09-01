# tests/test_config.py
# Тесты для модуля config.py
import pytest
import os
from config import get_config, reload_config, Config

# Пример теста для get_config

@pytest.mark.integration
def test_get_config_default():
    """
    Проверяет, что get_config возвращает объект Config с нужными атрибутами.
    """
    config = get_config()
    assert isinstance(config, Config)
    assert hasattr(config, "openai")
    assert hasattr(config.openai, "api_key_env_var")

# Добавьте тесты для reload_config и других функций
