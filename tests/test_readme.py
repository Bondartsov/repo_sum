# tests/test_readme.py
# Smoke-тест для README.md
import pytest
import os

def test_readme_exists_and_content():
    """
    Проверяет, что README.md существует и содержит ключевые слова.
    """
    assert os.path.exists('README.md')
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    assert 'OpenAI' in content or 'GPT' in content