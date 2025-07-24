"""
Модуль парсеров кода для различных языков программирования.
"""

from .base_parser import BaseParser
from .python_parser import PythonParser

__all__ = ["BaseParser", "PythonParser"]