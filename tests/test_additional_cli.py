#!/usr/bin/env python3
"""
Дополнительные тесты CLI интерфейса для проекта repo_sum.
Тесты T-001 и T-002 согласно техническому заданию.
"""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.functional
class TestAdditionalCLI:
    """Дополнительные тесты CLI интерфейса"""
    
    @pytest.fixture
    def main_script_path(self):
        """Путь к основному скрипту main.py"""
        return Path(__file__).parent.parent / "main.py"
    
    def test_t001_unknown_subcommand(self, main_script_path):
        """
        T-001 - CLI: неизвестная подкоманда
        
        Выполнить команду: `python main.py do-nothing`
        Ожидается: ненулевой код выхода + понятное сообщение об ошибке + 
                  краткая подсказка по доступным командам
        Не должно быть трассировки исключения
        """
        # Выполняем команду с неизвестной подкомандой
        result = subprocess.run(
            [sys.executable, str(main_script_path), "do-nothing"],
            capture_output=True,
            text=True
        )
        
        # Проверяем ненулевой код выхода
        assert result.returncode != 0, (
            f"Ожидался ненулевой код выхода для неизвестной команды, "
            f"получен: {result.returncode}"
        )
        
        # Объединяем stdout и stderr для анализа
        output = result.stdout + result.stderr
        output_lower = output.lower()
        
        # Проверяем, что есть сообщение об ошибке
        error_indicators = [
            "error", "ошибка", "invalid", "неверная", 
            "unknown", "неизвестная", "not found", "не найдена"
        ]
        has_error_message = any(indicator in output_lower for indicator in error_indicators)
        assert has_error_message, (
            f"Не найдено понятного сообщения об ошибке в выводе: {output}"
        )
        
        # Проверяем, что есть подсказка о доступных командах
        help_indicators = [
            "usage", "использование", "commands", "команды", 
            "help", "помощь", "--help", "analyze", "stats"
        ]
        has_help_info = any(indicator in output_lower for indicator in help_indicators)
        assert has_help_info, (
            f"Не найдена подсказка о доступных командах в выводе: {output}"
        )
        
        # Проверяем, что НЕТ трассировки исключения
        traceback_indicators = [
            "traceback", "трассировка", "file \"", "line ", 
            "exception", "исключение", ".py\", line"
        ]
        has_traceback = any(indicator in output_lower for indicator in traceback_indicators)
        assert not has_traceback, (
            f"Найдена трассировка исключения в выводе (не должна быть): {output}"
        )
        
        print(f"T-001 PASSED: Неизвестная команда корректно обработана")
        print(f"Код выхода: {result.returncode}")
        print(f"Вывод: {output[:200]}...")  # Первые 200 символов для отладки
    
    def test_t002_conflicting_flags(self, main_script_path):
        """
        T-002 - CLI: взаимоисключающие флаги
        
        Выполнить команду: `python main.py --generate-docs --run-web`
        Ожидается: отклонение конфликтующих флагов + понятное сообщение о конфликте + 
                  корректный код выхода
        """
        # Выполняем команду с конфликтующими флагами
        result = subprocess.run(
            [sys.executable, str(main_script_path), "--generate-docs", "--run-web"],
            capture_output=True,
            text=True
        )
        
        # Проверяем ненулевой код выхода
        assert result.returncode != 0, (
            f"Ожидался ненулевой код выхода для конфликтующих флагов, "
            f"получен: {result.returncode}"
        )
        
        # Объединяем stdout и stderr для анализа
        output = result.stdout + result.stderr
        output_lower = output.lower()
        
        # Проверяем, что есть сообщение о конфликте или ошибке
        conflict_indicators = [
            "error", "ошибка", "conflict", "конфликт", "invalid", "неверный",
            "mutually exclusive", "взаимоисключающие", "cannot", "нельзя",
            "not allowed", "недопустимо", "incompatible", "несовместимые"
        ]
        has_conflict_message = any(indicator in output_lower for indicator in conflict_indicators)
        assert has_conflict_message, (
            f"Не найдено сообщения о конфликте флагов в выводе: {output}"
        )
        
        # Проверяем, что упоминаются проблемные флаги
        flag_indicators = [
            "generate-docs", "run-web", "--generate", "--run"
        ]
        mentions_flags = any(indicator in output_lower for indicator in flag_indicators)
        
        # Альтернативно, может быть общее сообщение об ошибке опций
        option_indicators = [
            "option", "опция", "flag", "флаг", "argument", "аргумент"
        ]
        mentions_options = any(indicator in output_lower for indicator in option_indicators)
        
        assert mentions_flags or mentions_options, (
            f"Не найдено упоминания о проблемных флагах или опциях в выводе: {output}"
        )
        
        print(f"T-002 PASSED: Конфликтующие флаги корректно отклонены")
        print(f"Код выхода: {result.returncode}")
        print(f"Вывод: {output[:200]}...")  # Первые 200 символов для отладки


if __name__ == "__main__":
    # Запуск тестов напрямую для отладки
    pytest.main([__file__, "-v"])
