"""
Скрипт для запуска тестов RAG системы.

Предоставляет различные режимы запуска тестов:
- Быстрые unit тесты
- Интеграционные тесты  
- E2E тесты
- Тесты производительности
- Полный набор тестов
"""

import sys
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any


class RAGTestRunner:
    """Запускальщик тестов RAG системы"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.results = []
    
    def run_command(self, cmd: List[str], description: str) -> Dict[str, Any]:
        """Выполняет команду и возвращает результат"""
        print(f"\n{'='*60}")
        print(f"🧪 {description}")
        print(f"{'='*60}")
        print(f"Команда: {' '.join(cmd)}")
        print()
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.base_path.parent.parent,  # Корень проекта
                timeout=300  # 5 минут максимум
            )
            
            duration = time.time() - start_time
            
            success = result.returncode == 0
            
            # Выводим результат
            if success:
                print(f"✅ УСПЕХ за {duration:.2f}s")
            else:
                print(f"❌ ОШИБКА за {duration:.2f}s")
                print(f"Return code: {result.returncode}")
            
            if result.stdout:
                print(f"\n📄 STDOUT:")
                print(result.stdout)
            
            if result.stderr:
                print(f"\n⚠️ STDERR:")
                print(result.stderr)
            
            test_result = {
                'description': description,
                'command': ' '.join(cmd),
                'success': success,
                'duration': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            self.results.append(test_result)
            return test_result
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏰ TIMEOUT после {duration:.2f}s")
            
            test_result = {
                'description': description,
                'command': ' '.join(cmd),
                'success': False,
                'duration': duration,
                'return_code': -1,
                'stdout': '',
                'stderr': 'Timeout expired'
            }
            
            self.results.append(test_result)
            return test_result
        
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 ИСКЛЮЧЕНИЕ: {e}")
            
            test_result = {
                'description': description,
                'command': ' '.join(cmd),
                'success': False,
                'duration': duration,
                'return_code': -2,
                'stdout': '',
                'stderr': str(e)
            }
            
            self.results.append(test_result)
            return test_result
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Запускает быстрые unit тесты"""
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/rag/',
            '-v',
            '-m', 'unit and not slow',
            '--tb=short'
        ]
        return self.run_command(cmd, "Unit тесты RAG компонентов")
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Запускает интеграционные тесты"""
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/rag/test_rag_integration.py',
            '-v',
            '--tb=short'
        ]
        return self.run_command(cmd, "Интеграционные тесты RAG")
    
    def run_e2e_tests(self) -> Dict[str, Any]:
        """Запускает E2E тесты CLI"""
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/rag/test_rag_e2e_cli.py',
            '-v',
            '-m', 'not slow',
            '--tb=short'
        ]
        return self.run_command(cmd, "End-to-End тесты CLI")
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Запускает тесты производительности"""
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/rag/test_rag_performance.py',
            '-v',
            '-m', 'not stress and not benchmark',
            '--tb=short'
        ]
        return self.run_command(cmd, "Тесты производительности RAG")
    
    def run_stress_tests(self) -> Dict[str, Any]:
        """Запускает стресс-тесты"""
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/rag/test_rag_performance.py',
            '-v',
            '-m', 'stress',
            '--tb=short'
        ]
        return self.run_command(cmd, "Стресс-тесты RAG")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Запускает все RAG тесты"""
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/rag/',
            '-v',
            '--tb=short'
        ]
        return self.run_command(cmd, "Все тесты RAG системы")
    
    def run_smoke_tests(self) -> Dict[str, Any]:
        """Запускает дымовые тесты для быстрой проверки"""
        cmd = [
            sys.executable, '-m', 'pytest',
            'tests/rag/',
            '-v',
            '-m', 'smoke or (unit and not slow)',
            '--tb=line',
            '-x'  # Останавливаемся на первой ошибке
        ]
        return self.run_command(cmd, "Дымовые тесты RAG")
    
    def print_summary(self):
        """Выводит итоговую сводку по всем тестам"""
        print(f"\n{'='*80}")
        print(f"📊 ИТОГОВАЯ СВОДКА ТЕСТИРОВАНИЯ RAG СИСТЕМЫ")
        print(f"{'='*80}")
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r['success'])
        failed_tests = total_tests - successful_tests
        total_duration = sum(r['duration'] for r in self.results)
        
        print(f"Всего тестовых наборов: {total_tests}")
        print(f"Успешных: {successful_tests} ✅")
        print(f"Неудачных: {failed_tests} ❌")
        print(f"Общее время: {total_duration:.2f}s")
        print()
        
        # Детальная информация по каждому набору
        for i, result in enumerate(self.results, 1):
            status = "✅ УСПЕХ" if result['success'] else "❌ ОШИБКА"
            print(f"{i}. {result['description']}: {status} ({result['duration']:.2f}s)")
            
            if not result['success'] and result['stderr']:
                # Показываем первые строки ошибки
                error_lines = result['stderr'].split('\n')[:3]
                for line in error_lines:
                    if line.strip():
                        print(f"   💥 {line.strip()}")
        
        print(f"\n{'='*80}")
        
        if failed_tests == 0:
            print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
            return True
        else:
            print(f"⚠️ {failed_tests} НАБОРОВ ТЕСТОВ ЗАВЕРШИЛИСЬ С ОШИБКАМИ")
            return False


def main():
    """Главная функция скрипта"""
    parser = argparse.ArgumentParser(description="Запуск тестов RAG системы")
    parser.add_argument(
        'test_type',
        nargs='?',
        default='smoke',
        choices=['unit', 'integration', 'e2e', 'performance', 'stress', 'all', 'smoke'],
        help='Тип тестов для запуска'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Подробный вывод'
    )
    
    args = parser.parse_args()
    
    runner = RAGTestRunner()
    
    print("🚀 Запуск тестов RAG системы...")
    print(f"Тип тестов: {args.test_type}")
    print(f"Базовая директория: {runner.base_path}")
    
    # Выполняем тесты в зависимости от типа
    if args.test_type == 'unit':
        runner.run_unit_tests()
    elif args.test_type == 'integration':
        runner.run_integration_tests()
    elif args.test_type == 'e2e':
        runner.run_e2e_tests()
    elif args.test_type == 'performance':
        runner.run_performance_tests()
    elif args.test_type == 'stress':
        runner.run_stress_tests()
    elif args.test_type == 'smoke':
        runner.run_smoke_tests()
    elif args.test_type == 'all':
        # Запускаем все типы тестов последовательно
        runner.run_unit_tests()
        runner.run_integration_tests()
        runner.run_e2e_tests()
        runner.run_performance_tests()
    
    # Выводим итоговую сводку
    success = runner.print_summary()
    
    # Возвращаем соответствующий exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()