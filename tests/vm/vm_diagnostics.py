"""
VM Diagnostics - Комплексный диагностический инструмент для RAG-as-a-Service VM.

Этот скрипт объединяет все диагностические тесты в один удобный инструмент
для быстрого выявления и исправления проблем с VM подключением.

Автор: Claude (Cline) для диагностики VM проблем
Дата: 19 сентября 2025
Использование: python vm_diagnostics.py [--quick] [--fix] [--monitor]
"""

import sys
import os
import argparse
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# Добавляем путь к тестам для импорта
sys.path.insert(0, os.path.dirname(__file__))

# Импортируем наши диагностические классы
try:
    from test_vm_connectivity import VMConnectivityTester
    from test_vm_health_monitoring import VMHealthMonitor, VMHealthAlerts, HealthCheckResult
    from test_vm_firewall_config import VMFirewallTester, get_firewall_quick_fix_commands
except ImportError as e:
    print(f"❌ Ошибка импорта диагностических модулей: {e}")
    print("Убедитесь что находитесь в директории tests/vm/")
    sys.exit(1)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vm_diagnostics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VMDiagnosticSuite:
    """Комплексный набор диагностических инструментов для VM"""
    
    def __init__(self, vm_host: str = "10.61.11.54", vm_port: int = 8000):
        self.vm_host = vm_host
        self.vm_port = vm_port
        
        # Инициализируем диагностические инструменты
        self.connectivity_tester = VMConnectivityTester(vm_host, vm_port)
        self.health_monitor = VMHealthMonitor(vm_host, vm_port, check_interval=10)
        self.firewall_tester = VMFirewallTester(vm_host, vm_port)
        
        # Результаты диагностики
        self.diagnostic_results = {}
        self.critical_issues = []
        self.recommendations = []
    
    def print_banner(self):
        """Вывод заголовка диагностики"""
        print("╭─────────────────────────────────────────────────────╮")
        print("│  🏥 VM RAG-as-a-Service Diagnostic Suite v1.0      │")
        print("│                                                     │")
        print(f"│  Target VM: {self.vm_host}:{self.vm_port}             │")
        print(f"│  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}        │")
        print("╰─────────────────────────────────────────────────────╯")
        print()
    
    def run_quick_diagnostics(self) -> Dict[str, Any]:
        """Быстрая диагностика основных проблем (1-2 минуты)"""
        print("🚀 Запуск быстрой диагностики VM...")
        print("=" * 55)
        
        quick_results = {
            "test_suite": "quick_diagnostics",
            "vm_host": self.vm_host,
            "vm_port": self.vm_port,
            "start_time": datetime.now().isoformat(),
            "tests": {},
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "critical_issues": [],
                "quick_recommendations": []
            }
        }
        
        # 1. Тест базового TCP подключения
        print("1️⃣ Проверка TCP подключения...")
        tcp_result = self.connectivity_tester.test_basic_tcp_connection()
        quick_results["tests"]["tcp_connection"] = tcp_result
        self._process_test_result("TCP Connection", tcp_result, quick_results)
        
        # 2. Тест HTTP health endpoint
        print("2️⃣ Проверка HTTP health endpoint...")
        http_result = self.connectivity_tester.test_http_health_endpoint()
        quick_results["tests"]["http_health"] = http_result
        self._process_test_result("HTTP Health", http_result, quick_results)
        
        # 3. Быстрый ping тест
        print("3️⃣ Проверка ping связности...")
        ping_result = self.connectivity_tester.test_ping_connectivity()
        quick_results["tests"]["ping"] = ping_result
        self._process_test_result("Ping Test", ping_result, quick_results)
        
        # 4. Тест внешнего подключения
        print("4️⃣ Проверка внешнего подключения...")
        external_result = self.firewall_tester.test_external_connectivity()
        # Конвертируем FirewallTestResult в словарь
        external_dict = {
            "success": external_result.success,
            "error": external_result.error,
            "test_name": external_result.test_name,
            "details": external_result.details,
            "recommendations": external_result.recommendations
        }
        quick_results["tests"]["external_connectivity"] = external_dict
        self._process_test_result("External Connectivity", external_dict, quick_results)
        
        quick_results["end_time"] = datetime.now().isoformat()
        
        # Выводим результаты
        self._print_quick_results(quick_results)
        
        return quick_results
    
    def run_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """Полная диагностика всех компонентов (5-10 минут)"""
        print("🔬 Запуск комплексной диагностики VM...")
        print("=" * 55)
        
        comprehensive_results = {
            "test_suite": "comprehensive_diagnostics", 
            "vm_host": self.vm_host,
            "vm_port": self.vm_port,
            "start_time": datetime.now().isoformat(),
            "test_categories": {},
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "critical_issues": [],
                "all_recommendations": []
            }
        }
        
        # 1. Полное тестирование подключения
        print("📡 Тестирование сетевого подключения...")
        connectivity_results = self.connectivity_tester.run_all_connectivity_tests()
        comprehensive_results["test_categories"]["connectivity"] = connectivity_results
        
        # 2. Детальная проверка здоровья
        print("\n🏥 Проверка здоровья сервиса...")
        health_result = None
        try:
            import asyncio
            health_result = asyncio.run(self.health_monitor.check_health_detailed())
            comprehensive_results["test_categories"]["health"] = {
                "timestamp": health_result.timestamp,
                "success": health_result.success,
                "response_time_ms": health_result.response_time_ms,
                "service_status": health_result.service_status,
                "embedder_status": health_result.embedder_status,
                "vector_store_status": health_result.vector_store_status,
                "error": health_result.error
            }
            
            status = "✅ HEALTHY" if health_result.success else "❌ UNHEALTHY"
            print(f"  {status} Service Health Check ({health_result.response_time_ms:.1f}ms)")
            
        except Exception as e:
            print(f"  ❌ Health check failed: {e}")
            comprehensive_results["test_categories"]["health"] = {"error": str(e)}
        
        # 3. Тестирование firewall
        print("\n🔥 Тестирование firewall конфигурации...")
        firewall_results = self.firewall_tester.run_comprehensive_firewall_test()
        comprehensive_results["test_categories"]["firewall"] = firewall_results
        
        comprehensive_results["end_time"] = datetime.now().isoformat()
        
        # Анализируем результаты
        self._analyze_comprehensive_results(comprehensive_results)
        
        return comprehensive_results
    
    def start_monitoring_mode(self, duration_minutes: int = 10):
        """Режим мониторинга VM на определенное время"""
        print(f"📊 Запуск мониторинга VM на {duration_minutes} минут...")
        print("=" * 55)
        
        alerts = VMHealthAlerts(alert_threshold_failures=2)
        
        def monitoring_callback(result: HealthCheckResult):
            """Callback для обработки результатов мониторинга"""
            status = "✅ HEALTHY" if result.success else "❌ UNHEALTHY"
            timestamp = datetime.fromisoformat(result.timestamp).strftime("%H:%M:%S")
            
            print(f"[{timestamp}] {status} {result.response_time_ms:.1f}ms", end="")
            
            if result.service_status:
                print(f" | Service: {result.service_status}", end="")
            if result.embedder_status:
                print(f" | Embedder: {result.embedder_status}", end="")
            
            print()  # Новая строка
            
            # Проверяем алерты
            alert_messages = alerts.check_for_alerts(result)
            for alert in alert_messages:
                print(f"🚨 {alert}")
        
        try:
            # Запускаем мониторинг
            self.health_monitor.start_monitoring(callback=monitoring_callback)
            
            print("Мониторинг активен. Нажмите Ctrl+C для остановки...")
            time.sleep(duration_minutes * 60)
            
        except KeyboardInterrupt:
            print("\n⏹️ Мониторинг остановлен пользователем")
            
        finally:
            self.health_monitor.stop_monitoring()
            
            # Показываем финальный отчет
            report = self.health_monitor.get_monitoring_report()
            self._print_monitoring_report(report)
    
    def generate_fix_suggestions(self, diagnostic_results: Dict[str, Any]) -> List[str]:
        """Генерация предложений по исправлению проблем"""
        fix_suggestions = []
        
        # Анализируем результаты и предлагаем исправления
        if "connectivity" in diagnostic_results.get("test_categories", {}):
            connectivity = diagnostic_results["test_categories"]["connectivity"]
            if connectivity["summary"]["success_rate"] < 50:
                fix_suggestions.extend([
                    "🔧 Критические проблемы с подключением:",
                    "1. Проверьте что VM запущена: ping " + self.vm_host,
                    "2. Проверьте статус RAG сервиса на VM",
                    "3. Проверьте firewall правила (см. firewall тесты)"
                ])
        
        if "firewall" in diagnostic_results.get("test_categories", {}):
            firewall = diagnostic_results["test_categories"]["firewall"]
            if firewall["summary"]["failed_tests"] > 0:
                fix_suggestions.extend([
                    "🔥 Проблемы с firewall:",
                    *get_firewall_quick_fix_commands(self.vm_host, self.vm_port)
                ])
        
        if "health" in diagnostic_results.get("test_categories", {}):
            health = diagnostic_results["test_categories"]["health"]
            if health.get("error"):
                fix_suggestions.extend([
                    "🏥 Проблемы со здоровьем сервиса:",
                    "1. Перезапустить RAG сервис на VM",
                    "2. Проверить логи сервиса",
                    "3. Проверить доступность Qdrant на VM"
                ])
        
        return fix_suggestions
    
    def export_diagnostic_report(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Экспорт результатов диагностики в JSON файл"""
        if not filename:
            timestamp = int(time.time())
            test_suite = results.get("test_suite", "diagnostic")
            filename = f"vm_{test_suite}_report_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Отчет сохранен в {filename}")
        return filename
    
    def _process_test_result(self, test_name: str, result: Dict[str, Any], 
                           quick_results: Dict[str, Any]):
        """Обработка результата теста для быстрой диагностики"""
        success = result.get("success", False)
        error = result.get("error")
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
        
        if error:
            print(f"     Error: {error}")
            quick_results["summary"]["critical_issues"].append(f"{test_name}: {error}")
        
        # Обновляем статистику
        quick_results["summary"]["total_tests"] += 1
        if success:
            quick_results["summary"]["passed_tests"] += 1
        else:
            quick_results["summary"]["failed_tests"] += 1
    
    def _print_quick_results(self, results: Dict[str, Any]):
        """Вывод результатов быстрой диагностики"""
        summary = results["summary"]
        
        print("\n" + "=" * 55)
        print("📊 РЕЗУЛЬТАТЫ БЫСТРОЙ ДИАГНОСТИКИ")
        print("=" * 55)
        
        total = summary["total_tests"]
        passed = summary["passed_tests"]
        failed = summary["failed_tests"]
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"Всего тестов: {total}")
        print(f"Успешно: {passed} ✅")
        print(f"Неудачно: {failed} ❌")
        print(f"Успешность: {success_rate:.1f}%")
        
        if summary["critical_issues"]:
            print("\n🚨 Критические проблемы:")
            for issue in summary["critical_issues"]:
                print(f"  • {issue}")
        
        # Общие рекомендации
        if failed > 0:
            print("\n💡 Быстрые исправления:")
            if failed == total:
                print("  • VM может быть недоступна - проверьте ping")
                print("  • RAG сервис может быть остановлен")
                print("  • Firewall может блокировать подключения")
            else:
                print("  • Запустите полную диагностику: --comprehensive")
                print("  • Проверьте firewall: --firewall-only")
    
    def _analyze_comprehensive_results(self, results: Dict[str, Any]):
        """Анализ результатов комплексной диагностики"""
        print("\n" + "=" * 55)
        print("📊 РЕЗУЛЬТАТЫ КОМПЛЕКСНОЙ ДИАГНОСТИКИ")
        print("=" * 55)
        
        categories = results["test_categories"]
        
        # Анализируем каждую категорию
        for category, data in categories.items():
            if category == "connectivity":
                success_rate = data["summary"]["success_rate"]
                print(f"📡 Подключение: {success_rate:.1f}% ({data['summary']['passed_tests']}/{data['summary']['total_tests']})")
                
            elif category == "health":
                if data.get("success"):
                    status = data.get("service_status", "unknown")
                    response_time = data.get("response_time_ms", 0)
                    print(f"🏥 Здоровье сервиса: {status} ({response_time:.1f}ms)")
                else:
                    print(f"🏥 Здоровье сервиса: ERROR - {data.get('error', 'unknown error')}")
                
            elif category == "firewall":
                passed = data["summary"]["passed_tests"]
                total = data["summary"]["total_tests"]
                print(f"🔥 Firewall: {passed}/{total} тестов прошло")
        
        # Генерируем рекомендации
        fix_suggestions = self.generate_fix_suggestions(results)
        if fix_suggestions:
            print("\n💡 Рекомендации по исправлению:")
            for suggestion in fix_suggestions[:10]:  # Ограничиваем количество
                print(f"  {suggestion}")
    
    def _print_monitoring_report(self, report: Dict[str, Any]):
        """Вывод отчета мониторинга"""
        print("\n" + "=" * 55)
        print("📊 ОТЧЕТ МОНИТОРИНГА")
        print("=" * 55)
        
        metrics = report["metrics"]
        
        print(f"Продолжительность: {report.get('monitoring_duration_minutes', 0):.1f} минут")
        print(f"Всего проверок: {metrics['total_checks']}")
        print(f"Успешность: {metrics['success_rate']:.1f}%")
        print(f"Среднее время отклика: {metrics['avg_response_time_ms']:.1f}ms")
        print(f"Тренд производительности: {metrics['trend_direction']}")
        
        if metrics['last_error']:
            print(f"Последняя ошибка: {metrics['last_error']}")
        
        # Экспортируем историю мониторинга
        history_file = f"vm_monitoring_history_{int(time.time())}.json"
        self.health_monitor.export_health_history(history_file)


def main():
    """Главная функция диагностического инструмента"""
    parser = argparse.ArgumentParser(
        description="VM RAG-as-a-Service Diagnostic Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python vm_diagnostics.py                    # Быстрая диагностика
  python vm_diagnostics.py --comprehensive    # Полная диагностика
  python vm_diagnostics.py --monitor 5        # Мониторинг на 5 минут
  python vm_diagnostics.py --quick --fix      # Быстрая диагностика + исправления
        """
    )
    
    parser.add_argument("--vm-host", default="10.61.11.54", 
                       help="IP адрес VM (default: 10.61.11.54)")
    parser.add_argument("--vm-port", type=int, default=8000,
                       help="Порт VM сервиса (default: 8000)")
    
    # Режимы работы
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick", action="store_true", default=True,
                           help="Быстрая диагностика (default)")
    mode_group.add_argument("--comprehensive", action="store_true",
                           help="Полная диагностика всех компонентов")
    mode_group.add_argument("--monitor", type=int, metavar="MINUTES",
                           help="Режим мониторинга на N минут")
    
    # Дополнительные опции
    parser.add_argument("--fix", action="store_true",
                       help="Показать команды для исправления проблем")
    parser.add_argument("--export", metavar="FILENAME",
                       help="Экспортировать результаты в JSON файл")
    parser.add_argument("--verbose", action="store_true",
                       help="Подробный вывод")
    
    args = parser.parse_args()
    
    # Настройка логирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Создаем диагностический набор
    diagnostic_suite = VMDiagnosticSuite(args.vm_host, args.vm_port)
    diagnostic_suite.print_banner()
    
    results = None
    
    try:
        if args.monitor:
            # Режим мониторинга
            diagnostic_suite.start_monitoring_mode(args.monitor)
            
        elif args.comprehensive:
            # Полная диагностика
            results = diagnostic_suite.run_comprehensive_diagnostics()
            
        else:
            # Быстрая диагностика (default)
            results = diagnostic_suite.run_quick_diagnostics()
        
        # Экспорт результатов
        if results and args.export:
            diagnostic_suite.export_diagnostic_report(results, args.export)
        elif results:
            # Автоматический экспорт
            diagnostic_suite.export_diagnostic_report(results)
        
        # Показываем исправления
        if args.fix and results:
            fix_suggestions = diagnostic_suite.generate_fix_suggestions(results)
            if fix_suggestions:
                print("\n" + "=" * 55)
                print("🔧 КОМАНДЫ ДЛЯ ИСПРАВЛЕНИЯ ПРОБЛЕМ")
                print("=" * 55)
                for suggestion in fix_suggestions:
                    print(suggestion)
    
    except KeyboardInterrupt:
        print("\n⏹️ Диагностика прервана пользователем")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Ошибка во время диагностики: {e}")
        print(f"\n❌ Критическая ошибка диагностики: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
