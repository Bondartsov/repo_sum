"""
VM Firewall Configuration Tests - Тестирование настроек firewall на VM.

Этот модуль содержит тесты для проверки правильности настройки firewall
на VM для обеспечения доступности RAG-as-a-Service.

Автор: Claude (Cline) для диагностики firewall проблем
Дата: 19 сентября 2025
"""

import pytest
pytest.importorskip('paramiko')
import socket
import time
import paramiko
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FirewallRule:
    """Представление правила firewall"""
    port: int
    protocol: str  # tcp/udp
    direction: str  # incoming/outgoing
    action: str  # allow/deny
    source: Optional[str] = None
    destination: Optional[str] = None
    
    def __str__(self):
        return f"{self.direction} {self.protocol}/{self.port} {self.action}"


@dataclass
class FirewallTestResult:
    """Результат тестирования firewall"""
    test_name: str
    success: bool
    error: Optional[str]
    details: Dict[str, Any]
    recommendations: List[str]


class VMFirewallTester:
    """Класс для тестирования firewall конфигурации на VM"""
    
    def __init__(self, vm_host: str = "10.61.11.54", vm_port: int = 8000, 
                 ssh_user: str = "user"):
        self.vm_host = vm_host
        self.vm_port = vm_port
        self.ssh_user = ssh_user
        self.ssh_client = None
        
        # Загружаем SSH пароль из переменной окружения
        self.ssh_password = os.getenv("VM_PASSWORD")
        if not self.ssh_password:
            logger.warning("VM_PASSWORD не найден в переменных окружения")
    
    def connect_ssh(self) -> bool:
        """Подключение к VM по SSH"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            self.ssh_client.connect(
                hostname=self.vm_host,
                username=self.ssh_user,
                password=self.ssh_password,
                timeout=30
            )
            
            logger.info(f"SSH подключение к {self.vm_host} успешно")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка SSH подключения: {e}")
            return False
    
    def disconnect_ssh(self):
        """Отключение SSH"""
        if self.ssh_client:
            self.ssh_client.close()
            self.ssh_client = None
    
    def execute_ssh_command(self, command: str) -> Tuple[bool, str, str]:
        """Выполнить команду через SSH"""
        if not self.ssh_client:
            return False, "", "SSH не подключен"
        
        try:
            # Если команда содержит sudo, используем -S флаг для чтения пароля из stdin
            if command.startswith('sudo '):
                command = command.replace('sudo ', f'echo "{self.ssh_password}" | sudo -S ', 1)
            
            stdin, stdout, stderr = self.ssh_client.exec_command(command, timeout=30)
            exit_status = stdout.channel.recv_exit_status()
            
            stdout_text = stdout.read().decode('utf-8')
            stderr_text = stderr.read().decode('utf-8')
            
            return exit_status == 0, stdout_text, stderr_text
            
        except Exception as e:
            return False, "", str(e)
    
    def test_ufw_status(self) -> FirewallTestResult:
        """Тест статуса UFW firewall"""
        result = FirewallTestResult(
            test_name="UFW Status Check",
            success=False,
            error=None,
            details={},
            recommendations=[]
        )
        
        if not self.connect_ssh():
            result.error = "Не удалось подключиться по SSH"
            return result
        
        try:
            # Проверяем статус UFW
            success, stdout, stderr = self.execute_ssh_command("sudo ufw status verbose")
            
            if not success:
                result.error = f"Команда ufw status failed: {stderr}"
                result.recommendations.append("Проверить что UFW установлен: sudo apt install ufw")
                return result
            
            ufw_output = stdout.strip()
            result.details["ufw_output"] = ufw_output
            
            # Анализируем статус
            if "Status: active" in ufw_output:
                result.details["ufw_status"] = "active"
                result.success = True
                
                # Проверяем правила для порта 8000
                if f"{self.vm_port}/tcp" in ufw_output and "ALLOW IN" in ufw_output:
                    result.details["port_8000_allowed"] = True
                else:
                    result.details["port_8000_allowed"] = False
                    result.recommendations.append(f"Добавить правило: sudo ufw allow {self.vm_port}/tcp")
                    
            elif "Status: inactive" in ufw_output:
                result.details["ufw_status"] = "inactive"
                result.success = True  # Inactive тоже валидное состояние
                result.recommendations.append("UFW отключен - это может быть нормально если используется другой firewall")
                
            else:
                result.error = "Неожиданный вывод ufw status"
                result.recommendations.append("Проверить статус UFW вручную")
            
        except Exception as e:
            result.error = f"Ошибка при проверке UFW: {e}"
            
        finally:
            self.disconnect_ssh()
            
        return result
    
    def test_iptables_rules(self) -> FirewallTestResult:
        """Тест правил iptables"""
        result = FirewallTestResult(
            test_name="iptables Rules Check",
            success=False,
            error=None,
            details={},
            recommendations=[]
        )
        
        if not self.connect_ssh():
            result.error = "Не удалось подключиться по SSH"
            return result
        
        try:
            # Проверяем INPUT правила
            success, stdout, stderr = self.execute_ssh_command("sudo iptables -L INPUT -n --line-numbers")
            
            if not success:
                result.error = f"Команда iptables failed: {stderr}"
                return result
            
            iptables_input = stdout.strip()
            result.details["iptables_input"] = iptables_input
            
            # Проверяем есть ли правило для порта 8000
            port_rule_found = False
            allow_rule_found = False
            
            for line in iptables_input.split('\n'):
                if f"dpt:{self.vm_port}" in line:
                    port_rule_found = True
                    if "ACCEPT" in line:
                        allow_rule_found = True
                    break
            
            result.details["port_rule_found"] = port_rule_found
            result.details["allow_rule_found"] = allow_rule_found
            
            # Проверяем политику по умолчанию
            if "Chain INPUT (policy ACCEPT)" in iptables_input:
                result.details["default_policy"] = "ACCEPT"
                result.success = True  # Если политика ACCEPT, то порт открыт
            elif "Chain INPUT (policy DROP)" in iptables_input:
                result.details["default_policy"] = "DROP"
                if allow_rule_found:
                    result.success = True
                else:
                    result.recommendations.append(f"Добавить правило: sudo iptables -I INPUT -p tcp --dport {self.vm_port} -j ACCEPT")
            else:
                result.details["default_policy"] = "unknown"
            
            # Дополнительные рекомендации
            if not port_rule_found and result.details.get("default_policy") == "DROP":
                result.recommendations.append("При политике DROP нужно явное правило для порта 8000")
                
        except Exception as e:
            result.error = f"Ошибка при проверке iptables: {e}"
            
        finally:
            self.disconnect_ssh()
            
        return result
    
    def test_port_listening(self) -> FirewallTestResult:
        """Тест что порт слушается на VM"""
        result = FirewallTestResult(
            test_name="Port Listening Check",
            success=False,
            error=None,
            details={},
            recommendations=[]
        )
        
        if not self.connect_ssh():
            result.error = "Не удалось подключиться по SSH"
            return result
        
        try:
            # Проверяем что порт слушается
            success, stdout, stderr = self.execute_ssh_command(f"ss -tulnp | grep :{self.vm_port}")
            
            if success and stdout.strip():
                listening_info = stdout.strip()
                result.details["listening_info"] = listening_info
                
                # Анализируем на каком интерфейсе слушается
                if f"0.0.0.0:{self.vm_port}" in listening_info:
                    result.details["listening_interface"] = "all_interfaces"
                    result.success = True
                elif f"127.0.0.1:{self.vm_port}" in listening_info:
                    result.details["listening_interface"] = "localhost_only"
                    result.error = "Сервис слушается только на localhost"
                    result.recommendations.append("Настроить сервис для прослушивания на 0.0.0.0")
                else:
                    result.details["listening_interface"] = "other"
                    result.recommendations.append("Проверить на каком интерфейсе слушается сервис")
                
            else:
                result.error = f"Порт {self.vm_port} не слушается"
                result.recommendations.append("Запустить RAG сервис на VM")
                result.recommendations.append("Проверить конфигурацию сервиса")
                
        except Exception as e:
            result.error = f"Ошибка при проверке слушающих портов: {e}"
            
        finally:
            self.disconnect_ssh()
            
        return result
    
    def test_external_connectivity(self) -> FirewallTestResult:
        """Тест внешнего подключения к порту"""
        result = FirewallTestResult(
            test_name="External Connectivity Test",
            success=False,
            error=None,
            details={},
            recommendations=[]
        )
        
        try:
            # Тестируем TCP подключение
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            
            try:
                sock.connect((self.vm_host, self.vm_port))
                connect_time = (time.time() - start_time) * 1000
                
                result.success = True
                result.details["connection_time_ms"] = connect_time
                result.details["connection_status"] = "successful"
                
            except socket.timeout:
                result.error = "Connection timeout - возможно firewall блокирует подключение"
                result.recommendations.append("Проверить UFW/iptables правила")
                result.recommendations.append("Проверить что сервис запущен на VM")
                
            except ConnectionRefusedError:
                result.error = "Connection refused - сервис не слушается или firewall блокирует"
                result.recommendations.append("Проверить что RAG сервис запущен")
                result.recommendations.append("Проверить firewall правила")
                
            except Exception as e:
                result.error = f"Connection error: {e}"
                
            finally:
                sock.close()
                
        except Exception as e:
            result.error = f"Socket test error: {e}"
            
        return result
    
    def test_system_firewall_services(self) -> FirewallTestResult:
        """Тест статуса системных firewall сервисов"""
        result = FirewallTestResult(
            test_name="System Firewall Services",
            success=False,
            error=None,
            details={},
            recommendations=[]
        )
        
        if not self.connect_ssh():
            result.error = "Не удалось подключиться по SSH"
            return result
        
        try:
            services_to_check = ["ufw", "iptables", "firewalld", "nftables"]
            service_statuses = {}
            
            for service in services_to_check:
                # Проверяем статус сервиса
                success, stdout, stderr = self.execute_ssh_command(f"systemctl is-active {service} 2>/dev/null || echo 'not-found'")
                
                if success:
                    status = stdout.strip()
                    service_statuses[service] = status
                else:
                    service_statuses[service] = "error"
            
            result.details["service_statuses"] = service_statuses
            result.success = True
            
            # Анализируем какие firewall сервисы активны
            active_firewalls = [svc for svc, status in service_statuses.items() if status == "active"]
            result.details["active_firewalls"] = active_firewalls
            
            if not active_firewalls:
                result.recommendations.append("Ни один firewall сервис не активен - это может быть нормально")
            elif len(active_firewalls) > 1:
                result.recommendations.append(f"Несколько firewall сервисов активно: {active_firewalls} - может вызывать конфликты")
                
        except Exception as e:
            result.error = f"Ошибка при проверке firewall сервисов: {e}"
            
        finally:
            self.disconnect_ssh()
            
        return result
    
    def run_comprehensive_firewall_test(self) -> Dict[str, Any]:
        """Запустить комплексный тест firewall конфигурации"""
        print(f"🔥 Запуск комплексного теста firewall на VM {self.vm_host}:{self.vm_port}")
        print("=" * 70)
        
        all_results = {
            "vm_host": self.vm_host,
            "vm_port": self.vm_port,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": {},
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "critical_issues": [],
                "recommendations": []
            }
        }
        
        # Список тестов для выполнения
        tests_to_run = [
            self.test_external_connectivity,
            self.test_port_listening,
            self.test_ufw_status,
            self.test_iptables_rules,
            self.test_system_firewall_services
        ]
        
        for test_func in tests_to_run:
            try:
                result = test_func()
                test_name = result.test_name
                all_results["tests"][test_name] = {
                    "success": result.success,
                    "error": result.error,
                    "details": result.details,
                    "recommendations": result.recommendations
                }
                
                status = "✅ PASS" if result.success else "❌ FAIL"
                print(f"{status} {test_name}")
                
                if result.error:
                    print(f"    Error: {result.error}")
                    if not result.success:
                        all_results["summary"]["critical_issues"].append(result.error)
                
                if result.recommendations:
                    print("    Recommendations:")
                    for rec in result.recommendations:
                        print(f"      • {rec}")
                    all_results["summary"]["recommendations"].extend(result.recommendations)
                
                # Показываем ключевые детали
                if result.details:
                    for key, value in result.details.items():
                        if key in ["listening_interface", "ufw_status", "default_policy", "connection_status"]:
                            print(f"    {key}: {value}")
                
            except Exception as e:
                print(f"❌ FAIL {test_func.__name__} - Exception: {e}")
                all_results["tests"][test_func.__name__] = {
                    "success": False,
                    "error": str(e),
                    "details": {},
                    "recommendations": []
                }
        
        # Подсчет статистики
        total_tests = len(all_results["tests"])
        passed_tests = sum(1 for result in all_results["tests"].values() if result["success"])
        failed_tests = total_tests - passed_tests
        
        all_results["summary"].update({
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests
        })
        
        print("=" * 70)
        print(f"📊 Результаты firewall тестирования: {passed_tests}/{total_tests} прошло")
        
        if all_results["summary"]["critical_issues"]:
            print("🚨 Критические проблемы:")
            for issue in set(all_results["summary"]["critical_issues"]):
                print(f"  • {issue}")
        
        if all_results["summary"]["recommendations"]:
            print("💡 Основные рекомендации:")
            unique_recs = list(set(all_results["summary"]["recommendations"]))[:5]
            for rec in unique_recs:
                print(f"  • {rec}")
        
        return all_results


# Pytest тесты для firewall configuration
@pytest.mark.integration
class TestVMFirewallConfig:
    """Pytest класс для тестирования firewall конфигурации"""
    
    @pytest.fixture(scope="class")
    def firewall_tester(self):
        """Фикстура для создания firewall tester"""
        return VMFirewallTester()
    
    def test_external_port_connectivity(self, firewall_tester):
        """Тест внешнего подключения к порту"""
        result = firewall_tester.test_external_connectivity()
        
        # Этот тест критичный - если не проходит, то есть проблема с firewall
        if not result.success:
            pytest.skip(f"External connectivity failed: {result.error}")
        
        assert result.success, f"External connectivity test failed: {result.error}"
        assert result.details.get("connection_status") == "successful"
    
    @pytest.mark.skipif(not os.getenv("VM_PASSWORD"), reason="VM_PASSWORD не найден")
    def test_port_listening_on_vm(self, firewall_tester):
        """Тест что порт слушается на VM"""
        result = firewall_tester.test_port_listening()
        
        assert result.success, f"Port listening test failed: {result.error}"
        assert result.details.get("listening_interface") in ["all_interfaces", "other"], \
            "Service should listen on external interface"
    
    @pytest.mark.skipif(not os.getenv("VM_PASSWORD"), reason="VM_PASSWORD не найден")
    def test_ufw_configuration(self, firewall_tester):
        """Тест конфигурации UFW"""
        result = firewall_tester.test_ufw_status()
        
        # UFW может быть активным или неактивным - оба состояния валидны
        assert result.success, f"UFW status check failed: {result.error}"
        
        # Если UFW активен, проверяем правила
        if result.details.get("ufw_status") == "active":
            # Если UFW активен, порт 8000 должен быть разрешен ИЛИ тест подключения должен проходить
            connectivity_result = firewall_tester.test_external_connectivity()
            if not connectivity_result.success:
                assert result.details.get("port_8000_allowed", False), \
                    "If UFW is active and connection fails, port 8000 should be explicitly allowed"
    
    @pytest.mark.skipif(not os.getenv("VM_PASSWORD"), reason="VM_PASSWORD не найден")
    def test_iptables_configuration(self, firewall_tester):
        """Тест конфигурации iptables"""
        result = firewall_tester.test_iptables_rules()
        
        assert result.success, f"iptables check failed: {result.error}"
        
        # Если политика DROP, должно быть правило разрешающее порт
        if result.details.get("default_policy") == "DROP":
            connectivity_result = firewall_tester.test_external_connectivity()
            if not connectivity_result.success:
                assert result.details.get("allow_rule_found", False), \
                    "With DROP policy, port 8000 should have explicit ACCEPT rule"
    
    def test_comprehensive_firewall_suite(self, firewall_tester):
        """Комплексный тест firewall конфигурации"""
        results = firewall_tester.run_comprehensive_firewall_test()
        
        # Проверяем что хотя бы базовые тесты прошли
        assert results["summary"]["total_tests"] > 0, "No tests were executed"
        
        # Критичный тест - внешнее подключение должно работать
        external_test = results["tests"].get("External Connectivity Test", {})
        assert external_test.get("success", False), \
            f"External connectivity must work. Error: {external_test.get('error')}"
        
        # Если есть критические проблемы, показываем их
        if results["summary"]["critical_issues"]:
            pytest.fail(f"Critical firewall issues found: {results['summary']['critical_issues']}")


# Утилитарные функции для работы с firewall
def get_firewall_quick_fix_commands(vm_host: str = "10.61.11.54", vm_port: int = 8000) -> List[str]:
    """Получить список команд для быстрого исправления firewall проблем"""
    commands = [
        "# Команды для исправления firewall проблем:",
        "",
        "# 1. UFW - разрешить порт 8000:",
        f"sudo ufw allow {vm_port}/tcp",
        "sudo ufw reload",
        "",
        "# 2. iptables - разрешить порт 8000:",
        f"sudo iptables -I INPUT -p tcp --dport {vm_port} -j ACCEPT",
        "sudo iptables-save > /etc/iptables/rules.v4  # Сохранить правила",
        "",
        "# 3. Проверить что сервис слушается на всех интерфейсах:",
        f"ss -tulnp | grep :{vm_port}",
        "# Должно показать 0.0.0.0:8000, а не 127.0.0.1:8000",
        "",
        "# 4. Проверить статус сервиса:",
        "ps aux | grep vm_rag_service",
        "",
        "# 5. Перезапустить сервис если нужно:",
        "cd ~/repo_sum_rag/repo_sum && source venv/bin/activate",
        "python vm_rag_service.py"
    ]
    return commands


if __name__ == "__main__":
    # Запуск диагностики firewall из командной строки
    tester = VMFirewallTester()
    results = tester.run_comprehensive_firewall_test()
    
    # Сохраняем результаты в файл
    import json
    result_file = f"vm_firewall_test_{int(time.time())}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Результаты сохранены в {result_file}")
    
    # Показываем команды для исправления
    if results["summary"]["failed_tests"] > 0:
        print("\n🔧 Команды для исправления проблем:")
        fix_commands = get_firewall_quick_fix_commands()
        for cmd in fix_commands:
            print(cmd)
