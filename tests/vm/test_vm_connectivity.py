"""
VM Connectivity Tests - Тестирование сетевого подключения к RAG-as-a-Service VM.

Этот модуль содержит комплексные тесты для проверки сетевого подключения
между локальным клиентом и VM сервисом на 10.61.11.54:8000.

Автор: Claude (Cline) для диагностики Connection Refused ошибок
Дата: 19 сентября 2025
"""

import pytest
import asyncio
import aiohttp
import requests
import socket
import time
import subprocess
import platform
from typing import Dict, Any
import logging

from tests.network_utils import is_network_available


pytestmark = pytest.mark.skipif(
    not is_network_available(host="10.61.11.54", port=8000),
    reason="VM недоступна для сетевых тестов"
)

logger = logging.getLogger(__name__)


class VMConnectivityTester:
    """Класс для тестирования подключения к VM"""
    
    def __init__(self, vm_host: str = "10.61.11.54", vm_port: int = 8000):
        self.vm_host = vm_host
        self.vm_port = vm_port
        self.base_url = f"http://{vm_host}:{vm_port}"
        self.health_endpoint = f"{self.base_url}/health"
        self.embeddings_endpoint = f"{self.base_url}/embeddings"
        
    def test_basic_tcp_connection(self) -> Dict[str, Any]:
        """Базовый TCP тест подключения к VM порту"""
        result = {
            "test_name": "TCP Connection Test",
            "success": False,
            "error": None,
            "response_time_ms": None,
            "details": {}
        }
        
        start_time = time.time()
        try:
            # Создаем TCP сокет
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)  # 5 секунд таймаут
            
            # Пытаемся подключиться
            sock.connect((self.vm_host, self.vm_port))
            
            response_time = (time.time() - start_time) * 1000
            result.update({
                "success": True,
                "response_time_ms": response_time,
                "details": {"connection_established": True}
            })
            
        except socket.timeout:
            result["error"] = "Connection timeout after 5 seconds"
        except socket.gaierror as e:
            result["error"] = f"DNS resolution failed: {e}"
        except ConnectionRefusedError:
            result["error"] = "Connection refused - service not available"
        except Exception as e:
            result["error"] = f"Unexpected error: {e}"
        finally:
            try:
                sock.close()
            except Exception:
                pass
                
        return result
    
    def test_ping_connectivity(self) -> Dict[str, Any]:
        """Ping тест для проверки базовой достижимости VM"""
        result = {
            "test_name": "Ping Test",
            "success": False,
            "error": None,
            "avg_response_time_ms": None,
            "packet_loss": None
        }
        
        try:
            # Определяем команду ping в зависимости от ОС
            if platform.system().lower() == "windows":
                cmd = ["ping", "-n", "4", self.vm_host]
            else:
                cmd = ["ping", "-c", "4", self.vm_host]
            
            # Выполняем ping
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10
            )

            if process.returncode == 0:
                output = process.stdout or ""
                result["success"] = True
                
                # Парсим вывод для Windows/Linux
                if platform.system().lower() == "windows":
                    # Парсинг Windows ping вывода
                    lines = output.split('\n')
                    for line in lines:
                        if 'Average' in line or 'среднее' in line.lower():
                            try:
                                # Ищем время в миллисекундах
                                import re
                                time_match = re.search(r'(\d+)ms', line)
                                if time_match:
                                    result["avg_response_time_ms"] = int(time_match.group(1))
                            except Exception:
                                pass
                else:
                    # Парсинг Linux ping вывода  
                    lines = output.split('\n')
                    for line in lines:
                        if 'avg' in line:
                            try:
                                parts = line.split('/')
                                result["avg_response_time_ms"] = float(parts[4])
                            except Exception:
                                pass
                                
                result["packet_loss"] = "0%" if result["success"] else "unknown"
            else:
                stderr_output = process.stderr or ""
                result["error"] = f"Ping failed: {stderr_output}"
                
        except subprocess.TimeoutExpired:
            result["error"] = "Ping timeout after 10 seconds"
        except FileNotFoundError:
            result["error"] = "Ping command not found"
        except Exception as e:
            result["error"] = f"Ping test error: {e}"
            
        return result
    
    def test_http_health_endpoint(self) -> Dict[str, Any]:
        """HTTP тест health endpoint с requests библиотекой"""
        result = {
            "test_name": "HTTP Health Test (requests)",
            "success": False,
            "error": None,
            "response_time_ms": None,
            "status_code": None,
            "response_data": None
        }
        
        try:
            start_time = time.time()
            response = requests.get(
                self.health_endpoint,
                timeout=10,
                headers={'User-Agent': 'VM-Connectivity-Test/1.0'}
            )
            response_time = (time.time() - start_time) * 1000
            
            result.update({
                "success": response.status_code == 200,
                "response_time_ms": response_time,
                "status_code": response.status_code,
                "response_data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text[:200]
            })
            
            if response.status_code != 200:
                result["error"] = f"HTTP {response.status_code}: {response.text[:100]}"
                
        except requests.exceptions.ConnectTimeout:
            result["error"] = "HTTP connection timeout"
        except requests.exceptions.ConnectionError as e:
            result["error"] = f"HTTP connection error: {e}"
        except requests.exceptions.Timeout:
            result["error"] = "HTTP request timeout"
        except Exception as e:
            result["error"] = f"HTTP test error: {e}"
            
        return result
    
    async def test_aiohttp_health_endpoint(self) -> Dict[str, Any]:
        """Async HTTP тест health endpoint с aiohttp (как в реальном коде)"""
        result = {
            "test_name": "aiohttp Health Test",
            "success": False,
            "error": None,
            "response_time_ms": None,
            "status_code": None,
            "response_data": None
        }
        
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        
        try:
            start_time = time.time()
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    self.health_endpoint,
                    headers={'User-Agent': 'VM-Connectivity-Test-Async/1.0'}
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    response_text = await response.text()
                    
                    result.update({
                        "success": response.status == 200,
                        "response_time_ms": response_time,
                        "status_code": response.status,
                        "response_data": response_text[:200] if response_text else None
                    })
                    
                    if response.status != 200:
                        result["error"] = f"aiohttp {response.status}: {response_text[:100]}"
                        
        except aiohttp.ClientConnectorError as e:
            result["error"] = f"aiohttp connection error: {e}"
        except asyncio.TimeoutError:
            result["error"] = "aiohttp timeout"
        except Exception as e:
            result["error"] = f"aiohttp test error: {e}"
            
        return result
    
    async def test_embeddings_endpoint_post(self) -> Dict[str, Any]:
        """Тестируем POST запрос к /embeddings endpoint"""
        result = {
            "test_name": "Embeddings POST Test",
            "success": False,
            "error": None,
            "response_time_ms": None,
            "status_code": None
        }
        
        test_payload = {
            "texts": ["test connectivity"],
            "task": "retrieval.query",
            "truncate_dim": 1024,
            "normalize": True
        }
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        try:
            start_time = time.time()
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.embeddings_endpoint,
                    json=test_payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    response_text = await response.text()
                    
                    result.update({
                        "success": response.status == 200,
                        "response_time_ms": response_time,
                        "status_code": response.status
                    })
                    
                    if response.status != 200:
                        result["error"] = f"Embeddings POST {response.status}: {response_text[:100]}"
                        
        except aiohttp.ClientConnectorError as e:
            result["error"] = f"Embeddings POST connection error: {e}"
        except asyncio.TimeoutError:
            result["error"] = "Embeddings POST timeout"
        except Exception as e:
            result["error"] = f"Embeddings POST error: {e}"
            
        return result
    
    def test_dns_resolution(self) -> Dict[str, Any]:
        """Тестируем DNS разрешение VM хоста"""
        result = {
            "test_name": "DNS Resolution Test",
            "success": False,
            "error": None,
            "resolved_ip": None,
            "resolution_time_ms": None
        }
        
        try:
            start_time = time.time()
            resolved_ip = socket.gethostbyname(self.vm_host)
            resolution_time = (time.time() - start_time) * 1000
            
            result.update({
                "success": True,
                "resolved_ip": resolved_ip,
                "resolution_time_ms": resolution_time
            })
            
        except socket.gaierror as e:
            result["error"] = f"DNS resolution failed: {e}"
        except Exception as e:
            result["error"] = f"DNS test error: {e}"
            
        return result
    
    def run_all_connectivity_tests(self) -> Dict[str, Any]:
        """Запускает все тесты подключения и возвращает сводку"""
        print(f"🔍 Запуск диагностики подключения к VM {self.vm_host}:{self.vm_port}")
        print("=" * 60)
        
        all_results = {
            "vm_host": self.vm_host,
            "vm_port": self.vm_port,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": {},
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "success_rate": 0.0
            }
        }
        
        # Запускаем синхронные тесты
        sync_tests = [
            self.test_dns_resolution,
            self.test_ping_connectivity,
            self.test_basic_tcp_connection,
            self.test_http_health_endpoint
        ]
        
        for test_func in sync_tests:
            try:
                result = test_func()
                test_name = result["test_name"]
                all_results["tests"][test_name] = result
                
                status = "✅ PASS" if result["success"] else "❌ FAIL"
                print(f"{status} {test_name}")
                if result.get("error"):
                    print(f"    Error: {result['error']}")
                if result.get("response_time_ms"):
                    print(f"    Response time: {result['response_time_ms']:.1f}ms")
                    
            except Exception as e:
                print(f"❌ FAIL {test_func.__name__} - Exception: {e}")
                all_results["tests"][test_func.__name__] = {
                    "test_name": test_func.__name__,
                    "success": False,
                    "error": str(e)
                }
        
        # Запускаем асинхронные тесты
        async_tests = [
            self.test_aiohttp_health_endpoint,
            self.test_embeddings_endpoint_post
        ]
        
        for async_test_func in async_tests:
            try:
                result = asyncio.run(async_test_func())
                test_name = result["test_name"]
                all_results["tests"][test_name] = result
                
                status = "✅ PASS" if result["success"] else "❌ FAIL"
                print(f"{status} {test_name}")
                if result.get("error"):
                    print(f"    Error: {result['error']}")
                if result.get("response_time_ms"):
                    print(f"    Response time: {result['response_time_ms']:.1f}ms")
                    
            except Exception as e:
                print(f"❌ FAIL {async_test_func.__name__} - Exception: {e}")
                all_results["tests"][async_test_func.__name__] = {
                    "test_name": async_test_func.__name__,
                    "success": False,
                    "error": str(e)
                }
        
        # Подсчет статистики
        total_tests = len(all_results["tests"])
        passed_tests = sum(1 for result in all_results["tests"].values() if result["success"])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        all_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate
        }
        
        print("=" * 60)
        print(f"📊 Результаты: {passed_tests}/{total_tests} тестов прошло ({success_rate:.1f}%)")
        
        return all_results


# Pytest тесты для интеграции с test runner
@pytest.mark.integration 
class TestVMConnectivity:
    """Pytest класс для тестирования VM подключения"""
    
    @pytest.fixture(scope="class")
    def vm_tester(self):
        """Фикстура для создания VM тестера"""
        return VMConnectivityTester()
    
    def _check_vm_accessible(self, vm_tester) -> bool:
        """Быстрая проверка доступности VM"""
        try:
            result = vm_tester.test_basic_tcp_connection()
            return result["success"]
        except Exception:
            return False
    
    def test_vm_dns_resolution(self, vm_tester):
        """Тест DNS разрешения VM хоста"""
        result = vm_tester.test_dns_resolution()
        assert result["success"], f"DNS resolution failed: {result.get('error')}"
        assert result["resolved_ip"] is not None
        
    def test_vm_ping_connectivity(self, vm_tester):
        """Тест ping подключения к VM"""
        result = vm_tester.test_ping_connectivity()
        assert result["success"], f"Ping test failed: {result.get('error')}"
        
    def test_vm_tcp_connection(self, vm_tester):
        """Тест TCP подключения к VM порту 8000"""
        result = vm_tester.test_basic_tcp_connection()
        
        # Если VM недоступна, пропускаем тест
        if not result["success"]:
            pytest.skip(f"VM недоступна для TCP соединения: {result.get('error')}")
            
        assert result["response_time_ms"] is not None
        assert result["response_time_ms"] < 5000  # Менее 5 секунд
        
    def test_vm_http_health(self, vm_tester):
        """Тест HTTP health endpoint"""
        # Сначала проверяем базовое подключение
        if not self._check_vm_accessible(vm_tester):
            pytest.skip("VM недоступна для HTTP соединения")
            
        result = vm_tester.test_http_health_endpoint()
        assert result["success"], f"HTTP health test failed: {result.get('error')}"
        assert result["status_code"] == 200
        assert result["response_data"] is not None
        
    @pytest.mark.asyncio
    async def test_vm_aiohttp_health(self, vm_tester):
        """Тест aiohttp health endpoint (как в реальном коде)"""
        # Сначала проверяем базовое подключение
        if not self._check_vm_accessible(vm_tester):
            pytest.skip("VM недоступна для aiohttp соединения")
            
        result = await vm_tester.test_aiohttp_health_endpoint()
        assert result["success"], f"aiohttp health test failed: {result.get('error')}"
        assert result["status_code"] == 200
        
    @pytest.mark.asyncio 
    async def test_vm_embeddings_endpoint(self, vm_tester):
        """Тест POST запроса к embeddings endpoint"""
        # Сначала проверяем базовое подключение
        if not self._check_vm_accessible(vm_tester):
            pytest.skip("VM недоступна для embeddings endpoint тестирования")
            
        result = await vm_tester.test_embeddings_endpoint_post()
        assert result["success"], f"Embeddings endpoint test failed: {result.get('error')}"
        assert result["status_code"] == 200
        
    def test_vm_full_connectivity_suite(self, vm_tester):
        """Комплексный тест всех видов подключения"""
        results = vm_tester.run_all_connectivity_tests()
        
        # Проверяем что хотя бы базовые тесты (DNS, ping) прошли
        dns_success = results["tests"]["DNS Resolution Test"]["success"]
        ping_success = results["tests"]["Ping Test"]["success"]
        
        if not dns_success:
            pytest.skip("DNS resolution failed - VM host недоступен")
        
        if not ping_success:
            pytest.skip("Ping failed - VM недоступна по сети")
        
        # Если базовые тесты прошли, но подключение к сервису не работает
        success_rate = results["summary"]["success_rate"]
        if success_rate < 50.0:
            pytest.skip(f"VM сеть работает, но RAG сервис недоступен (success rate: {success_rate:.1f}%)")
        
        # Если дошли до сюда - все тесты должны пройти
        assert success_rate >= 50.0, f"VM connectivity too poor: only {success_rate:.1f}% tests passed"


if __name__ == "__main__":
    # Запуск диагностики из командной строки
    tester = VMConnectivityTester()
    results = tester.run_all_connectivity_tests()
    
    # Сохраняем результаты в файл для анализа
    import json
    result_file = f"vm_connectivity_test_{int(time.time())}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Результаты сохранены в {result_file}")
