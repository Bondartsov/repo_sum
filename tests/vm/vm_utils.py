"""
VM Testing Utilities - Вспомогательные функции для тестирования VM.

Этот модуль содержит общие утилиты и вспомогательные функции,
используемые во всех VM диагностических тестах.

Автор: Claude (Cline) для VM testing utilities
Дата: 19 сентября 2025
"""

import os
import time
import socket
import subprocess
import platform
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import signal
import sys

logger = logging.getLogger(__name__)


@dataclass
class VMConfig:
    """Конфигурация VM для тестирования"""
    host: str = "10.61.11.54"
    port: int = 8000
    ssh_user: str = "user"
    ssh_password: Optional[str] = None
    timeout_seconds: int = 30
    
    def __post_init__(self):
        # Загружаем пароль из переменной окружения если не указан
        if not self.ssh_password:
            # Сначала пробуем из переменной окружения
            self.ssh_password = os.getenv("VM_PASSWORD")
            
            # Если не найден, пробуем загрузить из .env файла
            if not self.ssh_password:
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                    self.ssh_password = os.getenv("VM_PASSWORD")
                    if self.ssh_password:
                        logger.info("SSH пароль загружен из .env файла")
                except ImportError:
                    logger.warning("python-dotenv не установлен, не удалось загрузить .env")
                except Exception as e:
                    logger.warning(f"Ошибка загрузки .env файла: {e}")
                    
            if not self.ssh_password:
                logger.warning("VM_PASSWORD не найден ни в переменных окружения, ни в .env файле")
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def health_endpoint(self) -> str:
        return f"{self.base_url}/health"
    
    @property
    def embeddings_endpoint(self) -> str:
        return f"{self.base_url}/embeddings"


class NetworkUtils:
    """Утилиты для работы с сетью"""
    
    @staticmethod
    def check_port_open(host: str, port: int, timeout: int = 5) -> Tuple[bool, float, Optional[str]]:
        """
        Проверить открыт ли порт на хосте
        
        Returns:
            Tuple[bool, float, Optional[str]]: (открыт, время_отклика_мс, ошибка)
        """
        start_time = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        try:
            result = sock.connect_ex((host, port))
            response_time = (time.time() - start_time) * 1000
            
            if result == 0:
                return True, response_time, None
            else:
                return False, response_time, f"Connection failed with code {result}"
                
        except socket.timeout:
            return False, timeout * 1000, "Connection timeout"
        except socket.gaierror as e:
            return False, 0, f"DNS resolution failed: {e}"
        except Exception as e:
            return False, 0, f"Socket error: {e}"
        finally:
            sock.close()
    
    @staticmethod
    def ping_host(host: str, count: int = 4, timeout: int = 10) -> Dict[str, Any]:
        """
        Выполнить ping хоста
        
        Returns:
            Dict с результатами ping
        """
        result = {
            "success": False,
            "packets_sent": count,
            "packets_received": 0,
            "packet_loss_percent": 100,
            "avg_response_time_ms": None,
            "min_response_time_ms": None,
            "max_response_time_ms": None,
            "error": None
        }
        
        try:
            # Определяем команду ping в зависимости от ОС
            if platform.system().lower() == "windows":
                cmd = ["ping", "-n", str(count), host]
            else:
                cmd = ["ping", "-c", str(count), host]
            
            # Выполняем ping
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout
            )
            
            if process.returncode == 0:
                output = process.stdout or ""
                result["success"] = True
                
                # Парсим результаты
                if platform.system().lower() == "windows":
                    # Windows ping parsing
                    lines = output.split('\n')
                    received_count = 0
                    times = []
                    
                    for line in lines:
                        if 'Reply from' in line and 'time=' in line:
                            received_count += 1
                            try:
                                import re
                                time_match = re.search(r'time=(\d+)ms', line)
                                if time_match:
                                    times.append(int(time_match.group(1)))
                            except Exception:
                                pass
                    
                    result["packets_received"] = received_count
                    if times:
                        result["avg_response_time_ms"] = sum(times) / len(times)
                        result["min_response_time_ms"] = min(times)
                        result["max_response_time_ms"] = max(times)
                else:
                    # Linux ping parsing
                    lines = output.split('\n')
                    for line in lines:
                        if 'packets transmitted' in line:
                            parts = line.split()
                            result["packets_received"] = int(parts[3])
                        elif 'min/avg/max' in line:
                            try:
                                times = line.split('=')[1].split('/')[0:3]
                                result["min_response_time_ms"] = float(times[0])
                                result["avg_response_time_ms"] = float(times[1])
                                result["max_response_time_ms"] = float(times[2])
                            except Exception:
                                pass
                
                # Вычисляем процент потерь
                if result["packets_received"] > 0:
                    result["packet_loss_percent"] = (
                        (count - result["packets_received"]) / count * 100
                    )
                    
            else:
                stderr_output = process.stderr or ""
                result["error"] = f"Ping failed: {stderr_output}"
                
        except subprocess.TimeoutExpired:
            result["error"] = f"Ping timeout after {timeout} seconds"
        except FileNotFoundError:
            result["error"] = "Ping command not found"
        except Exception as e:
            result["error"] = f"Ping error: {e}"
        
        return result
    
    @staticmethod
    def get_local_ip() -> str:
        """Получить локальный IP адрес"""
        try:
            # Создаем UDP соединение (не отправляем данные)
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(("8.8.8.8", 80))
            local_ip = sock.getsockname()[0]
            sock.close()
            return local_ip
        except Exception:
            return "127.0.0.1"
    
    @staticmethod
    def traceroute_to_host(host: str, max_hops: int = 10) -> List[Dict[str, Any]]:
        """
        Выполнить traceroute к хосту (упрощенная версия)
        
        Returns:
            List[Dict] с информацией о каждом hop
        """
        hops = []
        
        try:
            if platform.system().lower() == "windows":
                cmd = ["tracert", "-h", str(max_hops), host]
            else:
                cmd = ["traceroute", "-m", str(max_hops), host]
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60
            )

            if process.returncode == 0:
                lines = (process.stdout or "").split('\n')
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith('Tracing') and not line.startswith('traceroute'):
                        hops.append({
                            "hop_number": i,
                            "raw_line": line.strip(),
                            "parsed": False  # Можно добавить парсинг позже
                        })
            
        except Exception as e:
            logger.warning(f"Traceroute failed: {e}")
        
        return hops


class ProcessUtils:
    """Утилиты для работы с процессами"""
    
    @staticmethod
    def is_process_running(process_name: str) -> bool:
        """Проверить запущен ли процесс по имени"""
        try:
            if platform.system().lower() == "windows":
                cmd = ["tasklist", "/FI", f"IMAGENAME eq {process_name}"]
            else:
                cmd = ["pgrep", "-f", process_name]
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            stdout_output = process.stdout or ""
            return process.returncode == 0 and process_name in stdout_output
            
        except Exception:
            return False
    
    @staticmethod
    def get_process_info(process_name: str) -> List[Dict[str, Any]]:
        """Получить информацию о процессах по имени"""
        processes = []
        
        try:
            if platform.system().lower() == "windows":
                cmd = ["tasklist", "/FO", "CSV", "/FI", f"IMAGENAME eq {process_name}"]
            else:
                cmd = ["ps", "aux"]
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            if process.returncode == 0:
                lines = (process.stdout or "").split('\n')
                for line in lines:
                    if process_name in line:
                        processes.append({
                            "raw_line": line.strip(),
                            "parsed": False  # Можно добавить детальный парсинг
                        })
            
        except Exception as e:
            logger.warning(f"Failed to get process info: {e}")
        
        return processes


class FileUtils:
    """Утилиты для работы с файлами"""
    
    @staticmethod
    def ensure_directory_exists(directory: str) -> bool:
        """Убедиться что директория существует, создать если нет"""
        try:
            os.makedirs(directory, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            return False
    
    @staticmethod
    def save_json_report(data: Dict[str, Any], filename: str, 
                        directory: str = "vm_reports") -> str:
        """Сохранить отчет в JSON файл"""
        FileUtils.ensure_directory_exists(directory)
        
        filepath = os.path.join(directory, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Report saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            return ""
    
    @staticmethod
    def load_json_report(filepath: str) -> Optional[Dict[str, Any]]:
        """Загрузить отчет из JSON файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load report {filepath}: {e}")
            return None
    
    @staticmethod
    def cleanup_old_reports(directory: str = "vm_reports", 
                           max_age_hours: int = 24 * 7) -> int:
        """Очистить старые отчеты"""
        if not os.path.exists(directory):
            return 0
        
        cleaned_count = 0
        current_time = time.time()
        
        try:
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getmtime(filepath)
                    if file_age > max_age_hours * 3600:
                        os.remove(filepath)
                        cleaned_count += 1
                        
        except Exception as e:
            logger.error(f"Failed to cleanup old reports: {e}")
        
        return cleaned_count


class TimingUtils:
    """Утилиты для работы со временем и таймингом"""
    
    @staticmethod
    @contextmanager
    def timing_context(operation_name: str = "Operation"):
        """Context manager для измерения времени выполнения"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            logger.info(f"{operation_name} took {duration:.3f} seconds")
    
    @staticmethod
    def retry_with_backoff(func, max_retries: int = 3, 
                          base_delay: float = 1.0, 
                          backoff_factor: float = 2.0,
                          max_delay: float = 60.0):
        """
        Выполнить функцию с повторами и экспоненциальной задержкой
        
        Args:
            func: Функция для выполнения
            max_retries: Максимальное количество попыток
            base_delay: Базовая задержка в секундах
            backoff_factor: Множитель для увеличения задержки
            max_delay: Максимальная задержка
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")
        
        if last_exception:
            raise last_exception


class TestEnvironment:
    """Класс для управления тестовой средой"""
    
    def __init__(self, vm_config: Optional[VMConfig] = None):
        self.vm_config = vm_config or VMConfig()
        self.setup_logging()
    
    def setup_logging(self):
        """Настроить логирование для тестов"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Настраиваем базовое логирование
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('vm_tests.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def check_environment_ready(self) -> Dict[str, Any]:
        """Проверить готовность тестовой среды"""
        checks = {
            "vm_config_valid": self._check_vm_config(),
            "network_accessible": self._check_network_access(),
            "ssh_credentials": self._check_ssh_credentials(),
            "required_tools": self._check_required_tools()
        }
        
        checks["overall_ready"] = all(checks.values())
        return checks
    
    def _check_vm_config(self) -> bool:
        """Проверить конфигурацию VM"""
        return (
            self.vm_config.host is not None and
            self.vm_config.port > 0 and
            self.vm_config.ssh_user is not None
        )
    
    def _check_network_access(self) -> bool:
        """Проверить сетевой доступ к VM"""
        is_open, _, _ = NetworkUtils.check_port_open(
            self.vm_config.host, 
            self.vm_config.port, 
            timeout=5
        )
        return is_open
    
    def _check_ssh_credentials(self) -> bool:
        """Проверить наличие SSH учетных данных"""
        return self.vm_config.ssh_password is not None
    
    def _check_required_tools(self) -> bool:
        """Проверить наличие необходимых инструментов"""
        required_tools = ["ping", "curl"]
        
        if platform.system().lower() == "windows":
            required_tools.extend(["tasklist"])
        else:
            required_tools.extend(["ps", "ss"])
        
        for tool in required_tools:
            try:
                subprocess.run(
                    [tool],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=5
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False
        
        return True


class SignalHandler:
    """Обработчик сигналов для graceful shutdown"""
    
    def __init__(self):
        self.shutdown_requested = False
        self._original_handlers = {}
        
    def setup_handlers(self):
        """Настроить обработчики сигналов"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, requesting shutdown...")
            self.shutdown_requested = True
        
        for sig in [signal.SIGINT, signal.SIGTERM]:
            self._original_handlers[sig] = signal.signal(sig, signal_handler)
    
    def restore_handlers(self):
        """Восстановить оригинальные обработчики"""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
    
    @contextmanager
    def managed_execution(self):
        """Context manager для управляемого выполнения с обработкой сигналов"""
        self.setup_handlers()
        try:
            yield self
        finally:
            self.restore_handlers()


# Константы и конфигурация по умолчанию
DEFAULT_VM_CONFIG = VMConfig()

# Утилитарные функции верхнего уровня
def quick_vm_check(vm_host: str = "10.61.11.54", vm_port: int = 8000) -> Dict[str, Any]:
    """Быстрая проверка доступности VM"""
    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vm_host": vm_host,
        "vm_port": vm_port,
        "checks": {}
    }
    
    # Ping тест
    ping_result = NetworkUtils.ping_host(vm_host, count=2, timeout=10)
    result["checks"]["ping"] = ping_result["success"]
    
    # Port тест
    port_open, response_time, error = NetworkUtils.check_port_open(vm_host, vm_port, timeout=5)
    result["checks"]["port_open"] = port_open
    result["checks"]["port_response_time_ms"] = response_time
    
    # HTTP тест (базовый)
    try:
        import requests
        http_response = requests.get(f"http://{vm_host}:{vm_port}/health", timeout=10)
        result["checks"]["http_accessible"] = http_response.status_code == 200
    except Exception:
        result["checks"]["http_accessible"] = False
    
    # Общий статус
    result["overall_accessible"] = all([
        result["checks"]["ping"],
        result["checks"]["port_open"],
        result["checks"]["http_accessible"]
    ])
    
    return result


def setup_vm_test_environment() -> TestEnvironment:
    """Создать и настроить тестовую среду для VM"""
    env = TestEnvironment()
    
    # Проверяем готовность среды
    readiness = env.check_environment_ready()
    if not readiness["overall_ready"]:
        logger.warning("Test environment is not fully ready:")
        for check, status in readiness.items():
            if not status and check != "overall_ready":
                logger.warning(f"  - {check}: FAIL")
    
    return env


if __name__ == "__main__":
    # Демонстрация использования утилит
    print("🛠️ VM Testing Utilities Demo")
    print("=" * 40)
    
    # Быстрая проверка VM
    vm_status = quick_vm_check()
    print(f"VM Status: {'✅ ACCESSIBLE' if vm_status['overall_accessible'] else '❌ NOT ACCESSIBLE'}")
    
    # Проверка среды
    env = setup_vm_test_environment()
    readiness = env.check_environment_ready()
    print(f"Test Environment: {'✅ READY' if readiness['overall_ready'] else '⚠️ NOT READY'}")
    
    # Демонстрация ping
    ping_result = NetworkUtils.ping_host("10.61.11.54", count=3)
    if ping_result["success"]:
        # ИСПРАВЛЕНО: Проверяем что avg_response_time_ms не None
        if ping_result.get('avg_response_time_ms') is not None:
            print(f"Ping: ✅ {ping_result['avg_response_time_ms']:.1f}ms avg")
        else:
            print("Ping: ✅ Success (время недоступно)")
    else:
        print(f"Ping: ❌ {ping_result['error']}")
