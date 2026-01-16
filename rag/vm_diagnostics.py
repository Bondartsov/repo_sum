"""
Комплексная диагностика проблем подключения к VM сервису.

Этот модуль предоставляет функции для детальной диагностики connectivity issues,
включая проверку DNS, портов, HTTP доступности и latency измерения.

Автор: AI Assistant
Дата: 1 октября 2025
"""

import asyncio
import socket
import time
import logging
from typing import Dict, Any
import aiohttp

logger = logging.getLogger(__name__)


async def diagnose_vm_connection(
    host: str,
    port: int,
    timeout: float = 10.0
) -> Dict[str, Any]:
    """
    Выполняет комплексную диагностику VM подключения.
    
    Проверки выполняются последовательно:
    1. DNS resolution - может ли система разрешить имя хоста
    2. TCP port check - открыт ли порт на хосте
    3. HTTP health check - отвечает ли HTTP сервис
    4. Latency measurement - измерение времени отклика
    
    Args:
        host: Хост VM сервиса (IP или доменное имя)
        port: Порт VM сервиса
        timeout: Общий таймаут для всех проверок (секунды)
        
    Returns:
        Словарь с результатами диагностики:
        {
            'host_reachable': bool,        # DNS разрешается
            'port_open': bool,             # Порт доступен
            'http_responding': bool,       # HTTP сервис отвечает
            'latency_ms': Optional[int],   # Latency в миллисекундах
            'http_status': Optional[int],  # HTTP статус код
            'response_data': Optional[dict], # Данные ответа (если JSON)
            'recommendations': List[str],  # Рекомендации по устранению
            'diagnostic_commands': List[str], # Команды для ручной диагностики
            'errors': List[str],           # Список ошибок
            'success': bool                # Общий результат
        }
    
    Example:
        ```python
        diagnostics = await diagnose_vm_connection('10.61.11.54', 8000)
        
        if not diagnostics['success']:
            for recommendation in diagnostics['recommendations']:
                print(f"💡 {recommendation}")
            for cmd in diagnostics['diagnostic_commands']:
                print(f"🔧 {cmd}")
        ```
    """
    logger.info(f"Начало диагностики подключения к VM: {host}:{port}")
    
    start_time = time.time()
    diagnostics = {
        'host_reachable': False,
        'port_open': False,
        'http_responding': False,
        'latency_ms': None,
        'http_status': None,
        'response_data': None,
        'recommendations': [],
        'diagnostic_commands': [],
        'errors': [],
        'success': False
    }
    
    try:
        # 1. Проверка DNS resolution
        logger.debug(f"Шаг 1: Проверка DNS resolution для {host}")
        
        try:
            resolved_ip = socket.gethostbyname(host)
            diagnostics['host_reachable'] = True
            diagnostics['resolved_ip'] = resolved_ip
            logger.debug(f"DNS success: {host} -> {resolved_ip}")
        except socket.gaierror as e:
            error_msg = f"DNS не может разрешить {host}: {e}"
            diagnostics['errors'].append(error_msg)
            diagnostics['recommendations'].append(
                f"❌ DNS resolution failed для {host}. "
                "Проверьте: 1) Правильность адреса, 2) Доступность DNS сервера, "
                "3) Сетевое подключение"
            )
            diagnostics['diagnostic_commands'].extend([
                f"nslookup {host}",
                f"ping {host}",
                "ipconfig /all  # Windows",
                "cat /etc/resolv.conf  # Linux"
            ])
            logger.warning(error_msg)
            return diagnostics
        
        # 2. Проверка TCP порта
        logger.debug(f"Шаг 2: Проверка доступности порта {port}")
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(min(5.0, timeout / 2))
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                diagnostics['port_open'] = True
                logger.debug(f"Port {port} открыт на {host}")
            else:
                error_msg = f"Порт {port} закрыт на {host} (код: {result})"
                diagnostics['errors'].append(error_msg)
                diagnostics['recommendations'].append(
                    f"❌ Порт {port} недоступен на {host}. "
                    "Проверьте: 1) VM запущена, 2) Firewall не блокирует порт, "
                    "3) Сервис слушает на правильном порту"
                )
                diagnostics['diagnostic_commands'].extend([
                    f"telnet {host} {port}",
                    f"nc -zv {host} {port}  # netcat",
                    "netstat -an | findstr :{port}  # Windows",
                    "ss -tuln | grep :{port}  # Linux",
                    "python vm_start.py start  # Запуск VM"
                ])
                logger.warning(error_msg)
                return diagnostics
                
        except Exception as e:
            error_msg = f"Ошибка проверки порта: {e}"
            diagnostics["errors"].append(error_msg)
            diagnostics["recommendations"].append(
                f"❌ Не удалось проверить порт {port}: {e}"
            )
            logger.error(error_msg)
            return diagnostics
        
        # 3. HTTP health check с latency измерением
        logger.debug(f"Шаг 3: HTTP health check на http://{host}:{port}/health")
        
        http_start = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{host}:{port}/health",
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    latency = (time.time() - http_start) * 1000
                    diagnostics['latency_ms'] = int(latency)
                    diagnostics['http_status'] = response.status
                    
                    logger.debug(
                        f"HTTP response: status={response.status}, "
                        f"latency={latency:.0f}ms"
                    )
                    
                    if response.status == 200:
                        diagnostics['http_responding'] = True
                        
                        # Пытаемся прочитать JSON ответ
                        try:
                            data = await response.json()
                            diagnostics['response_data'] = data
                            logger.debug(f"Health check data: {data}")
                            
                            # Анализируем статус компонентов
                            if isinstance(data, dict):
                                status = data.get('status')
                                if status in ('healthy', 'ok', 'connected'):
                                    diagnostics['success'] = True
                                    diagnostics['recommendations'].append(
                                        f"✅ VM сервис работает корректно (latency: {latency:.0f}ms)"
                                    )
                                else:
                                    diagnostics['recommendations'].append(
                                        f"⚠️ VM отвечает, но статус: {status}. "
                                        "Проверьте компоненты внутри VM."
                                    )
                                    
                        except ValueError as e:
                            error_msg = f"Некорректный JSON ответ: {e}"
                            diagnostics['errors'].append(error_msg)
                            diagnostics['recommendations'].append(
                                "⚠️ HTTP сервис отвечает, но формат ответа некорректный. "
                                "Возможно, несовместимая версия API."
                            )
                            diagnostics['diagnostic_commands'].append(
                                f"curl -v http://{host}:{port}/health"
                            )
                            logger.warning(error_msg)
                    
                    elif response.status >= 500:
                        error_msg = f"HTTP {response.status}: Internal Server Error"
                        error_text = await response.text()
                        diagnostics['errors'].append(f"{error_msg}: {error_text[:200]}")
                        diagnostics['recommendations'].append(
                            f"❌ VM сервис вернул {response.status}. "
                            "Проверьте: 1) Логи VM сервиса, 2) Qdrant доступность, "
                            "3) Конфигурацию сервиса"
                        )
                        diagnostics['diagnostic_commands'].extend([
                            "# На VM:",
                            "journalctl -u rag-vm-service -n 50",
                            "docker ps | grep qdrant",
                            "systemctl status qdrant"
                        ])
                        logger.warning(error_msg)
                    
                    elif response.status >= 400:
                        error_msg = f"HTTP {response.status}: Client Error"
                        diagnostics['errors'].append(error_msg)
                        diagnostics['recommendations'].append(
                            f"❌ HTTP {response.status}. Проверьте endpoint URL и метод запроса."
                        )
                        logger.warning(error_msg)
                        
        except asyncio.TimeoutError:
            error_msg = f"HTTP timeout после {timeout}s"
            diagnostics['errors'].append(error_msg)
            diagnostics['recommendations'].append(
                f"❌ VM сервис не отвечает в срок ({timeout}s). "
                "Проверьте: 1) Загрузку VM (CPU/RAM), 2) Сетевую latency, "
                "3) Размер обрабатываемых данных"
            )
            diagnostics['diagnostic_commands'].extend([
                f"ping -c 5 {host}",
                "# На VM:",
                "top -b -n 1",
                "free -h",
                "iostat"
            ])
            logger.warning(error_msg)
            
        except aiohttp.ClientError as e:
            error_msg = f"HTTP client error: {e}"
            diagnostics['errors'].append(error_msg)
            diagnostics['recommendations'].append(
                f"❌ Ошибка HTTP клиента: {type(e).__name__}. "
                "Возможно, проблема с сетью или сервисом."
            )
            logger.error(error_msg)
    
    except Exception as e:
        error_msg = f"Неожиданная ошибка диагностики: {e}"
        diagnostics['errors'].append(error_msg)
        diagnostics['recommendations'].append(
            f"❌ Критическая ошибка диагностики: {type(e).__name__}: {e}"
        )
        logger.exception("Критическая ошибка в diagnose_vm_connection")
    
    # Финальный summary
    total_time = time.time() - start_time
    diagnostics['total_diagnostic_time_s'] = round(total_time, 2)
    
    # Добавляем общие рекомендации если нет успеха
    if not diagnostics['success'] and not diagnostics['recommendations']:
        diagnostics['recommendations'].append(
            "❌ VM сервис недоступен. Запустите: python vm_start.py start"
        )
        diagnostics['diagnostic_commands'].insert(0, "python vm_start.py start")
    
    logger.info(
        f"Диагностика завершена за {total_time:.2f}s: "
        f"success={diagnostics['success']}, "
        f"errors={len(diagnostics['errors'])}"
    )
    
    return diagnostics


def format_diagnostics_report(diagnostics: Dict[str, Any]) -> str:
    """
    Форматирует результаты диагностики в человекочитаемый отчёт.
    
    Args:
        diagnostics: Результаты diagnose_vm_connection()
        
    Returns:
        Форматированный текстовый отчёт
    """
    lines = [
        "=" * 60,
        "📊 ОТЧЁТ ДИАГНОСТИКИ VM ПОДКЛЮЧЕНИЯ",
        "=" * 60,
        ""
    ]
    
    # Статус проверок
    lines.append("🔍 Результаты проверок:")
    lines.append(f"  DNS Resolution: {'✅' if diagnostics['host_reachable'] else '❌'}")
    lines.append(f"  TCP Port: {'✅' if diagnostics['port_open'] else '❌'}")
    lines.append(f"  HTTP Service: {'✅' if diagnostics['http_responding'] else '❌'}")
    
    if diagnostics['latency_ms']:
        lines.append(f"  Latency: {diagnostics['latency_ms']}ms")
    
    if diagnostics['http_status']:
        lines.append(f"  HTTP Status: {diagnostics['http_status']}")
    
    lines.append("")
    
    # Ошибки
    if diagnostics['errors']:
        lines.append("❌ Обнаруженные ошибки:")
        for error in diagnostics['errors']:
            lines.append(f"  • {error}")
        lines.append("")
    
    # Рекомендации
    if diagnostics['recommendations']:
        lines.append("💡 Рекомендации:")
        for rec in diagnostics['recommendations']:
            lines.append(f"  {rec}")
        lines.append("")
    
    # Диагностические команды
    if diagnostics['diagnostic_commands']:
        lines.append("🔧 Команды для диагностики:")
        for cmd in diagnostics['diagnostic_commands']:
            lines.append(f"  {cmd}")
        lines.append("")
    
    # Итоговый статус
    status = "✅ SUCCESS" if diagnostics['success'] else "❌ FAILED"
    lines.append(f"Итоговый статус: {status}")
    lines.append(f"Время диагностики: {diagnostics.get('total_diagnostic_time_s', 0):.2f}s")
    lines.append("=" * 60)
    
    return "\n".join(lines)


async def quick_health_check(host: str, port: int, timeout: float = 5.0) -> bool:
    """
    Быстрая проверка доступности VM сервиса (только HTTP health check).
    
    Args:
        host: Хост VM сервиса
        port: Порт VM сервиса
        timeout: Таймаут проверки (секунды)
        
    Returns:
        True если сервис доступен и healthy
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://{host}:{port}/health",
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    status = data.get('status', '')
                    return status in ('healthy', 'ok', 'connected')
                return False
    except Exception:
        return False
