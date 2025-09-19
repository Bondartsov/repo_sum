"""
VM Health Monitoring Tests - Мониторинг состояния RAG-as-a-Service на VM.

Этот модуль содержит тесты для непрерывного мониторинга состояния 
VM сервиса и автоматического обнаружения проблем производительности.

Автор: Claude (Cline) для мониторинга VM сервиса
Дата: 19 сентября 2025
"""

import pytest
import asyncio
import aiohttp
import requests
import time
import json
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque
import statistics
import logging

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Результат проверки здоровья сервиса"""
    timestamp: str
    success: bool
    response_time_ms: float
    status_code: Optional[int]
    error: Optional[str]
    service_status: Optional[str]
    embedder_status: Optional[str]
    vector_store_status: Optional[str]
    memory_usage: Optional[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ServiceMetrics:
    """Метрики производительности сервиса"""
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    success_rate: float
    error_count: int
    total_checks: int
    uptime_percentage: float
    last_error: Optional[str]
    trend_direction: str  # "improving", "degrading", "stable"


class VMHealthMonitor:
    """Класс для мониторинга здоровья VM сервиса"""
    
    def __init__(self, vm_host: str = "10.61.11.54", vm_port: int = 8000, 
                 check_interval: int = 30, max_history: int = 100):
        self.vm_host = vm_host
        self.vm_port = vm_port
        self.check_interval = check_interval
        self.max_history = max_history
        
        self.base_url = f"http://{vm_host}:{vm_port}"
        self.health_endpoint = f"{self.base_url}/health"
        self.stats_endpoint = f"{self.base_url}/stats"
        
        # История проверок
        self.health_history = deque(maxlen=max_history)
        self.is_monitoring = False
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        # Метрики
        self.total_checks = 0
        self.successful_checks = 0
        self.failed_checks = 0
        
    async def check_health_detailed(self) -> HealthCheckResult:
        """Детальная проверка здоровья сервиса"""
        timestamp = datetime.now().isoformat()
        result = HealthCheckResult(
            timestamp=timestamp,
            success=False,
            response_time_ms=0.0,
            status_code=None,
            error=None,
            service_status=None,
            embedder_status=None,
            vector_store_status=None,
            memory_usage=None
        )
        
        timeout = aiohttp.ClientTimeout(total=15, connect=5)
        
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    self.health_endpoint,
                    headers={'User-Agent': 'VM-Health-Monitor/1.0'}
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    result.response_time_ms = response_time
                    result.status_code = response.status
                    
                    if response.status == 200:
                        health_data = await response.json()
                        
                        result.success = True
                        result.service_status = health_data.get("status", "unknown")
                        
                        # Извлекаем детали о компонентах
                        services = health_data.get("services", {})
                        
                        embedder_info = services.get("embedder", {})
                        result.embedder_status = embedder_info.get("status", "unknown")
                        
                        vector_store_info = services.get("vector_store", {})
                        result.vector_store_status = vector_store_info.get("status", "unknown")
                        
                        # Дополнительная информация о коллекции
                        collection_info = health_data.get("collection_info", {})
                        if collection_info:
                            result.memory_usage = {
                                "vectors_count": health_data.get("vector_count", 0),
                                "collection_status": health_data.get("collection_status", "unknown"),
                                "qdrant_status": health_data.get("qdrant_status", "unknown")
                            }
                            
                    else:
                        response_text = await response.text()
                        result.error = f"HTTP {response.status}: {response_text[:100]}"
                        
        except aiohttp.ClientConnectorError as e:
            result.error = f"Connection error: {e}"
        except asyncio.TimeoutError:
            result.error = "Health check timeout"
        except Exception as e:
            result.error = f"Health check error: {e}"
            
        return result
    
    async def check_service_stats(self) -> Dict[str, Any]:
        """Получить детальную статистику сервиса"""
        stats_result = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "stats_data": None,
            "error": None
        }
        
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    self.stats_endpoint,
                    headers={'User-Agent': 'VM-Stats-Monitor/1.0'}
                ) as response:
                    
                    if response.status == 200:
                        stats_data = await response.json()
                        stats_result.update({
                            "success": True,
                            "stats_data": stats_data
                        })
                    else:
                        response_text = await response.text()
                        stats_result["error"] = f"Stats HTTP {response.status}: {response_text[:100]}"
                        
        except aiohttp.ClientConnectorError as e:
            stats_result["error"] = f"Stats connection error: {e}"
        except asyncio.TimeoutError:
            stats_result["error"] = "Stats timeout"
        except Exception as e:
            stats_result["error"] = f"Stats error: {e}"
            
        return stats_result
    
    def calculate_service_metrics(self, window_minutes: int = 30) -> ServiceMetrics:
        """Рассчитать метрики производительности за определенное окно времени"""
        if not self.health_history:
            return ServiceMetrics(
                avg_response_time_ms=0.0,
                min_response_time_ms=0.0,
                max_response_time_ms=0.0,
                success_rate=0.0,
                error_count=0,
                total_checks=0,
                uptime_percentage=0.0,
                last_error=None,
                trend_direction="unknown"
            )
        
        # Фильтруем данные за последние N минут
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_checks = [
            check for check in self.health_history 
            if datetime.fromisoformat(check.timestamp) > cutoff_time
        ]
        
        if not recent_checks:
            recent_checks = list(self.health_history)
        
        # Рассчитываем метрики
        total_checks = len(recent_checks)
        successful_checks = sum(1 for check in recent_checks if check.success)
        failed_checks = total_checks - successful_checks
        
        response_times = [check.response_time_ms for check in recent_checks if check.success]
        
        avg_response_time = statistics.mean(response_times) if response_times else 0.0
        min_response_time = min(response_times) if response_times else 0.0
        max_response_time = max(response_times) if response_times else 0.0
        
        success_rate = (successful_checks / total_checks * 100) if total_checks > 0 else 0.0
        uptime_percentage = success_rate
        
        # Находим последнюю ошибку
        last_error = None
        for check in reversed(recent_checks):
            if not check.success and check.error:
                last_error = check.error
                break
        
        # Определяем тренд производительности
        trend_direction = self._calculate_trend(recent_checks)
        
        return ServiceMetrics(
            avg_response_time_ms=avg_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            success_rate=success_rate,
            error_count=failed_checks,
            total_checks=total_checks,
            uptime_percentage=uptime_percentage,
            last_error=last_error,
            trend_direction=trend_direction
        )
    
    def _calculate_trend(self, checks: List[HealthCheckResult]) -> str:
        """Вычислить тренд производительности"""
        if len(checks) < 10:
            return "insufficient_data"
        
        # Берем первую и вторую половину данных
        mid_point = len(checks) // 2
        first_half = checks[:mid_point]
        second_half = checks[mid_point:]
        
        # Считаем среднее время отклика для каждой половины
        first_avg = statistics.mean([c.response_time_ms for c in first_half if c.success])
        second_avg = statistics.mean([c.response_time_ms for c in second_half if c.success])
        
        if first_avg == 0 or second_avg == 0:
            return "insufficient_successful_data"
        
        # Определяем тренд
        change_percent = (second_avg - first_avg) / first_avg * 100
        
        if abs(change_percent) < 10:  # Менее 10% изменения
            return "stable"
        elif change_percent > 10:
            return "degrading"  # Время отклика увеличилось
        else:
            return "improving"  # Время отклика уменьшилось
    
    def start_monitoring(self, callback: Optional[Callable[[HealthCheckResult], None]] = None):
        """Запустить непрерывный мониторинг"""
        if self.is_monitoring:
            logger.warning("Мониторинг уже запущен")
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        def monitoring_loop():
            logger.info(f"Запуск мониторинга VM {self.vm_host}:{self.vm_port}")
            
            while not self.stop_event.is_set():
                try:
                    # Выполняем проверку здоровья
                    health_result = asyncio.run(self.check_health_detailed())
                    
                    # Добавляем в историю
                    self.health_history.append(health_result)
                    
                    # Обновляем счетчики
                    self.total_checks += 1
                    if health_result.success:
                        self.successful_checks += 1
                    else:
                        self.failed_checks += 1
                    
                    # Вызываем callback если предоставлен
                    if callback:
                        callback(health_result)
                    
                    # Логируем результат
                    status = "✅ HEALTHY" if health_result.success else "❌ UNHEALTHY"
                    logger.info(
                        f"{status} {health_result.timestamp} "
                        f"({health_result.response_time_ms:.1f}ms)"
                    )
                    
                    if not health_result.success:
                        logger.warning(f"Health check failed: {health_result.error}")
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                
                # Ждем до следующей проверки
                if not self.stop_event.wait(self.check_interval):
                    continue  # Продолжаем если не получили stop сигнал
                else:
                    break  # Выходим если получили stop сигнал
            
            logger.info("Мониторинг остановлен")
        
        self.monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Остановить мониторинг"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Получить отчет о мониторинге"""
        metrics = self.calculate_service_metrics()
        
        report = {
            "vm_host": self.vm_host,
            "vm_port": self.vm_port,
            "monitoring_status": "active" if self.is_monitoring else "stopped",
            "report_timestamp": datetime.now().isoformat(),
            "metrics": asdict(metrics),
            "total_history_records": len(self.health_history),
            "monitoring_duration_minutes": None
        }
        
        # Вычисляем продолжительность мониторинга
        if self.health_history:
            first_check = datetime.fromisoformat(self.health_history[0].timestamp)
            last_check = datetime.fromisoformat(self.health_history[-1].timestamp)
            duration = last_check - first_check
            report["monitoring_duration_minutes"] = duration.total_seconds() / 60
        
        return report
    
    def export_health_history(self, filename: str) -> None:
        """Экспортировать историю проверок в JSON файл"""
        history_data = {
            "export_timestamp": datetime.now().isoformat(),
            "vm_host": self.vm_host,
            "vm_port": self.vm_port,
            "total_records": len(self.health_history),
            "health_checks": [check.to_dict() for check in self.health_history]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"История здоровья экспортирована в {filename}")


# Pytest тесты для health monitoring
@pytest.mark.integration
class TestVMHealthMonitoring:
    """Pytest класс для тестирования мониторинга здоровья VM"""
    
    @pytest.fixture
    def health_monitor(self):
        """Фикстура для создания health monitor"""
        return VMHealthMonitor(check_interval=5, max_history=20)
    
    @pytest.mark.asyncio
    async def test_detailed_health_check(self, health_monitor):
        """Тест детальной проверки здоровья"""
        result = await health_monitor.check_health_detailed()
        
        assert isinstance(result, HealthCheckResult)
        assert result.timestamp is not None
        assert result.response_time_ms >= 0
        
        # Если сервис доступен, проверяем детали
        if result.success:
            assert result.status_code == 200
            assert result.service_status is not None
            assert result.embedder_status is not None
            assert result.vector_store_status is not None
    
    @pytest.mark.asyncio
    async def test_service_stats_check(self, health_monitor):
        """Тест получения статистики сервиса"""
        stats_result = await health_monitor.check_service_stats()
        
        assert "timestamp" in stats_result
        assert "success" in stats_result
        
        # Если статистика доступна, проверяем структуру
        if stats_result["success"]:
            assert stats_result["stats_data"] is not None
            assert isinstance(stats_result["stats_data"], dict)
    
    def test_metrics_calculation_empty_history(self, health_monitor):
        """Тест расчета метрик при пустой истории"""
        metrics = health_monitor.calculate_service_metrics()
        
        assert metrics.total_checks == 0
        assert metrics.success_rate == 0.0
        assert metrics.avg_response_time_ms == 0.0
    
    def test_metrics_calculation_with_data(self, health_monitor):
        """Тест расчета метрик с тестовыми данными"""
        # Добавляем тестовые данные
        test_checks = [
            HealthCheckResult(
                timestamp=datetime.now().isoformat(),
                success=True,
                response_time_ms=100.0,
                status_code=200,
                error=None,
                service_status="healthy",
                embedder_status="ready",
                vector_store_status="connected",
                memory_usage=None
            ),
            HealthCheckResult(
                timestamp=datetime.now().isoformat(),
                success=False,
                response_time_ms=0.0,
                status_code=None,
                error="Connection error",
                service_status=None,
                embedder_status=None,
                vector_store_status=None,
                memory_usage=None
            ),
            HealthCheckResult(
                timestamp=datetime.now().isoformat(),
                success=True,
                response_time_ms=150.0,
                status_code=200,
                error=None,
                service_status="healthy",
                embedder_status="ready",
                vector_store_status="connected",
                memory_usage=None
            )
        ]
        
        for check in test_checks:
            health_monitor.health_history.append(check)
        
        metrics = health_monitor.calculate_service_metrics()
        
        assert metrics.total_checks == 3
        assert metrics.success_rate == 100 * (2/3)  # 2 успешных из 3
        assert metrics.error_count == 1
        assert metrics.avg_response_time_ms == 125.0  # (100 + 150) / 2
        assert metrics.last_error == "Connection error"
    
    def test_monitoring_start_stop(self, health_monitor):
        """Тест запуска и остановки мониторинга"""
        assert not health_monitor.is_monitoring
        
        # Запускаем мониторинг
        health_monitor.start_monitoring()
        assert health_monitor.is_monitoring
        assert health_monitor.monitor_thread is not None
        
        # Даем мониторингу поработать немного
        time.sleep(2)
        
        # Останавливаем мониторинг
        health_monitor.stop_monitoring()
        assert not health_monitor.is_monitoring
    
    def test_monitoring_report_generation(self, health_monitor):
        """Тест генерации отчета мониторинга"""
        report = health_monitor.get_monitoring_report()
        
        assert "vm_host" in report
        assert "vm_port" in report
        assert "monitoring_status" in report
        assert "report_timestamp" in report
        assert "metrics" in report
        
        assert report["vm_host"] == health_monitor.vm_host
        assert report["vm_port"] == health_monitor.vm_port
    
    def test_health_history_export(self, health_monitor, tmp_path):
        """Тест экспорта истории проверок"""
        # Добавляем тестовую запись
        test_check = HealthCheckResult(
            timestamp=datetime.now().isoformat(),
            success=True,
            response_time_ms=100.0,
            status_code=200,
            error=None,
            service_status="healthy",
            embedder_status="ready",  
            vector_store_status="connected",
            memory_usage=None
        )
        health_monitor.health_history.append(test_check)
        
        # Экспортируем в временный файл
        export_file = tmp_path / "health_history_test.json"
        health_monitor.export_health_history(str(export_file))
        
        # Проверяем что файл создался и содержит данные
        assert export_file.exists()
        
        with open(export_file, 'r', encoding='utf-8') as f:
            exported_data = json.load(f)
        
        assert "export_timestamp" in exported_data
        assert "health_checks" in exported_data
        assert len(exported_data["health_checks"]) == 1
        assert exported_data["health_checks"][0]["success"] is True


# Утилиты для мониторинга
class VMHealthAlerts:
    """Класс для отправки уведомлений о проблемах с VM"""
    
    def __init__(self, alert_threshold_failures: int = 3, 
                 alert_threshold_response_time: float = 5000.0):
        self.alert_threshold_failures = alert_threshold_failures
        self.alert_threshold_response_time = alert_threshold_response_time
        self.consecutive_failures = 0
        self.alerts_sent = []
    
    def check_for_alerts(self, health_result: HealthCheckResult) -> List[str]:
        """Проверить нужно ли отправлять уведомления"""
        alerts = []
        
        if not health_result.success:
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= self.alert_threshold_failures:
                alert_msg = (
                    f"🚨 VM Service Alert: {self.consecutive_failures} consecutive failures. "
                    f"Last error: {health_result.error}"
                )
                alerts.append(alert_msg)
                self.alerts_sent.append({
                    "timestamp": health_result.timestamp,
                    "type": "consecutive_failures",
                    "message": alert_msg
                })
        else:
            # Сбрасываем счетчик при успешной проверке
            self.consecutive_failures = 0
            
            # Проверяем медленный отклик
            if health_result.response_time_ms > self.alert_threshold_response_time:
                alert_msg = (
                    f"⚠️ VM Service Slow Response: {health_result.response_time_ms:.1f}ms "
                    f"(threshold: {self.alert_threshold_response_time}ms)"
                )
                alerts.append(alert_msg)
                self.alerts_sent.append({
                    "timestamp": health_result.timestamp,
                    "type": "slow_response",
                    "message": alert_msg
                })
        
        return alerts


if __name__ == "__main__":
    # Пример использования мониторинга
    print("🏥 Запуск VM Health Monitor")
    
    monitor = VMHealthMonitor(check_interval=10)
    alerts = VMHealthAlerts()
    
    def health_callback(result: HealthCheckResult):
        """Callback для обработки результатов проверки здоровья"""
        alert_messages = alerts.check_for_alerts(result)
        for alert in alert_messages:
            print(f"📢 ALERT: {alert}")
    
    try:
        # Запускаем мониторинг на 60 секунд
        monitor.start_monitoring(callback=health_callback)
        print("Мониторинг запущен на 60 секунд...")
        time.sleep(60)
        
    finally:
        monitor.stop_monitoring()
        
        # Показываем отчет
        report = monitor.get_monitoring_report()
        print(f"\n📊 Отчет о мониторинге:")
        print(f"Всего проверок: {report['metrics']['total_checks']}")
        print(f"Успешность: {report['metrics']['success_rate']:.1f}%")
        print(f"Среднее время отклика: {report['metrics']['avg_response_time_ms']:.1f}ms")
        print(f"Тренд: {report['metrics']['trend_direction']}")
        
        # Экспортируем историю
        export_file = f"vm_health_history_{int(time.time())}.json"
        monitor.export_health_history(export_file)
        print(f"💾 История сохранена в {export_file}")
