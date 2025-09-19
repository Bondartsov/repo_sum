# VM Testing Suite - Система тестирования VM для RAG-as-a-Service

Этот каталог содержит комплексную систему тестирования и диагностики для VM, на которой работает RAG-as-a-Service система с Jina v3 эмбеддингами.

## 🎯 Цель

Система диагностики предназначена для:
- **Выявления проблем подключения** между локальным клиентом и VM
- **Мониторинга состояния** RAG сервиса на VM
- **Диагностики firewall** и сетевых настроек
- **Автоматического предложения исправлений** найденных проблем

## 📁 Структура файлов

```
tests/vm/
├── README.md                      # Этот файл - документация
├── vm_diagnostics.py              # 🚀 Главный диагностический инструмент
├── vm_utils.py                    # 🛠️ Вспомогательные утилиты
├── test_vm_connectivity.py        # 📡 Тестирование сетевого подключения
├── test_vm_health_monitoring.py   # 🏥 Мониторинг здоровья сервиса
└── test_vm_firewall_config.py     # 🔥 Тестирование firewall конфигурации
```

## 🚀 Быстрый старт

### 1. Базовые требования

```bash
# Убедитесь что у вас установлены зависимости
pip install -r requirements.txt

# Настройте переменную окружения для SSH доступа к VM
export VM_PASSWORD="ваш_пароль_от_vm"
# Или для Windows:
set VM_PASSWORD=ваш_пароль_от_vm
```

### 2. Быстрая диагностика (рекомендуется)

```bash
# Переходим в директорию VM тестов
cd tests/vm

# Запускаем быструю диагностику
python vm_diagnostics.py
```

### 3. Полная диагностика

```bash
# Для комплексной проверки всех компонентов
python vm_diagnostics.py --comprehensive

# Для мониторинга состояния на 5 минут
python vm_diagnostics.py --monitor 5

# Для диагностики с автоматическими исправлениями
python vm_diagnostics.py --comprehensive --fix
```

## 🔧 Основные команды

### vm_diagnostics.py - Главный диагностический инструмент

```bash
# Основные режимы
python vm_diagnostics.py                    # Быстрая диагностика (по умолчанию)
python vm_diagnostics.py --comprehensive    # Полная диагностика всех компонентов
python vm_diagnostics.py --monitor 10       # Мониторинг на 10 минут

# Дополнительные опции
python vm_diagnostics.py --fix              # Показать команды для исправления
python vm_diagnostics.py --export report.json  # Экспорт в файл
python vm_diagnostics.py --verbose          # Подробный вывод

# Настройка VM
python vm_diagnostics.py --vm-host 10.61.11.55 --vm-port 8001
```

### Отдельные тестовые модули

```bash
# Тестирование подключения (можно запустить отдельно)
python test_vm_connectivity.py

# Мониторинг здоровья сервиса
python test_vm_health_monitoring.py

# Проверка firewall конфигурации (требует SSH доступ)
python test_vm_firewall_config.py

# Демонстрация утилит
python vm_utils.py
```

### Pytest интеграция

```bash
# Запуск всех VM тестов через pytest
pytest tests/vm/ -v

# Запуск только integration тестов
pytest tests/vm/ -m integration -v

# Запуск с подробным выводом
pytest tests/vm/ -v -s
```

## 📊 Интерпретация результатов

### Типичные результаты быстрой диагностики

```
🚀 Запуск быстрой диагностики VM...
=======================================================
1️⃣ Проверка TCP подключения...
   ✅ PASS TCP Connection
2️⃣ Проверка HTTP health endpoint...
   ✅ PASS HTTP Health
3️⃣ Проверка ping связности...
   ✅ PASS Ping Test
4️⃣ Проверка внешнего подключения...
   ✅ PASS External Connectivity

=======================================================
📊 РЕЗУЛЬТАТЫ БЫСТРОЙ ДИАГНОСТИКИ
=======================================================
Всего тестов: 4
Успешно: 4 ✅
Неудачно: 0 ❌
Успешность: 100.0%
```

### Если есть проблемы

```
❌ FAIL TCP Connection
     Error: Connection refused - service not available

🚨 Критические проблемы:
  • TCP Connection: Connection refused - service not available

💡 Быстрые исправления:
  • VM может быть недоступна - проверьте ping
  • RAG сервис может быть остановлен
  • Firewall может блокировать подключения
```

## 🔧 Решение типичных проблем

### 1. Connection Refused

**Проблема**: `Cannot connect to host 10.61.11.54:8000`

**Решения**:
```bash
# 1. Проверить доступность VM
ping 10.61.11.54

# 2. Проверить статус RAG сервиса на VM (через SSH)
ssh user@10.61.11.54
ps aux | grep vm_rag_service

# 3. Перезапустить сервис
cd ~/repo_sum_rag/repo_sum && source venv/bin/activate
python vm_rag_service.py

# 4. Проверить firewall
sudo ufw allow 8000/tcp
sudo ufw reload
```

### 2. SSH Authentication Failed

**Проблема**: Тесты firewall не работают из-за SSH

**Решения**:
```bash
# Установить переменную окружения
export VM_PASSWORD="ваш_пароль"

# Проверить SSH доступ вручную
ssh user@10.61.11.54

# Некоторые тесты будут пропущены без SSH доступа
```

### 3. Медленные ответы

**Проблема**: Response time > 5000ms

**Решения**:
```bash
# Проверить нагрузку на VM
ssh user@10.61.11.54
top
free -h

# Перезапустить RAG сервис
cd ~/repo_sum_rag/repo_sum
python vm_start.py restart
```

## 📁 Создаваемые файлы

Диагностические инструменты создают следующие файлы:

```
vm_reports/                                    # Каталог отчетов
├── vm_quick_diagnostics_report_1234567890.json      # Быстрая диагностика
├── vm_comprehensive_diagnostics_report_1234567890.json  # Полная диагностика
├── vm_connectivity_test_1234567890.json             # Тесты подключения
├── vm_firewall_test_1234567890.json                 # Тесты firewall
└── vm_monitoring_history_1234567890.json            # История мониторинга

vm_diagnostics.log                             # Лог диагностики
vm_tests.log                                   # Лог тестов
```

## 🏥 Мониторинг в реальном времени

Для непрерывного мониторинга состояния VM:

```bash
# Мониторинг на 30 минут с алертами
python vm_diagnostics.py --monitor 30

# Пример вывода:
[14:30:15] ✅ HEALTHY 245.2ms | Service: healthy | Embedder: ready
[14:30:25] ✅ HEALTHY 198.7ms | Service: healthy | Embedder: ready
[14:30:35] ❌ UNHEALTHY 5000.0ms | Service: timeout
🚨 VM Service Alert: 3 consecutive failures. Last error: Health check timeout
```

## 🧪 Интеграция с pytest

Все тесты интегрированы с pytest и могут запускаться в составе общего test suite:

```bash
# Запуск всех VM тестов
pytest tests/vm/ -v

# Только быстрые тесты (без SSH)
pytest tests/vm/ -v -k "not ssh"

# Только integration тесты
pytest tests/vm/ -m integration -v
```

### Pytest маркеры

- `@pytest.mark.integration` - Интеграционные тесты (требуют VM доступ)
- `@pytest.mark.skipif(not os.getenv("VM_PASSWORD"))` - Тесты требующие SSH доступ

## 🛠️ Разработка и расширение

### Добавление новых тестов

1. **Создайте новый тестовый класс**:
```python
class VMCustomTester:
    def __init__(self, vm_host: str = "10.61.11.54", vm_port: int = 8000):
        self.vm_host = vm_host
        self.vm_port = vm_port
    
    def test_custom_functionality(self) -> Dict[str, Any]:
        # Ваша логика тестирования
        return {
            "test_name": "Custom Test",
            "success": True,
            "error": None,
            "details": {}
        }
```

2. **Интегрируйте в vm_diagnostics.py**:
```python
from your_module import VMCustomTester

# В методе run_comprehensive_diagnostics():
custom_tester = VMCustomTester(self.vm_host, self.vm_port)
custom_results = custom_tester.test_custom_functionality()
comprehensive_results["test_categories"]["custom"] = custom_results
```

### Использование утилит

Модуль `vm_utils.py` содержит полезные утилиты:

```python
from vm_utils import NetworkUtils, ProcessUtils, FileUtils, VMConfig

# Проверка порта
is_open, time_ms, error = NetworkUtils.check_port_open("10.61.11.54", 8000)

# Ping хоста
ping_result = NetworkUtils.ping_host("10.61.11.54", count=4)

# Сохранение отчета
FileUtils.save_json_report(data, "my_report.json")

# Конфигурация VM
config = VMConfig(host="10.61.11.55", port=8001)
```

## 📚 API Reference

### VMDiagnosticSuite

Основной класс для диагностики VM:

```python
suite = VMDiagnosticSuite("10.61.11.54", 8000)

# Быстрая диагностика (1-2 минуты)
quick_results = suite.run_quick_diagnostics()

# Полная диагностика (5-10 минут)
full_results = suite.run_comprehensive_diagnostics()

# Мониторинг (указанное время)
suite.start_monitoring_mode(duration_minutes=10)

# Генерация исправлений
fixes = suite.generate_fix_suggestions(results)

# Экспорт отчета
report_file = suite.export_diagnostic_report(results)
```

### VMConnectivityTester

Тестирование сетевого подключения:

```python
tester = VMConnectivityTester("10.61.11.54", 8000)

# Все тесты подключения
results = tester.run_all_connectivity_tests()

# Отдельные тесты
tcp_result = tester.test_basic_tcp_connection()
http_result = tester.test_http_health_endpoint()
ping_result = tester.test_ping_connectivity()
```

### VMHealthMonitor

Мониторинг здоровья сервиса:

```python
monitor = VMHealthMonitor("10.61.11.54", 8000, check_interval=30)

# Детальная проверка здоровья
health_result = await monitor.check_health_detailed()

# Запуск мониторинга
monitor.start_monitoring(callback=my_callback)

# Получение отчета
report = monitor.get_monitoring_report()

# Экспорт истории
monitor.export_health_history("health_history.json")
```

### VMFirewallTester

Тестирование firewall конфигурации:

```python
tester = VMFirewallTester("10.61.11.54", 8000, ssh_user="user")

# Комплексное тестирование firewall
firewall_results = tester.run_comprehensive_firewall_test()

# Отдельные тесты
ufw_result = tester.test_ufw_status()
iptables_result = tester.test_iptables_rules()
port_listening_result = tester.test_port_listening()
```

## 🚨 Troubleshooting

### Логи и отладка

1. **Включить подробное логирование**:
```bash
python vm_diagnostics.py --verbose
```

2. **Проверить лог файлы**:
```bash
tail -f vm_diagnostics.log
tail -f vm_tests.log
```

3. **Проверить переменные окружения**:
```bash
echo $VM_PASSWORD
# Или для Windows:
echo %VM_PASSWORD%
```

### Частые ошибки

| Ошибка | Причина | Решение |
|--------|---------|---------|
| `ImportError: No module named 'test_vm_connectivity'` | Неправильная директория | `cd tests/vm` |
| `VM_PASSWORD не найден` | Не установлена переменная | `export VM_PASSWORD="пароль"` |
| `Connection refused` | VM недоступна или сервис остановлен | Проверить VM и перезапустить сервис |
| `SSH connection failed` | Неправильный пароль или хост | Проверить SSH доступ вручную |

## 📞 Поддержка

Если у вас есть вопросы или проблемы:

1. **Проверьте FAQ** в этом документе
2. **Запустите полную диагностику** с `--fix` флагом
3. **Проверьте логи** `vm_diagnostics.log` и `vm_tests.log`
4. **Создайте issue** с подробным описанием проблемы и выводом диагностики

---

*Автор: Claude (Cline) для диагностики VM RAG-as-a-Service*  
*Дата: 19 сентября 2025*  
*Версия: 1.0*
