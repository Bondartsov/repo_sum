# 🔧 Implementation Plan: Исправление проблем таймаутов и Qdrant connectivity

**Дата:** 1 октября 2025  
**Версия:** 1.0.0  
**Статус:** Ready for Implementation  
**Автор:** AI Assistant

---

## 📋 Executive Summary

Обнаружены **2 критические проблемы** в RAG системе:

1. **Проблема индексации (P0)**: TimeoutError при индексации из-за конфликта outer timeout (30s) и retry логики (3 × 30s)
2. **Проблема Qdrant status (P0)**: Vector Store показывает статус "error" при `python main.py rag status --detailed`

Обе проблемы **НЕ ЗАДОКУМЕНТИРОВАНЫ** в Technical Debt.md и требуют немедленного исправления.

---

## 🔍 ПРОБЛЕМА #1: Таймауты при индексации RAG

### Root Cause Analysis

#### Трейсбек:
```
2025-10-01 11:01:58,639 - rag.remote_embedder - ERROR - TimeoutError в remote_embedder.py:109
EmbeddingException: Удалённый сервис эмбеддингов недоступен (провайдер: remote-vm)
```

#### Цепочка вызовов:
```
indexer_service.py:423
└─> indexer._index_chunks_batch()
    └─> self.embedder.embed_texts(texts, task=passage_task)
        └─> remote_embedder.py:94 embed_texts()
            └─> run_async_safe(self._async_embed_texts(...), timeout=30s)  ❌ OUTER TIMEOUT
                └─> _async_embed_texts()
                    └─> _make_request_with_retry()
                        ├─ Попытка 1: HTTP запрос (30s timeout)
                        ├─ Задержка 2s
                        ├─ Попытка 2: HTTP запрос (30s timeout)  
                        ├─ Задержка 4s (exponential backoff)
                        └─ Попытка 3: HTTP запрос (30s timeout)
                        ИТОГО: потенциально 96+ секунд!
```

#### Конфликт таймаутов:

| Уровень | Timeout | Проблема |
|---------|---------|----------|
| `run_async_safe()` | 30s | ❌ Outer timeout слишком короткий |
| HTTP request × 3 | 30s каждый | ⚠️ Не учитывается остаточное время |
| Exponential backoff | 2s, 4s, 8s | ⚠️ Добавляет задержки |
| **ИТОГО** | **96s** | **❌ Превышает outer timeout!** |

### Файлы с проблемами:

- `rag/remote_embedder.py:94-117` - outer timeout конфликт
- `rag/remote_embedder.py:167-223` - retry без учёта оставшегося времени
- `config.py:261-267` - `timeout_seconds: 60` не используется!
- `settings.json:33-37` - конфигурация неоптимальна

---

## 🔍 ПРОБЛЕМА #2: Qdrant Vector Store статус "error"

### Симптомы:
```bash
$ python main.py rag status --detailed
Компонент: Qdrant Vector Store
Статус: [red]error[/red]  ❌
Детали: HTTP 500: Internal Server Error
```

### Возможные причины:

#### Причина A: VM сервис недоступен (10.61.11.54:8000)
```
Признаки:
- ClientConnectorError: Connection refused
- Timeout при попытке подключения

Решение:
1. Запустить VM: python vm_start.py start
2. Проверить доступность: curl http://10.61.11.54:8000/health
3. Проверить firewall правила
```

#### Причина B: VM работает, но Qdrant внутри недоступна
```
Признаки:
- VM отвечает 200 OK
- collection_status: "unavailable"
- qdrant_status: "disconnected"

Решение:
1. Проверить Qdrant на VM: systemctl status qdrant (или docker ps)
2. Проверить логи Qdrant на VM
3. Проверить конфигурацию Qdrant (порт 6333)
```

#### Причина C: HTTP timeout при health check
```
Признаки:
- asyncio.TimeoutError в health_check
- Медленный ответ от VM (>30s)

Решение:
1. Увеличить timeout в RemoteServiceConfig
2. Оптимизировать health check endpoint на VM
3. Проверить сетевую latency между клиентом и VM
```

#### Причина D: Некорректный формат ответа от VM
```
Признаки:
- ValueError: Invalid JSON
- KeyError при парсинге response

Решение:
1. Проверить версию vm_rag_service.py на VM
2. Обновить контракт API между клиентом и VM
3. Добавить валидацию response schema
```

### Файлы с проблемами:

- `rag/remote_vector_store.py:154-195` - `_async_health_check()` 
- `rag/indexer_service.py:520-579` - `print_health_status()` отображение
- `main.py:858-959` - `rag status` команда
- `vm_rag_service.py` (на VM) - health endpoint implementation

---

## 🎯 IMPLEMENTATION ROADMAP

### ✅ **Фаза 0: Подготовка и документирование (1-2 часа)** ✅ ЗАВЕРШЕНО

- [x] 0.1 Обновить `rules/Technical Debt.md` ✅
  - [x] Добавить секцию "**8. Критическая проблема таймаутов при RAG индексации**"
  - [x] Добавить секцию "**9. Qdrant Vector Store показывает статус error**"
  - [x] Задокументировать root cause и симптомы
  - [x] Указать приоритет P0 для обеих проблем

- [x] 0.2 Запустить диагностические тесты ✅
  ```bash
  pytest tests/rag/test_vm_qdrant_connectivity.py -v --tb=short
  ```
  - [x] Проанализировать результаты - **Обнаружена проблема с async mock**
  - [x] Определить точную причину Qdrant error - **Причина A: VM недоступна (connection refused)**
  - [x] **БОНУС:** Исправить mock в test_vm_qdrant_connectivity.py (Mock вместо AsyncMock)
  - [x] **БОНУС:** Запустить тесты повторно - **10 passed in 11.59s** ✅

- [x] 0.3 Проверить статус VM вручную ✅
  ```bash
  # Проверка доступности VM
  curl http://10.61.11.54:8000/health
  # Результат: Connection refused - VM недоступна
  
  # Проверка Qdrant на VM (если доступ есть)
  ssh user@10.61.11.54
  docker ps | grep qdrant  # или systemctl status qdrant
  ```

---

### 🔥 **Фаза 1: Экстренные исправления - CRITICAL (День 1, 4-6 часов)** ✅ ЗАВЕРШЕНО (код реализован)

**Цель:** Разблокировать индексацию НЕМЕДЛЕННО

#### 1.1 Исправление таймаутов в remote_embedder.py ✅ РЕАЛИЗОВАНО

- [x] 1.1.1 Гармонизация timeout конфигурации ✅
  
  **Файл:** `rag/remote_embedder.py:94-128`
  
  **Реализовано:**
  ```python
  def embed_texts(self, texts, task=None, deadline_ms=None):
      # Используем конфигурацию вместо hardcode (1.1.3)
      if deadline_ms is None:
          deadline_ms = self.timeout_seconds * 1000  # 60s из config
      
      base_timeout = deadline_ms / 1000.0
      # Формула: base × retries + sum(delay × 2^i)
      backoff_total = sum(self.retry_delay * (2 ** i) for i in range(self.max_retries))
      total_timeout = (base_timeout * self.max_retries) + backoff_total
      # Результат: 194s = (60s × 3) + (2s + 4s + 8s)
      
      return run_async_safe(
          self._async_embed_texts(texts, task=task, deadline_ms=deadline_ms),
          timeout=total_timeout  # 194s гармонизированный timeout
      )
  ```
  **Статус:** ✅ Outer timeout (194s) теперь >= Inner timeout (180s + backoff)

- [x] 1.1.2 Добавить tracking оставшегося времени в retry ✅
  
  **Файл:** `rag/remote_embedder.py:167-290`
  
  **Реализовано:**
  ```python
  async def _make_request_with_retry(self, payload, deadline_ms):
      import time
      start_time = time.time()
      total_timeout = deadline_ms / 1000.0
      base_timeout = deadline_ms / 1000.0
      
      for attempt in range(self.max_retries):
          # ✅ Tracking оставшегося времени
          elapsed = time.time() - start_time
          remaining = total_timeout - elapsed
          
          if remaining <= 0:
              raise VMTimeoutError(
                  f"Исчерпано время retry: {elapsed:.1f}s из {total_timeout:.1f}s",
                  timeout_seconds=total_timeout,
                  elapsed_seconds=elapsed,
                  retry_attempt=attempt
              )
          
          # ✅ Адаптивный request timeout
          request_timeout_seconds = min(base_timeout, remaining)
          
          try:
              async with session.post(..., 
                  timeout=aiohttp.ClientTimeout(total=request_timeout_seconds)):
                  # ...
          except aiohttp.ClientConnectorError as e:
              # ✅ Специфичная обработка connection errors (1.2.2)
              if attempt < self.max_retries - 1:
                  delay = min(self.retry_delay * (2 ** attempt), remaining / 2)
                  if delay > 0:
                      await asyncio.sleep(delay)
                  else:
                      raise VMConnectionError(...)
              else:
                  raise VMConnectionError(...)
          except asyncio.TimeoutError as e:
              # ✅ Специфичная обработка timeout (1.2.2)
              if attempt < self.max_retries - 1:
                  delay = min(self.retry_delay * (2 ** attempt), remaining / 2)
                  if delay > 0:
                      await asyncio.sleep(delay)
                  else:
                      raise VMTimeoutError(...)
              else:
                  raise VMTimeoutError(...)
  ```
  **Статус:** ✅ Adaptive timeout + retry time tracking реализованы

- [x] 1.1.3 Использовать `remote_service.timeout_seconds` из конфигурации ✅
  
  **Файл:** `rag/remote_embedder.py:94-128`
  
  **Реализовано:** См. код выше в 1.1.1
  ```python
  if deadline_ms is None:
      deadline_ms = self.timeout_seconds * 1000  # 60s из config вместо hardcode 30s
  ```
  **Статус:** ✅ Config используется вместо hardcode

#### 1.2 Улучшение error handling ✅ РЕАЛИЗОВАНО

- [x] 1.2.1 Создать специфичные исключения ✅
  
  **Файл:** `rag/exceptions.py`
  
  **Реализовано:**
  ```python
  class VMConnectionError(EmbeddingException):
      """VM сервис недоступен (connection refused, DNS, firewall, etc.)"""
      def __init__(self, message: str, vm_host: str, vm_port: int, 
                   error_details: str = None, **kwargs):
          self.vm_host = vm_host
          self.vm_port = vm_port
          self.error_details = error_details
          super().__init__(message, **kwargs)
      
      def get_diagnostic_info(self) -> dict:
          return {
              "error_type": "vm_connection_error",
              "vm_host": self.vm_host,
              "vm_port": self.vm_port,
              "recommendation": "Проверьте: 1) VM запущена, 2) Firewall правила...",
              "troubleshooting_commands": [
                  f"curl http://{self.vm_host}:{self.vm_port}/health",
                  f"ping {self.vm_host}",
                  "python vm_start.py start"
              ]
          }
  
  class VMTimeoutError(EmbeddingException):
      """VM сервис превысил таймаут"""
      def __init__(self, message: str, timeout_seconds: float, 
                   elapsed_seconds: float = None, operation: str = "embedding",
                   retry_attempt: int = None, **kwargs):
          self.timeout_seconds = timeout_seconds
          self.elapsed_seconds = elapsed_seconds
          self.operation = operation
          self.retry_attempt = retry_attempt
          super().__init__(message, **kwargs)
      
      def get_diagnostic_info(self) -> dict:
          return {
              "error_type": "vm_timeout_error",
              "timeout_seconds": self.timeout_seconds,
              "elapsed_seconds": self.elapsed_seconds,
              "suggested_actions": [
                  "Увеличить remote_service.timeout_seconds в settings.json",
                  "Проверить latency: ping VM_IP",
                  "Проверить load на VM"
              ]
          }
  ```
  **Статус:** ✅ Два специфичных exception с diagnostic info

- [x] 1.2.2 Улучшить обработку ошибок в `embed_texts` ✅
  
  **Файл:** `rag/remote_embedder.py:167-290`
  
  **Реализовано:** Обработка встроена в `_make_request_with_retry` (см. 1.1.2):
  ```python
  except aiohttp.ClientConnectorError as e:
      if attempt < self.max_retries - 1:
          # ... retry логика
      else:
          raise VMConnectionError(
              f"VM embedder недоступен после {attempt+1} попыток: {e}",
              vm_host=self.service_host,
              vm_port=self.service_port,
              error_details=str(e),
              provider=self.provider_name,
              model_name=self.model_name
          )
  
  except asyncio.TimeoutError as e:
      if attempt < self.max_retries - 1:
          # ... retry логика
      else:
          raise VMTimeoutError(
              f"VM embedder timeout после {elapsed:.1f}s ({attempt+1} попыток)",
              timeout_seconds=total_timeout,
              elapsed_seconds=elapsed,
              operation="embedding",
              retry_attempt=attempt,
              provider=self.provider_name,
              model_name=self.model_name
          )
  ```
  **Статус:** ✅ Специфичные exceptions с контекстом

#### 1.3 Исправление Qdrant health check ✅ РЕАЛИЗОВАНО

- [x] 1.3.1 Добавить детальную диагностику в `_async_health_check` ✅
  
  **Файл:** `rag/remote_vector_store.py:454-551`
  
  **Реализовано:**
  ```python
  async def _async_health_check(self) -> Dict[str, Any]:
      health_info = {
          "status": "unknown",
          "components": {...},
          "error": None,
          "diagnostic": None  # ✅ НОВОЕ ПОЛЕ
      }
      
      start_time = time.time()
      
      try:
          # ... health check логика ...
      except aiohttp.ClientConnectorError as e:
          response_time_ms = (time.time() - start_time) * 1000
          health_info["diagnostic"] = {
              "error_type": "connection_refused",
              "vm_host": self.service_host,
              "vm_port": self.service_port,
              "recommendation": "VM сервис недоступен. Запустите: python vm_start.py start",
              "troubleshooting_commands": [
                  f"curl http://{self.service_host}:{self.service_port}/health",
                  f"ping {self.service_host}",
                  "python vm_start.py start"
              ],
              "response_time_ms": response_time_ms
          }
      except asyncio.TimeoutError as e:
          health_info["diagnostic"] = {
              "error_type": "timeout",
              "recommendation": f"VM не отвечает > {timeout}s. Проверьте latency.",
              "troubleshooting_commands": [...]
          }
      except ValueError as e:
          health_info["diagnostic"] = {
              "error_type": "invalid_response",
              "recommendation": "Некорректный JSON от VM. Проверьте версию API.",
              "troubleshooting_commands": [...]
          }
      # ... обработка HTTP errors
  ```
  **Статус:** ✅ Diagnostic info для 4 типов ошибок

- [x] 1.3.2 Улучшить отображение ошибок в `main.py` ✅
  
  **Файл:** `main.py:~1100-1140` (команда `rag status`)
  
  **Реализовано:**
  ```python
  console.print(components_table)
  
  # ✅ Показываем diagnostic таблицу при ошибках
  if health.get('status') != 'healthy':
      if vs_health.get('diagnostic'):
          diagnostic = vs_health['diagnostic']
          diag_table = Table(title="🔍 Диагностика: Qdrant Vector Store")
          diag_table.add_column("Параметр", style="cyan", no_wrap=True)
          diag_table.add_column("Значение", style="yellow")
          
          diag_table.add_row("Тип ошибки", diagnostic.get('error_type'))
          diag_table.add_row("VM адрес", f"{diagnostic.get('vm_host')}:{diagnostic.get('vm_port')}")
          diag_table.add_row("Response time", f"{diagnostic['response_time_ms']:.0f}ms")
          diag_table.add_row("Рекомендация", diagnostic.get('recommendation'))
          
          console.print(diag_table)
          
          # ✅ Troubleshooting команды
          if 'troubleshooting_commands' in diagnostic:
              console.print("[bold]💡 Команды для диагностики:[/bold]")
              for cmd in diagnostic['troubleshooting_commands']:
                  console.print(f"  • [cyan]{cmd}[/cyan]")
  ```
  **Статус:** ✅ Диагностическая таблица + troubleshooting команды

#### 1.4 Тестирование экстренных исправлений ⚠️ ЧАСТИЧНО

- [x] 1.4.1 Запустить тесты Qdrant connectivity ✅
  ```bash
  pytest tests/rag/test_vm_qdrant_connectivity.py -v
  ```
  **Результат:** ✅ **10 passed in 11.59s** - Все mock сценарии работают

- [ ] 1.4.2 Протестировать индексацию с новыми таймаутами ⚠️
  ```bash
  python main.py rag index /path/to/small/repo --batch-size 8
  ```
  **Блокер:** VM недоступна (10.61.11.54:8000 connection refused)
  **Требуется:** Запустить VM или использовать альтернативный endpoint

- [ ] 1.4.3 Проверить статус с улучшенной диагностикой ⚠️
  ```bash
  python main.py rag status --detailed
  ```
  **Блокер:** VM недоступна
  **Можно проверить:** Корректное отображение diagnostic info при ошибке

---

### 🏗️ **Фаза 2: Structural Fixes - HIGH (День 2-3, 8-12 часов)**

**Цель:** Устранить корневые причины и предотвратить регрессии

#### 2.1 Адаптивная retry стратегия

- [ ] 2.1.1 Создать класс `RetryPolicy`
  
  **Файл:** `rag/retry_policy.py` (новый)
  
  **Создать:**
  ```python
  """
  Адаптивная retry стратегия для HTTP запросов к VM.
  """
  import asyncio
  import time
  from typing import Optional, Callable, TypeVar, List
  from dataclasses import dataclass
  
  T = TypeVar('T')
  
  @dataclass
  class RetryConfig:
      """Конфигурация retry политики"""
      max_attempts: int = 3
      base_delay: float = 2.0
      max_delay: float = 30.0
      exponential_base: float = 2.0
      timeout_seconds: float = 60.0
      retryable_exceptions: tuple = (
          asyncio.TimeoutError,
          aiohttp.ClientError,
      )
  
  class RetryPolicy:
      """
      Адаптивная retry политика с учётом оставшегося времени.
      """
      def __init__(self, config: RetryConfig):
          self.config = config
      
      async def execute_with_retry(
          self,
          func: Callable[..., T],
          *args,
          **kwargs
      ) -> T:
          """
          Выполняет функцию с retry логикой.
          """
          start_time = time.time()
          last_exception = None
          
          for attempt in range(self.config.max_attempts):
              # Проверяем оставшееся время
              elapsed = time.time() - start_time
              remaining = self.config.timeout_seconds - elapsed
              
              if remaining <= 0:
                  raise asyncio.TimeoutError(
                      f"Retry timeout: {elapsed:.1f}s / {self.config.timeout_seconds:.1f}s"
                  )
              
              try:
                  # Выполняем функцию с ограничением по оставшемуся времени
                  return await asyncio.wait_for(
                      func(*args, **kwargs),
                      timeout=remaining
                  )
              
              except self.config.retryable_exceptions as e:
                  last_exception = e
                  
                  if attempt < self.config.max_attempts - 1:
                      # Вычисляем задержку (exponential backoff)
                      delay = min(
                          self.config.base_delay * (self.config.exponential_base ** attempt),
                          self.config.max_delay,
                          remaining / 2  # Не больше половины оставшегося времени
                      )
                      
                      await asyncio.sleep(delay)
                  else:
                      # Последняя попытка - пробрасываем ошибку
                      raise
          
          # Не должно произойти, но для type safety
          raise last_exception if last_exception else RuntimeError("Retry failed")
  ```

- [ ] 2.1.2 Интегрировать `RetryPolicy` в `remote_embedder.py`
  
  **Файл:** `rag/remote_embedder.py`
  
  **Заменить `_make_request_with_retry` на:**
  ```python
  from .retry_policy import RetryPolicy, RetryConfig
  
  def __init__(self, ...):
      # ... существующая инициализация
      
      # Создаем retry policy
      self.retry_policy = RetryPolicy(RetryConfig(
          max_attempts=self.max_retries,
          base_delay=self.retry_delay,
          timeout_seconds=self.timeout_seconds
      ))
  
  async def _make_request_with_retry(self, payload, deadline_ms):
      """Упрощённая версия с RetryPolicy"""
      
      async def _make_single_request():
          session = await get_shared_http_session()
          async with session.post(
              self.embeddings_endpoint,
              json=payload,
              headers={'Content-Type': 'application/json'}
          ) as response:
              if response.status == 200:
                  result = await response.json()
                  return result["embeddings"]
              else:
                  error_text = await response.text()
                  raise RuntimeError(f"HTTP {response.status}: {error_text}")
      
      # Используем retry policy
      return await self.retry_policy.execute_with_retry(_make_single_request)
  ```

#### 2.2 Circuit Breaker Pattern

- [ ] 2.2.1 Создать класс `CircuitBreaker`
  
  **Файл:** `rag/circuit_breaker.py` (новый)
  
  **Создать:**
  ```python
  """
  Circuit Breaker pattern для защиты от постоянных падений VM.
  """
  import asyncio
  import time
  from enum import Enum
  from typing import Optional, Callable, TypeVar
  from dataclasses import dataclass
  
  T = TypeVar('T')
  
  class CircuitState(Enum):
      CLOSED = "closed"        # Нормальная работа
      OPEN = "open"            # Сервис недоступен, skip requests
      HALF_OPEN = "half_open"  # Пробуем восстановление
  
  @dataclass
  class CircuitBreakerConfig:
      """Конфигурация Circuit Breaker"""
      failure_threshold: int = 5          # Порог неудач для open
      success_threshold: int = 2          # Порог успехов для closed
      timeout_seconds: float = 60.0       # Время до half_open
      half_open_max_calls: int = 1        # Макс вызовов в half_open
  
  class CircuitBreaker:
      """
      Circuit Breaker для защиты от каскадных падений.
      """
      def __init__(self, config: CircuitBreakerConfig):
          self.config = config
          self.state = CircuitState.CLOSED
          self.failure_count = 0
          self.success_count = 0
          self.last_failure_time: Optional[float] = None
          self.half_open_calls = 0
      
      async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
          """
          Выполняет функцию через Circuit Breaker.
          """
          # Проверяем текущее состояние
          if self.state == CircuitState.OPEN:
              # Проверяем timeout для перехода в half_open
              if (self.last_failure_time and 
                  time.time() - self.last_failure_time > self.config.timeout_seconds):
                  self.state = CircuitState.HALF_OPEN
                  self.half_open_calls = 0
              else:
                  raise RuntimeError(
                      f"Circuit breaker OPEN: VM сервис недоступен. "
                      f"Следующая попытка через {self._time_until_retry():.0f}s"
                  )
          
          if self.state == CircuitState.HALF_OPEN:
              if self.half_open_calls >= self.config.half_open_max_calls:
                  raise RuntimeError("Circuit breaker HALF_OPEN: ожидание результата пробного запроса")
              self.half_open_calls += 1
          
          # Пытаемся выполнить функцию
          try:
              result = await func(*args, **kwargs)
              self._on_success()
              return result
          except Exception as e:
              self._on_failure()
              raise
      
      def _on_success(self):
          """Обработка успешного вызова"""
          self.failure_count = 0
          
          if self.state == CircuitState.HALF_OPEN:
              self.success_count += 1
              if self.success_count >= self.config.success_threshold:
                  self.state = CircuitState.CLOSED
                  self.success_count = 0
      
      def _on_failure(self):
          """Обработка неудачного вызова"""
          self.failure_count += 1
          self.last_failure_time = time.time()
          self.success_count = 0
          
          if self.failure_count >= self.config.failure_threshold:
              self.state = CircuitState.OPEN
          
          if self.state == CircuitState.HALF_OPEN:
              self.state = CircuitState.OPEN
      
      def _time_until_retry(self) -> float:
          """Время до следующей попытки"""
          if not self.last_failure_time:
              return 0
          elapsed = time.time() - self.last_failure_time
          return max(0, self.config.timeout_seconds - elapsed)
      
      def get_state(self) -> dict:
          """Возвращает текущее состояние"""
          return {
              'state': self.state.value,
              'failure_count': self.failure_count,
              'success_count': self.success_count,
              'time_until_retry': self._time_until_retry() if self.state == CircuitState.OPEN else None
          }
  ```

- [ ] 2.2.2 Интегрировать Circuit Breaker в `remote_embedder.py`

#### 2.3 Comprehensive error messages и диагностика

- [ ] 2.3.1 Создать функцию `diagnose_vm_connection`
  
  **Файл:** `rag/vm_diagnostics.py` (новый)
  
  **Создать:**
  ```python
  """
  Диагностика проблем подключения к VM сервису.
  """
  import asyncio
  import socket
  import aiohttp
  from typing import Dict, Any
  
  async def diagnose_vm_connection(host: str, port: int) -> Dict[str, Any]:
      """
      Выполняет комплексную диагностику VM подключения.
      
      Returns:
          dict с результатами диагностики:
          {
              'host_reachable': bool,
              'port_open': bool,
              'http_responding': bool,
              'latency_ms': int,
              'recommendations': List[str]
          }
      """
      diagnostics = {
          'host_reachable': False,
          'port_open': False,
          'http_responding': False,
          'latency_ms': None,
          'recommendations': []
      }
      
      # 1. Проверка хоста (ping)
      try:
          socket.gethostbyname(host)
          diagnostics['host_reachable'] = True
      except socket.gaierror:
          diagnostics['recommendations'].append(
              f"DNS не может разрешить {host}. Проверьте сетевое подключение."
          )
          return diagnostics
      
      # 2. Проверка порта (socket)
      try:
          sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
          sock.settimeout(5)
          result = sock.connect_ex((host, port))
          sock.close()
          
          if result == 0:
              diagnostics['port_open'] = True
          else:
              diagnostics['recommendations'].append(
                  f"Порт {port} закрыт на {host}. Проверьте firewall и что VM сервис запущен."
              )
              return diagnostics
      except Exception as e:
          diagnostics['recommendations'].append(
              f"Ошибка проверки порта: {e}"
          )
          return diagnostics
      
      # 3. HTTP проверка с latency
      import time
      start_time = time.time()
      
      try:
          async with aiohttp.ClientSession() as session:
              async with session.get(
                  f"http://{host}:{port}/health",
                  timeout=aiohttp.ClientTimeout(total=10)
              ) as response:
                  latency = (time.time() - start_time) * 1000
                  diagnostics['latency_ms'] = int(latency)
                  
                  if response.status == 200:
                      diagnostics['http_responding'] = True
                  else:
                      diagnostics['recommendations
