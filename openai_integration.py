"""
Интеграция с OpenAI — минимальный аудит кода.
Изменения: поддержка полного текста отчёта, токенный лимит ↑ до 2048.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import Mock

import openai
import tiktoken
from openai import OpenAI

from config import get_config
from utils import (
    CodeChunk,
    GPTAnalysisRequest,
    GPTAnalysisResult,
    sanitize_text,
)

logger = logging.getLogger(__name__)

# Промпт вынесен в отдельный файл prompts/code_analysis_prompt.md


class RetryPolicy:
    """
    Политика повторных попыток для сетевых запросов.
    
    Инкапсулирует логику retry с экспоненциальными задержками
    и обработкой специфичных исключений OpenAI.
    """
    
    def __init__(
        self,
        attempts: int,
        delay: float,
        retryable_exceptions: tuple = (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError
        )
    ):
        """
        Args:
            attempts: Максимальное количество попыток
            delay: Задержка между попытками в секундах
            retryable_exceptions: Кортеж исключений, при которых делаем retry
        """
        self.attempts = max(1, attempts)
        self.delay = max(0.0, delay)
        self.retryable_exceptions = retryable_exceptions
    
    async def execute(self, func, *args, **kwargs):
        """
        Выполняет функцию с повторными попытками при сетевых ошибках.
        
        Args:
            func: Асинхронная функция для выполнения
            *args, **kwargs: Аргументы для передачи в функцию
            
        Returns:
            Результат успешного выполнения функции
            
        Raises:
            Exception: Последнее исключение после исчерпания всех попыток
        """
        last_exc = None
        
        for attempt in range(1, self.attempts + 1):
            try:
                return await func(*args, **kwargs)
            except self.retryable_exceptions as exc:
                last_exc = exc
                logger.warning(
                    f"Попытка {attempt}/{self.attempts} неудачна: {type(exc).__name__}: {str(exc)}"
                )
                if attempt < self.attempts:
                    await asyncio.sleep(self.delay)
        
        # Все попытки исчерпаны - пробрасываем последнее исключение
        raise last_exc


class OpenAITransport:
    """
    Реальный транспорт для OpenAI API.
    
    Выполняет настоящие HTTP-запросы к OpenAI через официальный SDK.
    """
    
    async def call_api(
        self,
        client: OpenAI,
        model: str,
        messages: list,
        temperature: float
    ) -> str:
        """
        Вызывает OpenAI API для генерации текста.
        
        Args:
            client: Клиент OpenAI
            model: Название модели (например, "gpt-4")
            messages: Список сообщений для чата
            temperature: Температура генерации
            
        Returns:
            Сгенерированный текст от модели
        """
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content


class OfflineTransport:
    """
    Оффлайн транспорт - возвращает заглушки без обращения к API.
    
    Используется в тестах и при отсутствии сетевого подключения.
    """
    
    def __init__(self):
        """Инициализация оффлайн транспорта"""
        self._last_request_context = None
    
    def set_context(self, request: GPTAnalysisRequest, code: str):
        """
        Сохраняет контекст запроса для формирования оффлайн-ответа.
        
        Args:
            request: Запрос на анализ
            code: Код для анализа
        """
        self._last_request_context = (request, code)
    
    async def call_api(self) -> str:
        """
        Возвращает оффлайн-заглушку без реального API вызова.
        
        Returns:
            Текст оффлайн-анализа
        """
        request, code = self._last_request_context or (None, "")
        
        if request is None:
            request = GPTAnalysisRequest(
                file_path="unknown",
                language="",
                chunks=[],
                context=""
            )
        
        return self._build_offline_response(request, code)
    
    def _build_offline_response(
        self,
        request: GPTAnalysisRequest,
        code: str
    ) -> str:
        """
        Формирует текст ответа для офлайн-режима.
        
        Args:
            request: Запрос на анализ
            code: Код для анализа
            
        Returns:
            Текст оффлайн-анализа
        """
        filename = Path(request.file_path).name
        lines = [
            f"🔍 Оффлайн-анализ файла {filename}",
            "⚠️ Анализ выполнен без обращения к OpenAI API.",
        ]

        if code:
            total_lines = code.count("\n") + 1
            lines.append(f"📄 Обработано строк кода: {total_lines}")

        if request.chunks:
            chunk_names = ", ".join(chunk.name for chunk in request.chunks[:3])
            lines.append(f"🧩 Ключевые элементы: {chunk_names}")

        lines.append("Документация сгенерирована автоматически (оффлайн режим)")
        lines.append("Рекомендуется повторить анализ при доступе к сети.")
        
        return "\n".join(lines)


def load_prompt_from_file(prompt_file: str) -> str:
    """Загружает промпт из файла"""
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.error(f"Файл промпта не найден: {prompt_file}")
        raise
    except Exception as e:
        logger.error(f"Ошибка загрузки промпта из {prompt_file}: {e}")
        raise




class GPTCache:
    """Кэширует результаты GPT‑анализов"""

    def __init__(self, cache_dir: str = "./cache") -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, request: GPTAnalysisRequest) -> str:
        """Генерация ключа кэша на основе содержимого запроса"""
        content = f"{request.file_path}_{request.language}_"
        content += "_".join([chunk.content for chunk in request.chunks])
        return hashlib.md5(content.encode()).hexdigest()

    def clear_expired_cache(self, days: int = 7) -> None:
        """Очистка устаревшего кэша"""
        cutoff = datetime.now() - timedelta(days=days)
        for cache_file in self.dir.glob("*.json"):
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                cached_at = datetime.fromisoformat(data.get("cached_at", ""))
                if cached_at < cutoff:
                    cache_file.unlink()
                    logger.debug("Удален устаревший кэш: %s", cache_file.name)
            except Exception as exc:
                logger.warning("Ошибка при проверке кэша %s: %s", cache_file.name, exc)

    def get_cached_result(self, key: str) -> Optional[GPTAnalysisResult]:
        file = self.dir / f"{key}.json"
        if not file.exists():
            return None
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            return GPTAnalysisResult(
                summary=data.get("summary", ""),
                key_components=data.get("key_components", []),
                analysis_per_chunk=data.get("analysis_per_chunk", {}),
                error=data.get("error"),
                full_text=data.get("full_text", ""),
            )
        except Exception as exc:
            logger.warning("Ошибка чтения кэша %s: %s", key, exc)
            return None

    def cache_result(self, key: str, res: GPTAnalysisResult) -> None:
        file = self.dir / f"{key}.json"
        data = {
            "summary": res.summary,
            "key_components": res.key_components,
            "analysis_per_chunk": res.analysis_per_chunk,
            "error": res.error,
            "full_text": res.full_text,
            "cached_at": datetime.now().isoformat(),
        }
        try:
            file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            logger.warning("Не удалось записать кэш %s: %s", key, exc)


class OpenAIManager:
    """Взаимодействие с OpenAI"""

    def analyze_chunk(self, request: GPTAnalysisRequest):
        """
        Совместимость с тестами: синхронная обёртка над analyze_code.
        """
        return asyncio.run(self.analyze_code(request))

    def __init__(self) -> None:
        """
        Инициализирует OpenAIManager с выбором транспорта и retry policy.
        
        Выбор транспорта основан на:
        1. Флаге force_online_for_tests (приоритет для тестов)
        2. Environment variables (OFFLINE_MODE, USE_MOCK_OPENAI)
        3. Наличии pytest_socket в sys.modules
        4. Флаге --disable-socket в sys.argv
        """
        self.config = get_config()
        
        # Проверяем offline режим с учетом конфига (флаг force_online_for_tests)
        self._offline_mode = _is_offline_mode(self.config)

        # Инициализируем параметры модели
        self.model = self.config.openai.model
        self.temperature = self.config.openai.temperature
        
        # Инициализируем API ключ
        api_key = self.config.openai.api_key
        
        # Выбор и инициализация транспорта
        if self._offline_mode:
            # Оффлайн транспорт - заглушки без реального API
            logger.info("[OpenAIManager] Используется OfflineTransport")
            self.transport = OfflineTransport()
            self.client = None  # Клиент не нужен для offline
            self.encoder = None  # Токенизатор не нужен
        else:
            # Онлайн транспорт - реальные API вызовы
            logger.info("[OpenAIManager] Используется OpenAITransport")
            if not api_key:
                raise ValueError("OPENAI_API_KEY не задан")
            
            self.transport = OpenAITransport()
            self.client = OpenAI(api_key=api_key)
            
            # Инициализируем токенизатор
            try:
                self.encoder = tiktoken.encoding_for_model(self.model)
            except Exception:
                self.encoder = tiktoken.get_encoding("cl100k_base")
        
        # Инициализируем RetryPolicy для онлайн-запросов
        self.retry_policy = RetryPolicy(
            attempts=self.config.openai.retry_attempts,
            delay=self.config.openai.retry_delay,
            retryable_exceptions=(
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError
            )
        )
        
        # Инициализируем кэш
        cache_dir = None
        if self._offline_mode:
            cache_dir = tempfile.mkdtemp(prefix="openai_offline_cache_")
        self.cache = GPTCache(cache_dir=cache_dir or "./cache")

    def count_tokens(self, text: str) -> int:
        """Подсчёт токенов в тексте"""
        if self.encoder is None:
            return max(1, len(text.split())) if text else 0

        return len(self.encoder.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Обрезка текста до указанного количества токенов"""
        if self.encoder is None:
            if not text:
                return ""
            words = text.split()
            if len(words) <= max_tokens:
                return text
            return " ".join(words[:max_tokens])

        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.encoder.decode(truncated_tokens)

    async def analyze_code(self, request: GPTAnalysisRequest) -> GPTAnalysisResult:
        """Основной метод анализа кода через GPT"""
        try:
            # Проверяем кэш
            cache_key = self.cache.get_cache_key(request)
            cached = self.cache.get_cached_result(cache_key)
            if cached:
                logger.debug("Найден кэшированный результат для %s", request.file_path)
                return cached

            # Подготавливаем код
            combined_code = self._combine_chunks(request.chunks)
            
            # Санитайзинг при необходимости
            if self.config.analysis.sanitize_enabled:
                combined_code = sanitize_text(combined_code, self.config.analysis.sanitize_patterns)

            # Формируем промпт
            prompt = self._build_analysis_prompt(request, combined_code)
            
            # Сохраняем контекст для офлайн-заглушки
            self._last_request_context = (request, combined_code)

            # Вызываем API или патч
            response_text = await self._call_openai_api(prompt)

            # Парсим ответ
            result = self._parse_gpt_response(response_text, request.chunks)

            # Кэшируем результат
            self.cache.cache_result(cache_key, result)
            
            return result

        except Exception as exc:
            error_msg = f"Ошибка анализа {request.file_path}: {str(exc)}"
            logger.error(error_msg)
            return GPTAnalysisResult(
                summary="Ошибка анализа",
                key_components=[],
                analysis_per_chunk={},
                error=error_msg
            )

    def _combine_chunks(self, chunks: List[CodeChunk]) -> str:
        """Объединение чанков кода в один текст"""
        if not chunks:
            return ""
        
        # Берём до 3-4 наиболее важных чанков
        important_chunks = chunks[:4]
        combined = []
        
        for chunk in important_chunks:
            combined.append(f"// --- {chunk.name} (строки {chunk.start_line}-{chunk.end_line}) ---")
            combined.append(chunk.content)
            combined.append("")
        
        return "\n".join(combined)

    def _build_analysis_prompt(self, request: GPTAnalysisRequest, code: str) -> str:
        """Построение промпта для анализа"""
        filename = Path(request.file_path).name
        
        # Подсчитываем метрики
        total_lines = code.count('\n') + 1 if code else 0
        functions_count = len([chunk for chunk in request.chunks if chunk.chunk_type == 'function'])
        classes_count = len([chunk for chunk in request.chunks if chunk.chunk_type == 'class'])
        
        # Убираем токен лимиты - позволяем OpenAI самому управлять размером

        # Загружаем промпт из файла
        prompt_template = load_prompt_from_file(self.config.prompts.code_analysis_prompt_file)
        
        return prompt_template.format(
            filename=filename,
            total_lines=total_lines,
            functions_count=functions_count,
            classes_count=classes_count,
            code_content=code
        )

    def _build_offline_response(self, request: GPTAnalysisRequest, code: str) -> str:
        """Формирует текст ответа для офлайн-режима без реального обращения к OpenAI."""

        filename = Path(request.file_path).name
        lines = [
            f"🔍 Оффлайн-анализ файла {filename}",
            "⚠️ Анализ выполнен без обращения к OpenAI API.",
        ]

        if code:
            total_lines = code.count("\n") + 1
            lines.append(f"📄 Обработано строк кода: {total_lines}")

        if request.chunks:
            chunk_names = ", ".join(chunk.name for chunk in request.chunks[:3])
            lines.append(f"🧩 Ключевые элементы: {chunk_names}")

        lines.append("Документация сгенерирована автоматически (оффлайн режим)")
        lines.append("Рекомендуется повторить анализ при доступе к сети.")
        return "\n".join(lines)

    def _offline_mock_response(self, *args, **kwargs) -> Mock:
        request, code = getattr(self, "_last_request_context", (None, ""))
        if request is None:
            request = GPTAnalysisRequest(file_path="unknown", language="", chunks=[], context="")

        content = self._build_offline_response(request, code)
        message = Mock()
        message.content = content
        choice = Mock()
        choice.message = message
        response = Mock()
        response.choices = [choice]
        return response

    async def _call_openai_api(self, prompt: str) -> str:
        """
        Вызывает OpenAI API через выбранный транспорт с retry policy.
        
        Args:
            prompt: Промпт для отправки в OpenAI
            
        Returns:
            Текст ответа от модели
            
        Raises:
            Exception: При ошибках после исчерпания всех попыток
        """
        # Проверка на monkey-patching для тестов (совместимость)
        override = self.__dict__.get("_call_openai_api")
        original = type(self).__dict__.get("_call_openai_api")
        if override is not None and override is not original:
            return await override(prompt)

        # Offline режим - используем OfflineTransport
        if self._offline_mode:
            if isinstance(self.transport, OfflineTransport):
                # Устанавливаем контекст для оффлайн-ответа
                request, code = getattr(self, "_last_request_context", (None, ""))
                self.transport.set_context(request, code)
                return await self.transport.call_api()
            else:
                # Fallback для совместимости (не должно происходить)
                request, code = getattr(self, "_last_request_context", (None, ""))
                if request is None:
                    request = GPTAnalysisRequest(
                        file_path="unknown",
                        language="",
                        chunks=[],
                        context=""
                    )
                return self._build_offline_response(request, code)
        
        # Online режим - используем OpenAITransport через RetryPolicy
        if self.client is None:
            raise RuntimeError("OpenAI клиент не инициализирован для онлайн-режима")
        
        # Формируем сообщения для API
        messages = [
            {
                "role": "system",
                "content": "Ты эксперт по анализу кода. Предоставляй краткие и точные описания.",
            },
            {"role": "user", "content": prompt},
        ]
        
        # Создаем функцию для вызова с ретраями
        async def api_call():
            return await self.transport.call_api(
                self.client,
                self.model,
                messages,
                self.temperature
            )
        
        # Выполняем с ретраями через RetryPolicy
        try:
            return await self.retry_policy.execute(api_call)
        except Exception as exc:
            # Логируем итоговую ошибку после всех попыток
            error_type = type(exc).__name__
            error_msg = str(exc)
            logger.error(f"[OpenAI API] Все попытки исчерпаны. {error_type}: {error_msg}")
            
            # Пробрасываем исключение наверх для обработки в analyze_code
            raise

    def _parse_gpt_response(self, text: str, chunks: List[CodeChunk]) -> GPTAnalysisResult:
        """
        Сохраняем *весь* text в full_text, а summary/keys — для краткой сводки.
        """
        summary = ""
        key_components: List[str] = []

        for line in text.splitlines():
            if line.startswith("🔍") or line.startswith("Назначение:"):
                summary = line.lstrip("🔍 ").replace("Назначение:", "").strip()
            if line.startswith("- ") and "Функция" in line:
                key_components.append(line.lstrip("- ").strip())

        if not summary:
            summary = text[:200] + "..." if len(text) > 200 else text

        return GPTAnalysisResult(
            summary=summary,
            key_components=key_components,
            analysis_per_chunk={chunk.name: summary for chunk in chunks[:3]},
            full_text=text,
            error=None,
        )

    def get_token_usage_stats(self) -> Dict:
        """Статистика использования токенов с обратной совместимостью.
        
        Возвращает:
            - used_today: суммарные токены за сегодня (синоним total_tokens)
            - requests_today: количество запросов за сегодня (синоним total_requests)
            - average_per_request: среднее число токенов на запрос (синоним average_tokens_per_request)
            - total_requests, total_tokens, average_tokens_per_request: сохранены для обратной совместимости
        """
        # TODO: заменить заглушку реальными счётчиками при наличии телеметрии
        total_requests = 0
        total_tokens = 0
        average_tokens_per_request = 0
        
        return {
            "used_today": total_tokens,
            "requests_today": total_requests,
            "average_per_request": average_tokens_per_request,
            # Обратная совместимость
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "average_tokens_per_request": average_tokens_per_request,
        }

    def clear_cache(self) -> int:
        """
        Очистка кэша OpenAI.
        
        Всегда очищает реальную директорию ./cache, независимо от того,
        в каком режиме (online/offline) работает менеджер.
        
        Returns:
            Количество удаленных файлов кэша
        """
        # Используем реальную cache директорию, а не self.cache.dir
        # которая может быть временной в offline режиме
        real_cache_dir = Path("./cache")
        
        # Если директория не существует, создаем её и возвращаем 0
        if not real_cache_dir.exists():
            real_cache_dir.mkdir(parents=True, exist_ok=True)
            return 0
        
        # Удаляем все .json файлы из cache директории
        cache_files = list(real_cache_dir.glob("*.json"))
        count = len(cache_files)
        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Не удалось удалить файл кэша {cache_file}: {e}")
        
        return count
def _is_offline_mode(config: Optional = None) -> bool:
    """
    Определяет, активирован ли офлайн-режим для OpenAI.
    
    Args:
        config: Опциональный объект конфигурации. Если передан,
                проверяется флаг force_online_for_tests.
    
    Returns:
        True если активирован офлайн-режим, False иначе
    """
    # Приоритет 1: Явный override для тестов
    # Если установлен флаг force_online_for_tests, игнорируем все остальные проверки
    if config is not None:
        try:
            if hasattr(config, 'openai') and hasattr(config.openai, 'force_online_for_tests'):
                if config.openai.force_online_for_tests:
                    logger.info("[offline] Флаг force_online_for_tests активирован - принудительный онлайн режим")
                    return False
        except Exception as e:
            logger.warning(f"[offline] Ошибка проверки force_online_for_tests: {e}")

    # Приоритет 2: Environment variables
    env_true = {"1", "true", "yes", "on"}

    if str(os.getenv("OFFLINE_MODE", "")).lower() in env_true:
        logger.info("[offline] Обнаружен OFFLINE_MODE - активируем офлайн режим")
        return True

    if str(os.getenv("USE_MOCK_OPENAI", "")).lower() in env_true:
        logger.info("[offline] Обнаружен USE_MOCK_OPENAI - активируем офлайн режим")
        return True

    # Приоритет 3: pytest_socket module (запрет сетевых соединений в тестах)
    if "pytest_socket" in sys.modules:
        logger.info("[offline] Обнаружен pytest_socket - активируем офлайн режим")
        return True

    # Приоритет 4: --disable-socket флаг
    if "--disable-socket" in sys.argv:
        logger.info("[offline] Обнаружен --disable-socket - активируем офлайн режим")
        return True

    return False
