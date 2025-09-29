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
    OpenAIError,
    sanitize_text,
)

logger = logging.getLogger(__name__)

# Промпт вынесен в отдельный файл prompts/code_analysis_prompt.md


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
        self.config = get_config()
        self._offline_mode = _is_offline_mode()

        api_key = self.config.openai.api_key
        if self._offline_mode:
            if not api_key:
                logger.warning(
                    "OpenAIManager запущен в офлайн-режиме без API ключа — используется заглушка"
                )
            self.client = Mock(name="OfflineOpenAIClient")
            self.client.chat = Mock()
            self.client.chat.completions = Mock()
            self.client.chat.completions.create = Mock(side_effect=self._offline_mock_response)
        else:
            if not api_key:
                raise ValueError("OPENAI_API_KEY не задан")
            self.client = OpenAI(api_key=api_key)
        self.model = self.config.openai.model
        self.temperature = self.config.openai.temperature
        if self._offline_mode:
            self.encoder = None
        else:
            try:
                self.encoder = tiktoken.encoding_for_model(self.model)
            except Exception:
                self.encoder = tiktoken.get_encoding("cl100k_base")

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
        override = self.__dict__.get("_call_openai_api")
        original = type(self).__dict__.get("_call_openai_api")
        if override is not None and override is not original:
            return await override(prompt)

        if self._offline_mode and (self.client is None or not isinstance(self.client, Mock)):
            request, code = getattr(self, "_last_request_context", (None, ""))
            if request is None:
                request = GPTAnalysisRequest(file_path="unknown", language="", chunks=[], context="")
            return self._build_offline_response(request, code)

        # Ретраи на случай временных ошибок сети/квот
        attempts = max(1, int(self.config.openai.retry_attempts))
        delay = max(0.0, float(self.config.openai.retry_delay))
        last_exc: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                if self.client is None:
                    raise RuntimeError("OpenAI клиент не инициализирован")

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Ты эксперт по анализу кода. Предоставляй краткие и точные описания.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                )
                return response.choices[0].message.content
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Ошибка вызова OpenAI (попытка %s/%s): %s", attempt, attempts, exc
                )
                if attempt < attempts:
                    await asyncio.sleep(delay)

        # Если все попытки исчерпаны — пробрасываем ошибку выше
        assert last_exc is not None
        raise last_exc

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
        """Очистка кэша OpenAI"""
        cache_files = list(self.cache.dir.glob("*.json"))
        count = len(cache_files)
        for cache_file in cache_files:
            cache_file.unlink()
        return count
def _is_offline_mode() -> bool:
    """Определяет, активирован ли офлайн-режим для OpenAI."""

    env_true = {"1", "true", "yes", "on"}

    if str(os.getenv("OFFLINE_MODE", "")).lower() in env_true:
        return True

    if str(os.getenv("USE_MOCK_OPENAI", "")).lower() in env_true:
        return True

    if "pytest_socket" in sys.modules:
        return True

    if "--disable-socket" in sys.argv:
        return True

    return False


