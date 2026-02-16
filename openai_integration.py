"""
Интеграция с LLM провайдерами — поддержка OpenAI, Google, Anthropic, GLM.
Минимальный аудит кода с автоматическим fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import tiktoken

from config import get_config
from llm_providers import (
    LLMProviderFactory,
    LLMProviderError,
    get_llm_factory,
)
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
    """Взаимодействие с LLM провайдерами с поддержкой fallback"""

    def analyze_chunk(self, request: GPTAnalysisRequest):
        """
        Совместимость с тестами: синхронная обёртка над analyze_code.
        """
        return asyncio.run(self.analyze_code(request))

    def __init__(self) -> None:
        self.config = get_config()
        
        # Проверяем доступность хотя бы одного провайдера
        self.llm_factory = get_llm_factory()
        
        try:
            self.primary_provider = self.llm_factory.get_primary_provider()
            self.model = self.config.llm.get_provider_model(self.config.llm.provider)
            self.temperature = self.config.llm.temperature
        except LLMProviderError as e:
            raise ValueError(f"Нет доступных LLM провайдеров: {e}")
        
        # Токенизатор для подсчёта токенов (используем cl100k_base как базовый)
        try:
            self.encoder = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            self.encoder = tiktoken.get_encoding("cl100k_base")

        self.cache = GPTCache()

    @property
    def llm(self):
        """Обратная совместимость - доступ к llm через config.llm"""
        return self.config.llm
    
    @property
    def openai(self):
        """Обратная совместимость - доступ к openai через config.openai"""
        return self.config.openai

    def count_tokens(self, text: str) -> int:
        """Подсчёт токенов в тексте"""
        return len(self.encoder.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Обрезка текста до указанного количества токенов"""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.encoder.decode(truncated_tokens)

    async def analyze_code(self, request: GPTAnalysisRequest) -> GPTAnalysisResult:
        """Основной метод анализа кода через LLM с автоматическим fallback"""
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
            
            # Вызываем LLM с fallback
            response = await self._call_llm_api(prompt)
            
            # Парсим ответ
            result = self._parse_gpt_response(response, request.chunks)
            
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

    async def _call_llm_api(self, prompt: str, system_prompt: str = None) -> str:
        """Вызов LLM с автоматическим fallback через фабрику провайдеров"""
        system = system_prompt or "Ты эксперт по анализу кода. Предоставляй краткие и точные описания."
        
        try:
            # Используем фабрику с fallback логикой
            response = await self.llm_factory.generate_with_fallback(prompt, system)
            logger.info(f"LLM запрос выполнен через провайдер: {response.provider}")
            return response.content
        except LLMProviderError as e:
            logger.error(f"Все LLM провайдеры недоступны: {e}")
            raise OpenAIError(f"Все LLM провайдеры недоступны: {e}")

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
            - provider: текущий активный провайдер
        """
        total_requests = 0
        total_tokens = 0
        average_tokens_per_request = 0
        
        # Получаем информацию о провайдерах
        provider_info = "unknown"
        try:
            primary = self.llm_factory.get_primary_provider()
            provider_info = primary.provider_name if primary else "unknown"
        except Exception:
            pass
        
        return {
            "used_today": total_tokens,
            "requests_today": total_requests,
            "average_per_request": average_tokens_per_request,
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "average_tokens_per_request": average_tokens_per_request,
            "provider": provider_info,
        }

    def clear_cache(self) -> int:
        """Очистка кэша OpenAI"""
        cache_files = list(self.cache.dir.glob("*.json"))
        count = len(cache_files)
        for cache_file in cache_files:
            cache_file.unlink()
        return count
