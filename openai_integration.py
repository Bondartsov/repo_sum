"""
Интеграция с OpenAI — минимальный аудит кода.
Изменения: поддержка полного текста отчёта, токенный лимит ↑ до 2048.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import openai
import tiktoken
from openai import OpenAI

from config import get_config
from utils import (
    CodeChunk,
    GPTAnalysisRequest,
    GPTAnalysisResult,
    OpenAIError,
)

logger = logging.getLogger(__name__)

# Промпт для минимального анализа кода
MINIMAL_ANALYSIS_PROMPT = """
Ты — автономный агент по описательному аудиту логики работы исходного кода.  
Не указывай на ошибки, не предлагай улучшения и не анализируй архитектурные паттерны.  
Фокусируйся исключительно на том, что делает файл, какие данные он принимает и куда передаёт, а также на перечислении его ключевых элементов.

Ответ должен содержать следующие разделы:

🔍 Общее описание логики  
- Краткое описание цели и роли файла в системе.  
- Источники входных данных (файлы, API, параметры и т.п.) и их формат.  
- Приёмники выходных данных (консоль, файлы, другие модули, БД, API) и формат результата.

🧩 Составные элементы  
- Список всех функций, классов и методов, присутствующих в файле.  
- Для каждого — имя и 2–3 предложения о его назначении и ключевой логике.

📥 Входы и 📤 Выходы  
- Подробно укажи, откуда именно поступают входные данные (например, аргументы командной строки, HTTP‑запрос, JSON‑файл).  
- Куда и в каком виде передаются результаты работы (например, вывод в консоль, запись в файл, отправка по API).

📝 Итоговая сводка  
- В конце дай 2–3 предложения, резюмирующие логику файла.  
  Например: "Итого: этот файл отвечает за сканирование файлов в указанной директории."

Начинай отчёт с заголовка  
`Audit Report: {filename}`  

Анализируй только то, что явно следует из кода, без домыслов о внешнем окружении.

Код для анализа:

{code_content}
"""

MAX_RESPONSE_TOKENS = 2048  # раньше было 1000


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

    def __init__(self) -> None:
        cfg = get_config()
        if not cfg.openai.api_key:
            raise ValueError("OPENAI_API_KEY не задан")
        self.client = OpenAI(api_key=cfg.openai.api_key)
        self.model = cfg.openai.model
        self.temperature = cfg.openai.temperature
        try:
            self.encoder = tiktoken.encoding_for_model(self.model)
        except Exception:
            self.encoder = tiktoken.get_encoding("cl100k_base")

        self.cache = GPTCache()

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
            
            # Формируем промпт
            prompt = self._build_analysis_prompt(request, combined_code)
            
            # Вызываем API
            response = await self._call_openai_api(prompt)
            
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
        
        # Ограничиваем размер кода
        max_code_tokens = 1500  # Оставляем место для промпта и ответа
        if self.count_tokens(code) > max_code_tokens:
            code = self.truncate_to_tokens(code, max_code_tokens)
            code += "\n\n... [код обрезан для экономии токенов] ..."

        return MINIMAL_ANALYSIS_PROMPT.format(
            filename=filename,
            total_lines=total_lines,
            functions_count=functions_count,
            classes_count=classes_count,
            code_content=code
        )

    async def _call_openai_api(self, prompt: str) -> str:
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
            max_tokens=MAX_RESPONSE_TOKENS,
        )
        return response.choices[0].message.content

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
        """Статистика использования токенов (заглушка)"""
        return {
            "total_requests": 0,
            "total_tokens": 0,
            "average_tokens_per_request": 0
        }

    def clear_cache(self) -> int:
        """Очистка кэша OpenAI"""
        cache_files = list(self.cache.dir.glob("*.json"))
        count = len(cache_files)
        for cache_file in cache_files:
            cache_file.unlink()
        return count
