"""
Утилиты и базовые структуры данных.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import json
import re
from datetime import datetime


@dataclass
class FileInfo:
    """Информация о файле"""
    path: str
    name: str
    size: int
    language: str
    extension: str
    modified_time: str
    encoding: str = "utf-8"


@dataclass
class CodeChunk:
    """Фрагмент кода для анализа"""
    name: str
    content: str
    start_line: int
    end_line: int
    chunk_type: str = "unknown"  # function, class, module, etc.
    # Для совместимости с CodeChunker
    line_start: int = 0
    line_end: int = 0
    tokens_estimate: int = 0
    element_type: str = ""  # алиас для chunk_type
    
    def __post_init__(self):
        # Синхронизируем поля для совместимости
        if self.line_start == 0 and self.start_line > 0:
            self.line_start = self.start_line
        if self.line_end == 0 and self.end_line > 0:
            self.line_end = self.end_line
        if not self.element_type and self.chunk_type:
            self.element_type = self.chunk_type
        elif self.element_type and not self.chunk_type:
            self.chunk_type = self.element_type


@dataclass
class ParsedElement:
    """Элемент кода (класс, функция, переменная и т.д.)"""
    name: str
    type: str  # "class", "function", "method", "variable", "constant", etc.
    line_number: int
    signature: str = ""
    docstring: Optional[str] = None
    comments: List[str] = field(default_factory=list)


@dataclass
class ParsedFile:
    """Результат парсинга файла"""
    file_info: FileInfo
    chunks: List[CodeChunk] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    # Для совместимости с парсерами
    elements: List[ParsedElement] = field(default_factory=list)
    global_comments: List[str] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)


@dataclass
class GPTAnalysisRequest:
    """Запрос к GPT для анализа"""
    file_path: str
    language: str
    chunks: List[CodeChunk]
    context: str = ""


@dataclass
class GPTAnalysisResult:
    """
    Итог ответа GPT.
    `full_text` — полный отчёт, отображаемый пользователю.
    """
    summary: str
    key_components: List[str]
    analysis_per_chunk: Dict[str, str]
    full_text: str = ""
    error: Optional[str] = None

    def __post_init__(self):
        self.key_components = self.key_components or []
        self.analysis_per_chunk = self.analysis_per_chunk or {}


# Исключения
class RepoSumError(Exception):
    """Базовое исключение для всех ошибок проекта"""
    pass


class FileParsingError(RepoSumError):
    """Ошибка парсинга файла"""
    pass


# Алиас для совместимости с парсерами
ParsingError = FileParsingError


class OpenAIError(RepoSumError):
    """Ошибка взаимодействия с OpenAI"""
    pass


class ConfigError(RepoSumError):
    """Ошибка конфигурации"""
    pass


# Утилитные функции
def ensure_directory_exists(path: str) -> None:
    """Создать директорию если её нет"""
    Path(path).mkdir(parents=True, exist_ok=True)


def compute_file_hash(path: str, block_size: int = 1 << 20) -> str:
    """SHA256 хэш содержимого файла, блочно, для больших файлов."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_index(index_path: str) -> Dict[str, Dict[str, str]]:
    """Читает/инициализирует индекс изменённых файлов.
    Структура: { file_path: {"hash": str, "analyzed_at": iso } }
    """
    p = Path(index_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def write_index(index_path: str, data: Dict[str, Dict[str, str]]) -> None:
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(index_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def sanitize_text(text: str, patterns: List[str]) -> str:
    """Маскирует секреты/PII по списку regex-паттернов."""
    if not patterns:
        return text
    masked = text
    for pat in patterns:
        try:
            masked = re.sub(pat, "[REDACTED]", masked, flags=re.MULTILINE)
        except re.error:
            # игнорируем некорректный паттерн
            continue
    return masked


class MetricsRecorder:
    """Сбор простых метрик сессии анализа."""
    def __init__(self) -> None:
        self.data: Dict[str, float] = {
            'total_requests': 0,
            'total_tokens': 0,
        }
        self.started_at: str = datetime.utcnow().isoformat()

    def add_request(self, tokens: int) -> None:
        self.data['total_requests'] += 1
        self.data['total_tokens'] += max(0, tokens)

    def snapshot(self) -> Dict[str, float]:
        return dict(self.data)


def create_error_parsed_file(file_info: FileInfo, error: Exception) -> ParsedFile:
    """Создает объект ParsedFile для файла с ошибкой парсинга
    Возвращает структуру с корректным заполнением поля parse_errors.
    """
    return ParsedFile(
        file_info=file_info,
        parse_errors=[str(error)]
    )


def create_error_gpt_result(error: Exception) -> GPTAnalysisResult:
    """Создает объект GPTAnalysisResult для случая ошибки анализа
    Поле error заполняется сообщением, full_text остаётся пустым,
    чтобы генератор Markdown использовал fallback‑разметку.
    """
    return GPTAnalysisResult(
        summary="Ошибка анализа",
        key_components=[],
        analysis_per_chunk={},
        full_text="",
        error=f"Ошибка анализа: {error}"
    )


def clean_filename(filename: str) -> str:
    """Очистить имя файла от недопустимых символов"""
    import re
    # Убираем недопустимые символы для имени файла
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Убираем множественные подчеркивания
    cleaned = re.sub(r'_+', '_', cleaned)
    # Убираем подчеркивания в начале и конце
    cleaned = cleaned.strip('_')
    return cleaned


def format_file_size(size: int) -> str:
    """Форматировать размер файла"""
    if size < 1024:
        return f"{size} bytes"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.1f} MB"


def get_language_from_extension(extension: str) -> str:
    """Определить язык программирования по расширению файла"""
    language_map = {
        '.py': 'python',
        '.js': 'javascript', 
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.cs': 'csharp',
        '.php': 'php',
        '.rb': 'ruby',
        '.go': 'go',
        '.rs': 'rust',
        '.kt': 'kotlin',
        '.swift': 'swift',
        '.m': 'objective-c',
        '.sh': 'bash',
        '.ps1': 'powershell',
        '.sql': 'sql',
        '.html': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.less': 'less',
        '.xml': 'xml',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.md': 'markdown',
        '.txt': 'text',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp'
    }
    return language_map.get(extension.lower(), 'unknown')


def truncate_text(text: str, max_length: int = 100) -> str:
    """Обрезать текст до указанной длины"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def count_lines(content: str) -> tuple:
    """
    Подсчитать количество строк разных типов
    Возвращает (total_lines, code_lines, comment_lines, blank_lines)
    """
    lines = content.split('\n')
    total_lines = len(lines)
    blank_lines = 0
    comment_lines = 0
    code_lines = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_lines += 1
        elif stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*'):
            comment_lines += 1
        else:
            code_lines += 1
    
    return total_lines, code_lines, comment_lines, blank_lines


# Алиас для совместимости
count_lines_in_text = count_lines


def setup_logging(level: str = "INFO") -> None:
    """Настройка логирования.

    Требования:
    - В консоли отображать только INFO и ERROR (исключая WARNING и DEBUG).
    - Файловый лог вести на заданном уровне (по умолчанию INFO).
    - Не дублировать хендлеры при повторной инициализации (актуально для Streamlit).
    """
    import logging
    import os

    # Директория для логов
    os.makedirs('logs', exist_ok=True)

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    root_level = level_map.get(level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)

    # Удаляем существующие хендлеры, чтобы избежать дублирования в Streamlit
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    # Единый форматтер
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Фильтр для консоли: пропускаем только INFO и ERROR
    class InfoErrorFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            # Показываем только INFO и ERROR
            if record.levelno not in (logging.INFO, logging.ERROR):
                return False
            # Не пугаем пользователя штатным failover'ом Qdrant (gRPC->HTTP):
            # начальный ERROR "Health check ..." скрываем в консоли,
            # а реальную ошибку после неудачного failover оставляем.
            try:
                msg = record.getMessage()
            except Exception:
                msg = str(record.msg)
            if (
                record.name.startswith('rag.vector_store')
                and record.levelno == logging.ERROR
                and 'Health check' in msg
                and 'после failover' not in msg
            ):
                return False
            return True

    # Консольный хендлер
    try:
        import sys
        import io
        stdout_buffer = getattr(sys.stdout, "buffer", None)
        if stdout_buffer is not None:
            utf8_stdout = io.TextIOWrapper(stdout_buffer, encoding="utf-8", errors="backslashreplace")
            stream_handler = logging.StreamHandler(stream=utf8_stdout)
        else:
            stream_handler = logging.StreamHandler()
    except Exception:
        stream_handler = logging.StreamHandler()
    # Уровень ставим INFO, а фильтр отсеет WARNING; DEBUG ниже хендлера и так не пройдет
    stream_handler.setLevel(logging.INFO)
    stream_handler.addFilter(InfoErrorFilter())
    stream_handler.setFormatter(formatter)

    # Файловый хендлер (оставляем информативным на уровне root_level)
    file_handler = logging.FileHandler('logs/app.log', encoding='utf-8')
    file_handler.setLevel(root_level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


def map_user_friendly_error(exc: Exception | dict) -> dict:
    """
    Преобразует исключения/ответы сервера в человеко‑читаемую структуру для UI/CLI.

    Возвращает словарь:
      {
        "title": "Validation error" | "Timeout" | "Connection issue" | "Server error" | "Unknown error",
        "message": "краткое описание",
        "recommendations": ["шаг1", "шаг2", ...],
        "code": 422|503|500|… (если известен),
        "details": [{"field":"...","issue":"...","id":"..."}] (если есть и безопасно)
      }

    Безопасность: не включает текст документов/векторов; в details только поля/причины/id.
    """
    import json
    import asyncio
    import re
    try:
        import aiohttp  # type: ignore
    except Exception:
        aiohttp = None  # type: ignore

    ALLOWED_DETAIL_KEYS = {"field", "issue", "id", "reason", "code"}

    def _sanitize_details(raw):
        out = []
        try:
            for item in raw or []:
                if isinstance(item, dict):
                    clean = {k: item.get(k) for k in ALLOWED_DETAIL_KEYS if k in item}
                    if "id" in clean:
                        try:
                            clean["id"] = str(clean["id"])
                        except Exception:
                            clean["id"] = "<invalid>"
                    # Удаляем потенциально чувствительные ключи, если они попадут в details
                    for banned in ("text", "content", "document", "vector", "payload"):
                        if banned in clean:
                            clean.pop(banned, None)
                    if clean:
                        out.append(clean)
        except Exception:
            pass
        return out

    def _base(title, message, code=None, recs=None, details=None):
        return {
            "title": str(title) if title else "Unknown error",
            "message": str(message) if message else "",
            "recommendations": list(recs or []),
            "code": int(code) if isinstance(code, int) else (code if isinstance(code, (int,)) else None),
            "details": _sanitize_details(details) if details else [],
        }

    def _timeout():
        return _base(
            "Timeout",
            "Превышено время ожидания ответа сервиса",
            504,
            [
                "Проверьте загрузку VM/сети",
                "Снизьте batch_size или повторите позже",
                "Порог p95 для /search ≤ профиля",
            ],
        )

    def _server(code=500):
        return _base(
            "Server error",
            "Внутренняя ошибка сервера",
            code,
            [
                "Проверьте серверные логи VM/Qdrant",
                "Повторите попытку позже",
            ],
        )

    def _conn():
        return _base(
            "Connection issue",
            "Сервис недоступен/подключение не установлено",
            503,
            [
                "Проверьте доступность хоста/порта",
                "Убедитесь, что VM сервис запущен",
            ],
        )

    def _validation(msg, details=None):
        return _base(
            "Validation error",
            msg or "Ошибка валидации входных данных",
            422,
            [
                "Проверьте заполненность text (не пустой)",
                "Пересоберите батч без отбракованных id",
                "См. логи/агрегаты по dropped_documents_total",
            ],
            details=details,
        )

    # 1) dict payload (например, сервер вернул {"error": {...}})
    if isinstance(exc, dict):
        data = exc
        payload = data.get("error") if isinstance(data.get("error"), dict) else data
        code = payload.get("code") or data.get("status")
        err_type = (payload.get("type") or "").lower()
        msg = payload.get("message") or data.get("message") or "Ошибка операции"
        details = payload.get("details") or data.get("details") or []
        if err_type == "validation_error" or code == 422:
            return _validation(msg, details)
        if isinstance(code, int):
            if code == 504:
                return _timeout()
            if 500 <= code <= 599:
                return _server(code)
        return _base("Unknown error", msg, code, ["Проверьте логи и параметры запроса"])

    # 2) Exception payload
    e = exc  # type: ignore
    name = type(e).__name__
    text = str(e)

    # Timeout типы
    if isinstance(e, asyncio.TimeoutError) or name in {"TimeoutException", "VMTimeoutError"}:
        return _timeout()

    # HTTP ошибки (aiohttp.ClientResponseError)
    if aiohttp and isinstance(e, getattr(aiohttp, "ClientResponseError", tuple())):
        status = getattr(e, "status", None)
        message = getattr(e, "message", None) or text
        if status == 422:
            # Пытаемся распарсить JSON из message ("HTTP 422: {json}")
            details = []
            parsed_msg = None
            try:
                idx = message.find("{")
                jtxt = message[idx:] if idx != -1 else ""
                data = json.loads(jtxt) if jtxt else {}
                payload = data.get("error", data)
                parsed_msg = payload.get("message")
                details = payload.get("details") or []
            except Exception:
                pass
            return _validation(parsed_msg or "Ошибка валидации входных данных", details)
        if status == 504:
            return _timeout()
        if isinstance(status, int) and 500 <= status <= 599:
            return _server(status)
        # Прочие HTTP статусы (4xx etc.)
        return _base("Unknown error", message, status, ["Проверьте логи и параметры запроса"])

    # Ошибки соединения
    if (aiohttp and isinstance(e, getattr(aiohttp, "ClientConnectorError", tuple()))) or \
       name in {"VectorStoreConnectionError", "ConnectionException"} or \
       "ClientConnectorError" in name or "ConnectionRefused" in text:
        return _conn()

    # Эвристика: встречается "HTTP 422" в тексте исключения
    if "HTTP 422" in text or "status 422" in text:
        details = []
        parsed_msg = None
        try:
            idx = text.find("{")
            jtxt = text[idx:] if idx != -1 else ""
            data = json.loads(jtxt) if jtxt else {}
            payload = data.get("error", data)
            parsed_msg = payload.get("message")
            details = payload.get("details") or []
        except Exception:
            pass
        return _validation(parsed_msg or "Ошибка валидации входных данных", details)

    # Эвристика: извлекаем HTTP 5xx код из строки
    m = re.search(r"HTTP\s+(5\d{2})", text)
    if m:
        code = int(m.group(1))
        if code == 504:
            return _timeout()
        return _server(code)

    # По умолчанию
    return _base("Unknown error", "Неизвестная ошибка", None, ["Проверьте логи и параметры запроса"])
