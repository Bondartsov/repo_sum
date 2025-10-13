"""
Модуль конфигурации для анализатора репозиториев.
"""

import json
import os
import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Загружаем переменные из .env файла
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv не установлен, переменные окружения должны быть установлены системно


# Utility функции для безопасной конверсии environment variables
def safe_int(env_var: str, default: str) -> int:
    """Безопасно конвертирует environment variable в int с fallback на default"""
    value = os.getenv(env_var, default)
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(f"Невалидное значение для {env_var}='{value}', используется default: {default}")
        return int(default)


def safe_float(env_var: str, default: str) -> float:
    """Безопасно конвертирует environment variable в float с fallback на default"""
    value = os.getenv(env_var, default)
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Невалидное значение для {env_var}='{value}', используется default: {default}")
        return float(default)


def safe_bool(env_var: str, default: str) -> bool:
    """Безопасно конвертирует environment variable в bool с fallback на default"""
    value = os.getenv(env_var, default)
    try:
        return value.lower() == "true"
    except (AttributeError, TypeError):
        logger.warning(f"Невалидное значение для {env_var}='{value}', используется default: {default}")
        return default.lower() == "true"


@dataclass
class OpenAIConfig:
    """Конфигурация OpenAI API"""
    api_key_env_var: str = "OPENAI_API_KEY"
    temperature: float = field(default_factory=lambda: safe_float("OPENAI_TEMPERATURE", "0.1"))
    retry_attempts: int = field(default_factory=lambda: safe_int("OPENAI_RETRY_ATTEMPTS", "3"))
    retry_delay: float = field(default_factory=lambda: safe_float("OPENAI_RETRY_DELAY", "1.0"))
    force_online_for_tests: bool = field(default_factory=lambda: safe_bool("FORCE_OPENAI_ONLINE_FOR_TESTS", "false"))

    @property
    def api_key(self) -> Optional[str]:
        """Получает API ключ из переменных окружения"""
        return os.getenv(self.api_key_env_var)

    @property
    def model(self) -> str:
        """Получает имя модели из переменных окружения"""
        return os.getenv("OPENAI_MODEL", "gpt-4.1-nano")


@dataclass
class TokenManagementConfig:
    """Конфигурация управления токенами"""
    enable_caching: bool = True
    cache_expiry_days: int = 7


# Значения по умолчанию для санитайзинга секретов
def _default_sanitize_patterns() -> List[str]:
    """Возвращает список регулярных выражений для маскировки секретов."""
    return [
        r"(?i)(api[_-]?key\s*[:=]\s*)(['\"]?[A-Za-z0-9\-_]{16,}['\"]?)",
        r"(?i)(secret[_-]?key\s*[:=]\s*)(['\"]?[A-Za-z0-9\-_]{16,}['\"]?)",
        r"(?i)(bearer\s+)([A-Za-z0-9\-_\.]{16,})",
        r"(?i)(password\s*[:=]\s*)(['\"][^'\"]+['\"])",
        r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+ PRIVATE KEY-----"
    ]


@dataclass
class AnalysisConfig:
    """Конфигурация анализа кода"""
    chunk_strategy: str = "logical"
    min_chunk_size: int = 100
    enable_fallback: bool = True
    languages_priority: List[str] = field(default_factory=lambda: ["python", "javascript", "java"])
    # Новые опции расширенного анализа
    enable_advanced_scoring: bool = False  # приоритизация чанков по «важности»
    sanitize_enabled: bool = True          # санитайзинг секретов перед отправкой в LLM
    sanitize_patterns: List[str] = field(default_factory=_default_sanitize_patterns)  # регулярные выражения для вырезания


@dataclass
class FileScannerConfig:
    """Конфигурация сканера файлов"""
    max_file_size: int = 10485760  # 10MB
    excluded_directories: List[str] = field(default_factory=lambda: [
        ".git", ".svn", ".hg",
        "node_modules", "venv", ".venv",
        "__pycache__", ".pytest_cache",
        "target", "build", "dist",
        ".idea", ".vscode",
        "logs", "tmp", "temp"
    ])
    supported_extensions: Dict[str, str] = field(default_factory=lambda: {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".h": "cpp",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".php": "php",
        ".rb": "ruby"
    })


@dataclass
class OutputConfig:
    """Конфигурация вывода"""
    default_output_dir: str = "./docs"
    file_template: str = "minimal_file.md"
    index_template: str = "index_template.md"
    format: str = "markdown"  # markdown|html
    templates_dir: str = "report_templates"


@dataclass
class PromptsConfig:
    """Конфигурация промптов"""
    code_analysis_prompt_file: str = "prompts/code_analysis_prompt.md"


@dataclass
class EmbeddingConfig:
    """Конфигурация эмбеддингов"""
    provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "sentence-transformers"))
    model_name: str = field(default_factory=lambda: os.getenv("EMB_MODEL_ID", "jinaai/jina-embeddings-v3"))
    precision: str = field(default_factory=lambda: os.getenv("FASTEMBED_PRECISION", "int8"))
    embedding_dim: int = field(default_factory=lambda: safe_int("EMB_DIM", "1024"))
    truncate_dim: int = field(default_factory=lambda: safe_int("EMB_TRUNCATE_DIM", os.getenv("EMB_DIM", "1024")))
    batch_size_min: int = field(default_factory=lambda: safe_int("EMBEDDING_BATCH_SIZE_MIN", "8"))
    batch_size_max: int = field(default_factory=lambda: safe_int("EMBEDDING_BATCH_SIZE_MAX", "128"))
    normalize_embeddings: bool = field(default_factory=lambda: safe_bool("EMB_L2_NORMALIZE", "true"))
    device: str = field(default_factory=lambda: os.getenv("FASTEMBED_DEVICE", "cpu"))
    warmup_enabled: bool = field(default_factory=lambda: safe_bool("EMBEDDING_WARMUP", "true"))
    num_workers: int = field(default_factory=lambda: safe_int("EMBEDDING_WORKERS", "4"))
    # Новые поля для Jina v3
    task_query: str = field(default_factory=lambda: os.getenv("EMB_TASK_QUERY", "retrieval.query"))
    task_passage: str = field(default_factory=lambda: os.getenv("EMB_TASK_PASSAGE", "retrieval.passage"))
    trust_remote_code: bool = field(default_factory=lambda: safe_bool("EMB_TRUST_REMOTE_CODE", "true"))
    pooling: str = field(default_factory=lambda: os.getenv("EMB_POOLING", "mean"))


@dataclass
class VectorStoreConfig:
    """Конфигурация Qdrant"""
    host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    port: int = field(default_factory=lambda: safe_int("QDRANT_PORT", "6333"))
    prefer_grpc: bool = field(default_factory=lambda: safe_bool("QDRANT_PREFER_GRPC", "true"))
    collection_name: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION_NAME", "code_chunks"))
    # Векторная размерность коллекции по умолчанию берётся из EMB_TRUNCATE_DIM,
        # иначе EMB_DIM, иначе явный EMBEDDING_DIMENSION. Это обеспечивает согласованность размерностей между компонентами.
    vector_size: int = field(
        default_factory=lambda: safe_int(
            "EMBEDDING_DIMENSION",
            os.getenv("EMB_TRUNCATE_DIM", os.getenv("EMB_DIM", "1024"))
        )
    )
    distance: str = field(default_factory=lambda: os.getenv("QDRANT_DISTANCE", "cosine"))
    # HNSW параметры
    hnsw_m: int = field(default_factory=lambda: safe_int("QDRANT_HNSW_M", "24"))
    hnsw_ef_construct: int = field(default_factory=lambda: safe_int("QDRANT_HNSW_EF_CONSTRUCT", "128"))
    search_hnsw_ef: int = field(default_factory=lambda: safe_int("QDRANT_SEARCH_HNSW_EF", "256"))
    # Квантование
    quantization_type: str = field(default_factory=lambda: os.getenv("QDRANT_QUANTIZATION_TYPE", "SQ"))
    enable_quantization: bool = field(default_factory=lambda: safe_bool("QDRANT_ENABLE_QUANTIZATION", "true"))
    # Репликация
    replication_factor: int = field(default_factory=lambda: safe_int("QDRANT_REPLICATION_FACTOR", "2"))
    write_consistency_factor: int = field(default_factory=lambda: safe_int("QDRANT_WRITE_CONSISTENCY_FACTOR", "1"))
    # Хранилище
    mmap: bool = field(default_factory=lambda: safe_bool("QDRANT_MMAP", "true"))


@dataclass
class QueryEngineConfig:
    """Конфигурация поиска"""
    max_results: int = field(default_factory=lambda: safe_int("SEARCH_MAX_RESULTS", "10"))
    rrf_enabled: bool = field(default_factory=lambda: safe_bool("SEARCH_RRF_ENABLED", "true"))
    use_hybrid: bool = field(default_factory=lambda: safe_bool("SEARCH_USE_HYBRID", "true"))
    mmr_enabled: bool = field(default_factory=lambda: safe_bool("SEARCH_MMR_ENABLED", "true"))
    mmr_lambda: float = field(default_factory=lambda: safe_float("SEARCH_MMR_LAMBDA", "0.7"))
    cache_ttl_seconds: int = field(default_factory=lambda: safe_int("CACHE_TTL_SECONDS", "300"))
    cache_max_entries: int = field(default_factory=lambda: safe_int("CACHE_MAX_ENTRIES", "1000"))
    score_threshold: float = field(default_factory=lambda: safe_float("SEARCH_SCORE_THRESHOLD", "0.5"))
    # Параллелизм
    concurrent_users_target: int = field(default_factory=lambda: safe_int("SEARCH_CONCURRENT_USERS", "20"))
    search_workers: int = field(default_factory=lambda: safe_int("SEARCH_WORKERS", "4"))
    embed_workers: int = field(default_factory=lambda: safe_int("EMBED_WORKERS", "4"))


@dataclass
class ParallelismConfig:
    """Управление потоками"""
    torch_num_threads: int = field(default_factory=lambda: safe_int("TORCH_NUM_THREADS", "4"))
    omp_num_threads: int = field(default_factory=lambda: safe_int("OMP_NUM_THREADS", "4"))
    mkl_num_threads: int = field(default_factory=lambda: safe_int("MKL_NUM_THREADS", "4"))


@dataclass
class RemoteServiceConfig:
    """Настройки удалённого RAG сервиса"""
    host: str = '10.61.11.54'
    port: int = 8000
    embeddings_endpoint: str = '/v1/embeddings'
    search_endpoint: str = '/v1/search_v2'
    index_endpoint: str = '/v1/index'
    health_endpoint: str = '/v1/health'
    timeout_seconds: int = 3600  # TIMEOUT FIX: 1 час (было 600s) - для больших батчей (512+ чанков)
    max_retries: int = 5  # HOTFIX: больше попыток (было 3)
    retry_delay: float = 10.0  # HOTFIX: больше задержка между попытками (было 2.0s)


@dataclass
class SparseConfig:
    """Конфигурация sparse поиска"""
    method: str = field(default_factory=lambda: os.getenv("SPARSE_METHOD", "SPLADE"))

@dataclass
class RetryProfile:
    """Профиль ретраев для конкретного эндпойнта"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 8.0
    exponential: bool = True  # jitter не используем на этом шаге


@dataclass
class TimeoutProfiles:
    """Пер-эндпойнтовые профили таймаутов и ретраев для RAG-клиентов"""
    # Таймауты
    health_total_sec: float = 2.0
    search_total_p95_sec: float = 10.0
    index_base_sec: float = 60.0
    index_per_batch_step_sec: float = 0.2
    # Ретраи
    retry_search: RetryProfile = field(
        default_factory=lambda: RetryProfile(
            max_attempts=3, base_delay=1.0, max_delay=8.0, exponential=True
        )
    )
    retry_index: RetryProfile = field(
        default_factory=lambda: RetryProfile(
            max_attempts=5, base_delay=2.0, max_delay=60.0, exponential=True
        )
    )


@dataclass
class RagConfig:
    remote_service: RemoteServiceConfig = field(default_factory=RemoteServiceConfig)
    """Конфигурация RAG системы"""
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    query_engine: QueryEngineConfig = field(default_factory=QueryEngineConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    sparse: SparseConfig = field(default_factory=SparseConfig)
    # Новое: пер-эндпойнтовые профили таймаутов и ретраев
    timeout_profiles: TimeoutProfiles = field(default_factory=TimeoutProfiles)


    @classmethod
    def from_dict(cls, data: dict) -> "RagConfig":
        """Создает экземпляр RagConfig из словаря с поддержкой timeout_profiles и .env override"""
        embeddings_data = data.get("embeddings", {}).copy()
        embeddings_data.pop("vector_size", None)

        vector_store_data = data.get("vector_store", data.get("qdrant", {})).copy()
        if "distance_metric" in vector_store_data:
            vector_store_data["distance"] = vector_store_data.pop("distance_metric")

        query_engine_data = data.get("query_engine", data.get("search", {})).copy()

        remote_service_data = data.get("remote_service", {}).copy()
        endpoints = remote_service_data.pop("endpoints", {})
        if endpoints:
            remote_service_data.setdefault("embeddings_endpoint", endpoints.get("embeddings", "/v1/embeddings"))
            remote_service_data.setdefault("search_endpoint", endpoints.get("search", "/v1/search_v2"))
            remote_service_data.setdefault("index_endpoint", endpoints.get("index", "/v1/index"))
            remote_service_data.setdefault("health_endpoint", endpoints.get("health", "/v1/health"))
        if "port" in remote_service_data:
            remote_service_data["port"] = int(remote_service_data["port"])
        if "timeout_seconds" in remote_service_data:
            remote_service_data["timeout_seconds"] = int(remote_service_data["timeout_seconds"])
        if "max_retries" in remote_service_data:
            remote_service_data["max_retries"] = int(remote_service_data["max_retries"])
        if "retry_delay" in remote_service_data:
            remote_service_data["retry_delay"] = float(remote_service_data["retry_delay"])

        remote_service = RemoteServiceConfig(**remote_service_data) if remote_service_data else RemoteServiceConfig()

        # === TimeoutProfiles: JSON + .env overrides с дефолтами ===
        tp_src = data.get("timeout_profiles", {}) or {}
        tp = TimeoutProfiles()

        # JSON-override простых полей
        simple_keys = ("health_total_sec", "search_total_p95_sec", "index_base_sec", "index_per_batch_step_sec")
        for key in simple_keys:
            if key in tp_src:
                try:
                    setattr(tp, key, float(tp_src[key]))
                except Exception:
                    logger.warning(f"Invalid value for timeout_profiles.{key}='{tp_src.get(key)}', using default")

        # JSON-override retry профилей
        rs = tp_src.get("retry_search", {}) or {}
        if isinstance(rs, dict):
            if "max_attempts" in rs:
                try: tp.retry_search.max_attempts = int(rs["max_attempts"])
                except Exception: logger.warning("Invalid retry_search.max_attempts in settings.json")
            if "base_delay" in rs:
                try: tp.retry_search.base_delay = float(rs["base_delay"])
                except Exception: logger.warning("Invalid retry_search.base_delay in settings.json")
            if "max_delay" in rs:
                try: tp.retry_search.max_delay = float(rs["max_delay"])
                except Exception: logger.warning("Invalid retry_search.max_delay in settings.json")
            if "exponential" in rs:
                try: tp.retry_search.exponential = bool(rs["exponential"])
                except Exception: logger.warning("Invalid retry_search.exponential in settings.json")

        ri = tp_src.get("retry_index", {}) or {}
        if isinstance(ri, dict):
            if "max_attempts" in ri:
                try: tp.retry_index.max_attempts = int(ri["max_attempts"])
                except Exception: logger.warning("Invalid retry_index.max_attempts in settings.json")
            if "base_delay" in ri:
                try: tp.retry_index.base_delay = float(ri["base_delay"])
                except Exception: logger.warning("Invalid retry_index.base_delay in settings.json")
            if "max_delay" in ri:
                try: tp.retry_index.max_delay = float(ri["max_delay"])
                except Exception: logger.warning("Invalid retry_index.max_delay in settings.json")
            if "exponential" in ri:
                try: tp.retry_index.exponential = bool(ri["exponential"])
                except Exception: logger.warning("Invalid retry_index.exponential in settings.json")

        # ENV overrides (при наличии)
        tp.health_total_sec = safe_float("RAG_TIMEOUT_HEALTH", str(tp.health_total_sec))
        tp.search_total_p95_sec = safe_float("RAG_TIMEOUT_SEARCH_P95", str(tp.search_total_p95_sec))
        tp.index_base_sec = safe_float("RAG_TIMEOUT_INDEX_BASE", str(tp.index_base_sec))
        tp.index_per_batch_step_sec = safe_float("RAG_TIMEOUT_INDEX_STEP", str(tp.index_per_batch_step_sec))

        tp.retry_search.max_attempts = safe_int("RAG_RETRY_SEARCH_MAX_ATTEMPTS", str(tp.retry_search.max_attempts))
        tp.retry_search.base_delay = safe_float("RAG_RETRY_SEARCH_BASE_DELAY", str(tp.retry_search.base_delay))
        tp.retry_search.max_delay = safe_float("RAG_RETRY_SEARCH_MAX_DELAY", str(tp.retry_search.max_delay))

        tp.retry_index.max_attempts = safe_int("RAG_RETRY_INDEX_MAX_ATTEMPTS", str(tp.retry_index.max_attempts))
        tp.retry_index.base_delay = safe_float("RAG_RETRY_INDEX_BASE_DELAY", str(tp.retry_index.base_delay))
        tp.retry_index.max_delay = safe_float("RAG_RETRY_INDEX_MAX_DELAY", str(tp.retry_index.max_delay))

        return cls(
            remote_service=remote_service,
            embeddings=EmbeddingConfig(**embeddings_data),
            vector_store=VectorStoreConfig(**vector_store_data),
            query_engine=QueryEngineConfig(**query_engine_data),
            sparse=SparseConfig(**data.get("sparse", {})),
            parallelism=ParallelismConfig(**data.get("parallelism", {})),
            timeout_profiles=tp
        )



@dataclass
class Config:
    """Основной класс конфигурации"""
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    token_management: TokenManagementConfig = field(default_factory=TokenManagementConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    file_scanner: FileScannerConfig = field(default_factory=FileScannerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    rag: RagConfig = field(default_factory=RagConfig)

    @classmethod
    def load_from_file(cls, config_path: str = "settings.json") -> "Config":
        """Загружает конфигурацию из JSON файла"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            openai=OpenAIConfig(**data.get("openai", {})),
            token_management=TokenManagementConfig(**data.get("token_management", {})),
            analysis=AnalysisConfig(**data.get("analysis", {})),
            file_scanner=FileScannerConfig(**data.get("file_scanner", {})),
            output=OutputConfig(**data.get("output", {})),
            prompts=PromptsConfig(**data.get("prompts", {})),
            rag=RagConfig.from_dict(data.get("rag", {}))
        )

    def validate(self, require_api_key: bool = True) -> bool:
        """Валидирует конфигурацию"""
        errors = []
        
        # Валидация OpenAI конфигурации
        if require_api_key and not self.openai.api_key:
            errors.append(f"OpenAI API ключ не найден в переменной окружения {self.openai.api_key_env_var}")
        
        if not 0 <= self.openai.temperature <= 2:
            errors.append("temperature должна быть в диапазоне 0-2")
        
        if self.openai.retry_attempts < 0:
            errors.append("retry_attempts должно быть неотрицательным числом")
        
        if self.openai.retry_delay < 0:
            errors.append("retry_delay должно быть неотрицательным числом")
        
        # Валидация управления токенами
        if self.token_management.cache_expiry_days <= 0:
            errors.append("cache_expiry_days должно быть положительным числом")
        
        # Валидация анализа
        if self.analysis.chunk_strategy not in ["logical", "size", "lines"]:
            errors.append("chunk_strategy должна быть одной из: logical, size, lines")
        
        if self.analysis.min_chunk_size <= 0:
            errors.append("min_chunk_size должно быть положительным числом")
        
        # Валидация сканера файлов
        if self.file_scanner.max_file_size <= 0:
            errors.append("max_file_size должно быть положительным числом")
        
        if not self.file_scanner.supported_extensions:
            errors.append("supported_extensions не может быть пустым")
        
        # Проверяем что все расширения начинаются с точки
        for ext in self.file_scanner.supported_extensions.keys():
            if not ext.startswith('.'):
                errors.append(f"Расширение файла должно начинаться с точки: {ext}")
        
        # Валидация исключенных директорий
        if not isinstance(self.file_scanner.excluded_directories, list):
            errors.append("excluded_directories должно быть списком")
        
        # Валидация вывода
        if not self.output.default_output_dir.strip():
            errors.append("default_output_dir не может быть пустым")
        
        # Валидация промптов
        if not self.prompts.code_analysis_prompt_file.strip():
            errors.append("code_analysis_prompt_file не может быть пустым")
        
        # Проверяем существование файла промпта
        prompt_path = Path(self.prompts.code_analysis_prompt_file)
        if not prompt_path.exists():
            errors.append(f"Файл промпта не найден: {self.prompts.code_analysis_prompt_file}")
        
        # Валидация RAG конфигурации
        # Валидация эмбеддингов
        if self.rag.embeddings.provider not in ["sentence-transformers", "fastembed", "remote-vm"]:
            errors.append("embeddings.provider должен быть 'sentence-transformers' или 'fastembed'")
        
        if not self.rag.embeddings.model_name.strip():
            errors.append("embeddings.model_name не может быть пустым")
        
        if self.rag.embeddings.precision not in ["int8", "float32"]:
            errors.append("embeddings.precision должен быть 'int8' или 'float32'")
        
        if self.rag.embeddings.truncate_dim != self.rag.embeddings.embedding_dim:
            errors.append("embeddings.truncate_dim должен совпадать с embeddings.embedding_dim (стандарт 1024d)")
        
        # Валидация новых полей Jina v3
        if self.rag.embeddings.task_query.strip() == "":
            errors.append("embeddings.task_query не может быть пустым")
        
        if self.rag.embeddings.task_passage.strip() == "":
            errors.append("embeddings.task_passage не может быть пустым")
        
        if self.rag.embeddings.pooling not in ["mean", "cls", "max"]:
            errors.append("embeddings.pooling должен быть 'mean', 'cls' или 'max'")
        
        if self.rag.embeddings.batch_size_min <= 0:
            errors.append("embeddings.batch_size_min должен быть положительным числом")
        
        if self.rag.embeddings.batch_size_max <= self.rag.embeddings.batch_size_min:
            errors.append("embeddings.batch_size_max должен быть больше batch_size_min")
        
        if self.rag.embeddings.device not in ["cpu", "cuda", "auto"]:
            errors.append("embeddings.device должен быть 'cpu', 'cuda' или 'auto'")
        
        if self.rag.embeddings.num_workers <= 0:
            errors.append("embeddings.num_workers должен быть положительным числом")
        
        # Валидация vector store (Qdrant)
        if not self.rag.vector_store.host.strip():
            errors.append("vector_store.host не может быть пустым")
        
        if not 1 <= self.rag.vector_store.port <= 65535:
            errors.append("vector_store.port должен быть в диапазоне 1-65535")
        
        if not self.rag.vector_store.collection_name.strip():
            errors.append("vector_store.collection_name не может быть пустым")
        
        if self.rag.vector_store.vector_size <= 0:
            errors.append("vector_store.vector_size должен быть положительным числом")
        if self.rag.embeddings.truncate_dim <= 0 or self.rag.embeddings.embedding_dim <= 0:
            errors.append("embeddings dimensions must be positive")
        elif self.rag.embeddings.truncate_dim > self.rag.embeddings.embedding_dim:
            errors.append("embeddings.truncate_dim cannot be greater than embeddings.embedding_dim")

        if self.rag.vector_store.vector_size != self.rag.embeddings.embedding_dim:
            errors.append("vector_store.vector_size must equal embeddings.embedding_dim (1024d for Jina v3)")

        
        if self.rag.vector_store.distance not in ["cosine", "dot", "euclidean"]:
            errors.append("vector_store.distance должен быть 'cosine', 'dot' или 'euclidean'")
        
        if self.rag.vector_store.hnsw_m <= 0:
            errors.append("vector_store.hnsw_m должен быть положительным числом")
        
        if self.rag.vector_store.hnsw_ef_construct <= 0:
            errors.append("vector_store.hnsw_ef_construct должен быть положительным числом")
        
        if self.rag.vector_store.search_hnsw_ef <= 0:
            errors.append("vector_store.search_hnsw_ef должен быть положительным числом")
        
        if self.rag.vector_store.quantization_type not in ["SQ", "PQ", "BQ"]:
            errors.append("vector_store.quantization_type должен быть 'SQ', 'PQ' или 'BQ'")
        
        if self.rag.vector_store.replication_factor <= 0:
            errors.append("vector_store.replication_factor должен быть положительным числом")
        
        if self.rag.vector_store.write_consistency_factor <= 0:
            errors.append("vector_store.write_consistency_factor должен быть положительным числом")
        if not self.rag.remote_service.host:
            errors.append("remote_service.host must not be empty")
        if self.rag.remote_service.port <= 0:
            errors.append("remote_service.port must be a positive integer")
        if self.rag.remote_service.timeout_seconds <= 0:
            errors.append("remote_service.timeout_seconds must be positive")
        if self.rag.remote_service.max_retries < 0:
            errors.append("remote_service.max_retries cannot be negative")
        if self.rag.remote_service.retry_delay < 0:
            errors.append("remote_service.retry_delay cannot be negative")

        
        # Валидация query engine
        if self.rag.query_engine.max_results <= 0:
            errors.append("query_engine.max_results должен быть положительным числом")
        
        if not 0 <= self.rag.query_engine.mmr_lambda <= 1:
            errors.append("query_engine.mmr_lambda должен быть в диапазоне 0-1")
        
        if self.rag.query_engine.cache_ttl_seconds <= 0:
            errors.append("query_engine.cache_ttl_seconds должен быть положительным числом")
        
        if self.rag.query_engine.cache_max_entries <= 0:
            errors.append("query_engine.cache_max_entries должен быть положительным числом")
        
        if self.rag.query_engine.concurrent_users_target <= 0:
            errors.append("query_engine.concurrent_users_target должен быть положительным числом")
        
        if self.rag.query_engine.search_workers <= 0:
            errors.append("query_engine.search_workers должен быть положительным числом")
        
        if self.rag.query_engine.embed_workers <= 0:
            errors.append("query_engine.embed_workers должен быть положительным числом")
        
        if not 0 <= self.rag.query_engine.score_threshold <= 1:
            errors.append("query_engine.score_threshold должен быть в диапазоне 0-1")

        # Валидация sparse конфигурации
        if self.rag.sparse.method not in ["SPLADE", "BM25"]:
            errors.append("sparse.method должен быть 'SPLADE' или 'BM25'")
        
        # Валидация parallelism
        if self.rag.parallelism.torch_num_threads <= 0:
            errors.append("parallelism.torch_num_threads должен быть положительным числом")
        
        if self.rag.parallelism.omp_num_threads <= 0:
            errors.append("parallelism.omp_num_threads должен быть положительным числом")
        
        if self.rag.parallelism.mkl_num_threads <= 0:
            errors.append("parallelism.mkl_num_threads должен быть положительным числом")
        
        if errors:
            raise ValueError("Ошибки конфигурации:\n" + "\n".join(f"- {error}" for error in errors))
        
        return True


# Глобальный экземпляр конфигурации
_config: Optional[Config] = None


def _harmonize_vector_dims(cfg: Config) -> None:
    try:
        emb_dim = int(cfg.rag.embeddings.embedding_dim)
        trunc_dim = int(cfg.rag.embeddings.truncate_dim)
        if trunc_dim <= 0 or trunc_dim > emb_dim:
            logger.warning(f"embeddings.truncate_dim={trunc_dim} вне диапазона, устанавливаю {emb_dim}")
            cfg.rag.embeddings.truncate_dim = emb_dim
            trunc_dim = emb_dim
        if int(cfg.rag.vector_store.vector_size) != trunc_dim:
            logger.info(f"Гармонизирую vector_store.vector_size: {cfg.rag.vector_store.vector_size} -> {trunc_dim}")
            cfg.rag.vector_store.vector_size = trunc_dim
    except Exception as e:
        logger.warning(f"Не удалось гармонизировать размерности: {e}")


def get_config(require_api_key: bool = False) -> Config:
    """Получает глобальный экземпляр конфигурации"""
    global _config
    if _config is None:
        _config = Config.load_from_file()
        _config.validate(require_api_key=require_api_key)
        _harmonize_vector_dims(_config)
    return _config


def reload_config(config_path: str = "settings.json", require_api_key: bool = True) -> Config:
    """Перезагружает конфигурацию"""
    global _config
    logger.debug(f"reload_config: старый _config id={id(_config)}")
    _config = Config.load_from_file(config_path)
    logger.debug(f"reload_config: загружен новый _config id={id(_config)}, api_key_length={len(_config.openai.api_key) if _config.openai.api_key else 0}")
    _config.validate(require_api_key=require_api_key)
    _harmonize_vector_dims(_config)
    return _config
