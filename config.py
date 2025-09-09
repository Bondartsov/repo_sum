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


@dataclass
class AnalysisConfig:
    """Конфигурация анализа кода"""
    chunk_strategy: str = "logical"
    min_chunk_size: int = 100
    enable_fallback: bool = True
    languages_priority: List[str] = field(default_factory=lambda: ["python", "javascript", "java"])
    # Новые опции расширенного анализа
    enable_advanced_scoring: bool = False  # приоритизация чанков по «важности»
    sanitize_enabled: bool = False         # санитайзинг секретов перед отправкой в LLM
    sanitize_patterns: List[str] = field(default_factory=list)  # регулярные выражения для вырезания


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
    provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "fastembed"))
    model_name: str = field(default_factory=lambda: os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    precision: str = field(default_factory=lambda: os.getenv("FASTEMBED_PRECISION", "int8"))
    truncate_dim: int = field(default_factory=lambda: safe_int("EMBEDDING_DIMENSION", "384"))
    batch_size_min: int = field(default_factory=lambda: safe_int("EMBEDDING_BATCH_SIZE_MIN", "8"))
    batch_size_max: int = field(default_factory=lambda: safe_int("EMBEDDING_BATCH_SIZE_MAX", "128"))
    normalize_embeddings: bool = field(default_factory=lambda: safe_bool("EMBEDDING_NORMALIZE", "true"))
    device: str = field(default_factory=lambda: os.getenv("FASTEMBED_DEVICE", "cpu"))
    warmup_enabled: bool = field(default_factory=lambda: safe_bool("EMBEDDING_WARMUP", "true"))
    num_workers: int = field(default_factory=lambda: safe_int("EMBEDDING_WORKERS", "4"))


@dataclass
class VectorStoreConfig:
    """Конфигурация Qdrant"""
    host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    port: int = field(default_factory=lambda: safe_int("QDRANT_PORT", "6333"))
    prefer_grpc: bool = field(default_factory=lambda: safe_bool("QDRANT_PREFER_GRPC", "true"))
    collection_name: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION_NAME", "code_chunks"))
    vector_size: int = field(default_factory=lambda: safe_int("EMBEDDING_DIMENSION", "384"))
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
class RagConfig:
    """Конфигурация RAG системы"""
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    query_engine: QueryEngineConfig = field(default_factory=QueryEngineConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "RagConfig":
        """Создает экземпляр RagConfig из словаря"""
        # Маппинг для совместимости с settings.json
        embeddings_data = data.get("embeddings", {}).copy()
        # Удаляем поле vector_size из embeddings (оно принадлежит vector_store)
        embeddings_data.pop("vector_size", None)
        
        vector_store_data = data.get("vector_store", data.get("qdrant", {})).copy()
        # Маппинг distance_metric -> distance для совместимости
        if "distance_metric" in vector_store_data:
            vector_store_data["distance"] = vector_store_data.pop("distance_metric")
        
        query_engine_data = data.get("query_engine", data.get("search", {})).copy()
        
        return cls(
            embeddings=EmbeddingConfig(**embeddings_data),
            vector_store=VectorStoreConfig(**vector_store_data),
            query_engine=QueryEngineConfig(**query_engine_data),
            parallelism=ParallelismConfig(**data.get("parallelism", {}))
        )


@dataclass
class Config:
    """Основной класс конфигурации"""
    openai: OpenAIConfig
    token_management: TokenManagementConfig
    analysis: AnalysisConfig
    file_scanner: FileScannerConfig
    output: OutputConfig
    prompts: PromptsConfig
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
        if self.rag.embeddings.provider not in ["sentence-transformers", "fastembed"]:
            errors.append("embeddings.provider должен быть 'sentence-transformers' или 'fastembed'")
        
        if not self.rag.embeddings.model_name.strip():
            errors.append("embeddings.model_name не может быть пустым")
        
        if self.rag.embeddings.precision not in ["int8", "float32"]:
            errors.append("embeddings.precision должен быть 'int8' или 'float32'")
        
        if not 256 <= self.rag.embeddings.truncate_dim <= 384:
            errors.append("embeddings.truncate_dim должен быть в диапазоне 256-384")
        
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


def get_config(require_api_key: bool = False) -> Config:
    """Получает глобальный экземпляр конфигурации"""
    global _config
    if _config is None:
        _config = Config.load_from_file()
        _config.validate(require_api_key=require_api_key)
    return _config


def reload_config(config_path: str = "settings.json", require_api_key: bool = True) -> Config:
    """Перезагружает конфигурацию"""
    global _config
    logger.debug(f"reload_config: старый _config id={id(_config)}")
    _config = Config.load_from_file(config_path)
    logger.debug(f"reload_config: загружен новый _config id={id(_config)}, api_key_length={len(_config.openai.api_key) if _config.openai.api_key else 0}")
    _config.validate(require_api_key=require_api_key)
    return _config
