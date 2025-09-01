"""
Общие фикстуры для тестов RAG системы.

Содержит переиспользуемые фикстуры для всех типов RAG тестов:
- Конфигурации
- Mock объекты
- Тестовые данные
- Утилиты для тестирования
"""

import pytest
import asyncio
import tempfile
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock

import numpy as np

from config import Config, RagConfig, EmbeddingConfig, VectorStoreConfig, QueryEngineConfig, ParallelismConfig
from rag.embedder import CPUEmbedder
from rag.vector_store import QdrantVectorStore
from rag.indexer_service import IndexerService
from rag.search_service import SearchService
from rag.query_engine import CPUQueryEngine


@pytest.fixture(scope="session")
def test_rag_settings_file():
    """Создает временный файл настроек для RAG тестов"""
    test_settings = {
        "openai": {
            "api_key_env_var": "OPENAI_API_KEY",
            "max_tokens_per_chunk": 4000,
            "temperature": 0.1
        },
        "token_management": {
            "enable_caching": True,
            "cache_expiry_days": 7
        },
        "analysis": {
            "chunk_strategy": "logical",
            "min_chunk_size": 100
        },
        "file_scanner": {
            "max_file_size": 10485760,
            "excluded_directories": [".git", "node_modules", "__pycache__"],
            "supported_extensions": {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript"
            }
        },
        "output": {
            "default_output_dir": "./docs",
            "format": "markdown"
        },
        "prompts": {
            "code_analysis_prompt_file": "prompts/code_analysis_prompt.md"
        },
        "rag": {
            "embeddings": {
                "provider": "fastembed",
                "model_name": "BAAI/bge-small-en-v1.5",
                "precision": "int8",
                "truncate_dim": 384,
                "batch_size_min": 8,
                "batch_size_max": 64,
                "normalize_embeddings": True,
                "device": "cpu",
                "warmup_enabled": True,
                "num_workers": 4
            },
            "vector_store": {
                "host": os.getenv("QDRANT_HOST", "localhost"),
                "port": int(os.getenv("QDRANT_PORT", "6333")),
                "prefer_grpc": False,
                "collection_name": "test_rag_collection",
                "vector_size": 384,
                "distance": "cosine",
                "hnsw_m": 16,
                "hnsw_ef_construct": 64,
                "search_hnsw_ef": 128,
                "quantization_type": "SQ",
                "enable_quantization": True,
                "replication_factor": 1,
                "write_consistency_factor": 1,
                "mmap": True
            },
            "query_engine": {
                "max_results": 10,
                "rrf_enabled": True,
                "use_hybrid": False,
                "mmr_enabled": True,
                "mmr_lambda": 0.7,
                "cache_ttl_seconds": 300,
                "cache_max_entries": 100,
                "score_threshold": 0.6,
                "concurrent_users_target": 10,
                "search_workers": 4,
                "embed_workers": 2
            },
            "parallelism": {
                "torch_num_threads": 2,
                "omp_num_threads": 2,
                "mkl_num_threads": 2
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_settings, f, indent=2)
        settings_path = f.name
    
    yield settings_path
    
    # Cleanup
    try:
        os.unlink(settings_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def test_rag_config():
    """Базовая конфигурация RAG для тестов"""
    return RagConfig(
        embeddings=EmbeddingConfig(
            provider="fastembed",
            model_name="BAAI/bge-small-en-v1.5",
            precision="int8",
            truncate_dim=384,
            batch_size_min=8,
            batch_size_max=64,
            normalize_embeddings=True,
            device="cpu",
            warmup_enabled=True,
            num_workers=4
        ),
        vector_store=VectorStoreConfig(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            prefer_grpc=False,
            collection_name="test_collection",
            vector_size=384,
            distance="cosine",
            hnsw_m=16,
            hnsw_ef_construct=64,
            search_hnsw_ef=128,
            quantization_type="SQ",
            enable_quantization=True,
            replication_factor=1,
            write_consistency_factor=1
        ),
        query_engine=QueryEngineConfig(
            max_results=10,
            rrf_enabled=True,
            use_hybrid=False,
            mmr_enabled=True,
            mmr_lambda=0.7,
            cache_ttl_seconds=300,
            cache_max_entries=100,
            score_threshold=0.6,
            concurrent_users_target=10,
            search_workers=4,
            embed_workers=2
        ),
        parallelism=ParallelismConfig(
            torch_num_threads=2,
            omp_num_threads=2,
            mkl_num_threads=2
        )
    )


@pytest.fixture
def minimal_rag_config():
    """Минимальная конфигурация RAG для быстрых тестов"""
    return RagConfig(
        embeddings=EmbeddingConfig(
            provider="fastembed",
            model_name="BAAI/bge-small-en-v1.5",
            batch_size_min=4,
            batch_size_max=16,
            warmup_enabled=False  # Отключаем для быстрых тестов
        ),
        vector_store=VectorStoreConfig(
            collection_name="minimal_test_collection",
            vector_size=384,
            enable_quantization=False,  # Отключаем для простоты
            hnsw_m=8,  # Уменьшенные параметры
            hnsw_ef_construct=32
        ),
        query_engine=QueryEngineConfig(
            max_results=5,
            rrf_enabled=False,  # Отключаем сложные алгоритмы
            mmr_enabled=False,
            cache_max_entries=10,
            concurrent_users_target=2
        )
    )


@pytest.fixture
def sample_code_texts():
    """Набор тестовых текстов кода для RAG тестов"""
    return [
        # Python функции
        "def authenticate_user(username, password):\n    return validate_credentials(username, password)",
        "def hash_password(password, salt):\n    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)",
        "def generate_token(user_id, username):\n    payload = {'user_id': user_id, 'username': username}\n    return jwt.encode(payload, secret_key)",
        
        # Python классы
        "class UserManager:\n    def __init__(self):\n        self.users = {}\n    def create_user(self, username, email):\n        pass",
        "class DatabaseConnection:\n    def __init__(self, db_path):\n        self.connection = sqlite3.connect(db_path)",
        "class AuthenticationError(Exception):\n    def __init__(self, message):\n        super().__init__(message)",
        
        # JavaScript функции
        "function connectToDatabase() {\n    return new DatabaseConnection();\n}",
        "const validateEmail = (email) => /^[^@]+@[^@]+\\.[^@]+$/.test(email);",
        "async function fetchUserData(userId) {\n    return await api.get(`/users/${userId}`);\n}",
        
        # SQL запросы
        "SELECT * FROM users WHERE active = true ORDER BY created_at DESC",
        "UPDATE users SET last_login = NOW() WHERE id = ?",
        "CREATE INDEX idx_users_email ON users(email) WHERE active = true"
    ]


@pytest.fixture
def sample_search_queries():
    """Набор тестовых поисковых запросов"""
    return [
        "user authentication function",
        "password hashing algorithm", 
        "database connection class",
        "JWT token generation",
        "email validation",
        "SQL query users",
        "error handling exception",
        "async API call",
        "user management system",
        "security validation"
    ]


@pytest.fixture
def mock_qdrant_client():
    """Стандартный mock Qdrant клиента для тестов"""
    with patch('rag.vector_store.QdrantClient') as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Настраиваем стандартные ответы
        mock_client.get_collection.return_value = Mock(
            vectors_count=100,
            indexed_vectors_count=100,
            points_count=100,
            status='green'
        )
        mock_client.create_collection.return_value = True
        mock_client.delete_collection.return_value = True
        mock_client.upsert.return_value = Mock(status='completed')
        mock_client.search.return_value = [
            Mock(
                id='test_1',
                score=0.95,
                payload={
                    'content': 'def authenticate_user(username, password):',
                    'file_path': 'auth/middleware.py',
                    'file_name': 'middleware.py',
                    'chunk_name': 'authenticate_user',
                    'chunk_type': 'function',
                    'language': 'python',
                    'start_line': 45,
                    'end_line': 50
                }
            ),
            Mock(
                id='test_2',
                score=0.87,
                payload={
                    'content': 'class UserManager:',
                    'file_path': 'auth/user.py',
                    'file_name': 'user.py', 
                    'chunk_name': 'UserManager',
                    'chunk_type': 'class',
                    'language': 'python',
                    'start_line': 20,
                    'end_line': 30
                }
            )
        ]
        mock_client.get_cluster_info.return_value = Mock(
            peer_id='test-peer-123',
            peers=[],
            raft_info={}
        )
        
        yield mock_client


@pytest.fixture
def mock_fastembed_embedder():
    """Mock FastEmbed эмбеддера"""
    with patch('rag.embedder.FASTEMBED_AVAILABLE', True):
        with patch('rag.embedder.TextEmbedding') as mock_text_embedding:
            mock_model = Mock()
            mock_text_embedding.return_value = mock_model
            
            # Функция генерации эмбеддингов
            def generate_embeddings(texts):
                return [np.random.random(384).astype(np.float32) for _ in texts]
            
            mock_model.embed = generate_embeddings
            
            yield mock_model


@pytest.fixture
def mock_sentence_transformers_embedder():
    """Mock Sentence Transformers эмбеддера"""
    with patch('rag.embedder.SENTENCE_TRANSFORMERS_AVAILABLE', True):
        with patch('rag.embedder.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_st.return_value = mock_model
            
            # Функция генерации эмбеддингов
            def generate_embeddings(texts, **kwargs):
                return np.array([np.random.random(384).astype(np.float32) for _ in texts])
            
            mock_model.encode = generate_embeddings
            
            yield mock_model


@pytest.fixture
def test_repository_path():
    """Путь к тестовому репозиторию с фиксированными файлами"""
    return "tests/fixtures/test_repo"


@pytest.fixture
def temp_test_repo():
    """Создает временный тестовый репозиторий"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Создаем структуру файлов
        test_files = {
            'auth/middleware.py': '''
def authenticate_request(token):
    """Middleware для аутентификации запросов"""
    if not token:
        raise AuthenticationError("Token required")
    return validate_token(token)

class AuthenticationError(Exception):
    pass
''',
            'auth/user.py': '''
class UserManager:
    """Менеджер пользователей"""
    def __init__(self):
        self.users = {}
    
    def create_user(self, username, email):
        user = User(username, email)
        self.users[username] = user
        return user

class User:
    def __init__(self, username, email):
        self.username = username
        self.email = email
''',
            'db/connection.py': '''
import sqlite3

class DatabaseConnection:
    """Подключение к базе данных"""
    def __init__(self, db_path):
        self.connection = sqlite3.connect(db_path)
    
    def execute_query(self, query, params=None):
        cursor = self.connection.cursor()
        return cursor.execute(query, params or [])
''',
            'utils/helpers.py': '''
def format_date(date_obj):
    """Форматирует дату в строку"""
    return date_obj.strftime("%Y-%m-%d %H:%M:%S")

def validate_email(email):
    """Валидация email адреса"""
    import re
    pattern = r'^[^@]+@[^@]+\\.[^@]+$'
    return bool(re.match(pattern, email))
'''
        }
        
        # Создаем файлы
        for file_path, content in test_files.items():
            full_path = Path(temp_dir) / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content.strip())
        
        yield temp_dir


@pytest.fixture
def sample_vector_points():
    """Тестовые векторные точки для индексации"""
    points = []
    
    for i in range(10):
        points.append({
            'id': f'test_point_{i}',
            'vector': np.random.random(384).tolist(),
            'payload': {
                'content': f'test content {i}',
                'file_path': f'test/file_{i}.py',
                'file_name': f'file_{i}.py',
                'chunk_name': f'test_function_{i}',
                'chunk_type': 'function',
                'language': 'python',
                'start_line': i * 10,
                'end_line': i * 10 + 5,
                'tokens_estimate': 20 + i,
                'indexed_at': '2024-01-01T12:00:00.000Z'
            }
        })
    
    return points


@pytest.fixture
def mock_file_scanner():
    """Mock FileScanner для тестов"""
    with patch('file_scanner.FileScanner') as mock_scanner_class:
        mock_scanner = Mock()
        mock_scanner_class.return_value = mock_scanner
        
        def create_file_info(path, name, language, size=100):
            file_info = Mock()
            file_info.path = path
            file_info.name = name
            file_info.language = language
            file_info.size = size
            return file_info
        
        # Стандартный набор файлов
        test_files = [
            create_file_info("auth/middleware.py", "middleware.py", "python", 150),
            create_file_info("auth/user.py", "user.py", "python", 200),
            create_file_info("db/connection.py", "connection.py", "python", 120),
            create_file_info("utils/helpers.py", "helpers.py", "python", 180)
        ]
        
        mock_scanner.scan_repository.return_value = test_files
        
        yield mock_scanner


@pytest.fixture
def mock_parser_registry():
    """Mock ParserRegistry для тестов"""
    with patch('parsers.base_parser.ParserRegistry') as mock_registry_class:
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        
        mock_parser = Mock()
        mock_registry.get_parser.return_value = mock_parser
        
        def create_parsed_file(file_info):
            parsed = Mock()
            parsed.file_info = file_info
            return parsed
        
        mock_parser.safe_parse.side_effect = create_parsed_file
        
        yield mock_registry


@pytest.fixture  
def mock_code_chunker():
    """Mock CodeChunker для тестов"""
    with patch('code_chunker.CodeChunker') as mock_chunker_class:
        mock_chunker = Mock()
        mock_chunker_class.return_value = mock_chunker
        
        def create_chunks(parsed_file):
            file_name = Path(parsed_file.file_info.path).name
            chunks = []
            
            # Создаем чанки в зависимости от типа файла
            if 'middleware' in file_name:
                chunk1 = Mock()
                chunk1.name = "authenticate_request"
                chunk1.chunk_type = "function"
                chunk1.start_line = 1
                chunk1.end_line = 8
                chunk1.content = "def authenticate_request(token):"
                chunk1.tokens_estimate = 25
                chunks.append(chunk1)
                
                chunk2 = Mock()
                chunk2.name = "AuthenticationError"
                chunk2.chunk_type = "class"
                chunk2.start_line = 10
                chunk2.end_line = 12
                chunk2.content = "class AuthenticationError(Exception):"
                chunk2.tokens_estimate = 15
                chunks.append(chunk2)
                
            elif 'user' in file_name:
                chunk1 = Mock()
                chunk1.name = "UserManager"
                chunk1.chunk_type = "class"
                chunk1.start_line = 1
                chunk1.end_line = 15
                chunk1.content = "class UserManager:"
                chunk1.tokens_estimate = 40
                chunks.append(chunk1)
                
            elif 'connection' in file_name:
                chunk1 = Mock()
                chunk1.name = "DatabaseConnection"
                chunk1.chunk_type = "class"
                chunk1.start_line = 3
                chunk1.end_line = 12
                chunk1.content = "class DatabaseConnection:"
                chunk1.tokens_estimate = 30
                chunks.append(chunk1)
                
            elif 'helpers' in file_name:
                chunk1 = Mock()
                chunk1.name = "format_date"
                chunk1.chunk_type = "function"
                chunk1.start_line = 1
                chunk1.end_line = 4
                chunk1.content = "def format_date(date_obj):"
                chunk1.tokens_estimate = 20
                chunks.append(chunk1)
                
                chunk2 = Mock()
                chunk2.name = "validate_email"
                chunk2.chunk_type = "function"
                chunk2.start_line = 6
                chunk2.end_line = 10
                chunk2.content = "def validate_email(email):"
                chunk2.tokens_estimate = 18
                chunks.append(chunk2)
            
            return chunks
        
        mock_chunker.chunk_parsed_file.side_effect = create_chunks
        
        yield mock_chunker


@pytest.fixture
def rag_test_environment(test_rag_config, mock_qdrant_client, mock_fastembed_embedder,
                        mock_file_scanner, mock_parser_registry, mock_code_chunker):
    """Полная тестовая среда RAG с всеми mocks"""
    return {
        'config': test_rag_config,
        'qdrant_client': mock_qdrant_client,
        'embedder_model': mock_fastembed_embedder,
        'file_scanner': mock_file_scanner,
        'parser_registry': mock_parser_registry,
        'code_chunker': mock_code_chunker
    }


@pytest.fixture
async def initialized_rag_components(test_rag_config, mock_qdrant_client, mock_fastembed_embedder):
    """Полностью инициализированные RAG компоненты"""
    # Создаем компоненты
    embedder = CPUEmbedder(test_rag_config.embeddings, test_rag_config.parallelism)
    vector_store = QdrantVectorStore(test_rag_config.vector_store)
    
    # Инициализируем векторное хранилище
    await vector_store.initialize_collection(recreate=True)
    
    # Создаем Query Engine
    query_engine = CPUQueryEngine(
        embedder=embedder,
        store=vector_store,
        qcfg=test_rag_config.query_engine
    )
    
    yield {
        'embedder': embedder,
        'vector_store': vector_store,
        'query_engine': query_engine
    }
    
    # Cleanup
    try:
        await query_engine.close()
        await vector_store.close()
    except Exception:
        pass


@pytest.fixture
def performance_test_data():
    """Данные для тестов производительности"""
    return {
        'small_dataset': 50,    # текстов
        'medium_dataset': 500,  # текстов  
        'large_dataset': 2000,  # текстов
        'concurrent_users': 10,
        'queries_per_user': 5,
        'performance_thresholds': {
            'index_rate_min': 10,      # файлов/сек
            'search_latency_max': 0.5, # секунд
            'memory_limit_mb': 500,    # МБ для 1000 документов
            'throughput_min': 20       # запросов/сек
        }
    }


@pytest.fixture
def mock_search_results():
    """Стандартные mock результаты поиска"""
    return [
        {
            'id': 'result_1',
            'score': 0.95,
            'payload': {
                'content': 'def authenticate_user(username, password):',
                'file_path': 'auth/middleware.py',
                'file_name': 'middleware.py',
                'chunk_name': 'authenticate_user',
                'chunk_type': 'function',
                'language': 'python',
                'start_line': 10,
                'end_line': 15
            }
        },
        {
            'id': 'result_2', 
            'score': 0.87,
            'payload': {
                'content': 'class UserManager:',
                'file_path': 'auth/user.py',
                'file_name': 'user.py',
                'chunk_name': 'UserManager', 
                'chunk_type': 'class',
                'language': 'python',
                'start_line': 25,
                'end_line': 35
            }
        },
        {
            'id': 'result_3',
            'score': 0.82,
            'payload': {
                'content': 'function connectToDatabase() {',
                'file_path': 'db/connection.js',
                'file_name': 'connection.js',
                'chunk_name': 'connectToDatabase',
                'chunk_type': 'function', 
                'language': 'javascript',
                'start_line': 5,
                'end_line': 10
            }
        }
    ]


@pytest.fixture
def clean_test_environment():
    """Очищает окружение перед каждым тестом"""
    # Сброс глобальных переменных если есть
    import gc
    gc.collect()
    
    yield
    
    # Cleanup после теста
    gc.collect()


# Утилиты для тестирования
def assert_valid_search_result(result: Dict[str, Any]) -> None:
    """Проверяет корректность структуры результата поиска"""
    required_fields = ['id', 'score', 'payload']
    for field in required_fields:
        assert field in result, f"Отсутствует обязательное поле: {field}"
    
    assert isinstance(result['score'], (int, float)), "Score должен быть числом"
    assert 0 <= result['score'] <= 1, f"Score должен быть в диапазоне 0-1: {result['score']}"
    
    payload = result['payload']
    payload_fields = ['content', 'file_path', 'chunk_name', 'language']
    for field in payload_fields:
        assert field in payload, f"Отсутствует поле payload: {field}"


def assert_performance_within_limits(metrics: Dict[str, Any], thresholds: Dict[str, float]) -> None:
    """Проверяет соответствие метрик производительности пороговым значениям"""
    if 'duration_sec' in metrics and 'max_duration' in thresholds:
        assert metrics['duration_sec'] <= thresholds['max_duration'], \
            f"Превышено время выполнения: {metrics['duration_sec']}s > {thresholds['max_duration']}s"
    
    if 'throughput_per_sec' in metrics and 'min_throughput' in thresholds:
        assert metrics['throughput_per_sec'] >= thresholds['min_throughput'], \
            f"Низкий throughput: {metrics['throughput_per_sec']} < {thresholds['min_throughput']}"
    
    if 'memory_peak_mb' in metrics and 'max_memory_mb' in thresholds:
        assert metrics['memory_peak_mb'] <= thresholds['max_memory_mb'], \
            f"Превышено потребление памяти: {metrics['memory_peak_mb']}MB > {thresholds['max_memory_mb']}MB"


# Параметризованные тесты
@pytest.fixture(params=[
    ('fastembed', 'BAAI/bge-small-en-v1.5'),
    ('sentence-transformers', 'all-MiniLM-L6-v2')
])
def embedder_provider_config(request, test_rag_config):
    """Параметризованная конфигурация для разных провайдеров эмбеддингов"""
    provider, model_name = request.param
    
    config = test_rag_config
    config.embeddings.provider = provider
    config.embeddings.model_name = model_name
    
    return config


@pytest.fixture(params=[8, 16, 32, 64])
def batch_size_config(request, test_rag_config):
    """Параметризованная конфигурация для разных размеров батчей"""
    batch_size = request.param
    
    config = test_rag_config  
    config.embeddings.batch_size_min = batch_size // 2
    config.embeddings.batch_size_max = batch_size
    
    return config


@pytest.fixture(params=['cosine', 'dot', 'euclidean'])
def distance_metric_config(request, test_rag_config):
    """Параметризованная конфигурация для разных метрик расстояния"""
    distance = request.param
    
    config = test_rag_config
    config.vector_store.distance = distance
    
    return config


# Фикстуры для автоматической очистки
@pytest.fixture(autouse=True)
def reset_environment_vars():
    """Автоматически сбрасывает переменные окружения после тестов"""
    original_env = os.environ.copy()
    
    yield
    
    # Восстанавливаем оригинальные переменные
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True, scope="function")
def cleanup_temp_files():
    """Автоматически очищает временные файлы после каждого теста"""
    yield
    
    # Принудительная сборка мусора для освобождения файловых дескрипторов
    import gc
    gc.collect()


# Маркеры для быстрого запуска тестов
def pytest_configure(config):
    """Конфигурация pytest для RAG тестов"""
    config.addinivalue_line(
        "markers", 
        "unit: Быстрые unit тесты компонентов RAG"
    )
    config.addinivalue_line(
        "markers",
        "integration_rag: Интеграционные тесты RAG компонентов"
    )
    config.addinivalue_line(
        "markers",
        "e2e_rag: End-to-End тесты RAG через CLI"
    )
    config.addinivalue_line(
        "markers", 
        "perf_rag: Тесты производительности RAG"
    )


def pytest_collection_modifyitems(config, items):
    """Автоматически добавляет маркеры к RAG тестам"""
    for item in items:
        # Добавляем маркер rag ко всем тестам в директории tests/rag/
        if "tests/rag/" in str(item.fspath):
            item.add_marker(pytest.mark.rag)
            
        # Добавляем маркеры по типам тестов
        if "integration" in item.name:
            item.add_marker(pytest.mark.integration_rag)
        elif "e2e" in item.name or "cli" in item.name:
            item.add_marker(pytest.mark.e2e_rag)
        elif "performance" in item.name or "perf" in item.name:
            item.add_marker(pytest.mark.perf_rag)
        else:
            item.add_marker(pytest.mark.unit)
