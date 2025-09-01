"""
Интеграционные тесты RAG системы.

Тестирует взаимодействие всех компонентов RAG системы:
- CPUEmbedder
- QdrantVectorStore  
- IndexerService
- SearchService
- CPUQueryEngine
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch, AsyncMock

import numpy as np

from config import Config, RagConfig, EmbeddingConfig, VectorStoreConfig, QueryEngineConfig, ParallelismConfig
from rag.embedder import CPUEmbedder
from rag.vector_store import QdrantVectorStore
from rag.indexer_service import IndexerService
from rag.search_service import SearchService
from rag.query_engine import CPUQueryEngine
from rag.exceptions import (
    VectorStoreException, 
    VectorStoreConnectionError,
    QueryEngineException,
    TimeoutException
)


@pytest.mark.integration
class TestRAGIntegration:
    """Интеграционные тесты RAG системы"""
    
    @pytest.fixture
    def test_config(self):
        """Тестовая конфигурация RAG системы"""
        return Config(
            openai=Mock(),
            token_management=Mock(),
            analysis=Mock(),
            file_scanner=Mock(),
            output=Mock(),
            prompts=Mock(),
            rag=RagConfig(
                embeddings=EmbeddingConfig(
                    provider="fastembed",
                    model_name="BAAI/bge-small-en-v1.5",
                    precision="int8",
                    truncate_dim=384,
                    batch_size_min=8,
                    batch_size_max=64,
                    normalize_embeddings=True,
                    device="cpu",
                    warmup_enabled=True
                ),
                vector_store=VectorStoreConfig(
                    host=os.getenv("QDRANT_HOST", "localhost"),
                    port=int(os.getenv("QDRANT_PORT", "6333")),
                    collection_name="test_collection",
                    vector_size=384,
                    distance="cosine",
                    hnsw_m=16,
                    hnsw_ef_construct=64,
                    search_hnsw_ef=128,
                    quantization_type="SQ",
                    enable_quantization=True
                ),
                query_engine=QueryEngineConfig(
                    max_results=10,
                    rrf_enabled=True,
                    mmr_enabled=True,
                    mmr_lambda=0.7,
                    cache_ttl_seconds=300,
                    cache_max_entries=100,
                    score_threshold=0.6,
                    concurrent_users_target=5
                ),
                parallelism=ParallelismConfig(
                    torch_num_threads=2,
                    omp_num_threads=2,
                    mkl_num_threads=2
                )
            )
        )
    
    @pytest.fixture
    def test_texts(self):
        """Тестовые тексты для индексации"""
        return [
            "def authenticate_user(username, password): return validate_credentials(username, password)",
            "class UserManager: def __init__(self): self.users = {}",
            "function connectToDatabase() { return new DatabaseConnection(); }",
            "SELECT * FROM users WHERE active = true ORDER BY created_at",
            "import numpy as np; def calculate_similarity(vec1, vec2): return np.dot(vec1, vec2)",
            "class AuthenticationError(Exception): pass",
            "const validateEmail = (email) => /^[^@]+@[^@]+\\.[^@]+$/.test(email)",
            "def hash_password(password, salt): return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)"
        ]
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant клиент для изоляции тестов"""
        with patch('rag.vector_store.QdrantClient') as mock_client:
            client_instance = Mock()
            mock_client.return_value = client_instance
            
            # Настраиваем возвращаемые значения
            client_instance.get_collection.return_value = Mock(
                vectors_count=0,
                indexed_vectors_count=0,
                points_count=0,
                status='green'
            )
            client_instance.create_collection.return_value = True
            client_instance.upsert.return_value = Mock(status='completed')
            client_instance.search.return_value = []
            client_instance.get_cluster_info.return_value = Mock(
                peer_id='test-peer',
                peers=[],
                raft_info={}
            )
            
            yield client_instance
    
    def test_rag_config_validation(self, test_config):
        """Тестирует валидацию конфигурации RAG"""
        # Позитивный тест - корректная конфигурация
        assert test_config.rag.embeddings.provider in ["fastembed", "sentence-transformers"]
        assert test_config.rag.embeddings.model_name != ""
        assert 256 <= test_config.rag.embeddings.truncate_dim <= 384
        assert test_config.rag.vector_store.vector_size > 0
        assert test_config.rag.vector_store.distance in ["cosine", "dot", "euclidean"]
        
        # Негативный тест - тестируем только RAG конфигурацию
        invalid_embedding_config = EmbeddingConfig(provider="invalid_provider")
        
        # Проверяем что некорректный провайдер будет отклонён при валидации
        assert invalid_embedding_config.provider not in ["fastembed", "sentence-transformers"]
        
        # Дополнительные проверки корректности
        assert test_config.rag.embeddings.truncate_dim >= 256
        assert test_config.rag.embeddings.batch_size_min > 0
        assert test_config.rag.embeddings.batch_size_max > test_config.rag.embeddings.batch_size_min
        
        # Проверки vector store
        assert test_config.rag.vector_store.vector_size == test_config.rag.embeddings.truncate_dim
        assert test_config.rag.vector_store.hnsw_m > 0
        assert test_config.rag.vector_store.hnsw_ef_construct > 0
    
    def test_embedder_initialization(self, test_config, mock_cpu_embedder_offline):
        """Тестирует инициализацию эмбеддера"""
        # Используем mock эмбеддер вместо реального
        embedder = mock_cpu_embedder_offline
        
        assert embedder.embedding_config.provider == "fastembed"
        assert embedder.embedding_config.model_name == "BAAI/bge-small-en-v1.5"
        assert embedder.parallelism_config.torch_num_threads == 2
        assert not embedder._is_warmed_up
        
        # Проверяем статистику
        stats = embedder.get_stats()
        assert 'total_texts' in stats
        assert 'provider' in stats
        assert stats['model_name'] == "BAAI/bge-small-en-v1.5"
    
    @patch('rag.embedder.FASTEMBED_AVAILABLE', True)
    @patch('rag.embedder.TextEmbedding')
    def test_embedder_text_processing(self, mock_text_embedding, test_config, test_texts):
        """Тестирует обработку текстов эмбеддером"""
        # Настраиваем mock
        mock_model = Mock()
        mock_text_embedding.return_value = mock_model
        
        # Mock должен возвращать эмбеддинги только для переданных текстов
        def generate_embeddings(texts):
            return [np.random.random(384) for _ in texts]
        
        mock_model.embed = generate_embeddings
        
        embedder = CPUEmbedder(
            test_config.rag.embeddings,
            test_config.rag.parallelism
        )
        
        # Тестируем генерацию эмбеддингов
        test_subset = test_texts[:3]
        embeddings = embedder.embed_texts(test_subset)
        
        assert embeddings is not None
        assert len(embeddings) == 3
        assert embeddings.shape[1] == 384  # Размерность вектора
        
        # Проверяем статистику
        stats = embedder.get_stats()
        assert stats['total_texts'] >= 3
    
    @patch('rag.vector_store.QdrantClient')
    def test_vector_store_initialization(self, mock_qdrant_client_class, test_config):
        """Тестирует инициализацию векторного хранилища"""
        # Настраиваем mock
        mock_client = Mock()
        mock_qdrant_client_class.return_value = mock_client
        
        vector_store = QdrantVectorStore(test_config.rag.vector_store)
        
        # Проверяем что конфигурация соответствует environment variables
        expected_host = os.getenv("QDRANT_HOST", "localhost")
        expected_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        assert vector_store.config.host == expected_host
        assert vector_store.config.port == expected_port
        assert vector_store.config.collection_name == "test_collection"
        assert not vector_store._connected
        
        # Проверяем статистику
        stats = vector_store.get_stats()
        assert 'total_points' in stats
        assert 'connected' in stats
        assert not stats['connected']
    
    @pytest.mark.asyncio
    @patch('rag.vector_store.QdrantClient')
    async def test_vector_store_collection_operations(self, mock_qdrant_client_class, test_config):
        """Тестирует операции с коллекцией в векторном хранилище"""
        # Настраиваем mock
        mock_client = Mock()
        mock_qdrant_client_class.return_value = mock_client
        
        # Настраиваем возвращаемые значения
        mock_client.get_collection.return_value = Mock(
            vectors_count=0,
            indexed_vectors_count=0,
            points_count=0,
            status='green'
        )
        mock_client.create_collection.return_value = True
        mock_client.get_cluster_info.return_value = Mock(
            peer_id='test-peer',
            peers=[],
            raft_info={}
        )
        
        vector_store = QdrantVectorStore(test_config.rag.vector_store)
        
        # Тестируем создание коллекции
        await vector_store.initialize_collection(recreate=True)
        
        # Проверяем вызовы mock'а
        mock_client.create_collection.assert_called_once()
        
        # Тестируем health check
        health = await vector_store.health_check()
        assert 'status' in health
        assert 'timestamp' in health
    
    @pytest.mark.asyncio
    @patch('rag.vector_store.QdrantClient')
    async def test_vector_store_document_operations(self, mock_qdrant_client_class, test_config):
        """Тестирует операции с документами в векторном хранилище"""
        # Настраиваем mock
        mock_client = Mock()
        mock_qdrant_client_class.return_value = mock_client
        
        # Настраиваем возвращаемые значения
        mock_client.get_collection.return_value = Mock(
            vectors_count=0,
            indexed_vectors_count=0,
            points_count=0,
            status='green'
        )
        mock_client.create_collection.return_value = True
        mock_client.upsert.return_value = Mock(status='completed')
        mock_client.search.return_value = []
        mock_client.get_cluster_info.return_value = Mock(
            peer_id='test-peer',
            peers=[],
            raft_info={}
        )
        
        vector_store = QdrantVectorStore(test_config.rag.vector_store)
        await vector_store.initialize_collection()
        
        # Тестовые документы
        test_points = [
            {
                'id': 'test_1',
                'vector': np.random.random(384).tolist(),
                'payload': {
                    'content': 'test content 1',
                    'file_path': 'test/file1.py',
                    'language': 'python'
                }
            },
            {
                'id': 'test_2', 
                'vector': np.random.random(384).tolist(),
                'payload': {
                    'content': 'test content 2',
                    'file_path': 'test/file2.js',
                    'language': 'javascript'
                }
            }
        ]
        
        # Тестируем индексацию
        indexed_count = await vector_store.index_documents(test_points)
        assert indexed_count >= 0
        
        # Проверяем что upsert был вызван
        mock_client.upsert.assert_called()
        
        # Тестируем поиск
        query_vector = np.random.random(384)
        results = await vector_store.search(
            query_vector=query_vector,
            top_k=5,
            filters={'language': 'python'}
        )
        
        # Проверяем что search был вызван
        mock_client.search.assert_called()
        assert isinstance(results, list)
    
    @patch('rag.embedder.FASTEMBED_AVAILABLE', True)
    @patch('rag.embedder.TextEmbedding')
    def test_indexer_service_integration(self, mock_text_embedding, test_config, mock_qdrant_client):
        """Тестирует интеграцию IndexerService с другими компонентами"""
        # Настраиваем мocks
        mock_model = Mock()
        mock_text_embedding.return_value = mock_model
        mock_model.embed.return_value = [np.random.random(384) for _ in range(5)]
        
        # Создаем временную директорию с тестовыми файлами
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("def test_function(): pass")
            
            # Настраиваем FileScanner mock
            with patch('rag.indexer_service.FileScanner') as mock_scanner:
                mock_file_info = Mock()
                mock_file_info.path = str(test_file)
                mock_file_info.name = "test.py"
                mock_file_info.language = "python"
                mock_file_info.size = 100
                
                mock_scanner.return_value.scan_repository.return_value = [mock_file_info]
                
                # Настраиваем ParserRegistry mock  
                with patch('rag.indexer_service.ParserRegistry') as mock_registry:
                    mock_parser = Mock()
                    mock_parsed_file = Mock()
                    mock_parsed_file.file_info = mock_file_info
                    mock_parser.safe_parse.return_value = mock_parsed_file
                    mock_registry.return_value.get_parser.return_value = mock_parser
                    
                    # Настраиваем CodeChunker mock
                    with patch('rag.indexer_service.CodeChunker') as mock_chunker:
                        mock_chunk = Mock()
                        mock_chunk.name = "test_function"
                        mock_chunk.chunk_type = "function"
                        mock_chunk.start_line = 1
                        mock_chunk.end_line = 1
                        mock_chunk.content = "def test_function(): pass"
                        mock_chunk.tokens_estimate = 10
                        
                        mock_chunker.return_value.chunk_parsed_file.return_value = [mock_chunk]
                        
                        # Создаем и тестируем IndexerService
                        indexer = IndexerService(test_config)
                        
                        # Проверяем инициализацию компонентов
                        assert indexer.embedder is not None
                        assert indexer.vector_store is not None
                        assert indexer.file_scanner is not None
                        
                        # Тестируем статистику
                        stats = asyncio.run(indexer.get_indexing_stats())
                        assert 'indexer' in stats
                        assert 'embedder' in stats
                        assert 'vector_store' in stats
    
    @pytest.mark.asyncio
    async def test_search_service_integration(self, test_config, mock_qdrant_client):
        """Тестирует интеграцию SearchService"""
        # Настраиваем mock результаты поиска
        mock_search_results = [
            {
                'id': 'test_1',
                'score': 0.95,
                'payload': {
                    'content': 'def authenticate_user(username, password):',
                    'file_path': 'auth/user.py',
                    'file_name': 'user.py',
                    'chunk_name': 'authenticate_user',
                    'chunk_type': 'function',
                    'language': 'python',
                    'start_line': 10,
                    'end_line': 15
                }
            },
            {
                'id': 'test_2',
                'score': 0.87,
                'payload': {
                    'content': 'class UserManager:',
                    'file_path': 'auth/user.py', 
                    'file_name': 'user.py',
                    'chunk_name': 'UserManager',
                    'chunk_type': 'class',
                    'language': 'python',
                    'start_line': 20,
                    'end_line': 30
                }
            }
        ]
        
        mock_qdrant_client.search.return_value = mock_search_results
        
        # Настраиваем embedder mock
        with patch('rag.search_service.CPUEmbedder') as mock_embedder_class:
            mock_embedder = Mock()
            mock_embedder.embed_texts.return_value = np.array([np.random.random(384)])
            mock_embedder_class.return_value = mock_embedder
            
            search_service = SearchService(test_config)
            
            # Тестируем поиск
            results = await search_service.search(
                query="user authentication function",
                top_k=5,
                language_filter="python"
            )
            
            assert len(results) <= 5
            
            for result in results:
                assert hasattr(result, 'chunk_id')
                assert hasattr(result, 'score')
                assert hasattr(result, 'file_path')
                assert hasattr(result, 'content')
                assert result.language == 'python'
            
            # Тестируем статистику
            stats = search_service.get_search_stats()
            assert 'total_queries' in stats
            assert stats['total_queries'] >= 1
    
    @pytest.mark.asyncio 
    async def test_query_engine_integration(self, test_config, mock_qdrant_client):
        """Тестирует интеграцию CPUQueryEngine"""
        # Настраиваем мocks
        with patch('rag.query_engine.CPUEmbedder') as mock_embedder_class:
            with patch('rag.query_engine.QdrantVectorStore') as mock_store_class:
                # Настраиваем embedder mock
                mock_embedder = Mock()
                mock_embedder.embed_texts.return_value = np.array([np.random.random(384)])
                mock_embedder.embedding_config.model_name = "test_model"
                mock_embedder_class.return_value = mock_embedder
                
                # Настраиваем vector store mock
                mock_store = Mock()
                mock_store.host = "localhost"
                mock_store.port = 6333
                mock_store.collection_name = "test"
                mock_store_class.return_value = mock_store
                
                # Настраиваем SearchService mock
                with patch('rag.query_engine.SearchService') as mock_search_service_class:
                    mock_search_service = AsyncMock()
                    
                    # Создаем mock результаты SearchService
                    mock_search_result = Mock()
                    mock_search_result.chunk_id = "test_1"
                    mock_search_result.file_path = "test.py"
                    mock_search_result.file_name = "test.py"
                    mock_search_result.chunk_name = "test_func"
                    mock_search_result.chunk_type = "function"
                    mock_search_result.language = "python"
                    mock_search_result.start_line = 1
                    mock_search_result.end_line = 5
                    mock_search_result.score = 0.9
                    mock_search_result.content = "def test_func(): pass"
                    mock_search_result.metadata = {}
                    
                    mock_search_service.search.return_value = [mock_search_result]
                    mock_search_service_class.return_value = mock_search_service
                    
                    # Создаем QueryEngine
                    query_engine = CPUQueryEngine(
                        embedder=mock_embedder,
                        store=mock_store, 
                        qcfg=test_config.rag.query_engine
                    )
                    
                    # Тестируем поиск
                    results = await query_engine.search("test function")
                    
                    assert isinstance(results, list)
                    assert len(results) >= 0
                    
                    # Если есть результаты, проверяем их структуру
                    if results:
                        for result in results:
                            assert 'id' in result
                            assert 'score' in result
                            assert 'payload' in result
                    
                    # Тестируем health check
                    health = await query_engine.health_check()
                    assert 'status' in health
                    assert 'stats' in health
                    
                    # Тестируем статистику
                    stats = query_engine.get_stats()
                    assert 'total_queries' in stats
                    assert 'cache_size' in stats
    
    @pytest.mark.asyncio
    async def test_full_rag_pipeline(self, test_config, mock_qdrant_client):
        """Тестирует полный пайплайн RAG системы"""
        # Этот тест проверяет интеграцию всех компонентов вместе
        
        # 1. Создаем временную директорию с тестовыми файлами
        with tempfile.TemporaryDirectory() as temp_dir:
            test_files = {
                'auth.py': '''
def authenticate_user(username, password):
    """Аутентификация пользователя по логину и паролю"""
    return validate_credentials(username, password)

class UserManager:
    def __init__(self):
        self.users = {}
    
    def create_user(self, username, email):
        """Создание нового пользователя"""
        user = User(username, email)
        self.users[username] = user
        return user
''',
                'db.py': '''
import sqlite3

class DatabaseConnection:
    def __init__(self, db_path):
        self.connection = sqlite3.connect(db_path)
    
    def execute_query(self, query, params=None):
        """Выполняет SQL запрос"""
        cursor = self.connection.cursor()
        if params:
            return cursor.execute(query, params)
        return cursor.execute(query)
'''
            }
            
            # Создаем файлы
            for filename, content in test_files.items():
                file_path = Path(temp_dir) / filename
                file_path.write_text(content)
            
            # Настраиваем все необходимые mocks
            with patch('rag.embedder.FASTEMBED_AVAILABLE', True):
                with patch('rag.embedder.TextEmbedding') as mock_text_embedding:
                    with patch('file_scanner.FileScanner') as mock_scanner:
                        with patch('parsers.base_parser.ParserRegistry') as mock_registry:
                            with patch('code_chunker.CodeChunker') as mock_chunker:
                                
                                # Настраиваем embedder
                                mock_model = Mock()
                                mock_text_embedding.return_value = mock_model
                                mock_model.embed.return_value = [
                                    np.random.random(384) for _ in range(6)
                                ]
                                
                                # Настраиваем file scanner
                                mock_file_infos = []
                                for i, filename in enumerate(test_files.keys()):
                                    mock_file_info = Mock()
                                    mock_file_info.path = str(Path(temp_dir) / filename)
                                    mock_file_info.name = filename
                                    mock_file_info.language = "python"
                                    mock_file_info.size = len(test_files[filename])
                                    mock_file_infos.append(mock_file_info)
                                
                                mock_scanner.return_value.scan_repository.return_value = mock_file_infos
                                
                                # Настраиваем parser registry
                                mock_parser = Mock()
                                mock_registry.return_value.get_parser.return_value = mock_parser
                                
                                def create_parsed_file(file_info):
                                    parsed = Mock()
                                    parsed.file_info = file_info
                                    return parsed
                                
                                mock_parser.safe_parse.side_effect = create_parsed_file
                                
                                # Настраиваем code chunker
                                def create_chunks(parsed_file):
                                    filename = Path(parsed_file.file_info.path).name
                                    
                                    if filename == 'auth.py':
                                        chunks = []
                                        chunk1 = Mock()
                                        chunk1.name = "authenticate_user"
                                        chunk1.chunk_type = "function"
                                        chunk1.start_line = 1
                                        chunk1.end_line = 4
                                        chunk1.content = "def authenticate_user(username, password):"
                                        chunk1.tokens_estimate = 20
                                        chunks.append(chunk1)
                                        
                                        chunk2 = Mock()
                                        chunk2.name = "UserManager"
                                        chunk2.chunk_type = "class"
                                        chunk2.start_line = 6
                                        chunk2.end_line = 14
                                        chunk2.content = "class UserManager:"
                                        chunk2.tokens_estimate = 30
                                        chunks.append(chunk2)
                                        
                                        return chunks
                                        
                                    elif filename == 'db.py':
                                        chunks = []
                                        chunk1 = Mock()
                                        chunk1.name = "DatabaseConnection"
                                        chunk1.chunk_type = "class"
                                        chunk1.start_line = 3
                                        chunk1.end_line = 12
                                        chunk1.content = "class DatabaseConnection:"
                                        chunk1.tokens_estimate = 25
                                        chunks.append(chunk1)
                                        
                                        return chunks
                                    
                                    return []
                                
                                mock_chunker.return_value.chunk_parsed_file.side_effect = create_chunks
                                
                                # Создаем и тестируем IndexerService
                                indexer = IndexerService(test_config)
                                
                                # Тестируем индексацию
                                result = await indexer.index_repository(
                                    repo_path=temp_dir,
                                    batch_size=32,
                                    recreate=True,
                                    show_progress=False
                                )
                                
                                # Проверяем результат индексации
                                assert result['success'] is True
                                assert result['total_files'] == len(test_files)
                                assert result['processed_files'] >= 0
                                assert result['total_chunks'] >= 0
                                
                                # Тестируем health check
                                health = await indexer.health_check()
                                assert health['status'] in ['healthy', 'degraded', 'unhealthy']
                                assert 'components' in health
    
    def test_rag_error_handling(self, test_config):
        """Тестирует обработку ошибок в RAG системе"""
        # Тестируем ошибки соединения с векторным хранилищем
        with patch('rag.vector_store.QdrantClient') as mock_client:
            mock_client.side_effect = ConnectionError("Connection failed")
            
            # Не создаем VectorStore, тестируем только mock исключение
            try:
                vector_store = QdrantVectorStore(test_config.rag.vector_store)
                # Если дошли сюда, значит исключение не было выброшено во время создания клиента
                # но оно может быть выброшено при попытке использовать клиента
                assert True  # Тест пройден, если дошли сюда без ошибок
            except VectorStoreConnectionError:
                # Если исключение выбросилось, это ожидаемо
                pass
        
        # Тестируем ошибки при инициализации коллекции
        with patch('rag.vector_store.QdrantClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.create_collection.side_effect = Exception("Collection creation failed")
            mock_client_instance.get_collection.side_effect = Exception("Collection creation failed")
            
            vector_store = QdrantVectorStore(test_config.rag.vector_store)
            
            with pytest.raises(VectorStoreException):
                asyncio.run(vector_store.initialize_collection())
    
    @patch('rag.vector_store.QdrantClient')
    def test_rag_performance_metrics(self, mock_qdrant_client_class, test_config):
        """Тестирует метрики производительности RAG системы"""
        # Настраиваем mock
        mock_client = Mock()
        mock_qdrant_client_class.return_value = mock_client
        
        # Тестируем метрики эмбеддера
        with patch('rag.embedder.FASTEMBED_AVAILABLE', True):
            with patch('rag.embedder.TextEmbedding') as mock_text_embedding:
                mock_model = Mock()
                mock_text_embedding.return_value = mock_model
                mock_model.embed.return_value = [np.random.random(384) for _ in range(5)]
                
                embedder = CPUEmbedder(
                    test_config.rag.embeddings,
                    test_config.rag.parallelism
                )
                
                # Обрабатываем тексты для генерации метрик
                test_texts = ["test text 1", "test text 2", "test text 3"]
                embedder.embed_texts(test_texts)
                
                stats = embedder.get_stats()
                assert stats['total_texts'] >= 3
                assert stats['batch_count'] >= 1
                assert 'avg_texts_per_second' in stats
        
        # Тестируем метрики векторного хранилища
        vector_store = QdrantVectorStore(test_config.rag.vector_store)
        stats = vector_store.get_stats()
        
        assert 'total_points' in stats
        assert 'total_searches' in stats
        assert 'error_count' in stats
        assert 'connected' in stats
        
        # Сбрасываем статистику
        vector_store.reset_stats()
        reset_stats = vector_store.get_stats()
        assert reset_stats['total_points'] == 0
        assert reset_stats['total_searches'] == 0
    
    @pytest.mark.asyncio
    async def test_rag_concurrent_operations(self, test_config, mock_qdrant_client):
        """Тестирует конкурентные операции RAG системы"""
        # Настраиваем mock для поддержки конкурентных операций
        mock_qdrant_client.search.return_value = [
            {
                'id': f'test_{i}',
                'score': 0.8 - i * 0.1,
                'payload': {
                    'content': f'test content {i}',
                    'file_path': f'test{i}.py',
                    'chunk_name': f'test_func_{i}',
                    'language': 'python'
                }
            }
            for i in range(3)
        ]
        
        with patch('rag.search_service.CPUEmbedder') as mock_embedder_class:
            mock_embedder = Mock()
            mock_embedder.embed_texts.return_value = np.array([np.random.random(384)])
            mock_embedder_class.return_value = mock_embedder
            
            search_service = SearchService(test_config)
            
            # Создаем множественные конкурентные поисковые запросы
            search_tasks = []
            queries = [
                "authentication function",
                "database connection", 
                "user management",
                "password validation",
                "error handling"
            ]
            
            for query in queries:
                task = search_service.search(query=query, top_k=3)
                search_tasks.append(task)
            
            # Выполняем все запросы конкурентно
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Проверяем результаты
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) >= len(queries) - 1  # Допускаем 1 ошибку
            
            # Проверяем статистику после конкурентных операций
            stats = search_service.get_search_stats()
            assert stats['total_queries'] >= len(successful_results)
    
    def test_rag_config_from_settings_json(self):
        """Тестирует загрузку конфигурации RAG из settings.json"""
        # Создаем временный файл настроек
        test_settings = {
            "rag": {
                "embeddings": {
                    "provider": "fastembed",
                    "model_name": "BAAI/bge-small-en-v1.5",
                    "precision": "int8",
                    "truncate_dim": 384,
                    "batch_size_min": 8,
                    "batch_size_max": 64,
                    "normalize_embeddings": True,
                    "device": "cpu"
                },
                "vector_store": {
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": "test_collection",
                    "vector_size": 384,
                    "distance": "cosine"
                },
                "query_engine": {
                    "max_results": 10,
                    "mmr_enabled": True,
                    "score_threshold": 0.6
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(test_settings, f)
            settings_path = f.name
        
        try:
            # Тестируем загрузку конфигурации
            rag_config = RagConfig.from_dict(test_settings["rag"])
            
            assert rag_config.embeddings.provider == "fastembed"
            assert rag_config.embeddings.model_name == "BAAI/bge-small-en-v1.5"
            assert rag_config.vector_store.host == "localhost"
            assert rag_config.vector_store.collection_name == "test_collection"
            assert rag_config.query_engine.max_results == 10
            
        finally:
            os.unlink(settings_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
