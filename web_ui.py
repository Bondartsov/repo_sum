#!/usr/bin/env python3
"""
Веб-интерфейс для анализатора репозиториев с генерацией MD документации.
Запуск: streamlit run web_ui.py
"""

import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
import json

# Настройка логирования для веб-интерфейса
logger = logging.getLogger(__name__)

# Импортируем компоненты программы
from config import get_config, reload_config
from file_scanner import FileScanner
from parsers.base_parser import ParserRegistry
from code_chunker import CodeChunker
from openai_integration import OpenAIManager
from doc_generator import DocumentationGenerator
from utils import (
    setup_logging,
    FileInfo,
    ParsedFile,
    GPTAnalysisRequest,
    GPTAnalysisResult,
    ensure_directory_exists,
    create_error_parsed_file,
    create_error_gpt_result,
)

# Импортируем RAG компоненты
try:
    from rag import (
        CPUEmbedder,
        QdrantVectorStore,
        CPUQueryEngine,
        IndexerService,
        SearchService,
        RagException,
        VectorStoreException,
        VectorStoreConnectionError
    )
    RAG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG компоненты недоступны: {e}")
    RAG_AVAILABLE = False
    # Заглушки для отсутствующих классов
    CPUEmbedder = None
    QdrantVectorStore = None
    CPUQueryEngine = None
    IndexerService = None
    SearchService = None
    RagException = Exception
    VectorStoreException = Exception
    VectorStoreConnectionError = Exception

# Настройка страницы
st.set_page_config(
    page_title="Анализатор репозиториев",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Настройка логирования для веб-интерфейса
setup_logging("INFO")


def validate_uploaded_file(uploaded_file) -> Tuple[bool, str]:
    """Валидирует загруженный файл на безопасность и корректность"""
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {'.zip'}
    
    # Проверка размера
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"Файл слишком большой: {uploaded_file.size / (1024*1024):.1f}MB. Максимум: {MAX_FILE_SIZE / (1024*1024):.0f}MB"
    
    # Проверка расширения
    file_ext = Path(uploaded_file.name).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"Неподдерживаемый тип файла: {file_ext}. Разрешены: {', '.join(ALLOWED_EXTENSIONS)}"
    
    return True, "OK"


def safe_extract_zip(zip_path: Path, extract_to: Path) -> Tuple[bool, str, Optional[str]]:
    """Безопасно извлекает ZIP архив с проверками на path traversal"""
    MAX_EXTRACTED_SIZE = 500 * 1024 * 1024  # 500MB
    MAX_FILES = 10000
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Проверяем содержимое архива
            file_count = 0
            total_size = 0
            
            for zip_info in zip_ref.infolist():
                file_count += 1
                total_size += zip_info.file_size
                
                # Проверка количества файлов
                if file_count > MAX_FILES:
                    return False, f"Слишком много файлов в архиве: {file_count}. Максимум: {MAX_FILES}", None
                
                # Проверка общего размера
                if total_size > MAX_EXTRACTED_SIZE:
                    return False, f"Архив слишком большой: {total_size / (1024*1024):.1f}MB. Максимум: {MAX_EXTRACTED_SIZE / (1024*1024):.0f}MB", None
                
                # Проверка на path traversal
                if '..' in zip_info.filename or zip_info.filename.startswith('/'):
                    return False, f"Потенциально опасный путь в архиве: {zip_info.filename}", None
                
                # Проверка на системные файлы
                if any(dangerous in zip_info.filename.lower() for dangerous in ['.exe', '.bat', '.sh', '.cmd']):
                    logger.warning(f"Подозрительный файл в архиве: {zip_info.filename}")
            
            # Извлекаем архив
            zip_ref.extractall(extract_to)
            
            # Находим корневую папку
            extracted_dirs = [d for d in extract_to.iterdir() if d.is_dir()]
            if extracted_dirs:
                repo_path = str(extracted_dirs[0])
            else:
                repo_path = str(extract_to)
            
            return True, "OK", repo_path
            
    except zipfile.BadZipFile:
        return False, "Поврежденный ZIP архив", None
    except Exception as e:
        return False, f"Ошибка извлечения архива: {e}", None


class WebRepositoryAnalyzer:
    """Адаптер основного анализатора для веб-интерфейса"""
    
    def __init__(self):
        # Загружаем конфигурацию без требования API ключа на старте
        self.config = get_config(require_api_key=False)
        self.file_scanner = FileScanner()
        self.parser_registry = ParserRegistry()
        self.code_chunker = CodeChunker()
        self.openai_manager = None
        self.doc_generator = DocumentationGenerator()
        
    def initialize_with_api_key(self, api_key: str) -> bool:
        logger.debug(f"initialize_with_api_key: api_key length={len(api_key) if api_key else 0}, env_key_set={bool(os.getenv('OPENAI_API_KEY'))}")
        """Инициализирует компоненты с API ключом"""
        try:
            # Проверяем что API ключ не пустой
            if not api_key or not api_key.strip():
                logger.error("API ключ пустой или не указан")
                return False
            
            # Устанавливаем переменную окружения
            os.environ['OPENAI_API_KEY'] = api_key.strip()
            logger.debug("initialize_with_api_key: API key set in environment")
            
            # Перезагружаем конфигурацию
            reload_config()
            self.config = get_config()
            
            # Создаем OpenAI manager с проверкой
            self.openai_manager = OpenAIManager()
            
            logger.info("OpenAI API инициализирован успешно")
            return True
            
        except ValueError as e:
            logger.exception("Ошибка валидации API ключа")
            return False
        except Exception as e:
            logger.exception("Ошибка инициализации с API ключом")
            return False
    
    async def analyze_repository(self, repo_path: str, output_path: str, progress_callback=None) -> Dict[str, Any]:
        """Анализирует репозиторий с обратными вызовами для прогресса"""
        try:
            logger.info(f"Начинаем анализ репозитория: {repo_path}")
            
            # Создаем выходную директорию
            ensure_directory_exists(output_path)
            
            # Сканируем файлы
            if progress_callback:
                progress_callback("Сканирование файлов...", 0)
            
            files_to_analyze = list(self.file_scanner.scan_repository(repo_path))
            
            if not files_to_analyze:
                return {'success': False, 'error': 'Не найдено файлов для анализа'}
            
            logger.info(f"Найдено {len(files_to_analyze)} файлов для анализа")
            
            # Анализируем файлы
            analyzed_files = []
            total_files = len(files_to_analyze)
            
            for i, file_info in enumerate(files_to_analyze):
                try:
                    if progress_callback:
                        progress = int((i / total_files) * 100)
                        progress_callback(f"Анализ: {Path(file_info.path).name}", progress)
                    
                    # Анализируем файл
                    parsed_file, gpt_result = await self._analyze_single_file(file_info)
                    analyzed_files.append((parsed_file, gpt_result))
                    
                except Exception as e:
                    logger.error(f"Ошибка при анализе {file_info.path}: {e}")
                    # Единообразно формируем результаты ошибок
                    analyzed_files.append((
                        create_error_parsed_file(file_info, e),
                        create_error_gpt_result(e)
                    ))
            
            # Генерируем документацию
            if progress_callback:
                progress_callback("Генерация документации...", 90)
            
            result = self.doc_generator.generate_complete_documentation(
                analyzed_files, output_path, repo_path
            )
            
            if progress_callback:
                progress_callback("Завершено!", 100)
            
            # Добавляем статистику токенов
            if self.openai_manager:
                token_stats = self.openai_manager.get_token_usage_stats()
                result['token_stats'] = token_stats
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка анализа: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _analyze_single_file(self, file_info: FileInfo):
        """Анализирует один файл"""
        # Получаем парсер
        parser = self.parser_registry.get_parser(file_info.path)
        if not parser:
            raise ValueError(f"Не найден парсер для файла {file_info.path}")
        
        # Парсим файл
        parsed_file = parser.safe_parse(file_info)
        
        # Разбиваем на чанки
        chunks = self.code_chunker.chunk_parsed_file(parsed_file)
        
        # Создаем запрос к GPT
        gpt_request = GPTAnalysisRequest(
            chunks=chunks,
            file_path=file_info.path,
            language=file_info.language
        )
        
        # Анализируем через GPT
        gpt_result = await self.openai_manager.analyze_code(gpt_request)
        
        return parsed_file, gpt_result
    
    def get_repository_stats(self, repo_path: str) -> Dict[str, Any]:
        """Получает статистику репозитория"""
        return self.file_scanner.get_repository_stats(repo_path)


@st.cache_resource
def get_analyzer():
    """Кэшированный экземпляр анализатора"""
    return WebRepositoryAnalyzer()


@st.cache_resource
def init_rag_components():
    """
    Инициализирует RAG компоненты с кэшированием.
    
    Returns:
        Tuple[Optional[SearchService], Optional[CPUQueryEngine], Optional[IndexerService], str]:
            Кортеж (search_service, query_engine, indexer_service, status_message)
    """
    if not RAG_AVAILABLE:
        return None, None, None, "RAG компоненты недоступны"
    
    try:
        # Загружаем конфигурацию
        config = get_config(require_api_key=False)
        
        # Инициализируем компоненты
        embedder = CPUEmbedder(config.rag.embeddings, config.rag.parallelism)
        vector_store = QdrantVectorStore(config.rag.vector_store)
        search_service = SearchService(config, silent_mode=True)
        query_engine = CPUQueryEngine(embedder, vector_store, config.rag.query_engine)
        indexer_service = IndexerService(config)
        
        logger.info("RAG компоненты успешно инициализированы")
        return search_service, query_engine, indexer_service, "RAG система готова"
        
    except Exception as e:
        logger.error(f"Ошибка инициализации RAG компонентов: {e}")
        return None, None, None, f"Ошибка RAG системы: {e}"


def get_current_api_key() -> Optional[str]:
    """
    Получает текущий API ключ из различных источников в правильном порядке приоритета.
    
    Returns:
        API ключ или None если не найден
    """
    # 1. Сначала проверяем session state (если пользователь ввел вручную)
    if 'manual_api_key' in st.session_state and st.session_state.manual_api_key:
        api_key = st.session_state.manual_api_key.strip()
        if api_key and not api_key.startswith('your_'):  # Проверяем что это не плейсхолдер
            return api_key
    
    # 2. Затем проверяем переменные окружения
    env_api_key = os.getenv('OPENAI_API_KEY', '').strip()
    if env_api_key and not env_api_key.startswith('your_'):  # Проверяем что это не плейсхолдер
        return env_api_key
    
    return None


def validate_api_key(api_key: str) -> tuple[bool, str]:
    """
    Валидирует API ключ OpenAI.
    
    Args:
        api_key: API ключ для проверки
        
    Returns:
        Кортеж (валидность, сообщение об ошибке)
    """
    if not api_key or not api_key.strip():
        return False, "API ключ пустой"
    
    api_key = api_key.strip()
    
    # Проверяем что это не плейсхолдер
    if api_key.startswith('your_') and api_key.endswith('_here'):
        return False, "Используется плейсхолдер вместо реального API ключа"
    
    # Проверяем формат OpenAI API ключа
    if not api_key.startswith('sk-'):
        return False, "API ключ должен начинаться с 'sk-'"
    
    if len(api_key) < 20:
        return False, "API ключ слишком короткий"
    
    return True, "OK"


def run_async(coro):
    """
    Выполняет асинхронную функцию в Streamlit.
    
    Args:
        coro: Асинхронная функция
        
    Returns:
        Результат выполнения функции
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def format_search_results_for_display(results, max_content_lines=10):
    """
    Форматирует результаты поиска для отображения в Streamlit.
    
    Args:
        results: Список результатов поиска
        max_content_lines: Максимальное количество строк контента
        
    Returns:
        Список отформатированных результатов
    """
    formatted_results = []
    
    for i, result in enumerate(results, 1):
        # Определяем цвет скора
        score_color = "🟢" if result.score > 0.8 else "🟡" if result.score > 0.6 else "🔴"
        
        # Обрезаем контент если слишком длинный
        content_lines = result.content.split('\n')
        if len(content_lines) > max_content_lines:
            content = '\n'.join(content_lines[:max_content_lines]) + '\n... (обрезано)'
        else:
            content = result.content
        
        formatted_result = {
            'index': i,
            'title': f"{score_color} {i}. {result.chunk_name}",
            'subtitle': f"{result.file_path}:{result.start_line}-{result.end_line} | Скор: {result.score:.3f}",
            'metadata': f"Язык: {result.language.title()}, Тип: {result.chunk_type}, Файл: {result.file_name}",
            'content': content,
            'language': result.language,
            'start_line': result.start_line,
            'original_result': result
        }
        
        formatted_results.append(formatted_result)
    
    return formatted_results


def main():
    """Основная функция веб-интерфейса"""
    
    st.title("📚 Анализатор репозиториев")
    st.markdown("Создание детальной MD документации для кода с помощью OpenAI GPT")
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Настройка API ключа
        st.subheader("🔑 OpenAI API")
        
        # Проверяем наличие API ключа в переменных окружения
        existing_api_key = os.getenv('OPENAI_API_KEY', '')
        
        # Состояние для хранения API ключа
        if 'api_key_source' not in st.session_state:
            st.session_state.api_key_source = 'env' if existing_api_key else 'input'
        
        # Кнопки выбора источника API ключа
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Взять из .env", disabled=not existing_api_key):
                st.session_state.api_key_source = 'env'
        with col2:
            if st.button("✍️ Ввести вручную"):
                st.session_state.api_key_source = 'input'
        
        # Отображение соответствующего интерфейса
        if st.session_state.api_key_source == 'env' and existing_api_key:
            st.success("✅ Используется API ключ из переменных окружения (.env)")
            api_key = existing_api_key
            # Очищаем manual_api_key чтобы get_current_api_key() использовал env
            st.session_state.manual_api_key = ""
        else:
            api_key = st.text_input(
                "API ключ",
                value=st.session_state.get('manual_api_key', ''),
                placeholder="sk-...",
                type="password",
                help="Получите API ключ на https://platform.openai.com/api-keys"
            )
            # Сохраняем введенный ключ в session_state для get_current_api_key()
            if api_key != st.session_state.get('manual_api_key', ''):
                st.session_state.manual_api_key = api_key
            
            if api_key:
                st.success("✅ API ключ введен")
            elif not existing_api_key:
                st.warning("⚠️ Введите OpenAI API ключ или создайте файл .env")
            else:
                st.info("ℹ️ Введите API ключ или нажмите 'Взять из .env'")
        
        # Дополнительные настройки
        st.subheader("🛠️ Параметры анализа")
        
        model_choice = st.selectbox(
            "Модель GPT",
            ["gpt-4.1-nano", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="gpt-4.1-nano - быстрая и экономичная модель (рекомендуется)"
        )
        
        # RAG система статус
        st.subheader("🔍 RAG Система")
        search_service, query_engine, indexer_service, rag_status = init_rag_components()
        
        if search_service is not None:
            st.success(f"✅ {rag_status}")
            
            # Информация о Jina v3 модели
            try:
                config = get_config(require_api_key=False)
                model_name = config.rag.embeddings.model_name
                vector_size = config.rag.embeddings.truncate_dim
                
                if "jinaai/jina-embeddings-v3" in model_name:
                    st.info(f"🚀 **Jina v3 Architecture**: {model_name} ({vector_size}d векторы, dual task)")
                else:
                    st.info(f"📊 **Embedding Model**: {model_name} ({vector_size}d векторы)")
            except:
                st.info("📊 **RAG Model**: Активна")
            
            if st.button("📊 Статистика RAG"):
                try:
                    stats = search_service.get_search_stats()  # Убираем run_async для синхронной функции
                    with st.expander("📈 Подробная статистика", expanded=True):
                        # Основные метрики поиска
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Всего запросов", stats.get('total_queries', 0))
                            st.metric("Попаданий в кэш", stats.get('cache_hits', 0))
                        with col2:
                            st.metric("Размер кэша", stats.get('cache_size', 0))
                            st.metric("Среднее время поиска", f"{stats.get('avg_search_time', 0):.3f}s")
                            
                        # Дополнительная статистика
                        col3, col4 = st.columns(2)
                        with col3:
                            st.metric("Промахи кэша", stats.get('cache_misses', 0))
                            st.metric("Коэф. попадания", f"{stats.get('cache_hit_rate', 0):.1%}")
                        with col4:
                            if stats.get('last_query_time'):
                                st.caption(f"Последний запрос: {stats['last_query_time'][:19].replace('T', ' ')}")
                            st.metric("Макс. размер кэша", stats.get('cache_max_size', 0))
                        
                        st.divider()
                        
                        # Техническая информация о модели (только для Jina v3)
                        try:
                            config = get_config(require_api_key=False)
                            if "jinaai/jina-embeddings-v3" in config.rag.embeddings.model_name:
                                st.markdown("**🔧 Jina v3 Technical Specs:**")
                                
                                tech_col1, tech_col2, tech_col3 = st.columns(3)
                                with tech_col1:
                                    st.metric("Параметры модели", "570M")
                                    st.metric("Размерность", f"{config.rag.embeddings.truncate_dim}d")
                                with tech_col2:
                                    st.metric("Task Query", config.rag.embeddings.task_query)
                                    st.metric("Task Passage", config.rag.embeddings.task_passage)
                                with tech_col3:
                                    st.metric("Trust Remote Code", "✅" if config.rag.embeddings.trust_remote_code else "❌")
                                    st.metric("L2 Normalize", "✅" if config.rag.embeddings.get('normalize_embeddings', True) else "❌")
                        except:
                            pass  # Ignore config errors in sidebar
                except Exception as e:
                    st.error(f"Ошибка получения статистики: {e}")
        else:
            st.error(f"❌ {rag_status}")
            st.info("💡 Установите Qdrant для RAG функций")
    
    # Основная область
    analyzer = get_analyzer()
    
    # Инициализация состояния сессии
    if 'analysis_completed' not in st.session_state:
        st.session_state.analysis_completed = False
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    
    # Вкладки интерфейса
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Анализ репозитория", "🔍 RAG: Поиск", "📊 Статистика", "❓ Справка"])
    
    with tab1:
        st.header("📁 Выберите репозиторий для анализа")
        
        # Выбор способа загрузки
        upload_method = st.radio(
            "Способ загрузки:",
            ["Путь к папке", "ZIP архив"],
            horizontal=True
        )
        
        repo_path = None
        temp_dir = None
        
        if upload_method == "Путь к папке":
            repo_path = st.text_input(
                "Путь к репозиторию",
                placeholder="C:/path/to/your/repository",
                help="Введите полный путь к папке с исходным кодом"
            )
            
            if repo_path and Path(repo_path).exists():
                st.success(f"✅ Папка найдена: {repo_path}")
            elif repo_path:
                st.error("❌ Папка не найдена")
                repo_path = None
        
        else:  # ZIP архив
            uploaded_file = st.file_uploader(
                "Загрузите ZIP архив с репозиторием",
                type=['zip'],
                help="Загрузите ZIP архив, содержащий исходный код"
            )
            
            if uploaded_file is not None:
                # Валидируем загруженный файл
                is_valid, error_msg = validate_uploaded_file(uploaded_file)
                if not is_valid:
                    st.error(f"❌ {error_msg}")
                    repo_path = None
                else:
                    # Создаем временную директорию
                    temp_dir = tempfile.mkdtemp()
                    zip_path = Path(temp_dir) / uploaded_file.name
                    
                    # Сохраняем загруженный файл
                    with open(zip_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Безопасно извлекаем архив
                    success, message, extracted_path = safe_extract_zip(zip_path, Path(temp_dir))
                    
                    if success:
                        repo_path = extracted_path
                        st.success(f"✅ Архив безопасно распакован: {uploaded_file.name}")
                    else:
                        st.error(f"❌ {message}")
                        repo_path = None
        
        # Информация о файлах для анализа
        if repo_path:
            st.subheader("📊 Информация о файлах")
            
            try:
                # Подсчитываем файлы для анализа
                scanner = FileScanner()
                total_files = scanner.count_files(repo_path)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Файлов для анализа", total_files)
                with col2:
                    st.metric("Поддерживаемых языков", len(scanner.supported_extensions))
                
                # Информация об исключениях
                with st.expander("ℹ️ Какие файлы исключаются из анализа"):
                    st.markdown("""
                    **🗂️ Исключаемые директории:**
                    - **Системы контроля версий**: `.git`, `.svn`, `.hg` - служебная информация
                    - **Зависимости**: `node_modules`, `venv`, `.venv` - сторонний код
                    - **Кэши**: `__pycache__`, `.pytest_cache` - временные файлы
                    - **Сборка**: `target`, `build`, `dist` - компилированный код  
                    - **IDE**: `.idea`, `.vscode` - настройки редакторов
                    - **Логи и временные**: `logs`, `tmp`, `temp` - служебная информация
                    
                    **🔸 Исключаемые файлы:**
                    - Скрытые файлы (начинающиеся с точки)
                    - Файлы больше 10MB (защита от больших бинарных файлов)
                    - Неподдерживаемые расширения файлов
                    
                    **💡 Результат:** Анализируется только ваш исходный код, исключая служебные файлы
                    """)
                    
                # Оценка стоимости для больших репозиториев
                if total_files > 100:
                    st.warning(f"⚠️ Найдено {total_files} файлов. Анализ может занять время и использовать много токенов OpenAI.")
                    
                    # Примерная оценка стоимости
                    estimated_tokens = total_files * 800  # примерная оценка
                    estimated_cost = estimated_tokens * 0.000001  # цена за токен для gpt-4.1-nano
                    st.info(f"💰 Примерная стоимость: {estimated_tokens:,} токенов (~${estimated_cost:.3f})")
                
            except Exception as e:
                st.error(f"❌ Ошибка подсчета файлов: {e}")
        
        # Предварительный просмотр файлов
        if repo_path:
            with st.expander("👀 Детальная статистика репозитория"):
                try:
                    stats = analyzer.get_repository_stats(repo_path)
                    
                    # Основные метрики
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Всего файлов", stats['total_files'])
                    with col2:
                        st.metric("Размер", f"{stats['total_size'] / 1024 / 1024:.1f} MB")
                    with col3:
                        st.metric("Языков", len(stats['languages']))
                    with col4:
                        if stats['total_files'] > 0:
                            avg_size = stats['total_size'] / stats['total_files'] / 1024
                            st.metric("Средний размер файла", f"{avg_size:.1f} KB")
                        else:
                            st.metric("Средний размер файла", "0 KB")
                    
                    # Разбивка по языкам с процентами
                    if stats['languages']:
                        st.write("**📊 Распределение по языкам программирования:**")
                        
                        # Сортируем по количеству файлов
                        sorted_languages = sorted(stats['languages'].items(), key=lambda x: x[1], reverse=True)
                        
                        for lang, count in sorted_languages:
                            percentage = (count / stats['total_files']) * 100
                            # Создаем прогресс бар для визуализации
                            progress_bar_html = f"""
                            <div style="background-color: #f0f2f6; border-radius: 10px; overflow: hidden; margin: 2px 0;">
                                <div style="background-color: #1f77b4; height: 20px; width: {percentage:.1f}%; 
                                           display: flex; align-items: center; padding-left: 8px; color: white; font-size: 12px;">
                                    <strong>{lang.title()}</strong>: {count} файлов ({percentage:.1f}%)
                                </div>
                            </div>
                            """
                            st.markdown(progress_bar_html, unsafe_allow_html=True)
                    
                    # Информация о кодировках (если есть разнообразие)
                    if len(stats.get('encoding_distribution', {})) > 1:
                        st.write("**🔤 Кодировки файлов:**")
                        for encoding, count in stats['encoding_distribution'].items():
                            st.write(f"• {encoding}: {count} файлов")
                    
                    # Топ самых больших файлов
                    if stats.get('largest_files'):
                        st.write("**📈 Самые большие файлы:**")
                        for i, file_info in enumerate(stats['largest_files'][:5], 1):
                            size_mb = file_info['size'] / 1024 / 1024
                            file_path = Path(file_info['path']).name  # Показываем только имя файла
                            st.write(f"{i}. **{file_path}** ({file_info['language'].title()}) - {size_mb:.2f} MB")
                
                except Exception as e:
                    st.error(f"❌ Ошибка анализа папки: {e}")
        
        # RAG индексация
        if search_service and indexer_service:
            enable_rag_indexing = st.checkbox(
                "📊 Индексировать в RAG систему",
                value=True,
                help="Параллельно с анализом создать векторный индекс для семантического поиска"
            )
        else:
            enable_rag_indexing = False
            if repo_path:
                st.info("ℹ️ RAG система недоступна - индексация отключена")
        
        # Кнопка запуска анализа
        if st.button("🚀 Начать анализ", type="primary", disabled=not (repo_path and api_key)):
            if not api_key:
                st.error("❌ Необходимо ввести OpenAI API ключ")
            elif not repo_path:
                st.error("❌ Необходимо выбрать репозиторий")
            else:
                # Инициализируем анализатор с API ключом
                if not analyzer.initialize_with_api_key(api_key):
                    st.error("❌ Ошибка инициализации с API ключом")
                else:
                    # ИСПРАВЛЕНИЕ: передаем repo_path напрямую, а не создаем web_output
                    # Документация будет создана в корне анализируемого репозитория
                    
                    # Запускаем анализ
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(message: str, progress: int):
                        progress_bar.progress(progress)
                        status_text.text(message)
                    
                    try:
                        # Запускаем асинхронный анализ - передаем repo_path как output_dir
                        result = asyncio.run(analyzer.analyze_repository(
                            repo_path,
                            repo_path,  # ИСПРАВЛЕНИЕ: используем repo_path для создания SUMMARY_REPORT_ внутри репозитория
                            progress_callback=update_progress
                        ))
                        
                        # Параллельная RAG индексация если включена
                        if enable_rag_indexing and indexer_service:
                            try:
                                status_text.text("Индексация в RAG систему...")
                                progress_bar.progress(95)
                                
                                # Запускаем индексацию репозитория
                                indexing_result = run_async(indexer_service.index_repository(
                                    repo_path,
                                    batch_size=512,
                                    recreate=False,
                                    show_progress=False
                                ))
                                
                                if indexing_result and indexing_result.get('success', False):
                                    st.success(f"🎯 RAG индексация завершена: {indexing_result.get('indexed_chunks', 0)} чанков")
                                    result['rag_indexing'] = indexing_result
                                else:
                                    st.warning("⚠️ RAG индексация завершена с ошибками")
                                    result['rag_indexing'] = {'success': False, 'error': 'Индексация не удалась'}
                                    
                            except Exception as rag_error:
                                st.warning(f"⚠️ Ошибка RAG индексации: {rag_error}")
                                logger.exception("Ошибка RAG индексации")
                                result['rag_indexing'] = {'success': False, 'error': str(rag_error)}
                        
                        st.session_state.analysis_result = result
                        st.session_state.analysis_completed = True
                        
                        # Очищаем временную директорию если она была создана
                        if temp_dir and Path(temp_dir).exists():
                            shutil.rmtree(temp_dir)
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка анализа: {e}")
                        if temp_dir and Path(temp_dir).exists():
                            shutil.rmtree(temp_dir)
        
        # Отображение результатов
        if st.session_state.analysis_completed and st.session_state.analysis_result:
            result = st.session_state.analysis_result
            
            if result.get('success', True):
                st.success("🎉 Анализ завершен успешно!")
                
                # Основная статистика результатов
                st.subheader("📈 Детальная статистика анализа")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Найдено файлов", result.get('scanned_files', result.get('total_files', 0)))
                with col2:
                    st.metric("Проанализировано", result.get('successful', 0))
                with col3:
                    st.metric("С ошибками", result.get('failed', 0))
                with col4:
                    success_rate = 0
                    total = result.get('total_files', 0)
                    if total > 0:
                        success_rate = (result.get('successful', 0) / total) * 100
                    st.metric("Успешность", f"{success_rate:.1f}%")
                
                # Информация о токенах и затратах
                if 'token_stats' in result:
                    token_stats = result['token_stats']
                    used_tokens = token_stats.get('used_today', 0)
                    estimated_cost = used_tokens * 0.000001  # примерная стоимость для gpt-4.1-nano
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"🔢 Использовано токенов: {used_tokens:,}")
                    with col2:
                        st.info(f"💰 Примерная стоимость: ${estimated_cost:.4f}")
                
                # Статус RAG индексации
                if 'rag_indexing' in result:
                    st.subheader("🔍 Статус RAG индексации")
                    rag_result = result['rag_indexing']
                    
                    if rag_result.get('success', False):
                        st.success("✅ RAG индексация выполнена успешно!")
                        
                        # Детальная статистика RAG
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Проиндексировано чанков", rag_result.get('indexed_chunks', 0))
                        with col2:
                            st.metric("Обработано файлов", rag_result.get('processed_files', 0))
                        with col3:
                            processing_time = rag_result.get('processing_time', 0)
                            st.metric("Время индексации", f"{processing_time:.1f}s")
                        
                        # Дополнительная информация о RAG
                        if rag_result.get('indexed_chunks', 0) > 0:
                            st.info("💡 Теперь вы можете использовать семантический поиск по коду во вкладке 'RAG: Поиск'")
                    else:
                        st.warning(f"⚠️ RAG индексация завершилась с ошибкой: {rag_result.get('error', 'неизвестная ошибка')}")
                        st.info("ℹ️ Анализ кода выполнен успешно, но векторный индекс не создан. Вы можете создать его отдельно во вкладке 'RAG: Поиск'")
                
                # Ссылка на результаты
                output_path = result.get('output_directory', './web_output')
                if Path(output_path).exists():
                    st.success(f"📁 Документация сохранена в: `{output_path}`")
                    
                    # Создаем ZIP архив с результатами
                    zip_path = Path(output_path).parent / "documentation.zip"
                    try:
                        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            for file_path in Path(output_path).rglob('*'):
                                if file_path.is_file():
                                    arcname = file_path.relative_to(output_path)
                                    zipf.write(file_path, arcname)
                        
                        # Кнопка скачивания
                        with open(zip_path, 'rb') as f:
                            st.download_button(
                                label="📥 Скачать документацию (ZIP)",
                                data=f.read(),
                                file_name="documentation.zip",
                                mime="application/zip"
                            )
                    
                    except Exception as e:
                        st.warning(f"Не удалось создать ZIP архив: {e}")
                
                # Кнопка для нового анализа
                if st.button("🔄 Новый анализ"):
                    st.session_state.analysis_completed = False
                    st.session_state.analysis_result = None
                    st.rerun()
            
            else:
                st.error(f"❌ Ошибка анализа: {result.get('error', 'Неизвестная ошибка')}")
    
    with tab2:
        st.header("🔍 Семантический поиск по коду")
        
        # Инициализация RAG компонентов
        search_service, query_engine, indexer_service, rag_status = init_rag_components()
        
        # Показываем статус RAG системы
        if search_service is not None:
            st.success(f"✅ {rag_status}")
        else:
            st.error(f"❌ {rag_status}")
            st.info("💡 Для использования RAG функций необходимо установить Qdrant и настроить RAG систему")
        
        # НОВОЕ: Standalone RAG индексация
        st.subheader("📚 Индексация репозитория")
        st.markdown("*Создание векторного индекса для семантического поиска (не требует OpenAI API)*")
        
        # Выбор репозитория для индексации
        col1, col2 = st.columns([3, 1])
        with col1:
            index_repo_path = st.text_input(
                "Путь к репозиторию для индексации",
                placeholder="C:/path/to/your/repository",
                help="Укажите путь к папке с исходным кодом для создания векторного индекса"
            )
        with col2:
            recreate_index = st.checkbox(
                "Пересоздать индекс",
                value=False,
                help="Удалить существующий индекс и создать новый"
            )
        
        # Кнопка standalone индексации
        if st.button(
            "🔄 Индексировать репозиторий", 
            type="secondary", 
            disabled=not (indexer_service and index_repo_path and Path(index_repo_path).exists())
        ):
            if not indexer_service:
                st.error("❌ RAG система недоступна")
            elif not index_repo_path:
                st.warning("⚠️ Укажите путь к репозиторию")
            elif not Path(index_repo_path).exists():
                st.error("❌ Репозиторий не найден")
            else:
                try:
                    with st.spinner("Индексация репозитория в RAG систему..."):
                        indexing_result = run_async(indexer_service.index_repository(
                            index_repo_path,
                            batch_size=512,
                            recreate=recreate_index,
                            show_progress=False
                        ))
                        
                        if indexing_result and indexing_result.get('success', False):
                            st.success(f"🎯 Индексация завершена успешно!")
                            
                            # Отображаем статистику индексации
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Проиндексировано чанков", indexing_result.get('indexed_chunks', 0))
                            with col2:
                                st.metric("Обработано файлов", indexing_result.get('processed_files', 0))
                            with col3:
                                st.metric("Время индексации", f"{indexing_result.get('processing_time', 0):.1f}s")
                                
                            st.info("💡 Теперь можете использовать семантический поиск по этому репозиторию")
                            
                        else:
                            error_msg = indexing_result.get('error', 'Неизвестная ошибка') if indexing_result else 'Индексация не удалась'
                            st.error(f"❌ Ошибка индексации: {error_msg}")
                            
                except Exception as e:
                    st.error(f"❌ Ошибка индексации: {e}")
                    logger.exception("Ошибка standalone RAG индексации")
        
        st.divider()
        
        # Разделы RAG интерфейса
        rag_mode = st.radio(
            "Выберите режим:",
            ["🔍 Семантический поиск", "💬 Q&A по репозиторию"],
            horizontal=True
        )
        
        if rag_mode == "🔍 Семантический поиск":
            st.subheader("🔍 Поиск по коду")
            
            # Поисковый интерфейс
            query = st.text_input(
                "Введите запрос для поиска по коду",
                placeholder="например: authentication middleware, database connection, error handling"
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                top_k = st.slider("Количество результатов", 1, 20, 10)
            with col2:
                lang_filter = st.selectbox(
                    "Язык",
                    ["все", "python", "javascript", "typescript", "cpp", "csharp", "java", "go", "rust"]
                )
            with col3:
                chunk_type = st.selectbox(
                    "Тип",
                    ["все", "function", "class", "imports", "other"]
                )
            
            # Кнопка поиска
            if st.button("🔍 Поиск", type="primary", disabled=not search_service or not query.strip()):
                if not query.strip():
                    st.warning("⚠️ Введите поисковый запрос")
                elif not search_service:
                    st.error("❌ RAG система недоступна")
                else:
                    try:
                        with st.spinner("Выполнение семантического поиска..."):
                            # Подготовка параметров поиска
                            language_filter = None if lang_filter == "все" else lang_filter
                            chunk_type_filter = None if chunk_type == "все" else chunk_type
                            
                            # Выполнение поиска
                            results = run_async(search_service.search(
                                query=query,
                                top_k=top_k,
                                language_filter=language_filter,
                                chunk_type_filter=chunk_type_filter,
                                min_score=0.5
                            ))
                            
                            # Отображение результатов
                            if results:
                                st.success(f"🎯 Найдено {len(results)} результатов")
                                
                                # Форматирование результатов для отображения
                                formatted_results = format_search_results_for_display(results)
                                
                                for result in formatted_results:
                                    with st.expander(f"{result['title']} - {result['subtitle']}", expanded=False):
                                        st.caption(result['metadata'])
                                        
                                        # Отображение кода с подсветкой синтаксиса
                                        st.code(
                                            result['content'],
                                            language=result['language'],
                                            line_numbers=True
                                        )
                                        
                                        # Дополнительная информация
                                        st.caption(f"📍 Строки: {result['start_line']}-{result['original_result'].end_line}")
                            else:
                                st.info("🔍 Результаты не найдены. Попробуйте изменить запрос или параметры поиска.")
                                
                    except Exception as e:
                        st.error(f"❌ Ошибка поиска: {e}")
                        logger.exception("Ошибка выполнения семантического поиска")
        
        elif rag_mode == "💬 Q&A по репозиторию":
            st.subheader("💬 Q&A по репозиторию")
            
            # Инициализация истории чата
            if "rag_chat_history" not in st.session_state:
                st.session_state.rag_chat_history = []
            
            # Отображение истории чата
            for i, (question, answer, context_files) in enumerate(st.session_state.rag_chat_history):
                with st.container():
                    st.markdown(f"**❓ Вопрос {i+1}:** {question}")
                    st.markdown(f"**💡 Ответ:** {answer}")
                    if context_files:
                        st.caption(f"📚 Использованные файлы: {', '.join(context_files)}")
                    st.divider()
            
            # Поле ввода нового вопроса
            question = st.text_area(
                "Задайте вопрос о коде репозитория",
                placeholder="Как работает аутентификация в этом проекте?\nКакие есть API endpoints?\nКак устроена архитектура базы данных?",
                height=100
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                context_limit = st.number_input("Контекст (файлы)", 1, 10, 5)
            
            # Кнопка отправки вопроса
            if st.button("💬 Ответить", type="primary", disabled=not search_service or not query_engine or not question.strip()):
                # Получаем актуальный API ключ
                current_api_key = get_current_api_key()
                
                if not question.strip():
                    st.warning("⚠️ Введите вопрос")
                elif not search_service or not query_engine:
                    st.error("❌ RAG система недоступна")
                elif not current_api_key:
                    st.error("❌ Необходим OpenAI API ключ для Q&A")
                else:
                    # Валидируем API ключ
                    is_valid, error_msg = validate_api_key(current_api_key)
                    if not is_valid:
                        st.error(f"❌ Ошибка API ключа: {error_msg}")
                        return
                    try:
                        with st.spinner("Поиск релевантного кода и генерация ответа..."):
                            # 1. Семантический поиск релевантного кода с retry логикой
                            search_results = None
                            max_retries = 2
                            
                            for attempt in range(max_retries):
                                try:
                                    search_results = run_async(search_service.search(
                                        query=question,
                                        top_k=context_limit,
                                        min_score=0.6
                                    ))
                                    break  # Успешно выполнен, выходим из цикла
                                except Exception as search_error:
                                    logger.warning(f"Попытка поиска {attempt + 1}/{max_retries} неудачна: {search_error}")
                                    if attempt == max_retries - 1:
                                        # Последняя попытка неудачна, пробрасываем исключение
                                        raise search_error
                                    # Небольшая пауза перед повтором
                                    import time
                                    time.sleep(0.5)
                            
                            if search_results:
                                # 2. Формирование контекста из найденного кода
                                context_parts = []
                                context_files = []
                                
                                for result in search_results:
                                    context_parts.append(f"""
**Файл:** {result.file_path} (строки {result.start_line}-{result.end_line})
**Тип:** {result.chunk_type}
**Код:**
```{result.language}
{result.content}
```
""")
                                    if result.file_name not in context_files:
                                        context_files.append(result.file_name)
                                
                                context = "\n---\n".join(context_parts)
                                
                                # 3. Формирование промпта с контекстом
                                prompt_with_context = f"""
Ты - опытный разработчик, анализирующий кодовую базу. Используй предоставленный контекст кода для ответа на вопрос пользователя.

**КОНТЕКСТ ИЗ КОДА РЕПОЗИТОРИЯ:**
{context}

**ВОПРОС ПОЛЬЗОВАТЕЛЯ:**
{question}

**ИНСТРУКЦИИ:**
- Отвечай на русском языке
- Используй только информацию из предоставленного контекста кода
- Если контекста недостаточно для полного ответа, так и скажи
- Приводи примеры кода из контекста при необходимости
- Структурируй ответ для лучшего понимания

**ОТВЕТ:**
"""
                                
                                # 4. Вызов OpenAI с контекстом
                                if not analyzer.initialize_with_api_key(current_api_key):
                                    st.error("❌ Ошибка инициализации OpenAI API")
                                else:
                                    try:
                                        response = analyzer.openai_manager.client.chat.completions.create(
                                            model=analyzer.openai_manager.model,
                                            messages=[
                                                {"role": "user", "content": prompt_with_context}
                                            ],
                                            temperature=0.1
                                        )
                                        
                                        answer = response.choices[0].message.content.strip()
                                        
                                        # 5. Сохранение в истории и отображение
                                        st.session_state.rag_chat_history.append((question, answer, context_files))
                                        
                                        # Отображение нового ответа
                                        st.success("✅ Ответ сгенерирован!")
                                        st.rerun()
                                        
                                    except Exception as openai_error:
                                        st.error(f"❌ Ошибка OpenAI API: {openai_error}")
                            else:
                                st.warning("🔍 Не найдено релевантного кода для ответа на вопрос. Попробуйте переформулировать вопрос.")
                                
                    except Exception as e:
                        st.error(f"❌ Ошибка Q&A: {e}")
                        logger.exception("Ошибка выполнения Q&A")
            
            # Кнопка очистки истории
            if st.session_state.rag_chat_history:
                if st.button("🗑️ Очистить историю", type="secondary"):
                    st.session_state.rag_chat_history = []
                    st.rerun()
    
    with tab3:
        st.header("📊 Статистика")
        
        if api_key:
            try:
                if analyzer.initialize_with_api_key(api_key):
                    token_stats = analyzer.openai_manager.get_token_usage_stats()
                    
                    st.metric(
                        "Использовано сегодня", 
                        token_stats.get('used_today', 0),
                        help="Количество токенов, использованных сегодня"
                    )
                else:
                    st.warning("⚠️ Не удалось подключиться к OpenAI API")
            
            except Exception as e:
                st.error(f"❌ Ошибка получения статистики: {e}")
        else:
            st.info("ℹ️ Введите OpenAI API ключ для просмотра статистики")
    
    with tab4:
        st.header("❓ Справка по использованию")
        
        st.markdown("""
        ### 🚀 Как использовать анализатор:
        
        1. **Получите OpenAI API ключ:**
           - Зарегистрируйтесь на [OpenAI Platform](https://platform.openai.com/)
           - Создайте API ключ в разделе "API Keys"
           - Введите ключ в боковой панели
        
        2. **Выберите репозиторий:**
           - Укажите путь к папке с кодом
           - Или загрузите ZIP архив
        
        3. **Запустите анализ:**
           - Нажмите кнопку "Начать анализ"
           - Дождитесь завершения
           - Скачайте результаты
        
        ### 📋 Поддерживаемые языки:
        """)
        
        languages = {
            "Python": ".py",
            "JavaScript": ".js, .jsx", 
            "TypeScript": ".ts, .tsx",
            "Java": ".java",
            "C++": ".cpp, .cc, .cxx, .h, .hpp",
            "C#": ".cs",
            "Go": ".go",
            "Rust": ".rs",
            "PHP": ".php",
            "Ruby": ".rb"
        }
        
        for lang, ext in languages.items():
            st.write(f"• **{lang}**: {ext}")
        
        st.markdown("""
        ### 💰 Примерная стоимость:
        
        - **gpt-4.1-nano**: ~$0.001 за 1000 токенов
        - Средний файл: ~500-1500 токенов
        - Репозиторий (20 файлов): ~$0.01-0.03
        
        ### ⚙️ Настройки:
        
        - Используйте **gpt-4.1-nano** для экономии
        - Ограничьте количество файлов
        - Проверяйте статистику токенов
        
        ### 🔒 Безопасность:
        
        - API ключ не сохраняется
        - Код отправляется только в OpenAI
        - Локальное кэширование результатов
        """)


if __name__ == "__main__":
    main()
