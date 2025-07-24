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
from typing import Optional, Dict, Any
import logging
import json

# Импортируем компоненты программы
from config import get_config, reload_config
from file_scanner import FileScanner
from parsers.base_parser import ParserRegistry
from code_chunker import CodeChunker
from openai_integration import OpenAIManager
from doc_generator import DocumentationGenerator
from utils import setup_logging, FileInfo, ParsedFile, GPTAnalysisRequest, GPTAnalysisResult, ensure_directory_exists

# Настройка страницы
st.set_page_config(
    page_title="Анализатор репозиториев",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Настройка логирования для веб-интерфейса
setup_logging("DEBUG")
logger = logging.getLogger(__name__)


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
        logger.debug(f"initialize_with_api_key: аргумент api_key={api_key!r}, os.getenv до установки={os.getenv('OPENAI_API_KEY')}, config.openai.api_key до reload={self.config.openai.api_key}")
        """Инициализирует компоненты с API ключом"""
        try:
            # Проверяем что API ключ не пустой
            if not api_key or not api_key.strip():
                logger.error("API ключ пустой или не указан")
                return False
            
            # Устанавливаем переменную окружения
            os.environ['OPENAI_API_KEY'] = api_key.strip()
            logger.debug(f"initialize_with_api_key: os.getenv после установки={os.getenv('OPENAI_API_KEY')}")
            
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
                    # Создаем пустой результат для файла с ошибкой
                    empty_parsed = ParsedFile(file_info, [], [], [], [str(e)])
                    empty_gpt = GPTAnalysisResult("", [], {}, f"Ошибка анализа: {e}")
                    analyzed_files.append((empty_parsed, empty_gpt))
            
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
        else:
            api_key = st.text_input(
                "API ключ",
                value="",
                placeholder="sk-...",
                type="password",
                help="Получите API ключ на https://platform.openai.com/api-keys"
            )
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
        
        max_files = st.number_input(
            "Максимум файлов для анализа",
            min_value=1,
            max_value=100,
            value=20,
            help="Ограничение для избежания больших расходов"
        )
    
    # Основная область
    analyzer = get_analyzer()
    
    # Инициализация состояния сессии
    if 'analysis_completed' not in st.session_state:
        st.session_state.analysis_completed = False
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    
    # Вкладки интерфейса
    tab1, tab2, tab3 = st.tabs(["📁 Анализ репозитория", "📊 Статистика", "❓ Справка"])
    
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
                # Создаем временную директорию
                temp_dir = tempfile.mkdtemp()
                zip_path = Path(temp_dir) / uploaded_file.name
                
                # Сохраняем загруженный файл
                with open(zip_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Распаковываем архив
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Находим корневую папку (обычно первая папка в архиве)
                    extracted_dirs = [d for d in Path(temp_dir).iterdir() if d.is_dir()]
                    if extracted_dirs:
                        repo_path = str(extracted_dirs[0])
                        st.success(f"✅ Архив распакован: {uploaded_file.name}")
                    else:
                        repo_path = temp_dir
                        st.success(f"✅ Файлы извлечены в: {temp_dir}")
                        
                except Exception as e:
                    st.error(f"❌ Ошибка распаковки архива: {e}")
                    repo_path = None
        
        # Предварительный просмотр файлов
        if repo_path:
            with st.expander("👀 Предварительный просмотр"):
                try:
                    stats = analyzer.get_repository_stats(repo_path)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Всего файлов", stats['total_files'])
                    with col2:
                        st.metric("Размер", f"{stats['total_size'] / 1024 / 1024:.1f} MB")
                    with col3:
                        st.metric("Языков", len(stats['languages']))
                    
                    if stats['languages']:
                        st.write("**Языки программирования:**")
                        for lang, count in sorted(stats['languages'].items(), key=lambda x: x[1], reverse=True)[:5]:
                            st.write(f"• {lang.title()}: {count} файлов")
                
                except Exception as e:
                    st.error(f"Ошибка анализа папки: {e}")
        
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
                
                # Статистика результатов
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Обработано файлов", result.get('total_files', 0))
                with col2:
                    st.metric("Успешно", result.get('successful', 0))
                with col3:
                    st.metric("С ошибками", result.get('failed', 0))
                
                # Статистика токенов
                if 'token_stats' in result:
                    token_stats = result['token_stats']
                    st.info(f"🔢 Использовано токенов: {token_stats.get('used_today', 0)}")
                
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
    
    with tab3:
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
