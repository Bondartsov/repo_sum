"""
Тестовый файл для демонстрации RAG поиска по SQL коду.
SQLAlchemy импорты помечены type: ignore - не требуют установки для тестирования.

Модуль подключения к базе данных.

Управляет соединениями с различными типами БД,
пулом соединений и миграциями.

Тестовый файл для демонстрации RAG поиска по базам данных.
SQLAlchemy импорты с fallback на заглушки для избежания лишних зависимостей.
"""

import os
import logging
from typing import Optional, Dict, Any, Generator
from contextlib import contextmanager
from urllib.parse import quote_plus

# SQLAlchemy импорты с fallback заглушками
try:
    # Реальные SQLAlchemy импорты (если установлена)
    from sqlalchemy import create_engine, event, pool  # type: ignore
    from sqlalchemy.engine import Engine  # type: ignore
    from sqlalchemy.orm import sessionmaker, Session  # type: ignore
    from sqlalchemy.pool import QueuePool  # type: ignore
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    # Заглушки для тестирования без SQLAlchemy зависимости
    class MockSQLAlchemyType:
        """Базовая заглушка для SQLAlchemy типов"""
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
        
        def __call__(self, *args, **kwargs):
            return MockSQLAlchemyType(*args, **kwargs)
    
    # Заглушка для create_engine
    def create_engine(*args, **kwargs):
        """Заглушка для create_engine"""
        return MockEngine()
    
    class MockEngine:
        """Заглушка для Engine"""
        def __init__(self):
            self.pool = MockPool()
        
        def connect(self):
            return MockConnection()
        
        def dispose(self):
            pass
        
        def table_names(self):
            return []
    
    class MockConnection:
        """Заглушка для Connection"""
        def execute(self, query):
            return MockResult()
        
        def close(self):
            pass
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
    
    class MockResult:
        """Заглушка для Result"""
        def fetchone(self):
            return None
        
        def fetchall(self):
            return []
    
    class MockPool:
        """Заглушка для Pool"""
        def size(self):
            return 0
        
        def checkedin(self):
            return 0
        
        def checkedout(self):
            return 0
        
        def overflow(self):
            return 0
        
        def invalid(self):
            return 0
    
    class MockSession:
        """Заглушка для Session"""
        def __init__(self, *args, **kwargs):
            pass
        
        def add(self, obj):
            pass
        
        def commit(self):
            pass
        
        def rollback(self):
            pass
        
        def close(self):
            pass
        
        def execute(self, query):
            return MockResult()
        
        def query(self, model):
            return MockQuery()
    
    class MockQuery:
        """Заглушка для Query"""
        def filter(self, *args):
            return self
        
        def first(self):
            return None
        
        def all(self):
            return []
    
    # Заглушка для sessionmaker
    def sessionmaker(*args, **kwargs):
        """Заглушка для sessionmaker"""
        return MockSession
    
    # Заглушки для остальных объектов
    Engine = MockEngine
    Session = MockSession
    QueuePool = MockPool
    
    # Заглушка для event модуля
    class MockEvent:
        """Заглушка для event"""
        @staticmethod
        def listens_for(target, identifier):
            def decorator(func):
                return func
            return decorator
    
    event = MockEvent()
    pool = MockSQLAlchemyType()
    
    SQLALCHEMY_AVAILABLE = False


logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Конфигурация подключения к базе данных"""
    
    def __init__(self, 
                 db_type: str = "postgresql",
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "myapp",
                 username: str = "postgres",
                 password: str = "",
                 pool_size: int = 10,
                 max_overflow: int = 20,
                 pool_timeout: int = 30,
                 pool_recycle: int = 3600,
                 echo: bool = False):
        """
        Инициализация конфигурации БД.
        
        Args:
            db_type: Тип БД (postgresql, mysql, sqlite)
            host: Хост БД
            port: Порт БД
            database: Название базы данных
            username: Имя пользователя
            password: Пароль
            pool_size: Размер пула соединений
            max_overflow: Максимальное переполнение пула
            pool_timeout: Таймаут получения соединения
            pool_recycle: Время жизни соединения
            echo: Логирование SQL запросов
        """
        self.db_type = db_type
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.echo = echo
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Создает конфигурацию из переменных окружения"""
        return cls(
            db_type=os.getenv("DB_TYPE", "postgresql"),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "myapp"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),
            echo=os.getenv("DB_ECHO", "false").lower() == "true"
        )
    
    def get_database_url(self) -> str:
        """Формирует URL для подключения к БД"""
        if self.db_type == "sqlite":
            return f"sqlite:///{self.database}"
        
        # Экранируем пароль для URL
        escaped_password = quote_plus(self.password) if self.password else ""
        auth_part = f"{self.username}:{escaped_password}@" if self.username else ""
        
        if self.db_type == "postgresql":
            return f"postgresql://{auth_part}{self.host}:{self.port}/{self.database}"
        elif self.db_type == "mysql":
            return f"mysql+pymysql://{auth_part}{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Неподдерживаемый тип БД: {self.db_type}")


class DatabaseConnection:
    """Менеджер подключений к базе данных"""
    
    def __init__(self, config: DatabaseConfig):
        """
        Инициализация менеджера подключений.
        
        Args:
            config: Конфигурация БД
        """
        self.config = config
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None
        self._is_connected = False
    
    def connect(self) -> None:
        """Устанавливает соединение с базой данных"""
        if self._is_connected:
            logger.warning("Соединение с БД уже установлено")
            return
        
        try:
            database_url = self.config.get_database_url()
            logger.info(f"Подключение к БД: {self.config.db_type}://{self.config.host}:{self.config.port}/{self.config.database}")
            
            # Создаем движок с пулом соединений
            self.engine = create_engine(
                database_url,
                poolclass=QueuePool if self.config.db_type != "sqlite" else None,
                pool_size=self.config.pool_size if self.config.db_type != "sqlite" else None,
                max_overflow=self.config.max_overflow if self.config.db_type != "sqlite" else None,
                pool_timeout=self.config.pool_timeout if self.config.db_type != "sqlite" else None,
                pool_recycle=self.config.pool_recycle if self.config.db_type != "sqlite" else None,
                echo=self.config.echo,
                connect_args=self._get_connect_args()
            )
            
            # Добавляем слушатели событий
            self._setup_event_listeners()
            
            # Создаем фабрику сессий
            self.session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )
            
            # Тестируем соединение
            self._test_connection()
            
            self._is_connected = True
            logger.info("Успешное подключение к БД")
            
        except Exception as e:
            logger.error(f"Ошибка подключения к БД: {e}")
            raise
    
    def disconnect(self) -> None:
        """Закрывает соединение с базой данных"""
        if not self._is_connected:
            return
        
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("Соединение с БД закрыто")
            
            self.engine = None
            self.session_factory = None
            self._is_connected = False
            
        except Exception as e:
            logger.error(f"Ошибка при закрытии соединения с БД: {e}")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Контекстный менеджер для работы с сессией БД.
        
        Yields:
            Сессия SQLAlchemy
        """
        if not self._is_connected or not self.session_factory:
            raise RuntimeError("БД не подключена. Вызовите connect() сначала.")
        
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Ошибка в сессии БД: {e}")
            raise
        finally:
            session.close()
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Возвращает информацию о текущем соединении"""
        if not self.engine:
            return {'connected': False}
        
        pool = self.engine.pool if hasattr(self.engine, 'pool') else None
        
        return {
            'connected': self._is_connected,
            'database_url': self.config.get_database_url().split('@')[-1] if '@' in self.config.get_database_url() else self.config.get_database_url(),
            'pool_info': {
                'size': pool.size() if pool else None,
                'checked_in': pool.checkedin() if pool else None,
                'checked_out': pool.checkedout() if pool else None,
                'overflow': pool.overflow() if pool else None,
                'invalid': pool.invalid() if pool else None
            } if pool else None
        }
    
    def _get_connect_args(self) -> Dict[str, Any]:
        """Возвращает дополнительные аргументы подключения"""
        connect_args = {}
        
        if self.config.db_type == "sqlite":
            connect_args["check_same_thread"] = False
        elif self.config.db_type == "postgresql":
            connect_args["connect_timeout"] = 10
        elif self.config.db_type == "mysql":
            connect_args["connect_timeout"] = 10
            connect_args["charset"] = "utf8mb4"
        
        return connect_args
    
    def _setup_event_listeners(self) -> None:
        """Настраивает слушатели событий движка"""
        if not self.engine:
            return
        
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Настройки для SQLite"""
            if self.config.db_type == "sqlite":
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()
        
        @event.listens_for(self.engine, "checkout")
        def ping_connection(dbapi_connection, connection_record, connection_proxy):
            """Проверка соединения перед использованием"""
            connection_record.info['pid'] = os.getpid()
        
        @event.listens_for(self.engine, "connect")
        def set_connection_options(dbapi_connection, connection_record):
            """Дополнительные настройки соединения"""
            if self.config.db_type == "postgresql":
                with dbapi_connection.cursor() as cursor:
                    cursor.execute("SET timezone TO 'UTC'")
    
    def _test_connection(self) -> None:
        """Тестирует соединение с БД"""
        if not self.engine:
            raise RuntimeError("Движок БД не инициализирован")
        
        try:
            with self.engine.connect() as connection:
                if self.config.db_type == "sqlite":
                    connection.execute("SELECT 1")
                elif self.config.db_type == "postgresql":
                    connection.execute("SELECT version()")
                elif self.config.db_type == "mysql":
                    connection.execute("SELECT VERSION()")
                
        except Exception as e:
            raise RuntimeError(f"Тест соединения с БД не прошел: {e}")


class DatabaseHealthCheck:
    """Проверка состояния базы данных"""
    
    def __init__(self, connection: DatabaseConnection):
        self.connection = connection
    
    def check_connection(self) -> Dict[str, Any]:
        """Проверяет соединение с БД"""
        try:
            if not self.connection._is_connected:
                return {
                    'status': 'disconnected',
                    'error': 'БД не подключена'
                }
            
            with self.connection.get_session() as session:
                # Простой запрос для проверки
                if self.connection.config.db_type == "sqlite":
                    result = session.execute("SELECT 1").fetchone()
                elif self.connection.config.db_type == "postgresql":
                    result = session.execute("SELECT current_timestamp").fetchone()
                elif self.connection.config.db_type == "mysql":
                    result = session.execute("SELECT NOW()").fetchone()
                else:
                    result = None
                
                return {
                    'status': 'healthy',
                    'response_time': 'fast',
                    'connection_info': self.connection.get_connection_info()
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def check_tables_exist(self, table_names: list) -> Dict[str, Any]:
        """Проверяет существование таблиц"""
        try:
            if not self.connection.engine:
                return {'status': 'error', 'error': 'Движок БД не инициализирован'}
            
            existing_tables = self.connection.engine.table_names()
            missing_tables = [name for name in table_names if name not in existing_tables]
            
            return {
                'status': 'healthy' if not missing_tables else 'warning',
                'existing_tables': existing_tables,
                'missing_tables': missing_tables,
                'total_tables': len(existing_tables)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }


# Глобальные экземпляры
_db_config: Optional[DatabaseConfig] = None
_db_connection: Optional[DatabaseConnection] = None


def get_database_config() -> DatabaseConfig:
    """Получает глобальную конфигурацию БД"""
    global _db_config
    if _db_config is None:
        _db_config = DatabaseConfig.from_env()
    return _db_config


def get_database_connection() -> DatabaseConnection:
    """Получает глобальное соединение с БД"""
    global _db_connection
    if _db_connection is None:
        config = get_database_config()
        _db_connection = DatabaseConnection(config)
    return _db_connection


def init_database() -> None:
    """Инициализирует подключение к БД"""
    connection = get_database_connection()
    connection.connect()


def close_database() -> None:
    """Закрывает подключение к БД"""
    global _db_connection
    if _db_connection:
        _db_connection.disconnect()
        _db_connection = None


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Удобная функция для получения сессии БД"""
    connection = get_database_connection()
    with connection.get_session() as session:
        yield session