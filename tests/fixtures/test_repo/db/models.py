"""
Тестовый файл для демонстрации RAG поиска по SQL коду.
SQLAlchemy импорты помечены type: ignore - не требуют установки для тестирования.

Модели базы данных для приложения.

Содержит SQLAlchemy модели для работы с пользователями,
статьями, комментариями и другими сущностями.

Тестовый файл для демонстрации RAG поиска по базам данных.
SQLAlchemy импорты с fallback на заглушки для избежания лишних зависимостей.
"""

from datetime import datetime
from typing import Optional, List
import uuid

# SQLAlchemy импорты с fallback заглушками
try:
    # Реальные SQLAlchemy импорты (если установлена)
    from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Index  # type: ignore
    from sqlalchemy.ext.declarative import declarative_base  # type: ignore
    from sqlalchemy.orm import relationship, Session  # type: ignore
    from sqlalchemy.dialects.postgresql import UUID  # type: ignore
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
    
    # Заглушки для Column и типов данных
    Column = MockSQLAlchemyType
    Integer = MockSQLAlchemyType
    String = MockSQLAlchemyType
    Text = MockSQLAlchemyType
    DateTime = MockSQLAlchemyType
    Boolean = MockSQLAlchemyType
    ForeignKey = MockSQLAlchemyType
    Index = MockSQLAlchemyType
    UUID = MockSQLAlchemyType
    
    # Заглушка для declarative_base
    def declarative_base():
        """Заглушка для declarative_base"""
        class MockBase:
            __tablename__ = None
            __table_args__ = ()
            
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        return MockBase
    
    # Заглушки для relationship и Session
    def relationship(*args, **kwargs):
        """Заглушка для relationship"""
        return None
    
    class Session:
        """Заглушка для Session"""
        def __init__(self, *args, **kwargs):
            pass
        
        def add(self, obj):
            pass
        
        def commit(self):
            pass
        
        def refresh(self, obj):
            pass
        
        def query(self, model):
            return MockQuery()
        
        def close(self):
            pass
        
        def rollback(self):
            pass
    
    class MockQuery:
        """Заглушка для Query"""
        def filter(self, *args):
            return self
        
        def first(self):
            return None
        
        def all(self):
            return []
        
        def order_by(self, *args):
            return self
        
        def limit(self, n):
            return self
        
        def offset(self, n):
            return self
    
    SQLALCHEMY_AVAILABLE = False


Base = declarative_base()


class User(Base):
    """Модель пользователя"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Отношения
    articles = relationship("Article", back_populates="author", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="author", cascade="all, delete-orphan")
    
    # Индексы
    __table_args__ = (
        Index('idx_users_email', 'email'),
        Index('idx_users_username', 'username'),
        Index('idx_users_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"
    
    def to_dict(self) -> dict:
        """Преобразует модель в словарь"""
        return {
            'id': self.id,
            'uuid': str(self.uuid),
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }


class Article(Base):
    """Модель статьи"""
    __tablename__ = 'articles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    title = Column(String(500), nullable=False)
    slug = Column(String(500), unique=True, nullable=False)
    content = Column(Text, nullable=False)
    excerpt = Column(Text)
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    is_published = Column(Boolean, default=False, nullable=False)
    is_featured = Column(Boolean, default=False, nullable=False)
    view_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    published_at = Column(DateTime)
    
    # Отношения
    author = relationship("User", back_populates="articles")
    comments = relationship("Comment", back_populates="article", cascade="all, delete-orphan")
    tags = relationship("ArticleTag", back_populates="article", cascade="all, delete-orphan")
    
    # Индексы
    __table_args__ = (
        Index('idx_articles_slug', 'slug'),
        Index('idx_articles_author_id', 'author_id'),
        Index('idx_articles_published', 'is_published', 'published_at'),
        Index('idx_articles_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Article(id={self.id}, title='{self.title[:30]}...', author_id={self.author_id})>"
    
    def to_dict(self) -> dict:
        """Преобразует модель в словарь"""
        return {
            'id': self.id,
            'uuid': str(self.uuid),
            'title': self.title,
            'slug': self.slug,
            'content': self.content,
            'excerpt': self.excerpt,
            'author_id': self.author_id,
            'is_published': self.is_published,
            'is_featured': self.is_featured,
            'view_count': self.view_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'published_at': self.published_at.isoformat() if self.published_at else None
        }


class Comment(Base):
    """Модель комментария"""
    __tablename__ = 'comments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    content = Column(Text, nullable=False)
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    article_id = Column(Integer, ForeignKey('articles.id'), nullable=False)
    parent_id = Column(Integer, ForeignKey('comments.id'))  # Для вложенных комментариев
    is_approved = Column(Boolean, default=True, nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Отношения
    author = relationship("User", back_populates="comments")
    article = relationship("Article", back_populates="comments")
    parent = relationship("Comment", remote_side=[id])
    replies = relationship("Comment", cascade="all, delete-orphan")
    
    # Индексы
    __table_args__ = (
        Index('idx_comments_article_id', 'article_id'),
        Index('idx_comments_author_id', 'author_id'),
        Index('idx_comments_parent_id', 'parent_id'),
        Index('idx_comments_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Comment(id={self.id}, article_id={self.article_id}, author_id={self.author_id})>"


class Tag(Base):
    """Модель тега"""
    __tablename__ = 'tags'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    slug = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    color = Column(String(7))  # HEX цвет
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Отношения
    articles = relationship("ArticleTag", back_populates="tag")
    
    def __repr__(self):
        return f"<Tag(id={self.id}, name='{self.name}')>"


class ArticleTag(Base):
    """Связь между статьями и тегами (многие-ко-многим)"""
    __tablename__ = 'article_tags'
    
    article_id = Column(Integer, ForeignKey('articles.id'), primary_key=True)
    tag_id = Column(Integer, ForeignKey('tags.id'), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Отношения
    article = relationship("Article", back_populates="tags")
    tag = relationship("Tag", back_populates="articles")


class DatabaseManager:
    """Менеджер для работы с базой данных"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create_user(self, username: str, email: str, password_hash: str, 
                   full_name: Optional[str] = None) -> User:
        """
        Создает нового пользователя.
        
        Args:
            username: Имя пользователя
            email: Email адрес
            password_hash: Хеш пароля
            full_name: Полное имя
            
        Returns:
            Созданный пользователь
        """
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            full_name=full_name
        )
        
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        
        return user
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Получает пользователя по имени.
        
        Args:
            username: Имя пользователя
            
        Returns:
            Пользователь или None
        """
        return self.session.query(User).filter(User.username == username).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Получает пользователя по email.
        
        Args:
            email: Email адрес
            
        Returns:
            Пользователь или None
        """
        return self.session.query(User).filter(User.email == email).first()
    
    def create_article(self, title: str, slug: str, content: str, author_id: int,
                      excerpt: Optional[str] = None, is_published: bool = False) -> Article:
        """
        Создает новую статью.
        
        Args:
            title: Заголовок статьи
            slug: Слаг статьи
            content: Содержимое статьи
            author_id: ID автора
            excerpt: Отрывок статьи
            is_published: Опубликована ли статья
            
        Returns:
            Созданная статья
        """
        article = Article(
            title=title,
            slug=slug,
            content=content,
            author_id=author_id,
            excerpt=excerpt,
            is_published=is_published,
            published_at=datetime.utcnow() if is_published else None
        )
        
        self.session.add(article)
        self.session.commit()
        self.session.refresh(article)
        
        return article
    
    def get_published_articles(self, limit: int = 10, offset: int = 0) -> List[Article]:
        """
        Получает список опубликованных статей.
        
        Args:
            limit: Количество статей
            offset: Смещение
            
        Returns:
            Список статей
        """
        return (
            self.session.query(Article)
            .filter(Article.is_published == True)
            .order_by(Article.published_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
    
    def search_articles(self, query: str, limit: int = 10) -> List[Article]:
        """
        Поиск статей по содержимому.
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            
        Returns:
            Список найденных статей
        """
        return (
            self.session.query(Article)
            .filter(
                Article.is_published == True,
                Article.title.contains(query) | Article.content.contains(query)
            )
            .order_by(Article.published_at.desc())
            .limit(limit)
            .all()
        )
    
    def add_comment(self, content: str, author_id: int, article_id: int,
                   parent_id: Optional[int] = None) -> Comment:
        """
        Добавляет комментарий к статье.
        
        Args:
            content: Содержимое комментария
            author_id: ID автора комментария
            article_id: ID статьи
            parent_id: ID родительского комментария (для ответов)
            
        Returns:
            Созданный комментарий
        """
        comment = Comment(
            content=content,
            author_id=author_id,
            article_id=article_id,
            parent_id=parent_id
        )
        
        self.session.add(comment)
        self.session.commit()
        self.session.refresh(comment)
        
        return comment
    
    def get_article_comments(self, article_id: int) -> List[Comment]:
        """
        Получает комментарии к статье.
        
        Args:
            article_id: ID статьи
            
        Returns:
            Список комментариев
        """
        return (
            self.session.query(Comment)
            .filter(
                Comment.article_id == article_id,
                Comment.is_approved == True,
                Comment.is_deleted == False
            )
            .order_by(Comment.created_at.asc())
            .all()
        )
    
    def create_tag(self, name: str, slug: str, description: Optional[str] = None,
                  color: Optional[str] = None) -> Tag:
        """
        Создает новый тег.
        
        Args:
            name: Название тега
            slug: Слаг тега
            description: Описание тега
            color: Цвет тега в HEX формате
            
        Returns:
            Созданный тег
        """
        tag = Tag(
            name=name,
            slug=slug,
            description=description,
            color=color
        )
        
        self.session.add(tag)
        self.session.commit()
        self.session.refresh(tag)
        
        return tag
    
    def add_tag_to_article(self, article_id: int, tag_id: int) -> ArticleTag:
        """
        Добавляет тег к статье.
        
        Args:
            article_id: ID статьи
            tag_id: ID тега
            
        Returns:
            Связь статьи и тега
        """
        article_tag = ArticleTag(article_id=article_id, tag_id=tag_id)
        
        self.session.add(article_tag)
        self.session.commit()
        
        return article_tag