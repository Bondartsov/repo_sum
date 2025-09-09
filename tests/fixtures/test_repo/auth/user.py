"""
Модуль для работы с пользователями системы.

Содержит классы и функции для управления пользователями,
их регистрации, аутентификации и управления профилями.
"""

import hashlib
import os
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class UserProfile:
    """Профиль пользователя с базовой информацией"""
    user_id: int
    username: str
    email: str
    full_name: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    roles: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)


class PasswordValidator:
    """Валидатор паролей с настраиваемыми правилами"""
    
    def __init__(self):
        self.min_length = 8
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_digits = True
        self.require_special_chars = True
        self.special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """
        Валидирует пароль по заданным правилам.
        
        Args:
            password: Пароль для проверки
            
        Returns:
            Словарь с результатами валидации
        """
        errors = []
        
        if len(password) < self.min_length:
            errors.append(f"Пароль должен содержать минимум {self.min_length} символов")
        
        if self.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Пароль должен содержать минимум одну заглавную букву")
        
        if self.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Пароль должен содержать минимум одну строчную букву")
        
        if self.require_digits and not re.search(r'\d', password):
            errors.append("Пароль должен содержать минимум одну цифру")
        
        if self.require_special_chars and not any(c in self.special_chars for c in password):
            errors.append("Пароль должен содержать минимум один специальный символ")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'strength_score': self._calculate_strength(password)
        }
    
    def _calculate_strength(self, password: str) -> int:
        """Вычисляет силу пароля от 0 до 100"""
        score = 0
        
        # Длина пароля
        score += min(len(password) * 2, 20)
        
        # Разнообразие символов
        if re.search(r'[a-z]', password):
            score += 10
        if re.search(r'[A-Z]', password):
            score += 10
        if re.search(r'\d', password):
            score += 10
        if any(c in self.special_chars for c in password):
            score += 15
        
        # Отсутствие повторяющихся символов
        unique_chars = len(set(password))
        score += min(unique_chars * 2, 25)
        
        # Отсутствие простых паттернов
        if not self._has_common_patterns(password):
            score += 10
        
        return min(score, 100)
    
    def _has_common_patterns(self, password: str) -> bool:
        """Проверяет наличие общих слабых паттернов"""
        common_patterns = [
            r'123',
            r'abc',
            r'qwerty',
            r'password',
            r'(.)\1{2,}'  # Повторяющиеся символы
        ]
        
        for pattern in common_patterns:
            if re.search(pattern, password.lower()):
                return True
        
        return False


class UserManager:
    """Менеджер для управления пользователями"""
    
    def __init__(self):
        self.password_validator = PasswordValidator()
        self.users_storage = {}  # В реальном приложении это будет БД
        self.next_user_id = 1
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """
        Хеширует пароль с использованием соли.
        
        Args:
            password: Исходный пароль
            salt: Соль для хеширования (генерируется автоматически если не указана)
            
        Returns:
            Кортеж (хеш_пароля, соль)
        """
        if salt is None:
            salt = os.urandom(32).hex()
        
        # Используем PBKDF2 для безопасного хеширования
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 100,000 итераций
        ).hex()
        
        return password_hash, salt
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """
        Проверяет пароль против хеша.
        
        Args:
            password: Пароль для проверки
            password_hash: Сохранённый хеш пароля
            salt: Соль, использованная при хешировании
            
        Returns:
            True если пароль корректный
        """
        test_hash, _ = self.hash_password(password, salt)
        return test_hash == password_hash
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: str,
        roles: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Создаёт нового пользователя.
        
        Args:
            username: Имя пользователя
            email: Email адрес
            password: Пароль
            full_name: Полное имя
            roles: Роли пользователя
            
        Returns:
            Результат создания пользователя
        """
        # Валидация входных данных
        errors = []
        
        if not username or len(username) < 3:
            errors.append("Имя пользователя должно содержать минимум 3 символа")
        
        if not self._is_valid_email(email):
            errors.append("Некорректный email адрес")
        
        if self._username_exists(username):
            errors.append("Пользователь с таким именем уже существует")
        
        if self._email_exists(email):
            errors.append("Пользователь с таким email уже существует")
        
        # Валидация пароля
        password_validation = self.password_validator.validate_password(password)
        if not password_validation['is_valid']:
            errors.extend(password_validation['errors'])
        
        if errors:
            return {
                'success': False,
                'errors': errors
            }
        
        # Создание пользователя
        user_id = self.next_user_id
        self.next_user_id += 1
        
        password_hash, salt = self.hash_password(password)
        
        user_profile = UserProfile(
            user_id=user_id,
            username=username,
            email=email,
            full_name=full_name,
            created_at=datetime.utcnow(),
            roles=roles or ['user']
        )
        
        self.users_storage[user_id] = {
            'profile': user_profile,
            'password_hash': password_hash,
            'password_salt': salt
        }
        
        return {
            'success': True,
            'user_id': user_id,
            'message': 'Пользователь успешно создан'
        }
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserProfile]:
        """
        Аутентифицирует пользователя по имени и паролю.
        
        Args:
            username: Имя пользователя
            password: Пароль
            
        Returns:
            Профиль пользователя если аутентификация успешна, иначе None
        """
        for user_data in self.users_storage.values():
            profile = user_data['profile']
            
            if profile.username == username and profile.is_active:
                if self.verify_password(password, user_data['password_hash'], user_data['password_salt']):
                    # Обновляем время последнего входа
                    profile.last_login = datetime.utcnow()
                    return profile
        
        return None
    
    def get_user_by_id(self, user_id: int) -> Optional[UserProfile]:
        """Получает пользователя по ID"""
        user_data = self.users_storage.get(user_id)
        return user_data['profile'] if user_data else None
    
    def update_user_preferences(self, user_id: int, preferences: Dict[str, Any]) -> bool:
        """
        Обновляет настройки пользователя.
        
        Args:
            user_id: ID пользователя
            preferences: Новые настройки
            
        Returns:
            True если обновление успешно
        """
        user_data = self.users_storage.get(user_id)
        if user_data:
            user_data['profile'].preferences.update(preferences)
            return True
        
        return False
    
    def deactivate_user(self, user_id: int) -> bool:
        """Деактивирует пользователя"""
        user_data = self.users_storage.get(user_id)
        if user_data:
            user_data['profile'].is_active = False
            return True
        
        return False
    
    def get_users_by_role(self, role: str) -> List[UserProfile]:
        """Получает всех пользователей с определённой ролью"""
        users = []
        for user_data in self.users_storage.values():
            profile = user_data['profile']
            if role in profile.roles:
                users.append(profile)
        
        return users
    
    def _is_valid_email(self, email: str) -> bool:
        """Проверяет корректность email адреса"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, email))
    
    def _username_exists(self, username: str) -> bool:
        """Проверяет существование пользователя с таким именем"""
        for user_data in self.users_storage.values():
            if user_data['profile'].username == username:
                return True
        
        return False
    
    def _email_exists(self, email: str) -> bool:
        """Проверяет существование пользователя с таким email"""
        for user_data in self.users_storage.values():
            if user_data['profile'].email == email:
                return True
        
        return False


# Глобальный экземпляр менеджера пользователей
user_manager = UserManager()


def get_current_user_roles(user_id: int) -> List[str]:
    """
    Получает роли текущего пользователя.
    
    Args:
        user_id: ID пользователя
        
    Returns:
        Список ролей пользователя
    """
    user = user_manager.get_user_by_id(user_id)
    return user.roles if user else []


def check_user_permission(user_id: int, required_permission: str) -> bool:
    """
    Проверяет наличие у пользователя определённого разрешения.
    
    Args:
        user_id: ID пользователя
        required_permission: Требуемое разрешение
        
    Returns:
        True если у пользователя есть разрешение
    """
    user_roles = get_current_user_roles(user_id)
    
    # Простая система разрешений на основе ролей
    role_permissions = {
        'admin': ['read', 'write', 'delete', 'manage_users', 'admin'],
        'moderator': ['read', 'write', 'delete'],
        'user': ['read', 'write'],
        'guest': ['read']
    }
    
    for role in user_roles:
        if role in role_permissions and required_permission in role_permissions[role]:
            return True
    
    return False