"""
Middleware для аутентификации пользователей.

Обеспечивает проверку токенов и управление сессиями.
"""

import jwt
import time
from typing import Optional, Dict, Any
from functools import wraps
from flask import request, jsonify, current_app


class AuthenticationError(Exception):
    """Исключение для ошибок аутентификации"""
    pass


class TokenManager:
    """Менеджер JWT токенов для аутентификации"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """
        Инициализация менеджера токенов.
        
        Args:
            secret_key: Секретный ключ для подписи токенов
            algorithm: Алгоритм шифрования
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = 3600  # 1 час
    
    def generate_token(self, user_id: int, username: str) -> str:
        """
        Генерирует JWT токен для пользователя.
        
        Args:
            user_id: ID пользователя
            username: Имя пользователя
            
        Returns:
            JWT токен
        """
        payload = {
            'user_id': user_id,
            'username': username,
            'exp': time.time() + self.token_expiry,
            'iat': time.time()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Проверяет JWT токен и возвращает payload.
        
        Args:
            token: JWT токен для проверки
            
        Returns:
            Декодированный payload
            
        Raises:
            AuthenticationError: При невалидном токене
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Токен истёк")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Невалидный токен")


def authenticate_request(f):
    """
    Декоратор для проверки аутентификации запроса.
    
    Проверяет наличие и валидность Bearer токена в заголовках.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({'error': 'Отсутствует заголовок Authorization'}), 401
        
        try:
            scheme, token = auth_header.split(' ', 1)
            if scheme.lower() != 'bearer':
                return jsonify({'error': 'Неподдерживаемая схема аутентификации'}), 401
        except ValueError:
            return jsonify({'error': 'Неверный формат заголовка Authorization'}), 401
        
        try:
            token_manager = TokenManager(current_app.config['SECRET_KEY'])
            payload = token_manager.verify_token(token)
            
            # Добавляем информацию о пользователе в request
            request.user = {
                'user_id': payload['user_id'],
                'username': payload['username']
            }
            
        except AuthenticationError as e:
            return jsonify({'error': str(e)}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function


def require_permission(permission: str):
    """
    Декоратор для проверки прав доступа.
    
    Args:
        permission: Название требуемого разрешения
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(request, 'user'):
                return jsonify({'error': 'Пользователь не аутентифицирован'}), 401
            
            # Здесь должна быть логика проверки прав доступа
            # Для примера просто проверяем, что пользователь аутентифицирован
            user_permissions = get_user_permissions(request.user['user_id'])
            
            if permission not in user_permissions:
                return jsonify({'error': f'Недостаточно прав: требуется {permission}'}), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def get_user_permissions(user_id: int) -> list:
    """
    Получает список разрешений для пользователя.
    
    Args:
        user_id: ID пользователя
        
    Returns:
        Список разрешений пользователя
    """
    # Заглушка - в реальном приложении это будет запрос к БД
    default_permissions = ['read', 'write']
    
    if user_id == 1:  # Администратор
        return default_permissions + ['admin', 'delete']
    
    return default_permissions