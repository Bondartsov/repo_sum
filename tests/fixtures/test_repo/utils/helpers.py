"""
Утилиты и вспомогательные функции для приложения.

Содержит общие функции для работы со строками, датами,
файлами и другие полезные утилиты.
"""

import re
import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
from urllib.parse import urlparse
import mimetypes


def generate_random_string(length: int = 32, 
                         use_uppercase: bool = True,
                         use_lowercase: bool = True, 
                         use_digits: bool = True,
                         use_symbols: bool = False) -> str:
    """
    Генерирует случайную строку заданной длины.
    
    Args:
        length: Длина строки
        use_uppercase: Использовать заглавные буквы
        use_lowercase: Использовать строчные буквы
        use_digits: Использовать цифры
        use_symbols: Использовать символы
        
    Returns:
        Случайная строка
    """
    chars = ""
    
    if use_uppercase:
        chars += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if use_lowercase:
        chars += "abcdefghijklmnopqrstuvwxyz"
    if use_digits:
        chars += "0123456789"
    if use_symbols:
        chars += "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    if not chars:
        raise ValueError("Необходимо выбрать хотя бы один тип символов")
    
    return ''.join(secrets.choice(chars) for _ in range(length))


def generate_slug(text: str, max_length: int = 50) -> str:
    """
    Генерирует слаг из текста для URL.
    
    Args:
        text: Исходный текст
        max_length: Максимальная длина слага
        
    Returns:
        Слаг для URL
    """
    # Удаляем HTML теги если есть
    text = re.sub(r'<[^>]+>', '', text)
    
    # Приводим к нижнему регистру
    slug = text.lower()
    
    # Заменяем пробелы и специальные символы на дефисы
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)
    
    # Удаляем дефисы в начале и конце
    slug = slug.strip('-')
    
    # Обрезаем до максимальной длины
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip('-')
    
    return slug


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Обрезает текст до заданной длины с добавлением суффикса.
    
    Args:
        text: Исходный текст
        max_length: Максимальная длина
        suffix: Суффикс для обрезанного текста
        
    Returns:
        Обрезанный текст
    """
    if len(text) <= max_length:
        return text
    
    # Обрезаем по словам если возможно
    truncated = text[:max_length - len(suffix)]
    
    # Ищем последний пробел
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.7:  # Если пробел не слишком близко к началу
        truncated = truncated[:last_space]
    
    return truncated + suffix


def sanitize_filename(filename: str) -> str:
    """
    Очищает имя файла от недопустимых символов.
    
    Args:
        filename: Исходное имя файла
        
    Returns:
        Очищенное имя файла
    """
    # Удаляем недопустимые символы
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Удаляем точки в начале и конце
    sanitized = sanitized.strip('.')
    
    # Ограничиваем длину
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        max_name_length = 255 - len(ext)
        sanitized = name[:max_name_length] + ext
    
    return sanitized


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Вычисляет хеш файла.
    
    Args:
        file_path: Путь к файлу
        algorithm: Алгоритм хеширования (md5, sha1, sha256)
        
    Returns:
        Хеш файла в шестнадцатеричном формате
    """
    hash_algo = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_algo.update(chunk)
    
    return hash_algo.hexdigest()


def format_file_size(size_bytes: int) -> str:
    """
    Форматирует размер файла в читаемый вид.
    
    Args:
        size_bytes: Размер в байтах
        
    Returns:
        Отформатированный размер
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def parse_user_agent(user_agent: str) -> Dict[str, Optional[str]]:
    """
    Парсит строку User-Agent для извлечения информации о браузере.
    
    Args:
        user_agent: Строка User-Agent
        
    Returns:
        Словарь с информацией о браузере
    """
    result = {
        'browser': None,
        'version': None,
        'os': None,
        'device': None
    }
    
    if not user_agent:
        return result
    
    # Определяем браузер
    browser_patterns = {
        'Chrome': r'Chrome/(\d+\.\d+)',
        'Firefox': r'Firefox/(\d+\.\d+)',
        'Safari': r'Safari/(\d+\.\d+)',
        'Edge': r'Edg/(\d+\.\d+)',
        'Opera': r'OPR/(\d+\.\d+)'
    }
    
    for browser, pattern in browser_patterns.items():
        match = re.search(pattern, user_agent)
        if match:
            result['browser'] = browser
            result['version'] = match.group(1)
            break
    
    # Определяем ОС
    os_patterns = {
        'Windows': r'Windows NT (\d+\.\d+)',
        'macOS': r'Mac OS X (\d+_\d+)',
        'Linux': r'Linux',
        'Android': r'Android (\d+\.\d+)',
        'iOS': r'OS (\d+_\d+)'
    }
    
    for os_name, pattern in os_patterns.items():
        if re.search(pattern, user_agent):
            result['os'] = os_name
            break
    
    # Определяем устройство
    if re.search(r'Mobile', user_agent):
        result['device'] = 'Mobile'
    elif re.search(r'Tablet', user_agent):
        result['device'] = 'Tablet'
    else:
        result['device'] = 'Desktop'
    
    return result


def validate_email(email: str) -> bool:
    """
    Валидирует email адрес.
    
    Args:
        email: Email адрес для проверки
        
    Returns:
        True если email корректный
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """
    Валидирует URL.
    
    Args:
        url: URL для проверки
        
    Returns:
        True если URL корректный
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def format_datetime(dt: datetime, format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Форматирует дату и время.
    
    Args:
        dt: Объект datetime
        format_string: Строка формата
        
    Returns:
        Отформатированная дата
    """
    return dt.strftime(format_string)


def time_ago(dt: datetime) -> str:
    """
    Возвращает время в формате "время назад".
    
    Args:
        dt: Объект datetime
        
    Returns:
        Строка вида "5 минут назад"
    """
    now = datetime.utcnow()
    diff = now - dt
    
    if diff.days > 0:
        if diff.days == 1:
            return "вчера"
        elif diff.days < 30:
            return f"{diff.days} дн. назад"
        elif diff.days < 365:
            months = diff.days // 30
            return f"{months} мес. назад"
        else:
            years = diff.days // 365
            return f"{years} лет назад"
    
    seconds = diff.seconds
    
    if seconds < 60:
        return "только что"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} мин. назад"
    else:
        hours = seconds // 3600
        return f"{hours} ч. назад"


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Глубоко объединяет два словаря.
    
    Args:
        dict1: Первый словарь
        dict2: Второй словарь
        
    Returns:
        Объединенный словарь
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Преобразует вложенный словарь в плоский.
    
    Args:
        d: Исходный словарь
        parent_key: Родительский ключ
        sep: Разделитель
        
    Returns:
        Плоский словарь
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def retry_on_exception(max_attempts: int = 3, 
                      delay: float = 1.0, 
                      backoff: float = 2.0,
                      exceptions: tuple = (Exception,)) -> Callable:
    """
    Декоратор для повторных попыток выполнения функции при ошибках.
    
    Args:
        max_attempts: Максимальное количество попыток
        delay: Начальная задержка между попытками
        backoff: Множитель для увеличения задержки
        exceptions: Кортеж исключений для повтора
        
    Returns:
        Декоратор
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise e
                    
                    print(f"Попытка {attempt} не удалась: {e}. Повтор через {current_delay}s...")
                    import time
                    time.sleep(current_delay)
                    
                    attempt += 1
                    current_delay *= backoff
            
        return wrapper
    return decorator


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Безопасный парсинг JSON с возвратом значения по умолчанию.
    
    Args:
        json_str: JSON строка
        default: Значение по умолчанию
        
    Returns:
        Распарсенные данные или значение по умолчанию
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def get_mime_type(file_path: Union[str, Path]) -> str:
    """
    Определяет MIME тип файла.
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        MIME тип
    """
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or 'application/octet-stream'


def is_safe_filename(filename: str) -> bool:
    """
    Проверяет безопасность имени файла.
    
    Args:
        filename: Имя файла
        
    Returns:
        True если имя файла безопасно
    """
    # Проверяем на недопустимые символы
    dangerous_chars = '<>:"/\\|?*'
    if any(char in filename for char in dangerous_chars):
        return False
    
    # Проверяем на зарезервированные имена Windows
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
        'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2',
        'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    name_without_ext = os.path.splitext(filename)[0].upper()
    if name_without_ext in reserved_names:
        return False
    
    return True