"""
Валидаторы для различных типов данных.

Содержит функции для валидации пользовательского ввода,
данных форм и API запросов.
"""

import re
import ipaddress
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Union, Callable
from urllib.parse import urlparse


class ValidationError(Exception):
    """Исключение для ошибок валидации"""
    
    def __init__(self, message: str, field: Optional[str] = None, code: Optional[str] = None):
        self.message = message
        self.field = field
        self.code = code
        super().__init__(message)


class ValidationResult:
    """Результат валидации"""
    
    def __init__(self, is_valid: bool = True, errors: Optional[List[str]] = None):
        self.is_valid = is_valid
        self.errors = errors or []
    
    def add_error(self, error: str) -> None:
        """Добавляет ошибку валидации"""
        self.errors.append(error)
        self.is_valid = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует результат в словарь"""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors
        }


class BaseValidator:
    """Базовый класс для валидаторов"""
    
    def __init__(self, required: bool = False, allow_empty: bool = False):
        self.required = required
        self.allow_empty = allow_empty
    
    def validate(self, value: Any) -> ValidationResult:
        """
        Выполняет валидацию значения.
        
        Args:
            value: Значение для валидации
            
        Returns:
            Результат валидации
        """
        result = ValidationResult()
        
        # Проверка на обязательность
        if self.required and (value is None or (isinstance(value, str) and not value.strip())):
            result.add_error("Поле обязательно для заполнения")
            return result
        
        # Проверка на пустое значение
        if not self.allow_empty and isinstance(value, str) and not value.strip():
            result.add_error("Поле не может быть пустым")
            return result
        
        # Если значение None и поле не обязательное, считаем валидным
        if value is None and not self.required:
            return result
        
        # Выполняем специфичную валидацию
        return self._validate_value(value, result)
    
    def _validate_value(self, value: Any, result: ValidationResult) -> ValidationResult:
        """Переопределяется в наследниках для специфичной валидации"""
        return result


class StringValidator(BaseValidator):
    """Валидатор для строк"""
    
    def __init__(self, min_length: Optional[int] = None, max_length: Optional[int] = None,
                 pattern: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = re.compile(pattern) if pattern else None
    
    def _validate_value(self, value: Any, result: ValidationResult) -> ValidationResult:
        if not isinstance(value, str):
            result.add_error("Значение должно быть строкой")
            return result
        
        # Проверка минимальной длины
        if self.min_length is not None and len(value) < self.min_length:
            result.add_error(f"Минимальная длина: {self.min_length} символов")
        
        # Проверка максимальной длины
        if self.max_length is not None and len(value) > self.max_length:
            result.add_error(f"Максимальная длина: {self.max_length} символов")
        
        # Проверка паттерна
        if self.pattern and not self.pattern.match(value):
            result.add_error("Значение не соответствует требуемому формату")
        
        return result


class EmailValidator(BaseValidator):
    """Валидатор для email адресов"""
    
    EMAIL_PATTERN = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    def _validate_value(self, value: Any, result: ValidationResult) -> ValidationResult:
        if not isinstance(value, str):
            result.add_error("Email должен быть строкой")
            return result
        
        if not self.EMAIL_PATTERN.match(value):
            result.add_error("Некорректный формат email адреса")
        
        # Дополнительные проверки
        if len(value) > 254:
            result.add_error("Email адрес слишком длинный")
        
        local_part, domain = value.split('@', 1)
        if len(local_part) > 64:
            result.add_error("Локальная часть email слишком длинная")
        
        return result


class PasswordValidator(BaseValidator):
    """Валидатор для паролей"""
    
    def __init__(self, min_length: int = 8, require_uppercase: bool = True,
                 require_lowercase: bool = True, require_digits: bool = True,
                 require_special: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length
        self.require_uppercase = require_uppercase
        self.require_lowercase = require_lowercase
        self.require_digits = require_digits
        self.require_special = require_special
    
    def _validate_value(self, value: Any, result: ValidationResult) -> ValidationResult:
        if not isinstance(value, str):
            result.add_error("Пароль должен быть строкой")
            return result
        
        if len(value) < self.min_length:
            result.add_error(f"Пароль должен содержать минимум {self.min_length} символов")
        
        if self.require_uppercase and not re.search(r'[A-Z]', value):
            result.add_error("Пароль должен содержать заглавные буквы")
        
        if self.require_lowercase and not re.search(r'[a-z]', value):
            result.add_error("Пароль должен содержать строчные буквы")
        
        if self.require_digits and not re.search(r'\d', value):
            result.add_error("Пароль должен содержать цифры")
        
        if self.require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', value):
            result.add_error("Пароль должен содержать специальные символы")
        
        # Проверка на слабые пароли
        weak_patterns = [
            r'123',
            r'abc',
            r'qwerty',
            r'password',
            r'(.)\1{3,}'  # Повторяющиеся символы
        ]
        
        for pattern in weak_patterns:
            if re.search(pattern, value.lower()):
                result.add_error("Пароль содержит слабые элементы")
                break
        
        return result


class NumberValidator(BaseValidator):
    """Валидатор для чисел"""
    
    def __init__(self, min_value: Optional[Union[int, float]] = None,
                 max_value: Optional[Union[int, float]] = None,
                 integer_only: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.integer_only = integer_only
    
    def _validate_value(self, value: Any, result: ValidationResult) -> ValidationResult:
        if not isinstance(value, (int, float)):
            try:
                if self.integer_only:
                    value = int(value)
                else:
                    value = float(value)
            except (ValueError, TypeError):
                result.add_error("Значение должно быть числом")
                return result
        
        if self.integer_only and not isinstance(value, int):
            result.add_error("Значение должно быть целым числом")
        
        if self.min_value is not None and value < self.min_value:
            result.add_error(f"Минимальное значение: {self.min_value}")
        
        if self.max_value is not None and value > self.max_value:
            result.add_error(f"Максимальное значение: {self.max_value}")
        
        return result


class DateValidator(BaseValidator):
    """Валидатор для дат"""
    
    def __init__(self, min_date: Optional[date] = None,
                 max_date: Optional[date] = None,
                 date_format: str = "%Y-%m-%d", **kwargs):
        super().__init__(**kwargs)
        self.min_date = min_date
        self.max_date = max_date
        self.date_format = date_format
    
    def _validate_value(self, value: Any, result: ValidationResult) -> ValidationResult:
        parsed_date = None
        
        if isinstance(value, str):
            try:
                parsed_date = datetime.strptime(value, self.date_format).date()
            except ValueError:
                result.add_error(f"Некорректный формат даты. Ожидается: {self.date_format}")
                return result
        elif isinstance(value, datetime):
            parsed_date = value.date()
        elif isinstance(value, date):
            parsed_date = value
        else:
            result.add_error("Значение должно быть датой")
            return result
        
        if self.min_date and parsed_date < self.min_date:
            result.add_error(f"Дата не может быть раньше {self.min_date}")
        
        if self.max_date and parsed_date > self.max_date:
            result.add_error(f"Дата не может быть позже {self.max_date}")
        
        return result


class UrlValidator(BaseValidator):
    """Валидатор для URL"""
    
    def __init__(self, schemes: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.schemes = schemes or ['http', 'https']
    
    def _validate_value(self, value: Any, result: ValidationResult) -> ValidationResult:
        if not isinstance(value, str):
            result.add_error("URL должен быть строкой")
            return result
        
        try:
            parsed = urlparse(value)
        except Exception:
            result.add_error("Некорректный URL")
            return result
        
        if not parsed.scheme:
            result.add_error("URL должен содержать схему (http, https)")
        elif parsed.scheme not in self.schemes:
            result.add_error(f"Поддерживаемые схемы: {', '.join(self.schemes)}")
        
        if not parsed.netloc:
            result.add_error("URL должен содержать доменное имя")
        
        return result


class IPValidator(BaseValidator):
    """Валидатор для IP адресов"""
    
    def __init__(self, version: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.version = version  # 4, 6 или None для любой версии
    
    def _validate_value(self, value: Any, result: ValidationResult) -> ValidationResult:
        if not isinstance(value, str):
            result.add_error("IP адрес должен быть строкой")
            return result
        
        try:
            ip = ipaddress.ip_address(value)
            
            if self.version == 4 and not isinstance(ip, ipaddress.IPv4Address):
                result.add_error("Ожидается IPv4 адрес")
            elif self.version == 6 and not isinstance(ip, ipaddress.IPv6Address):
                result.add_error("Ожидается IPv6 адрес")
                
        except ValueError:
            result.add_error("Некорректный IP адрес")
        
        return result


class ChoiceValidator(BaseValidator):
    """Валидатор для выбора из списка допустимых значений"""
    
    def __init__(self, choices: List[Any], case_sensitive: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.choices = choices
        self.case_sensitive = case_sensitive
    
    def _validate_value(self, value: Any, result: ValidationResult) -> ValidationResult:
        choices_to_check = self.choices
        
        if not self.case_sensitive and isinstance(value, str):
            value = value.lower()
            choices_to_check = [
                choice.lower() if isinstance(choice, str) else choice
                for choice in self.choices
            ]
        
        if value not in choices_to_check:
            result.add_error(f"Допустимые значения: {', '.join(map(str, self.choices))}")
        
        return result


class FileValidator(BaseValidator):
    """Валидатор для файлов"""
    
    def __init__(self, allowed_extensions: Optional[List[str]] = None,
                 max_size: Optional[int] = None,  # в байтах
                 min_size: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.allowed_extensions = [ext.lower() for ext in (allowed_extensions or [])]
        self.max_size = max_size
        self.min_size = min_size
    
    def _validate_value(self, value: Any, result: ValidationResult) -> ValidationResult:
        # Ожидаем объект с атрибутами filename и content_length
        if not hasattr(value, 'filename'):
            result.add_error("Некорректный объект файла")
            return result
        
        filename = getattr(value, 'filename', '')
        if not filename:
            result.add_error("Имя файла не указано")
            return result
        
        # Проверка расширения
        if self.allowed_extensions:
            file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
            if file_ext not in self.allowed_extensions:
                result.add_error(f"Разрешенные расширения: {', '.join(self.allowed_extensions)}")
        
        # Проверка размера файла
        file_size = getattr(value, 'content_length', 0)
        
        if self.max_size and file_size > self.max_size:
            max_size_mb = self.max_size / (1024 * 1024)
            result.add_error(f"Максимальный размер файла: {max_size_mb:.1f} MB")
        
        if self.min_size and file_size < self.min_size:
            min_size_kb = self.min_size / 1024
            result.add_error(f"Минимальный размер файла: {min_size_kb:.1f} KB")
        
        return result


class FormValidator:
    """Валидатор для форм с множественными полями"""
    
    def __init__(self, field_validators: Dict[str, BaseValidator]):
        self.field_validators = field_validators
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """
        Валидирует данные формы.
        
        Args:
            data: Данные для валидации
            
        Returns:
            Словарь с результатами валидации для каждого поля
        """
        results = {}
        
        for field_name, validator in self.field_validators.items():
            field_value = data.get(field_name)
            results[field_name] = validator.validate(field_value)
        
        return results
    
    def is_valid(self, data: Dict[str, Any]) -> bool:
        """
        Проверяет, являются ли данные валидными.
        
        Args:
            data: Данные для проверки
            
        Returns:
            True если все поля валидны
        """
        results = self.validate(data)
        return all(result.is_valid for result in results.values())
    
    def get_errors(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Получает ошибки валидации.
        
        Args:
            data: Данные для валидации
            
        Returns:
            Словарь с ошибками для каждого поля
        """
        results = self.validate(data)
        return {
            field: result.errors
            for field, result in results.items()
            if not result.is_valid
        }


# Предустановленные валидаторы
def create_user_registration_validator() -> FormValidator:
    """Создает валидатор для регистрации пользователя"""
    return FormValidator({
        'username': StringValidator(min_length=3, max_length=50, required=True),
        'email': EmailValidator(required=True),
        'password': PasswordValidator(required=True),
        'full_name': StringValidator(min_length=2, max_length=100),
        'age': NumberValidator(min_value=13, max_value=120, integer_only=True)
    })


def create_article_validator() -> FormValidator:
    """Создает валидатор для статьи"""
    return FormValidator({
        'title': StringValidator(min_length=5, max_length=200, required=True),
        'content': StringValidator(min_length=100, required=True),
        'category': ChoiceValidator(['tech', 'business', 'science', 'art'], required=True),
        'tags': StringValidator(max_length=500),
        'publish_date': DateValidator()
    })