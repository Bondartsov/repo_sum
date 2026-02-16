"""
Модуль LLM провайдеров с поддержкой множественных провайдеров и fallback.

Поддерживаемые провайдеры:
- OpenAI (GPT-4, GPT-3.5)
- Google Gemini
- Anthropic Claude
- GLM (Zhipu AI)

Fallback логика: при недоступности основного провайдера автоматически
переключается на следующий в списке fallback.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from config import LLMConfig, get_config

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Базовый класс ошибок LLM провайдера"""
    pass


class ProviderNotAvailableError(LLMProviderError):
    """Провайдер недоступен"""
    pass


@dataclass
class LLMResponse:
    """Ответ от LLM провайдера"""
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None


class BaseLLMProvider(ABC):
    """Базовый абстрактный класс для LLM провайдеров"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Имя провайдера (openai, google, anthropic, glm)"""
        pass
    
    @property
    @abstractmethod
    def requires_api_key(self) -> bool:
        """Требуется ли API ключ для этого провайдера"""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Генерирует ответ на основе промпта"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Проверяет доступность провайдера"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Валидирует конфигурацию провайдера"""
        pass
    
    def _get_temperature(self) -> float:
        return self.config.temperature
    
    def _get_max_tokens(self) -> int:
        return self.config.max_tokens


class OpenAIProvider(BaseLLMProvider):
    """Провайдер OpenAI"""
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    @property
    def requires_api_key(self) -> bool:
        return True
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        try:
            from openai import OpenAI
            
            if self._client is None:
                api_key = self.config.llm.get_provider_api_key("openai")
                if not api_key:
                    raise ProviderNotAvailableError("OpenAI API ключ не найден")
                self._client = OpenAI(api_key=api_key)
            
            model = self.config.llm.get_provider_model("openai")
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self._get_temperature(),
                max_tokens=self._get_max_tokens(),
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider=self.provider_name,
                model=model,
                tokens_used=response.usage.total_tokens if response.usage else None,
            )
        except Exception as e:
            raise ProviderNotAvailableError(f"OpenAI недоступен: {e}")
    
    def is_available(self) -> bool:
        api_key = self.config.llm.get_provider_api_key("openai")
        return bool(api_key and api_key.startswith("sk-"))
    
    def validate_config(self) -> bool:
        api_key = self.config.llm.get_provider_api_key("openai")
        return bool(api_key and api_key.startswith("sk-"))


class GoogleProvider(BaseLLMProvider):
    """Провайдер Google Gemini"""
    
    @property
    def provider_name(self) -> str:
        return "google"
    
    @property
    def requires_api_key(self) -> bool:
        return True
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        try:
            import google.genai as genai  # type: ignore[import-untyped]
            
            api_key = self.config.llm.get_provider_api_key("google")
            if not api_key:
                raise ProviderNotAvailableError("Google API ключ не найден")
            
            if self._client is None:
                self._client = genai.Client(api_key=api_key)
            
            model = self.config.llm.get_provider_model("google")
            
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            response = self._client.models.generate_content(
                model=model,
                contents=full_prompt,
                config={
                    "temperature": self._get_temperature(),
                    "max_output_tokens": self._get_max_tokens(),
                }
            )
            
            return LLMResponse(
                content=response.text,
                provider=self.provider_name,
                model=model,
            )
        except Exception as e:
            raise ProviderNotAvailableError(f"Google Gemini недоступен: {e}")
    
    def is_available(self) -> bool:
        api_key = self.config.llm.get_provider_api_key("google")
        return bool(api_key and api_key.startswith("AIza"))
    
    def validate_config(self) -> bool:
        api_key = self.config.llm.get_provider_api_key("google")
        return bool(api_key and api_key.startswith("AIza"))


class AnthropicProvider(BaseLLMProvider):
    """Провайдер Anthropic Claude"""
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    @property
    def requires_api_key(self) -> bool:
        return True
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        try:
            from anthropic import Anthropic
            
            api_key = self.config.llm.get_provider_api_key("anthropic")
            if not api_key:
                raise ProviderNotAvailableError("Anthropic API ключ не найден")
            
            if self._client is None:
                self._client = Anthropic(api_key=api_key)
            
            model = self.config.llm.get_provider_model("anthropic")
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self._client.messages.create(
                model=model,
                messages=messages,
                temperature=self._get_temperature(),
                max_tokens=self._get_max_tokens(),
                system=system_prompt,
            )
            
            return LLMResponse(
                content=response.content[0].text,
                provider=self.provider_name,
                model=model,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens if response.usage else None,
            )
        except Exception as e:
            raise ProviderNotAvailableError(f"Anthropic Claude недоступен: {e}")
    
    def is_available(self) -> bool:
        api_key = self.config.llm.get_provider_api_key("anthropic")
        return bool(api_key and api_key.startswith("sk-ant-"))
    
    def validate_config(self) -> bool:
        api_key = self.config.llm.get_provider_api_key("anthropic")
        return bool(api_key and api_key.startswith("sk-ant-"))


class GLMProvider(BaseLLMProvider):
    """Провайдер GLM (Zhipu AI)"""
    
    @property
    def provider_name(self) -> str:
        return "glm"
    
    @property
    def requires_api_key(self) -> bool:
        return True
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        try:
            import requests
            
            api_key = self.config.llm.get_provider_api_key("glm")
            if not api_key:
                raise ProviderNotAvailableError("GLM API ключ не найден")
            
            model = self.config.llm.get_provider_model("glm")
            
            url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": self._get_temperature(),
                "max_tokens": self._get_max_tokens(),
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                provider=self.provider_name,
                model=model,
                tokens_used=data.get("usage", {}).get("total_tokens") if "usage" in data else None,
            )
        except Exception as e:
            raise ProviderNotAvailableError(f"GLM недоступен: {e}")
    
    def is_available(self) -> bool:
        api_key = self.config.llm.get_provider_api_key("glm")
        return bool(api_key)
    
    def validate_config(self) -> bool:
        api_key = self.config.llm.get_provider_api_key("glm")
        return bool(api_key)


class LLMProviderFactory:
    """Фабрика для создания LLM провайдеров с fallback логикой"""
    
    PROVIDERS: Dict[str, type] = {
        "openai": OpenAIProvider,
        "google": GoogleProvider,
        "anthropic": AnthropicProvider,
        "glm": GLMProvider,
    }
    
    def __init__(self):
        self.config = get_config()
        self.llm_config = self.config.llm
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._init_providers()
    
    def _init_providers(self):
        """Инициализирует все доступные провайдеры"""
        for name, provider_class in self.PROVIDERS.items():
            try:
                provider = provider_class(self.config)
                if provider.validate_config():
                    self._providers[name] = provider
                    logger.info(f"LLM провайдер {name} инициализирован")
                else:
                    logger.warning(f"LLM провайдер {name} не валиден - пропускаем")
            except Exception as e:
                logger.warning(f"Ошибка инициализации {name}: {e}")
    
    def get_primary_provider(self) -> BaseLLMProvider:
        """Возвращает основной провайдер"""
        primary = self.llm_config.provider.lower()
        
        if primary in self._providers:
            return self._providers[primary]
        
        for fallback in self.llm_config.fallback_list:
            if fallback in self._providers:
                logger.info(f"Основной провайдер {primary} недоступен, используем fallback: {fallback}")
                return self._providers[fallback]
        
        raise LLMProviderError("Нет доступных LLM провайдеров")
    
    def get_provider(self, provider_name: str) -> Optional[BaseLLMProvider]:
        """Возвращает провайдер по имени"""
        return self._providers.get(provider_name.lower())
    
    def get_all_providers(self) -> List[BaseLLMProvider]:
        """Возвращает все доступные провайдеры"""
        return list(self._providers.values())
    
    def get_fallback_chain(self) -> List[BaseLLMProvider]:
        """Возвращает цепочку fallback провайдеров"""
        chain = []
        primary = self.llm_config.provider.lower()
        
        if primary in self._providers:
            chain.append(self._providers[primary])
        
        for fallback in self.llm_config.fallback_list:
            if fallback in self._providers and fallback != primary:
                chain.append(self._providers[fallback])
        
        return chain
    
    async def generate_with_fallback(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """
        Генерирует ответ с автоматическим fallback.
        Перебирает провайдеры по порядку until успешного ответа.
        """
        errors = []
        
        for provider in self.get_fallback_chain():
            try:
                logger.info(f"Пробуем LLM провайдер: {provider.provider_name}")
                result = await provider.generate(prompt, system_prompt)
                logger.info(f"Успешный ответ от {provider.provider_name}")
                return result
            except ProviderNotAvailableError as e:
                logger.warning(f"Провайдер {provider.provider_name} недоступен: {e}")
                errors.append(f"{provider.provider_name}: {e}")
            except Exception as e:
                logger.error(f"Ошибка {provider.provider_name}: {e}")
                errors.append(f"{provider.provider_name}: {e}")
        
        raise LLMProviderError(f"Все LLM провайдеры недоступны: {'; '.join(errors)}")


_llm_factory: Optional[LLMProviderFactory] = None


def get_llm_factory() -> LLMProviderFactory:
    """Получает глобальный экземпляр фабрики LLM провайдеров"""
    global _llm_factory
    if _llm_factory is None:
        _llm_factory = LLMProviderFactory()
    return _llm_factory
