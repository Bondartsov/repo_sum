"""
Mock объекты для тестирования RAG системы.

Содержит фиктивные реализации компонентов RAG системы,
которые не требуют сетевых соединений и внешних зависимостей.
"""

from .mock_cpu_embedder import MockCPUEmbedder, is_socket_disabled

__all__ = ['MockCPUEmbedder', 'is_socket_disabled']
