"""Тесты для проверки доступности удалённого VM сервиса."""

from __future__ import annotations

import os

import pytest

from tests.conftest import check_vm_availability


@pytest.mark.vm
def test_vm_is_reachable(request: pytest.FixtureRequest) -> None:
    """Проверяет доступность VM и пропускает тесты, если сервис отключён."""

    config = request.config
    vm_host = config.getoption("--vm-host") or os.getenv("RAG_SERVICE_HOST", "10.61.11.54")
    vm_port = config.getoption("--vm-port") or int(os.getenv("RAG_SERVICE_PORT", "8000"))

    is_available = check_vm_availability(vm_host, vm_port, timeout=0.5)

    if not is_available:
        pytest.skip(
            f"VM endpoint {vm_host}:{vm_port} недоступен. "
            "Запустите сервис перед выполнением @pytest.mark.vm тестов."
        )

    assert is_available, f"VM {vm_host}:{vm_port} должна быть доступна для выполнения теста"
