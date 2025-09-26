"""
Утилиты для стабильного запуска CLI в функциональных тестах.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
SETTINGS_TEST = PROJECT_ROOT / "settings-test.json"


def _build_command(args: Iterable[str], use_test_config: bool) -> List[str]:
    cmd = [sys.executable, str(MAIN_SCRIPT)]
    if use_test_config:
        cmd.extend(["--config", str(SETTINGS_TEST), "--offline"])
    cmd.extend(args)
    return cmd


def run_cli(
    args: Iterable[str],
    *,
    use_test_config: bool = True,
    env: Optional[dict] = None,
    cwd: Optional[Path] = None,
    **popen_kwargs,
) -> subprocess.CompletedProcess:
    """Запускает main.py с предсказуемой кодировкой и offline параметрами."""
    command = _build_command(args, use_test_config=use_test_config)
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    defaults = dict(capture_output=True, text=True, encoding="utf-8", errors="replace")
    defaults.update(popen_kwargs)
    workdir = Path(cwd) if cwd else PROJECT_ROOT
    return subprocess.run(command, cwd=str(workdir), env=run_env, **defaults)
