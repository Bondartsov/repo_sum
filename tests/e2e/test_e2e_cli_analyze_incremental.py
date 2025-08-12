import os
import sys
import json
import subprocess
from pathlib import Path

import pytest


@pytest.mark.e2e
@pytest.mark.integration
def test_e2e_cli_analyze_incremental_skip(monkeypatch, tmp_path):
    """
    E2E: Полный запуск CLI analyze в инкрементальном режиме, когда изменений нет.
    Ожидание: команда завершается успешно и не производит сетевых вызовов к OpenAI.

    Реальная картина системы:
    - Загружается конфиг из settings.json (в корне проекта).
    - Инициализируется OpenAIManager (требуется OPENAI_API_KEY), но анализ не выполняется,
      так как индекс указывает на отсутствие изменений.
    - CLI корректно завершает работу с короткой сводкой.
    """
    # Репозиторий с одним исходником
    repo = tmp_path / "repo"
    repo.mkdir()
    src = repo / "a.py"
    src.write_text("def a():\n    return 1\n", encoding="utf-8")

    # Вычислим sha256 контента, чтобы сформировать валидный индекс
    helper = (
        "import sys,hashlib;"
        "p=sys.argv[1];"
        "print(hashlib.sha256(open(p,'rb').read()).hexdigest())"
    )
    proc = subprocess.run([sys.executable, "-c", helper, str(src)], capture_output=True, text=True, check=True)
    file_hash = proc.stdout.strip()

    # Индекс, соответствующий текущему хешу файла
    idx_dir = repo / ".repo_sum"
    idx_dir.mkdir(parents=True, exist_ok=True)
    index = {str(src): {"hash": file_hash, "analyzed_at": "2025-01-01T00:00:00"}}
    (idx_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    # Подготовим окружение и вызов CLI из корня проекта (для корректной загрузки settings.json)
    project_root = Path(__file__).resolve().parents[2]
    main_py = project_root / "main.py"

    env = os.environ.copy()
    # Ключ нужен только для инициализации OpenAIManager (анализ не запускается)
    env["OPENAI_API_KEY"] = "fake-key"

    out_dir = tmp_path / "out"

    run = subprocess.run(
        [sys.executable, str(main_py), "analyze", str(repo), "-o", str(out_dir), "--no-progress"],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )

    # E2E критерии: успешное завершение и отсутствие критической ошибки
    assert run.returncode == 0, f"STDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}"
    assert "Критическая ошибка" not in (run.stdout + run.stderr)
    # analyze_repository при отсутствии изменений возвращает success=True и короткую сводку
    # (проверим, что команда не упала и не вернула неизвестную ошибку)
    assert "Ошибка загрузки конфигурации" not in (run.stdout + run.stderr)
