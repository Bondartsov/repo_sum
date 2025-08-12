import os
import sys
import subprocess
from pathlib import Path

import pytest


@pytest.mark.e2e
@pytest.mark.integration
def test_e2e_cli_analyze_generates_docs_without_openai(monkeypatch, tmp_path):
    """
    E2E: Полный запуск CLI analyze без инкремента и без сети (OPENAI_API_KEY=fake).
    Ожидаем, что:
      - Команда завершается успешно (returncode == 0).
      - Создаётся SUMMARY_REPORT_<repo>/README.md.
      - Создаются отчёты по файлам; содержимое может включать "Ошибка анализа" (реальная картина при недоступном OpenAI).
    """
    # Подготовка репозитория
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def foo():\n    return 42\n", encoding="utf-8")
    (repo / "b.js").write_text("class B {}\n", encoding="utf-8")

    # Корень проекта и путь до main.py
    project_root = Path(__file__).resolve().parents[2]
    main_py = project_root / "main.py"

    # Окружение: фейковый ключ для инициализации OpenAIManager
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "fake-key"

    out_dir = tmp_path / "out"

    # Запуск CLI (неинкрементально, с отключенным прогрессом), из корня проекта (для settings.json/prompts)
    proc = subprocess.run(
        [sys.executable, str(main_py), "analyze", str(repo), "-o", str(out_dir), "--no-progress", "--no-incremental"],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "Ошибка загрузки конфигурации" not in (proc.stdout + proc.stderr)
    assert "Критическая ошибка" not in (proc.stdout + proc.stderr)

    # Проверяем артефакты вывода
    summary_root = out_dir / f"SUMMARY_REPORT_{repo.name}"
    index_md = summary_root / "README.md"
    assert summary_root.exists() and summary_root.is_dir()
    assert index_md.exists()

    # Проверим, что создан хотя бы один отчёт по файлам
    md_files = list(summary_root.rglob("*.md"))
    # README.md не считаем; должен быть ещё хотя бы один файл отчёта
    md_reports = [p for p in md_files if p.name.lower() != "readme.md"]
    assert md_reports, f"Не найдены файл(ы) отчётов, найдено только: {[p.name for p in md_files]}"

    # В одном из отчётов может присутствовать "Ошибка анализа" — это реальная картина без OpenAI
    content_any = "\n".join(p.read_text(encoding="utf-8")[:2000] for p in md_reports)
    assert ("Документация сгенерирована автоматически" in content_any) or ("Ошибка анализа" in content_any)
