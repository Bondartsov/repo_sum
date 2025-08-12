import os
import sys
import json
import subprocess
from pathlib import Path

import pytest


@pytest.mark.functional
def test_cli_analyze_incremental_no_changes_success(monkeypatch, tmp_path):
    """
    Functional: analyze должен отработать в инкрементальном режиме без анализа кода,
    если все файлы уже проиндексированы без изменений (избегаем реальных OpenAI вызовов).
    """
    # Репозиторий с одним файлом
    repo = tmp_path / "repo"
    repo.mkdir()
    src = repo / "a.py"
    src.write_text("def a():\n    return 1\n", encoding="utf-8")

    # Готовим индекс с актуальным хешем, чтобы analyze вернул "Нет изменений — отчёты актуальны"
    # compute_file_hash доступен только из python, поэтому просто сформируем индекс-заглушку;
    # analyze_repository сверяет хеши; если индекс не совпадет — анализ пойдет.
    # Для стабильности — вычислим хеш через одноразовый помощник в подпроцессе python.
    helper = (
        "import json,sys,hashlib;"
        "p=sys.argv[1];"
        "h=hashlib.sha256(open(p,'rb').read()).hexdigest();"
        "print(h)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", helper, str(src)],
        capture_output=True,
        text=True,
        check=True,
    )
    file_hash = proc.stdout.strip()

    idx_dir = repo / ".repo_sum"
    idx_dir.mkdir(parents=True, exist_ok=True)
    index = {str(src): {"hash": file_hash, "analyzed_at": "2025-01-01T00:00:00"}}
    (idx_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    # Выходная директория
    out_dir = tmp_path / "out"

    # CLI вызов
    project_root = Path(__file__).resolve().parents[1]
    main_py = project_root / "main.py"

    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "fake-key")

    # analyze по умолчанию incremental=True
    proc2 = subprocess.run(
        [sys.executable, str(main_py), "analyze", str(repo), "-o", str(out_dir), "--no-progress"],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )
    # Должен завершиться успешно
    assert proc2.returncode == 0
    # В stdout обычно печатается "Анализ завершен успешно!", но при отсутствии изменений
    # analyze_repository возвращает success=True и короткую сводку — проверим, что не было критической ошибки:
    assert "Критическая ошибка" not in (proc2.stdout + proc2.stderr)


@pytest.mark.functional
def test_cli_stats_outputs_tables(tmp_path):
    """
    Functional: команда stats печатает таблицы общей статистики и, при наличии,
    статистику по языкам.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("print('x')\n", encoding="utf-8")
    (repo / "b.js").write_text("console.log('y')\n", encoding="utf-8")

    project_root = Path(__file__).resolve().parents[1]
    main_py = project_root / "main.py"

    proc = subprocess.run(
        [sys.executable, str(main_py), "stats", str(repo)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0
    out = proc.stdout
    assert "Общая статистика" in out
    # Таблица по языкам может печататься при наличии статистики
    assert ("По языкам программирования" in out) or ("Самые большие файлы" in out)


@pytest.mark.functional
def test_cli_token_stats_handles_error_gracefully(monkeypatch, tmp_path):
    """
    Functional: token_stats должен корректно обработать несоответствие формы статистики.
    В текущей реализации OpenAIManager.get_token_usage_stats возвращает другие поля,
    из-за чего команда выведет сообщение об ошибке (и не упадёт).
    """
    project_root = Path(__file__).resolve().parents[1]
    main_py = project_root / "main.py"

    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "fake-key")

    proc = subprocess.run(
        [sys.executable, str(main_py), "token-stats"],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )
    # Команда не должна аварийно завершаться
    assert proc.returncode == 0
    # Ожидаем сообщение об ошибке получения статистики (из-за несовпадения ключей)
    assert "Ошибка при получении статистики" in (proc.stdout + proc.stderr)


@pytest.mark.functional
def test_cli_subcommands_help(tmp_path):
    """
    Functional/Smoke: --help для ключевых подкоманд.
    """
    project_root = Path(__file__).resolve().parents[1]
    main_py = project_root / "main.py"

    subcommands = [
        ["analyze", "--help"],
        ["stats", "--help"],
        ["clear-cache", "--help"],
        ["token-stats", "--help"],
    ]

    for args in subcommands:
        proc = subprocess.run(
        [sys.executable, str(main_py), *args],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        )
        assert proc.returncode == 0, f"Команда {' '.join(args)} завершилась с ошибкой"
        assert ("Options" in proc.stdout) or ("Опции" in proc.stdout) or ("help" in proc.stdout.lower())


@pytest.mark.functional
def test_cli_settings_validation_error(tmp_path):
    """
    Functional: при передаче некорректного файла настроек через -c/--config
    CLI должен завершиться с кодом 1 и сообщением об ошибке загрузки конфигурации.
    """
    project_root = Path(__file__).resolve().parents[1]
    main_py = project_root / "main.py"

    # Некорректный JSON
    bad_cfg = tmp_path / "bad_settings.json"
    bad_cfg.write_text("{ invalid json", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(main_py), "-c", str(bad_cfg), "stats", str(tmp_path)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "Ошибка загрузки конфигурации" in (proc.stdout + proc.stderr)
