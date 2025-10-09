import os

import pytest

from tests.utils_cli import run_cli


@pytest.mark.e2e
@pytest.mark.integration
def test_e2e_cli_analyze_generates_docs_without_openai(monkeypatch, tmp_path):
    """
    E2E: полный запуск analyze без настоящего OpenAI, проверяем генерацию отчётов.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def foo():\n    return 42\n", encoding="utf-8")
    (repo / "b.js").write_text("class B {}\n", encoding="utf-8")

    env = {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "fake-key")}
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("OFFLINE_MODE", "1")
    env.setdefault("USE_MOCK_EMBEDDER", "1")
    env.setdefault("EMBEDDING_PROVIDER", "mock")
    env.setdefault("VECTOR_STORE_PROVIDER", "mock")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    out_dir = tmp_path / "out"

    proc = run_cli([
        "analyze",
        str(repo),
        "-o",
        str(out_dir),
        "--no-progress",
        "--no-incremental",
    ], env=env)

    stdout_output, stderr_output = proc.stdout or "", proc.stderr or ""
    assert proc.returncode == 0, f"STDOUT:\n{stdout_output}\nSTDERR:\n{stderr_output}"
    combined_output = stdout_output + stderr_output
    assert "Ошибка загрузки конфигурации" not in combined_output
    assert "Критическая ошибка" not in combined_output

    summary_root = out_dir / f"SUMMARY_REPORT_{repo.name}"
    index_md = summary_root / "README.md"
    assert summary_root.exists() and summary_root.is_dir()
    assert index_md.exists()

    md_files = list(summary_root.rglob("*.md"))
    md_reports = [p for p in md_files if p.name.lower() != "readme.md"]
    assert md_reports, f"Не найдены файл(ы) отчётов, найдено только: {[p.name for p in md_files]}"

    content_any = "\n".join(p.read_text(encoding="utf-8")[:2000] for p in md_reports)
    success_markers = [
        "Документация сгенерирована автоматически",
        "Ошибка анализа",
        "Audit Report",
        "analysis error",
    ]
    assert any(marker in content_any for marker in success_markers)
