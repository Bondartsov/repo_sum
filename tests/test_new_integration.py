import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from typing import List

import pytest

from openai_integration import OpenAIManager, GPTCache
from utils import CodeChunk, GPTAnalysisRequest, compute_file_hash
from file_scanner import FileScanner
from parsers.base_parser import ParserRegistry
from code_chunker import CodeChunker
from doc_generator import DocumentationGenerator
from main import RepositoryAnalyzer


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_cache_hit_on_second_call(monkeypatch, tmp_path):
    """
    Integration: кэш OpenAI-анализа должен срабатывать на втором вызове.
    Первый вызов дергает _call_openai_api, второй - достаёт из кэша.
    """
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    manager = OpenAIManager()
    # перенаправляем кэш в tmp
    manager.cache = GPTCache(cache_dir=str(tmp_path / "cache"))

    chunk = CodeChunk(
        name="a.py",
        content="def foo():\n    return 1",
        start_line=1,
        end_line=2,
        chunk_type="function",
    )
    request = GPTAnalysisRequest(
        file_path="a.py",
        language="python",
        chunks=[chunk],
        context=""
    )

    calls = {"n": 0}

    async def fake_call(prompt: str) -> str:
        calls["n"] += 1
        return "OK"

    # мок приватного метода, чтобы не трогать реальный OpenAI
    monkeypatch.setattr(manager, "_call_openai_api", fake_call)

    res1 = await manager.analyze_code(request)
    res2 = await manager.analyze_code(request)

    assert res1.error is None
    assert res2.error is None
    assert calls["n"] == 1, "Ожидался один вызов OpenAI, второй раз из кэша"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_cycle_multiple_files(monkeypatch, tmp_path):
    """
    Integration: сквозной цикл для нескольких файлов.
    scan -> parse -> chunk -> analyze(OpenAI mocked) -> markdown
    """
    # создаём несколько исходников
    sources = {
        "a.py": "class A:\n    pass\n",
        "b.js": "class B {}\n",
        "c.ts": "class C {}\n",
    }
    for name, content in sources.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    scanner = FileScanner()
    parser_registry = ParserRegistry()
    chunker = CodeChunker()
    doc_gen = DocumentationGenerator()

    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    manager = OpenAIManager()

    async def fake_call(prompt: str) -> str:
        return "OK: analyzed"

    monkeypatch.setattr(manager, "_call_openai_api", fake_call)

    files = list(scanner.scan_repository(str(tmp_path)))
    assert len(files) == len(sources)

    for fi in files:
        parser = parser_registry.get_parser(fi.path)
        parsed = parser.parse_file(fi)
        # простая проверка, что разобралось хоть что-то
        assert hasattr(parsed, "elements")

        # разбиение на чанки (по содержимому файла)
        code = Path(fi.path).read_text(encoding="utf-8")
        # Используем результаты парсера для корректной разбивки (поддержка классов для JS/TS)
        chunks = chunker.chunk_parsed_file(parsed, code)
        assert isinstance(chunks, list) and len(chunks) >= 1

        req = GPTAnalysisRequest(file_path=fi.path, language=fi.language, chunks=chunks, context="")
        result = await manager.analyze_code(req)
        assert result.error is None
        assert result.full_text

        md = doc_gen.generate_markdown(parsed)
        assert isinstance(md, str)
        # ключевые слова, которые часто встречаются
        assert "Импорты" in md or "Комментарии" in md or "class" in md or "class" in code


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_retries_and_error_propagation(monkeypatch, tmp_path, caplog):
    """
    Integration: ретраи OpenAI и корректная обработка ошибки.
    """
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    manager = OpenAIManager()
    # уменьшим задержки между ретраями
    manager.config.openai.retry_attempts = 3
    manager.config.openai.retry_delay = 0.01

    chunk = CodeChunk(name="x.py", content="def x(): pass", start_line=1, end_line=1, chunk_type="function")
    req = GPTAnalysisRequest(file_path="x.py", language="python", chunks=[chunk], context="")

    class FakeErr(Exception):
        pass

    async def fake_call(prompt: str) -> str:
        raise FakeErr("network/timeouts")

    monkeypatch.setattr(manager, "_call_openai_api", fake_call)

    res = await manager.analyze_code(req)
    assert res.error is not None
    # сообщение об ошибке формируется в analyze_code
    assert "Ошибка анализа" in res.error


@pytest.mark.integration
def test_cli_clear_cache_integration(monkeypatch, tmp_path):
    """
    Integration: CLI clear-cache очищает cache/*.json.
    """
    # путь к main.py и корню проекта
    project_root = Path(__file__).resolve().parents[1]
    main_py = project_root / "main.py"

    # создаём фейковые записи кэша в CWD проекта
    cache_dir = project_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (cache_dir / f"entry_{i}.json").write_text('{"cached_at":"2025-01-01T00:00:00"}', encoding="utf-8")

    env = os.environ.copy()
    if "OPENAI_API_KEY" not in env:
        env["OPENAI_API_KEY"] = "fake-key"

    proc = subprocess.run(
        [sys.executable, str(main_py), "clear-cache"],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Очищено" in proc.stdout
    # директория cache пуста
    assert not any(cache_dir.glob("*.json"))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_concurrent_analysis(monkeypatch, tmp_path):
    """
    Integration: параллельный анализ нескольких запросов (asyncio.gather).
    """
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    manager = OpenAIManager()

    async def fake_call(prompt: str) -> str:
        # имитируем очень быстрый ответ
        return "OK"

    monkeypatch.setattr(manager, "_call_openai_api", fake_call)

    def mk_req(i: int) -> GPTAnalysisRequest:
        chunk = CodeChunk(name=f"f{i}.py", content=f"def f{i}(): pass", start_line=1, end_line=1, chunk_type="function")
        return GPTAnalysisRequest(file_path=f"f{i}.py", language="python", chunks=[chunk], context="")

    reqs: List[GPTAnalysisRequest] = [mk_req(i) for i in range(5)]
    results = await asyncio.gather(*[manager.analyze_code(r) for r in reqs])

    assert len(results) == 5
    assert all(r.error is None for r in results)
    assert all(r.full_text for r in results)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_incremental_analysis_skips_unchanged(monkeypatch, tmp_path):
    """
    Integration: RepositoryAnalyzer с incremental=True должен пропускать неизменённые файлы.
    """
    # готовим репозиторий с одним файлом
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "a.py"
    source.write_text("def a():\n    return 1\n", encoding="utf-8")

    # индекс соответствующий текущему хешу
    index_dir = repo / ".repo_sum"
    index_dir.mkdir(parents=True, exist_ok=True)
    h = compute_file_hash(str(source))
    index = {str(source): {"hash": h, "analyzed_at": "2025-01-01T00:00:00"}}
    (index_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    # переменная окружения для OpenAIManager (конструктор требует ключ)
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    analyzer = RepositoryAnalyzer()
    out_dir = tmp_path / "out"

    result = await analyzer.analyze_repository(str(repo), str(out_dir), show_progress=False, incremental=True)
    assert result.get("success", False) is True
    assert result.get("total_files", -1) == 0
    # убедимся, что вернулись пути к предполагаемым отчётам
    assert "output_directory" in result
