import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQ_FILE = PROJECT_ROOT / "requirements.txt"

# Упрощённое сопоставление импортов с пакетами PyPI
MODULE_TO_PKG = {
    "openai": "openai",
    "tiktoken": "tiktoken",
    "chardet": "chardet",
    "click": "click",
    "rich": "rich",
    "streamlit": "streamlit",
    "dotenv": "python-dotenv",
    # тесты
    "pytest": "pytest",
    "pytest_asyncio": "pytest-asyncio",
    "hypothesis": "hypothesis",
    # научные и системные
    "numpy": "numpy",
    "psutil": "psutil",
    "cachetools": "cachetools",
    # RAG и ML
    "fastembed": "fastembed",
    "qdrant_client": "qdrant-client",
    "sentence_transformers": "sentence-transformers",
    "torch": "torch",
    "faiss": "faiss-cpu",
    "rank_bm25": "rank-bm25",
    "nltk": "nltk",
    # инфраструктура
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "prometheus_client": "prometheus-client",
    "onnxruntime": "onnxruntime",
}

IGNORE_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build"}
IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)")


def iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if any(seg in IGNORE_DIRS for seg in path.parts):
            continue
        yield path


def collect_top_imports(root: Path):
    modules = set()
    for py in iter_python_files(root):
        try:
            for line in py.read_text(encoding="utf-8").splitlines():
                m = IMPORT_RE.match(line)
                if not m:
                    continue
                mod = m.group(1).split(".")[0]
                # пропускаем локальные импорты проекта
                if mod in {"config", "utils", "openai_integration", "file_scanner", "doc_generator", "code_chunker", "parsers", "tests", "web_ui", "run_web", "main"}:
                    continue
                modules.add(mod)
        except Exception:
            continue
    return sorted(modules)


def parse_requirements(req_path: Path):
    pkgs = set()
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        import re as _re
        base = _re.split(r"[<>=]", line)[0].strip()
        # сохраняем extras (например, qdrant-client[fastembed])
        pkgs.add(base)
    return pkgs


def main():
    used_modules = collect_top_imports(PROJECT_ROOT)
    declared_pkgs = parse_requirements(REQ_FILE)

    missing = []
    for mod in used_modules:
        pkg = MODULE_TO_PKG.get(mod)
        if pkg and pkg not in declared_pkgs:
            missing.append((mod, pkg))

    extra = []
    for pkg in declared_pkgs:
        # грубая проверка: если пакет не мапится ни на один импорт — отметить как потенциальный лишний
        if pkg not in MODULE_TO_PKG.values():
            extra.append(pkg)

    print("Used modules:", used_modules)
    print("Declared packages:", sorted(declared_pkgs))

    if missing:
        print("MISSING packages (module -> pkg):")
        for mod, pkg in missing:
            print(f"  - {mod} -> {pkg}")
        sys.exit(1)

    if extra:
        print("POTENTIALLY EXTRA packages in requirements (check manually):")
        for pkg in sorted(extra):
            print(f"  - {pkg}")

    print("Requirements look consistent.")


if __name__ == "__main__":
    main()


