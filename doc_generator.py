"""
Генерация Markdown‑документации из результатов анализа кода.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import get_config
from utils import ParsedFile, GPTAnalysisResult, ensure_directory_exists, clean_filename

logger = logging.getLogger(__name__)


class MarkdownGenerator:
    """Генератор Markdown‑файлов"""

    def __init__(self) -> None:
        self.config = get_config()
        self.output_config = self.config.output

    # ---------- Публичные методы ----------

    def generate_file_documentation(
        self,
        parsed_file: ParsedFile,
        gpt_result: GPTAnalysisResult,
        output_dir: str,
        custom_filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Создать MD‑отчёт по одному файлу.

        :param parsed_file: структура с результатами парсинга
        :param gpt_result: ответ GPT
        :param output_dir: куда сохранять
        :param custom_filename: уже готовое имя файла (если None – будет сгенерировано)
        """
        try:
            ensure_directory_exists(output_dir)

            # ---------- имя выходного файла ----------
            if custom_filename:
                file_name = custom_filename
            else:
                file_name = self._fallback_filename(parsed_file.file_info.path)

            output_path = Path(output_dir) / file_name

            # ---------- содержимое ----------
            content = self._generate_file_content(parsed_file, gpt_result)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.debug("Сгенерирована документация: %s", output_path)
            return str(output_path)

        except Exception as exc:  # pragma: no cover
            logger.error("Ошибка генерирования MD для %s: %s", parsed_file.file_info.path, exc)
            return None

    def generate_index_file(
        self,
        generated_files: List[Dict],
        output_dir: str,
        repo_path: str
    ) -> Optional[str]:
        """Создаёт README со списком всех отчётов"""
        try:
            index_path = Path(output_dir) / "README.md"
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(self._generate_index_content(generated_files, repo_path))
            return str(index_path)
        except Exception as exc:  # pragma: no cover
            logger.error("Ошибка создания README: %s", exc)
            return None

    # ---------- Внутренние методы ----------

    def _generate_file_content(self, parsed_file: ParsedFile, gpt_result: GPTAnalysisResult) -> str:
        """
        Если GPT вернул полный отчёт – отдаём его без изменений.
        Иначе – используем старый формат.
        """
        if gpt_result and gpt_result.full_text and not gpt_result.error:
            return gpt_result.full_text

        # ---------- fallback‑формат ----------
        parts: List[str] = [
            f"# {Path(parsed_file.file_info.path).name}",
            "",
            f"**Путь:** `{parsed_file.file_info.path}`",
            f"**Язык:** {parsed_file.file_info.language}",
            f"**Размер:** {self._format_file_size(parsed_file.file_info.size)}",
            ""
        ]

        if gpt_result and gpt_result.summary:
            parts += ["## Анализ кода", "", gpt_result.summary, ""]

        parts += [
            "---",
            f"*Документация сгенерирована автоматически {datetime.now():%Y-%m-%d %H:%M:%S}*"
        ]
        return "\n".join(parts)

    # ---------- Служебное ----------
    def _fallback_filename(self, file_path: str) -> str:
        safe = clean_filename(file_path.replace("/", "_").replace("\\", "_"))
        return f"summary_{safe}.md"

    def _generate_index_content(self, generated_files: List[Dict], repo_path: str) -> str:
        repo_name = Path(repo_path).name
        total = len(generated_files)
        success = len([f for f in generated_files if f["success"]])
        failed = total - success

        parts = [
            f"# Документация кода: {repo_name}",
            "",
            f"- **Файлов проанализировано:** {total}",
            f"- **Успешно:** {success}",
        ]
        if failed:
            parts.append(f"- **С ошибками:** {failed}")
        parts.append("")
        parts.append(f"*Сгенерировано {datetime.now():%Y-%m-%d %H:%M:%S}*")
        return "\n".join(parts)

    def _format_file_size(self, size: int) -> str:
        if size < 1024:
            return f"{size} B"
        if size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size / (1024 * 1024):.1f} MB"


class DocumentationGenerator:
    """Координатор генерации всей документации"""

    def __init__(self) -> None:
        self.md = MarkdownGenerator()
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_complete_documentation(
        self,
        files_data: List[tuple],
        output_dir: str,
        repo_path: str
    ) -> Dict:
        """
        Создать SUMMARY_REPORT_<repo> с вложенными подпапками.
        """
        repo_name = Path(repo_path).name
        summary_root = Path(repo_path) / f"SUMMARY_REPORT_{repo_name}"
        ensure_directory_exists(str(summary_root))

        generated: List[Dict] = []

        for parsed_file, gpt_result in files_data:
            src_path = Path(parsed_file.file_info.path).resolve()
            repo_path_resolved = Path(repo_path).resolve()
            
            try:
                rel_path = src_path.relative_to(repo_path_resolved)          # например src/main.py
            except ValueError:
                # Если файл не в поддиректории repo_path, используем только имя файла
                rel_path = Path(src_path.name)
                
            rel_parent = rel_path.parent                       # src или .
            target_dir = summary_root / rel_parent
            ensure_directory_exists(str(target_dir))

            # формируем имя «report_<top-folder>_<filename>.md»
            prefix_parts = list(rel_parent.parts)
            prefix = "_".join(prefix_parts).lower()
            if prefix and prefix != ".":
                custom_name = f"report_{prefix}_{src_path.name}.md"
            else:
                custom_name = f"report_{src_path.name}.md"

            doc_path = self.md.generate_file_documentation(
                parsed_file,
                gpt_result,
                str(target_dir),
                custom_filename=custom_name
            )

            generated.append({
                "original_path": parsed_file.file_info.path,
                "doc_path": doc_path,
                "language": parsed_file.file_info.language,
                "success": bool(doc_path)
            })

        index_path = self.md.generate_index_file(generated, str(summary_root), repo_path)

        return {
            "total_files": len(generated),
            "successful": len([f for f in generated if f["success"]]),
            "failed": len([f for f in generated if not f["success"]]),
            "output_directory": str(summary_root),
            "index_file": index_path,
            "success": True
        }
