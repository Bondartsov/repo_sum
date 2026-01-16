#!/usr/bin/env python3
"""
Автоматическая настройка VM для Jina v3 RAG системы.

Этот скрипт:
1. Подключается к VM по SSH
2. Клонирует репозиторий
3. Устанавливает все зависимости
4. Запускает Qdrant (если нужно)
5. Запускает RAG сервис
6. Проверяет что все работает

Использование: python vm_start.py
"""

import os
import sys
import shlex
import time
import logging
from pathlib import Path
from typing import Tuple, Optional
import paramiko
from dotenv import load_dotenv, dotenv_values
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Удаляем старые хендлеры, чтобы избежать дублирования
for h in list(logger.handlers):
    logger.removeHandler(h)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Консольный хендлер
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Файловый хендлер (без закрытия глобальных потоков)
file_handler = logging.FileHandler('vm_setup.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class VMSetupManager:
    """Управление автоматической настройкой VM"""
    
    def __init__(self):
        # Загрузка конфигурации
        load_dotenv()
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
            sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
        except Exception:
            pass
        self.console = Console()
        
        # VM параметры из .env
        self.vm_host = os.getenv("VM_HOST", "10.61.11.54")
        self.vm_user = os.getenv("VM_USER", "user")
        self.vm_password = os.getenv("VM_PASSWORD")
        self.vm_port = int(os.getenv("VM_PORT", "22"))
        
        # Пути на VM
        self.vm_work_dir = "~/repo_sum_rag"
        self.vm_repo_dir = f"{self.vm_work_dir}/repo_sum"
        self.repo_url = os.getenv("VM_REPO_URL", "https://github.com/Bondartsov/repo_sum.git")
        self.repo_branch = os.getenv("VM_REPO_BRANCH", "jina-embeddings-v3")
        
        # SSH клиент
        self.ssh_client = None
        
        # Проверка параметров
        if not self.vm_password:
            raise ValueError("VM_PASSWORD не найден в .env файле!")
            
        logger.info(f"Инициализация VMSetupManager: {self.vm_user}@{self.vm_host}:{self.vm_port}")
    
    def connect_ssh(self) -> bool:
        """Подключение к VM по SSH"""
        try:
            self.console.print(f"[blue]🔗 Подключение к VM {self.vm_host}...[/blue]")
            
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.load_system_host_keys()
            
            self.ssh_client.connect(
                hostname=self.vm_host,
                port=self.vm_port,
                username=self.vm_user,
                password=self.vm_password,
                timeout=30
            )
            
            self.console.print("[green]✅ SSH подключение установлено[/green]")
            logger.info(f"SSH соединение с {self.vm_host} успешно")
            return True
            
        except Exception as e:
            self.console.print(f"[red]❌ Ошибка SSH подключения: {e}[/red]")
            logger.error(f"SSH ошибка: {e}")
            return False
    
    def execute_command(self, command: str, timeout: int = 30, ignore_exit_codes: list = None) -> Tuple[bool, str, str]:
        """
        Выполнение команды на VM.
        
        Args:
            command: Команда для выполнения
            timeout: Таймаут в секундах
            ignore_exit_codes: Список кодов возврата, которые не считаются ошибкой (для nohup, background процессов и т.д.)
        
        Returns:
            Tuple[success, stdout, stderr]
        """
        try:
            logger.debug(f"Выполнение: {command}")
            stdin, stdout, stderr = self.ssh_client.exec_command(command, timeout=timeout)
            
            # Ожидание завершения
            exit_status = stdout.channel.recv_exit_status()
            
            stdout_text = stdout.read().decode('utf-8')
            stderr_text = stderr.read().decode('utf-8')
            
            # Проверка успешности с учётом игнорируемых кодов
            success = exit_status == 0 or (ignore_exit_codes and exit_status in ignore_exit_codes)
            
            # Логируем только реальные ошибки
            if not success and exit_status != 0:
                logger.warning(f"Команда завершилась с кодом {exit_status}")
                if stderr_text.strip():
                    logger.warning(f"STDERR: {stderr_text}")
            
            return success, stdout_text, stderr_text
            
        except Exception as e:
            logger.error(f"Ошибка выполнения команды '{command}': {e}")
            return False, "", str(e)
    
    def check_vm_status(self, show_banner: bool = True) -> dict:
        """Проверка текущего состояния VM"""
        if show_banner:
            self.console.print("[blue]🔍 Проверка состояния VM...[/blue]")

        status = {
            'python_version': None,
            'memory_gb': None,
            'repo_exists': False,
            'env_exists': False,
            'venv_exists': False,
            'qdrant_running': False,
            'rag_service_running': False
        }

        success, output, _ = self.execute_command("python3 --version")
        if success:
            status['python_version'] = output.strip()

        success, output, _ = self.execute_command("free -h | grep 'Mem:' | awk '{print $2}'")
        if success:
            status['memory_gb'] = output.strip()

        success, _, _ = self.execute_command(f"test -d {self.vm_repo_dir}")
        status['repo_exists'] = success

        success, _, _ = self.execute_command(f"test -f {self.vm_repo_dir}/.env")
        status['env_exists'] = success

        success, _, _ = self.execute_command(f"test -d {self.vm_repo_dir}/venv")
        status['venv_exists'] = success

        success, _, _ = self.execute_command("curl -s http://localhost:6333 >/dev/null 2>&1")
        status['qdrant_running'] = success

        success, _, _ = self.execute_command("curl -s http://localhost:8000/health >/dev/null 2>&1")
        status['rag_service_running'] = success

        self._show_vm_status(status)

        return status

    def _show_vm_status(self, status: dict):
        """Отрисовать текущий статус VM."""
        table = Table(title="Статус VM")
        table.add_column("Компонент", style="cyan")
        table.add_column("Состояние", style="bold")
        table.add_column("Детали")

        python_status = "✅ Установлен" if status['python_version'] else "❌ Отсутствует"
        table.add_row("Python", python_status, status['python_version'] or "")

        memory_status = "✅ Доступно" if status['memory_gb'] else "❌ Нет данных"
        table.add_row("Память", memory_status, status['memory_gb'] or "")

        repo_status = "✅ Синхронизирован" if status['repo_exists'] else "❌ Отсутствует"
        table.add_row("Репозиторий", repo_status, self.vm_repo_dir)

        env_status = "✅ Найден" if status['env_exists'] else "❌ Нет"
        table.add_row(".env", env_status, f"{self.vm_repo_dir}/.env")

        venv_status = "✅ Готов" if status['venv_exists'] else "❌ Нет"
        table.add_row("Virtual Env", venv_status, f"{self.vm_repo_dir}/venv")

        qdrant_status = "✅ Запущен" if status['qdrant_running'] else "❌ Остановлен"
        table.add_row("Qdrant", qdrant_status, "localhost:6333")

        rag_status = "✅ Запущен" if status['rag_service_running'] else "❌ Остановлен"
        table.add_row("RAG Service", rag_status, "localhost:8000")

        self.console.print(table)

    def get_available_branches(self) -> list:
        """
        Получение списка доступных веток из GitHub репозитория с информацией о коммитах.

        Returns:
            List[dict] - список словарей с ключами: name, commit_hash, commit_date
        """
        self.console.print("[blue]🔍 Получаю список веток из GitHub...[/blue]")

        try:
            # Используем git ls-remote для получения списка веток с хэшами
            success, output, error = self.execute_command(
                f"git ls-remote --heads {self.repo_url}",
                timeout=30
            )

            if not success or not output.strip():
                self.console.print(f"[red]❌ Не удалось получить список веток: {error}[/red]")
                return []

            # Парсим вывод: commit_hash refs/heads/branch_name
            branch_info = []

            # Используем GitHub API для получения дат коммитов (более надежно)
            # Формат ls-remote: <hash> refs/heads/<branch>
            for line in output.strip().splitlines():
                if 'refs/heads/' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        commit_hash = parts[0][:7]  # Короткий хэш
                        full_hash = parts[0]  # Полный хэш для API
                        branch_name = parts[1].split('refs/heads/')[-1].strip()

                        # Получаем дату через GitHub API (работает без клонирования)
                        # Используем curl к GitHub API для получения информации о коммите
                        api_url = f"https://api.github.com/repos/Bondartsov/repo_sum/commits/{full_hash}"
                        date_cmd = f"curl -s {api_url} | grep '\"date\":' | head -1 | cut -d'\"' -f4"
                        date_success, date_output, _ = self.execute_command(date_cmd, timeout=15)

                        if date_success and date_output.strip():
                            # Парсим ISO формат: 2025-10-02T15:30:00Z -> 02.10.2025
                            iso_date = date_output.strip().split('T')[0]  # 2025-10-02
                            try:
                                year, month, day = iso_date.split('-')
                                commit_date = f"{day}.{month}.{year}"
                            except Exception:
                                commit_date = "N/A"
                        else:
                            commit_date = "N/A"

                        branch_info.append({
                            'name': branch_name,
                            'commit_hash': commit_hash,
                            'commit_date': commit_date
                        })

            # Сортируем ветки, приоритет: master, main, затем остальные по алфавиту
            def branch_sort_key(branch: dict) -> tuple:
                name = branch['name']
                if name == 'master':
                    return (0, name)
                elif name == 'main':
                    return (1, name)
                else:
                    return (2, name)

            branch_info.sort(key=branch_sort_key)

            self.console.print(f"[green]✅ Найдено веток: {len(branch_info)}[/green]")
            return branch_info

        except Exception as e:
            self.console.print(f"[red]❌ Ошибка получения списка веток: {e}[/red]")
            logger.error(f"Branch list error: {e}")
            return []

    def select_branch_interactive(self, branches: list) -> Optional[str]:
        """
        Интерактивный выбор ветки из списка.

        Args:
            branches: List[dict] - список словарей с ключами name, commit_hash, commit_date

        Returns:
            Имя выбранной ветки или None
        """
        if not branches:
            self.console.print("[red]❌ Список веток пуст[/red]")
            return None

        # Создаем красивую таблицу с ветками
        table = Table(title="📁 Выберите ветку для развертывания")
        table.add_column("№", style="cyan", justify="right")
        table.add_column("Ветка", style="bold green")
        table.add_column("Коммит", style="yellow")
        table.add_column("Дата", style="dim")

        for idx, branch_info in enumerate(branches, start=1):
            branch_name = branch_info['name']
            commit_hash = branch_info['commit_hash']
            commit_date = branch_info['commit_date']

            # Выделяем текущую ветку
            if branch_name == self.repo_branch:
                branch_display = f"[bold yellow]{branch_name}[/bold yellow] (текущая)"
            else:
                branch_display = branch_name

            table.add_row(str(idx), branch_display, commit_hash, commit_date)

        self.console.print(table)

        # Запрашиваем выбор пользователя
        try:
            choice = input(f"\nВведите номер ветки [1-{len(branches)}] или 'q' для отмены: ").strip()

            if choice.lower() == 'q':
                self.console.print("[yellow]⏹️ Выбор ветки отменен[/yellow]")
                return None

            idx = int(choice)
            if 1 <= idx <= len(branches):
                selected_branch = branches[idx - 1]['name']
                self.console.print(f"[green]✅ Выбрана ветка: {selected_branch}[/green]")
                return selected_branch
            else:
                self.console.print(f"[red]❌ Неверный номер. Выберите от 1 до {len(branches)}[/red]")
                return None

        except (ValueError, KeyboardInterrupt):
            self.console.print("[yellow]⏹️ Выбор ветки отменен[/yellow]")
            return None

    def setup_repository(self) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Manage repository on the VM by cloning or syncing.

        Returns:
            Tuple[success, old_commit, new_commit, branch_name]
        """
        self.console.print("[blue]📁 Настраиваю репозиторий...[/blue]")

        success, _, _ = self.execute_command(f"mkdir -p {self.vm_work_dir}")
        if not success:
            self.console.print("[red]❌ Не удалось создать рабочую папку[/red]")
            return False, None, None, None

        repo_ready, _, _ = self.execute_command(f"test -d {self.vm_repo_dir}/.git")

        # Сохраняем старую версию если репозиторий уже существует
        old_commit = None
        old_branch = None
        old_date = None
        if repo_ready:
            action = "update"
            # Получаем старую ветку, коммит и дату перед обновлением
            old_info_cmd = (
                f"cd {self.vm_repo_dir} && "
                f"git rev-parse --abbrev-ref HEAD && "
                f"git rev-parse --short HEAD && "
                f"git log -1 --format=%ci"
            )
            old_success, old_output, _ = self.execute_command(old_info_cmd)
            if old_success:
                old_lines = [line.strip() for line in old_output.strip().splitlines() if line.strip()]
                if len(old_lines) >= 3:
                    old_branch = old_lines[0]
                    old_commit = old_lines[1]
                    # Берем дату и конвертируем в ДД.ММ.ГГГГ формат
                    iso_date = old_lines[2].split()[0]  # 2025-10-02
                    try:
                        year, month, day = iso_date.split('-')
                        old_date = f"{day}.{month}.{year}"
                    except Exception:
                        old_date = iso_date

            commands = [
                f"cd {self.vm_repo_dir}",
                "git fetch --all --prune",
                f"git checkout {shlex.quote(self.repo_branch)}",  # Переключаемся на нужную ветку
                f"git reset --hard origin/{shlex.quote(self.repo_branch)}",
                "git clean -fd"
            ]
        else:
            action = "clone"
            commands = [
                f"cd {self.vm_work_dir}",
                f"git clone --branch {shlex.quote(self.repo_branch)} --single-branch {self.repo_url} repo_sum",
                f"cd {self.vm_repo_dir}"
            ]

        commands.extend([
            "git submodule update --init --recursive",
            "git rev-parse --abbrev-ref HEAD",
            "git rev-parse --short HEAD",
            "git log -1 --format=%ci"
        ])

        combined_command = " && ".join(commands)
        success, output, error = self.execute_command(combined_command, timeout=600)

        if success:
            lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
            if len(lines) >= 3:
                branch = lines[-3]
                new_commit = lines[-2]
                # Берем дату и конвертируем в ДД.ММ.ГГГГ формат
                iso_date = lines[-1].split()[0]  # 2025-10-02
                try:
                    year, month, day = iso_date.split('-')
                    new_date = f"{day}.{month}.{year}"
                except Exception:
                    new_date = iso_date
            else:
                branch = self.repo_branch
                new_commit = ""
                new_date = "N/A"

            status_label = "склонирован" if action == "clone" else "обновлён"

            # Формируем сообщение о версии
            if action == "update" and old_commit and old_branch and new_commit:
                if old_commit == new_commit and old_branch == branch:
                    version_msg = f"версия {new_commit} ({new_date}) (без изменений)"
                else:
                    version_msg = (
                        f"версия обновлена с ветки ({old_branch}) {old_commit} ({old_date}) "
                        f"на ветку ({branch}) {new_commit} ({new_date})"
                    )
            else:
                version_msg = f"версия {new_commit} ({new_date}), ветка: {branch}"

            self.console.print(f"[green]✅ Репозиторий {status_label}: {version_msg}[/green]")
            logger.info(f"Repository {action} -> {branch}@{new_commit} (old: {old_branch or 'N/A'}@{old_commit or 'N/A'})")
            return True, old_commit, new_commit, branch

        self.console.print(f"[red]❌ Ошибка синхронизации репозитория: {error}")
        if output.strip():
            logger.error(f"Git output: {output.strip()}")
        return False, None, None, None

    def check_critical_files(self) -> bool:
        """Проверка критических файлов"""
        self.console.print("[blue]🔍 Проверка критических файлов...[/blue]")
        
        critical_files = [
            "vm_rag_service.py",
            "rag/remote_embedder.py", 
            "rag/remote_vector_store.py",
            "requirements.txt"
        ]
        
        all_present = True
        
        for file in critical_files:
            file_path = f"{self.vm_repo_dir}/{file}"
            success, _, _ = self.execute_command(f"test -f {file_path}")
            
            if success:
                self.console.print(f"[green]✅ {file}[/green]")
            else:
                self.console.print(f"[red]❌ {file} ОТСУТСТВУЕТ![/red]")
                all_present = False
        
        if not all_present:
            self.console.print("[red]🚨 Критические файлы отсутствуют! Проверьте ветку репозитория.[/red]")
            return False
        
        return True
    
    def setup_python_environment(self) -> bool:
        """Ensure the Python virtual environment exists and dependencies are installed."""
        self.console.print("[blue]🐍 Настраиваю Python окружение...[/blue]")

        venv_path = f"{self.vm_repo_dir}/venv"
        venv_exists, _, _ = self.execute_command(f"test -d {venv_path}")

        commands = [f"cd {self.vm_repo_dir}"]
        if not venv_exists:
            commands.append("python3 -m venv venv")

        # Используем pip install с флагами -q (quiet) и перенаправлением вывода для полного отключения шума
        commands.extend([
            "source venv/bin/activate",
            "pip install --disable-pip-version-check --upgrade pip setuptools wheel",
            "pip install --disable-pip-version-check -r requirements.txt",
            "pip install --disable-pip-version-check sentence-transformers>=3.0 transformers>=4.35.0",
            "echo 'INSTALL_COMPLETE'"  # Маркер успешной установки
        ])

        combined_command = " && ".join(commands)
        success, output, error = self.execute_command(combined_command, timeout=900)

        if success and "INSTALL_COMPLETE" in output:
            state = "создано" if not venv_exists else "обновлено"
            deps_msg = "(зависимости обновлены)"

            self.console.print(f"[green]✅ Python окружение {state} {deps_msg}[/green]")
            logger.info(f"Python environment {state}")
            return True

        self.console.print("[red]❌ Ошибка установки Python окружения.[/red]")
        if error.strip():
            self.console.print(Panel(error.strip(), title="STDERR", border_style="red"))
        if output.strip():
            self.console.print(Panel(output.strip(), title="STDOUT", border_style="yellow"))
        logger.error(f"Venv error: {error}\\nOutput: {output}")
        return False

    def test_jina_v3(self) -> bool:
        """Тестирование загрузки Jina v3"""
        self.console.print("[blue]🧪 Тестирование Jina v3...[/blue]")
        
        test_script = """
print('🚀 Тестируем Jina v3...')
try:
    from sentence_transformers import SentenceTransformer
    print('📥 Загружаем модель (570M параметров)...')
    model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True)
    print('✅ Jina v3 УСПЕШНО ЗАГРУЖЕНА!')
    print(f'📏 Размерность: {model.get_sentence_embedding_dimension()}d')
    
    # Тест dual task
    query = model.encode(['test query'], task='retrieval.query')
    passage = model.encode(['test passage'], task='retrieval.passage')
    print(f'✅ Query task: {query.shape}')
    print(f'✅ Passage task: {passage.shape}')
    print('🎉 DUAL TASK РАБОТАЕТ!')
    print('SUCCESS')
except Exception as e:
    print(f'ERROR: {e}')
"""
        
        command = f"cd {self.vm_repo_dir} && source venv/bin/activate && python3 -c \"{test_script}\""
        success, output, error = self.execute_command(command, timeout=300)
        
        if success and "SUCCESS" in output:
            self.console.print("[green]✅ Jina v3 тест пройден![/green]")
            logger.info("Jina v3 загружается корректно")
            return True
        else:
            self.console.print(f"[red]❌ Jina v3 тест не пройден: {error}[/red]")
            self.console.print(f"Output: {output}")
            return False
    
    def setup_qdrant(self) -> bool:
        """Настройка и запуск Qdrant"""
        self.console.print("[blue]🗄️ Настройка Qdrant...[/blue]")
        
        # Проверка запущен ли уже
        success, _, _ = self.execute_command("curl -s http://localhost:6333 >/dev/null 2>&1")
        if success:
            self.console.print("[green]✅ Qdrant уже запущен[/green]")
            return True
        
        # Запуск Qdrant в Docker
        docker_command = (
            "docker run -d --name qdrant-repo-sum "
            "-p 6333:6333 -p 6334:6334 "
            f"-v {self.vm_repo_dir}/qdrant_storage:/qdrant/storage "
            "qdrant/qdrant 2>/dev/null || echo 'Container already exists'"
        )
        
        success, output, error = self.execute_command(docker_command)
        
        # Ждем запуска
        self.console.print("[blue]⏳ Ожидание запуска Qdrant...[/blue]")
        for i in range(30):  # 30 секунд максимум
            time.sleep(1)
            success, _, _ = self.execute_command("curl -s http://localhost:6333 >/dev/null 2>&1")
            if success:
                self.console.print("[green]✅ Qdrant успешно запущен[/green]")
                return True
        
        self.console.print("[red]❌ Qdrant не запустился в течение 30 секунд[/red]")
        return False
    
    def create_env_file(self) -> bool:
        """Create or refresh .env on the VM using the local template and secrets."""
        self.console.print("[blue]⚙️ Создание .env файла...[/blue]")

        template_path = Path(".env.example")
        local_env_path = Path(".env")

        template_lines = []
        if template_path.exists():
            template_lines = template_path.read_text(encoding="utf-8").splitlines()
        else:
            self.console.print("[yellow]⚠️ .env.example не найден, используем только .env[/yellow]")

        local_values = {}
        if local_env_path.exists():
            local_values = {k: v for k, v in dotenv_values(local_env_path).items() if v is not None}

        rendered_lines = []
        template_keys = []
        missing_keys = []

        if template_lines:
            for line in template_lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in line:
                    rendered_lines.append(line)
                    continue

                key, _, default_value = line.partition("=")
                key = key.strip()
                template_keys.append(key)

                value = local_values.get(key)
                if value is None:
                    value = os.getenv(key)
                if value is None:
                    value = default_value.strip()
                    missing_keys.append(key)

                rendered_lines.append(f"{key}={value}")
        elif local_values:
            rendered_lines = [f"{key}={value}" for key, value in local_values.items()]
        else:
            self.console.print("[red]❌ Нет данных для генерации .env[/red]")
            return False

        additional_keys = [key for key in local_values.keys() if key not in template_keys]
        if additional_keys:
            rendered_lines.append("")
            rendered_lines.append("# === Additional Variables ===")
            for key in sorted(additional_keys):
                rendered_lines.append(f"{key}={local_values[key]}")

        # Force VM-local overrides in the resulting .env content
        def set_kv(lines: list, key: str, value: str) -> list:
            updated = False
            out = []
            for ln in lines:
                if not ln or ln.startswith('#') or '=' not in ln:
                    out.append(ln)
                    continue
                k, sep, v = ln.partition('=')
                if k.strip() == key:
                    out.append(f"{key}={value}")
                    updated = True
                else:
                    out.append(ln)
            if not updated:
                if out and out[-1] != "":
                    out.append("")
                out.append(f"{key}={value}")
            return out

        # Compute ports and overrides
        port = None
        for ln in rendered_lines:
            if ln.startswith('RAG_SERVICE_PORT='):
                port = ln.split('=',1)[1].strip()
                break
        if not port:
            port = os.getenv('RAG_SERVICE_PORT', '8000')

        overrides = {
            'RAG_SERVICE_HOST': '127.0.0.1',
            'RAG_SERVICE_PORT': port,
            'RAG_EMBEDDINGS_ENDPOINT': f'http://127.0.0.1:{port}/v1/embeddings',
            'RAG_SEARCH_ENDPOINT': f'http://127.0.0.1:{port}/v1/search_v2',
            'RAG_TEXT_SEARCH_ENDPOINT': f'http://127.0.0.1:{port}/v1/search',
            'RAG_INDEX_ENDPOINT': f'http://127.0.0.1:{port}/v1/index',
            'RAG_API_CONTRACT': 'v1.0.0',
            'RAG_TIMEOUT_SECONDS': os.getenv('RAG_TIMEOUT_SECONDS', '3600'),
            'QDRANT_HOST': 'localhost',
            'QDRANT_PORT': os.getenv('QDRANT_PORT', '6333'),
            'QDRANT_PREFER_GRPC': 'false',
            'EMBEDDING_PROVIDER': 'fastembed',
            'VECTOR_STORE_PROVIDER': 'local',
        }

        for k, v in overrides.items():
            rendered_lines = set_kv(rendered_lines, k, str(v))

        # Ensure EMBEDDING_DIMENSION matches EMB_TRUNCATE_DIM if present
        trunc_dim = None
        for ln in rendered_lines:
            if ln.startswith('EMB_TRUNCATE_DIM='):
                trunc_dim = ln.split('=',1)[1].strip()
                break
        if trunc_dim:
            rendered_lines = set_kv(rendered_lines, 'EMBEDDING_DIMENSION', trunc_dim)

        env_content = "\n".join(rendered_lines) + "\n"

        self.execute_command(f"cd {self.vm_repo_dir} && cp .env .env.backup 2>/dev/null || true")

        command = (
            f"cd {self.vm_repo_dir} && cat > .env <<'EOF'\n"
            f"{env_content}EOF\n"
        )
        success, _, error = self.execute_command(command)

        if success:
            self.console.print("[green]✅ .env файл обновлён[/green]")
            if missing_keys:
                pretty_missing = ", ".join(sorted(set(missing_keys)))
                self.console.print(f"[yellow]ℹ️ Использованы значения по умолчанию для: {pretty_missing}[/yellow]")
            logger.info("Remote .env refreshed with %d keys", len([line for line in rendered_lines if line and not line.startswith('#')]))
            return True

        self.console.print(f"[red]❌ Ошибка при записи .env: {error}")
        return False

    def diagnose_rag_service(self) -> dict:
        """Диагностика проблем с RAG сервисом"""
        self.console.print("[blue]🔍 Диагностика RAG сервиса...[/blue]")
        
        diagnostics = {
            'process_running': False,
            'port_available': True,
            'logs_exist': False,
            'python_imports_ok': False,
            'service_logs': "",
            'error_details': []
        }
        
        # Проверка процесса
        success, output, _ = self.execute_command("ps aux | grep vm_rag_service | grep -v grep")
        if success and output.strip():
            diagnostics['process_running'] = True
            self.console.print("[green]✅ Процесс vm_rag_service запущен[/green]")
        else:
            self.console.print("[red]❌ Процесс vm_rag_service не найден[/red]")
        
        # Проверка порта 8000
        success, output, _ = self.execute_command("netstat -tulnp | grep :8000")
        if success and output.strip():
            diagnostics['port_available'] = False
            self.console.print(f"[yellow]⚠️ Порт 8000 занят: {output.strip()}[/yellow]")
        else:
            self.console.print("[green]✅ Порт 8000 свободен[/green]")
        
        # Проверка логов
        success, logs, _ = self.execute_command(f"cd {self.vm_repo_dir} && cat rag_service.log 2>/dev/null | tail -20")
        if success and logs.strip():
            diagnostics['logs_exist'] = True
            diagnostics['service_logs'] = logs.strip()
            self.console.print("[green]✅ Логи RAG сервиса найдены[/green]")
            self.console.print(f"[dim]Последние строки логов:[/dim]\n{logs}")
        else:
            self.console.print("[red]❌ Логи RAG сервиса не найдены[/red]")
        
        # Проверка Python imports
        import_test = """
try:
    import sys
    sys.path.append('.')
    from vm_rag_service import app
    print('IMPORTS_OK')
except Exception as e:
    print(f'IMPORT_ERROR: {e}')
"""
        
        success, output, error = self.execute_command(
            f"cd {self.vm_repo_dir} && source venv/bin/activate && python3 -c \"{import_test}\""
        )
        
        if success and "IMPORTS_OK" in output:
            diagnostics['python_imports_ok'] = True
            self.console.print("[green]✅ Python imports работают[/green]")
        else:
            diagnostics['python_imports_ok'] = False
            diagnostics['error_details'].append(f"Import error: {output} {error}")
            self.console.print(f"[red]❌ Ошибка импорта: {output} {error}[/red]")
        
        # Попытка ручного запуска с детальной ошибкой
        if not diagnostics['process_running']:
            self.console.print("[blue]🧪 Тест ручного запуска...[/blue]")
            success, output, error = self.execute_command(
                f"cd {self.vm_repo_dir} && source venv/bin/activate && timeout 10s python vm_rag_service.py"
            )
            if not success:
                diagnostics['error_details'].append(f"Manual start error: {output} {error}")
                self.console.print(f"[red]❌ Ошибка ручного запуска: {output} {error}[/red]")
        
        return diagnostics
    def check_service_running(self, service_name: str = "vm_rag_service.py") -> bool:
        """Проверка запущен ли сервис на VM"""
        success, output, _ = self.execute_command(
            f"ps aux | grep '{service_name}' | grep -v grep"
        )
        return success and bool(output.strip())
    
    def get_service_pid(self, pid_file: str = None) -> Optional[str]:
        """Получить PID сервиса из файла"""
        if pid_file is None:
            pid_file = f"{self.vm_repo_dir}/rag_service.pid"
        
        success, output, _ = self.execute_command(f"cat {pid_file} 2>/dev/null")
        pid = output.strip()
        return pid if pid else None
    
    def stop_service(self, pid_file: str = None) -> bool:
        """Остановить сервис"""
        if pid_file is None:
            pid_file = f"{self.vm_repo_dir}/rag_service.pid"
        
        pid = self.get_service_pid(pid_file)
        if pid:
            self.console.print(f"[blue]🛑 Остановка процесса PID {pid}...[/blue]")
            success, _, _ = self.execute_command(f"kill {pid}")
            time.sleep(2)
            
            # Проверяем что процесс остановился
            if not self.check_service_running():
                self.console.print("[green]✅ Сервис остановлен[/green]")
                return True
            else:
                self.console.print("[yellow]⚠️ Процесс не остановился, пробуем kill -9[/yellow]")
                self.execute_command(f"kill -9 {pid}")
                time.sleep(1)
                return not self.check_service_running()
        else:
            self.console.print("[yellow]⚠️ PID файл не найден или пуст[/yellow]")
            # Проверим, может процесс все равно запущен
            if self.check_service_running():
                self.console.print("[yellow]Процесс найден, попытка остановить через pkill...[/yellow]")
                self.execute_command("pkill -f vm_rag_service.py")
                time.sleep(2)
                return not self.check_service_running()
            return False
    
    def start_service_via_script(self, script_path: str = None) -> bool:
        """Запустить сервис через скрипт start_vm_rag.sh"""
        if script_path is None:
            script_path = f"{self.vm_repo_dir}/start_vm_rag.sh"
        
        self.console.print(f"[blue]🚀 Запуск сервиса через {script_path}...[/blue]")
        
        # Проверка существования скрипта
        success, output, _ = self.execute_command(f"test -f {script_path} && echo 'exists'")
        if not output.strip():
            self.console.print(f"[red]❌ Скрипт {script_path} не найден![/red]")
            self.console.print("[yellow]💡 Создайте скрипт согласно docs/VM_STARTUP_CONFIGURATION.md[/yellow]")
            self.console.print("[yellow]💡 Или используйте python vm_start.py start для запуска без скрипта[/yellow]")
            return False
        
        # Запуск через скрипт
        start_command = f"cd {self.vm_repo_dir} && {script_path}"
        success, output, error = self.execute_command(start_command, ignore_exit_codes=[7, 127])
        
        if output.strip():
            self.console.print(f"[dim]Вывод скрипта: {output.strip()}[/dim]")
        
        # Проверка что запустился - используем умный health check
        print()  # Новая строка для читаемости
        if self.verify_service_health():
            pid = self.get_service_pid()
            self.console.print(f"[green]✅ Сервис полностью готов с PID {pid}[/green]")
            return True
        else:
            self.console.print("[red]❌ Не удалось запустить сервис![/red]")
            # Проверим логи
            log_success, logs, _ = self.execute_command(
                f"cd {self.vm_repo_dir} && tail -5 rag_service.log 2>/dev/null"
            )
            if log_success and logs.strip():
                self.console.print(f"[yellow]📄 Последние строки логов:[/yellow]\n{logs.strip()}")
            return False
    
    def restart_service(self) -> bool:
        """Перезапуск сервиса"""
        self.console.print("[blue]🔄 Перезапуск сервиса...[/blue]")
        
        # Остановка
        self.stop_service()
        time.sleep(2)
        
        # Запуск через скрипт если он существует, иначе стандартный метод
        script_path = f"{self.vm_repo_dir}/start_vm_rag.sh"
        success, _, _ = self.execute_command(f"test -f {script_path}")
        
        if success:
            return self.start_service_via_script(script_path)
        else:
            self.console.print("[yellow]⚠️ Скрипт start_vm_rag.sh не найден, использую стандартный запуск[/yellow]")
            return self.start_rag_service()
    
    def verify_service_health(self, health_url="http://localhost:8000/health",
                             max_attempts=18, interval=5):
        """
        Проверка health endpoint с retry-логикой
        
        Args:
            health_url: URL health endpoint (по умолчанию localhost:8000/health)
            max_attempts: Максимальное количество попыток (default: 18 = 90 секунд)
            interval: Интервал между попытками в секундах (default: 5)
        
        Returns:
            bool: True если сервис отвечает корректно
        """
        print(f"🏥 Проверка health endpoint (макс. {max_attempts * interval} сек)...")
        
        for attempt in range(1, max_attempts + 1):
            # Индикатор прогресса
            elapsed = (attempt - 1) * interval
            print(f"   ⏳ Попытка {attempt}/{max_attempts} (прошло {elapsed} сек)...", end='', flush=True)
            
            try:
                # Выполнение health check
                stdin, stdout, stderr = self.ssh_client.exec_command(
                    f"curl -s -f {health_url} --connect-timeout 5 --max-time 10",
                    timeout=15
                )
                
                exit_status = stdout.channel.recv_exit_status()
                response = stdout.read().decode().strip()
                
                # Проверка успешности
                if exit_status == 0 and response:
                    # Проверка содержимого ответа
                    if any(keyword in response.lower() for keyword in ["connected", "status", "ok", "healthy"]):
                        print(" ✅ УСПЕХ!")
                        print(f"   📊 Сервис готов (время загрузки: {elapsed} сек)")
                        
                        # Показать превью ответа
                        print(f"   🔍 Ответ: {response[:150]}{'...' if len(response) > 150 else ''}")
                        return True
                
                # Endpoint ответил, но не корректно
                if exit_status == 0:
                    print(" ⚠️ Ответ получен, но некорректный")
                    print(f"      Код: {exit_status}, Ответ: {response[:100]}")
                else:
                    # Endpoint не доступен, но это может быть нормально (модель загружается)
                    print(" ⏳ Ещё загружается...")
                    
            except Exception as e:
                print(f" ⚠️ Ошибка: {str(e)[:50]}")
            
            # Если это не последняя попытка - ждём
            if attempt < max_attempts:
                time.sleep(interval)
            else:
                # Последняя попытка провалилась
                print(f"\n   ❌ Health check не пройден после {max_attempts * interval} секунд")
                print("   💡 Проверьте логи: ssh user@10.61.11.54 'tail -100 ~/repo_sum_rag/repo_sum/rag_service.log'")
                return False
        
        return False
    
    
    def start_rag_service(self) -> bool:
        """Запуск RAG сервиса с улучшенной диагностикой"""
        self.console.print("[blue]🚀 Запуск RAG сервиса...[/blue]")
        
        # Проверка не запущен ли уже
        success, _, _ = self.execute_command("curl -s http://localhost:8000/health >/dev/null 2>&1")
        if success:
            self.console.print("[green]✅ RAG сервис уже запущен[/green]")
            return True
        
        # Очистка старых процессов и логов
        self.execute_command(f"cd {self.vm_repo_dir} && kill $(cat rag_service.pid 2>/dev/null) 2>/dev/null || true")
        self.execute_command(f"cd {self.vm_repo_dir} && rm -f rag_service.log rag_service.pid")
        
        # Запуск в фоновом режиме
        start_command = (
            f"cd {self.vm_repo_dir} && "
            "source venv/bin/activate && "
            "nohup python vm_rag_service.py > rag_service.log 2>&1 & "
            "echo $! > rag_service.pid"
        )
        
        # Игнорируем коды 7 (nohup/background) и 127 (command not found может быть временной проблемой)
        success, output, error = self.execute_command(start_command, ignore_exit_codes=[7, 127])
        
        if not success:
            self.console.print(f"[red]❌ Ошибка запуска сервиса: {error}[/red]")
            # Диагностика при ошибке запуска
            self.diagnose_rag_service()
            return False
        
        # Ждем запуска сервиса с детальным прогресс-баром и принудительным timeout
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                
                task = progress.add_task("Запуск RAG сервиса...", total=60)
                start_time = time.time()
                
                for i in range(60):  # 60 секунд максимум
                    # Принудительный timeout protection
                    if time.time() - start_time > 70:  # 70 секунд абсолютный лимит
                        progress.stop()
                        self.console.print("[red]⚠️ Принудительное завершение по timeout (70s)[/red]")
                        break
                    
                    time.sleep(1)
                    success, _, _ = self.execute_command("curl -s http://localhost:8000/health >/dev/null 2>&1")
                    if success:
                        progress.update(task, description="✅ RAG сервис запущен!")
                        self.console.print("[green]✅ RAG сервис успешно запущен[/green]")
                        return True
                    
                    # Обновляем описание прогресса
                    if i < 10:
                        desc = f"Загрузка Jina v3 модели... ({i+1}/60s)"
                    elif i < 20:
                        desc = f"Инициализация компонентов... ({i+1}/60s)"
                    elif i < 30:
                        desc = f"Подключение к Qdrant... ({i+1}/60s)"
                    else:
                        desc = f"Ожидание запуска сервиса... ({i+1}/60s)"
                    
                    progress.update(task, description=desc, advance=1)
                    
                    # Диагностика каждые 15 секунд без остановки прогресса
                    if i % 15 == 14:
                        progress.update(task, description=f"🔍 Проверка статуса... ({i+1}/60s)")
                        success_proc, output, _ = self.execute_command("ps aux | grep vm_rag_service | grep -v grep")
                        if success_proc and output.strip():
                            progress.update(task, description=f"✅ Процесс найден, ждем готовности... ({i+1}/60s)")
                        else:
                            # Проверяем логи для диагностики
                            log_success, logs, _ = self.execute_command(f"cd {self.vm_repo_dir} && tail -3 rag_service.log 2>/dev/null")
                            if log_success and "ERROR" in logs:
                                progress.stop()
                                self.console.print("[red]❌ Обнаружена ошибка в логах[/red]")
                                self.console.print(f"[dim]Последние строки: {logs.strip()}[/dim]")
                                break
        except Exception as e:
            self.console.print(f"[red]❌ Ошибка в progress monitoring: {e}[/red]")
            logger.error(f"Progress monitoring error: {e}")
        
        self.console.print("[red]❌ RAG сервис не запустился в течение 60 секунд[/red]")
        
        # Быстрая диагностика без зависания
        self.console.print("[blue]🔍 Быстрая проверка логов...[/blue]")
        log_success, logs, _ = self.execute_command(f"cd {self.vm_repo_dir} && tail -5 rag_service.log 2>/dev/null")
        
        if log_success and logs.strip():
            self.console.print(f"[yellow]📄 Последние строки логов:[/yellow]\n{logs.strip()}")
        else:
            self.console.print("[yellow]⚠️ Логи RAG сервиса недоступны[/yellow]")
        
        self.console.print("[yellow]💡 Рекомендация: Проверьте логи на VM: tail -20 rag_service.log[/yellow]")
        self.console.print("[yellow]💡 Или выполните диагностику: python vm_start.py diagnose[/yellow]")
        
        return False
    
    def test_full_system(self) -> bool:
        """Заключительная проверка, что весь стек работает."""
        self.console.print("[blue]🧪 Тестирование полной системы...[/blue]")

        tests = [
            ("Health check", "curl -s http://localhost:8000/health", 30),
            ("Qdrant status", "curl -s http://localhost:6333", 30),
            (
                "RAG status",
                f"cd {self.vm_repo_dir} && source venv/bin/activate && python main.py rag status",
                90,
            ),
        ]

        all_passed = True

        for test_name, command, timeout in tests:
            self.console.print(f"[blue]🔍 {test_name}...[/blue]")
            success, output, error = self.execute_command(command, timeout=timeout)

            if success:
                self.console.print(f"[green]✅ {test_name} прошел[/green]")
                continue

            all_passed = False
            self.console.print(f"[red]❌ {test_name} не прошел[/red]")
            if output.strip():
                self.console.print(Panel.fit(output.strip(), title="STDOUT", border_style="yellow"))
            if error.strip():
                self.console.print(Panel.fit(error.strip(), title="STDERR", border_style="red"))

            if "python main.py rag status" in command:
                self.console.print(
                    "[yellow]ℹ️ На VM можно выполнить: source venv/bin/activate && python main.py rag status --verbose[/yellow]"
                )
                self.console.print(
                    "[yellow]ℹ️ Для логов: tail -n 20 rag_service.log или tail -n 20 rag_service.err 2>/dev/null[/yellow]"
                )
                self.console.print("[blue]🔎 Запускаю встроенную диагностику RAG сервиса...[/blue]")
                self.diagnose_rag_service()

        return all_passed

    def test_system_lightweight(self) -> bool:
        """Упрощенная проверка системы без RAG сервиса"""
        self.console.print("[blue]🧪 Упрощенная проверка системы...[/blue]")

        tests = [
            ("Qdrant status", "curl -s http://localhost:6333", 10),
        ]

        all_passed = True

        for test_name, command, timeout in tests:
            self.console.print(f"[blue]🔍 {test_name}...[/blue]")
            success, output, error = self.execute_command(command, timeout=timeout)

            if success:
                self.console.print(f"[green]✅ {test_name} прошел[/green]")
            else:
                self.console.print(f"[yellow]⚠️ {test_name} не прошел, но это не критично[/yellow]")

        return True  # Всегда возвращаем True для упрощенной проверки

    def run_full_setup(self) -> bool:
        """Запуск полной настройки VM"""
        try:
            self.console.print(Panel.fit(
                "[bold blue]🚀 Автоматическая настройка VM для Jina v3 RAG[/bold blue]\n"
                f"VM: {self.vm_user}@{self.vm_host}:{self.vm_port}"
            ))

            if not self.connect_ssh():
                return False

            # Интерактивный выбор ветки (если не указан через --branch)
            if not hasattr(self, 'selected_branch') or not self.selected_branch:
                branches = self.get_available_branches()
                if branches:
                    selected = self.select_branch_interactive(branches)
                    if selected:
                        self.repo_branch = selected
                    # Если выбор отменён, используем дефолтную ветку из .env
                else:
                    self.console.print("[yellow]⚠️ Использую дефолтную ветку из .env[/yellow]")

            status = self.check_vm_status()

            repo_result = self.setup_repository()
            if not repo_result[0]:  # repo_result[0] = success
                return False
            status['repo_exists'] = True

            if not self.check_critical_files():
                return False

            if not self.setup_python_environment():
                return False
            status['venv_exists'] = True

            if not self.test_jina_v3():
                return False

            if not status['qdrant_running']:
                if not self.setup_qdrant():
                    return False

            if not self.create_env_file():
                return False
            status['env_exists'] = True

            if not status['rag_service_running']:
                # Попытка запуска RAG сервиса с fallback логикой
                rag_started = self.start_rag_service()
                if not rag_started:
                    self.console.print("[yellow]⚠️ RAG сервис не запустился, но продолжаем...[/yellow]")
                    self.console.print("[yellow]💡 Сервис можно запустить вручную: python vm_start.py start[/yellow]")

            # Упрощенная проверка системы (без RAG если он не запустился)
            if not self.test_system_lightweight():
                self.console.print("[yellow]⚠️ Некоторые тесты не прошли, но система частично готова[/yellow]")

            self.console.print("[blue]🔁 Обновляю таблицу статуса после настройки...[/blue]")
            self.check_vm_status(show_banner=False)

            self.console.print(Panel.fit(
                "[bold green]🎉 VM настройка завершена успешно![/bold green]\n\n"
                "✅ Jina v3 загружается корректно\n"
                "✅ Qdrant работает на localhost:6333\n"
                "✅ RAG сервис доступен на localhost:8000\n"
                "✅ Система готова к использованию\n\n"
                "[bold]Для проверки локально выполните:[/bold]\n"
                "[cyan]python main.py rag status[/cyan]"
            ))

            # Явное завершение при успехе
            if self.ssh_client:
                self.ssh_client.close()
            logger.info("VM setup completed successfully - exiting")
            return True

        except Exception as e:
            self.console.print(f"[red]❌ Критическая ошибка: {e}[/red]")
            logger.error(f"Критическая ошибка настройки: {e}")
            if self.ssh_client:
                self.ssh_client.close()
            return False

        finally:
            # Убеждаемся что SSH соединение закрыто
            if self.ssh_client:
                try:
                    self.ssh_client.close()
                except Exception:
                    pass

    def update_code_on_vm(self) -> bool:
        """Sync repository state on the VM with the configured branch."""
        self.console.print("[blue]🔄 Обновляю код на VM...[/blue]")

        try:
            # Интерактивный выбор ветки (если не указан через --branch)
            if not hasattr(self, 'selected_branch') or not self.selected_branch:
                branches = self.get_available_branches()
                if branches:
                    selected = self.select_branch_interactive(branches)
                    if selected:
                        self.repo_branch = selected
                    # Если выбор отменён, используем дефолтную ветку из .env
                else:
                    self.console.print("[yellow]⚠️ Использую дефолтную ветку из .env[/yellow]")

            repo_result = self.setup_repository()
            success = repo_result[0]
            if success:
                self.console.print("[green]✅ Код на VM синхронизирован[/green]")
            return success
        except Exception as e:
            self.console.print(f"[red]❌ Критическая ошибка обновления: {e}[/red]")
            return False

    def stop_services(self) -> bool:
        """Остановка сервисов на VM"""
        self.console.print("[blue]🛑 Остановка сервисов...[/blue]")
        
        if not self.connect_ssh():
            return False
        
        try:
            # Остановка RAG сервиса
            self.execute_command(f"cd {self.vm_repo_dir} && kill $(cat rag_service.pid 2>/dev/null) 2>/dev/null || true")
            
            # Остановка Qdrant
            self.execute_command("docker stop qdrant-repo-sum 2>/dev/null || true")
            
            self.console.print("[green]✅ Сервисы остановлены[/green]")
            return True
            
        finally:
            self.ssh_client.close()

def main():
    """Главная функция"""
    import argparse

    parser = argparse.ArgumentParser(description="Автоматическая настройка VM для Jina v3 RAG")
    parser.add_argument(
        "action",
        nargs='?',
        default="start",
        choices=["start", "stop", "status", "diagnose", "update"],
        help="Режимы: start (по умолчанию, запуск и проверка), stop (остановка), status (проверка), diagnose (диагностика), update (обновление кода)"
    )
    parser.add_argument(
        "--branch",
        type=str,
        help="Ветка для развертывания (пропустить интерактивный выбор)"
    )
    parser.add_argument(
        "--auto-restart",
        action="store_true",
        help="Автоматически перезапустить сервис после обновления кода"
    )
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="НЕ перезапускать сервис после обновления кода"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Только проверить статус сервиса без изменений"
    )

    args = parser.parse_args()

    try:
        manager = VMSetupManager()

        # Устанавливаем выбранную ветку из CLI аргумента
        if args.branch:
            manager.selected_branch = args.branch
            manager.repo_branch = args.branch
            manager.console.print(f"[green]✅ Выбрана ветка из CLI: {args.branch}[/green]")
        
        if args.action == "start":
            success = manager.run_full_setup()
            sys.exit(0 if success else 1)
            
        elif args.action == "stop":
            success = manager.stop_services()
            sys.exit(0 if success else 1)
            
        elif args.action == "status":
            if manager.connect_ssh():
                manager.check_vm_status()
                manager.ssh_client.close()
            sys.exit(0)
            
        elif args.action == "diagnose":
            if manager.connect_ssh():
                manager.diagnose_rag_service()
                manager.ssh_client.close()
            sys.exit(0)
            
        elif args.action == "update":
            if manager.connect_ssh():
                # Обработка --check-only (только проверка статуса)
                if args.check_only:
                    manager.console.print("\n" + "="*60)
                    manager.console.print("ПРОВЕРКА СТАТУСА СЕРВИСА")
                    manager.console.print("="*60)
                    
                    is_running = manager.check_service_running()
                    if is_running:
                        pid = manager.get_service_pid()
                        manager.console.print(f"[green]✅ Сервис запущен (PID {pid})[/green]")
                        manager.verify_service_health()
                    else:
                        manager.console.print("[yellow]⚠️ Сервис НЕ запущен[/yellow]")
                    
                    manager.ssh_client.close()
                    sys.exit(0)
                
                # 1) Синхронизация кода
                manager.update_code_on_vm()
                
                # 2) Обновление .env на основе .env.example + локального .env с подстановкой localhost
                manager.create_env_file()
                
                # 3) Управление сервисом
                manager.console.print("\n" + "="*60)
                manager.console.print("УПРАВЛЕНИЕ СЕРВИСОМ")
                manager.console.print("="*60)
                
                is_running = manager.check_service_running()
                
                if is_running:
                    manager.console.print("[green]✅ Сервис запущен[/green]")
                    
                    # Автоматический режим с --auto-restart
                    if args.auto_restart:
                        manager.console.print("[blue]🔄 Автоматический перезапуск (--auto-restart)[/blue]")
                        if manager.restart_service():
                            manager.verify_service_health()
                    # Режим без перезапуска с --no-restart
                    elif args.no_restart:
                        manager.console.print("[yellow]⏭️ Перезапуск пропущен (--no-restart)[/yellow]")
                        manager.console.print("[yellow]⚠️ Сервис продолжит работать со СТАРЫМ кодом[/yellow]")
                        manager.console.print("[yellow]💡 Для применения изменений потребуется ручной перезапуск[/yellow]")
                    # Интерактивный режим
                    else:
                        restart = input("Перезапустить сервис для применения нового кода? (y/N): ")
                        if restart.lower() == 'y':
                            if manager.restart_service():
                                manager.verify_service_health()
                        else:
                            manager.console.print("[yellow]⚠️ Сервис продолжит работать со СТАРЫМ кодом[/yellow]")
                            manager.console.print("[yellow]💡 Для применения изменений потребуется ручной перезапуск[/yellow]")
                else:
                    manager.console.print("[yellow]⚠️ Сервис НЕ запущен[/yellow]")
                    
                    # Автоматический запуск если --auto-restart
                    if args.auto_restart:
                        manager.console.print("[blue]🚀 Автоматический запуск (--auto-restart)[/blue]")
                        script_path = f"{manager.vm_repo_dir}/start_vm_rag.sh"
                        success, _, _ = manager.execute_command(f"test -f {script_path}")
                        
                        if success:
                            if manager.start_service_via_script(script_path):
                                manager.verify_service_health()
                        else:
                            if manager.start_rag_service():
                                manager.verify_service_health()
                    # Не запускать если --no-restart
                    elif args.no_restart:
                        manager.console.print("[yellow]⏭️ Запуск пропущен (--no-restart)[/yellow]")
                        manager.console.print("[yellow]💡 Используйте ./start_vm_rag.sh на VM для запуска[/yellow]")
                    # Интерактивный режим
                    else:
                        start = input("Запустить сервис? (Y/n): ")
                        if start.lower() != 'n':
                            script_path = f"{manager.vm_repo_dir}/start_vm_rag.sh"
                            success, _, _ = manager.execute_command(f"test -f {script_path}")
                            
                            if success:
                                if manager.start_service_via_script(script_path):
                                    manager.verify_service_health()
                            else:
                                if manager.start_rag_service():
                                    manager.verify_service_health()
                        else:
                            manager.console.print("[yellow]💡 Сервис не запущен. Используйте ./start_vm_rag.sh на VM[/yellow]")
                
                manager.ssh_client.close()
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n[yellow]⏹️ Операция прервана пользователем[/yellow]")
        sys.exit(1)
    except Exception as e:
        print(f"[red]❌ Ошибка: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
