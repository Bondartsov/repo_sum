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
import time
import logging
from pathlib import Path
from typing import Tuple, Optional
import paramiko
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vm_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VMSetupManager:
    """Управление автоматической настройкой VM"""
    
    def __init__(self):
        # Загрузка конфигурации
        load_dotenv()
        self.console = Console()
        
        # VM параметры из .env
        self.vm_host = os.getenv("VM_HOST", "10.61.11.54")
        self.vm_user = os.getenv("VM_USER", "user")
        self.vm_password = os.getenv("VM_PASSWORD")
        self.vm_port = int(os.getenv("VM_PORT", "22"))
        
        # Пути на VM
        self.vm_work_dir = "~/repo_sum_rag"
        self.vm_repo_dir = f"{self.vm_work_dir}/repo_sum"
        
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
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
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
    
    def execute_command(self, command: str, timeout: int = 300) -> Tuple[bool, str, str]:
        """Выполнение команды на VM"""
        try:
            logger.debug(f"Выполнение: {command}")
            stdin, stdout, stderr = self.ssh_client.exec_command(command, timeout=timeout)
            
            # Ожидание завершения
            exit_status = stdout.channel.recv_exit_status()
            
            stdout_text = stdout.read().decode('utf-8')
            stderr_text = stderr.read().decode('utf-8')
            
            success = exit_status == 0
            
            if not success:
                logger.warning(f"Команда завершилась с кодом {exit_status}")
                logger.warning(f"STDERR: {stderr_text}")
            
            return success, stdout_text, stderr_text
            
        except Exception as e:
            logger.error(f"Ошибка выполнения команды '{command}': {e}")
            return False, "", str(e)
    
    def check_vm_status(self) -> dict:
        """Проверка текущего состояния VM"""
        self.console.print("[blue]🔍 Проверка состояния VM...[/blue]")
        
        status = {
            'python_version': None,
            'memory_gb': None,
            'repo_exists': False,
            'venv_exists': False,
            'qdrant_running': False,
            'rag_service_running': False
        }
        
        # Проверка Python
        success, output, _ = self.execute_command("python3 --version")
        if success:
            status['python_version'] = output.strip()
        
        # Проверка памяти
        success, output, _ = self.execute_command("free -h | grep 'Mem:' | awk '{print $2}'")
        if success:
            status['memory_gb'] = output.strip()
        
        # Проверка репозитория
        success, _, _ = self.execute_command(f"test -d {self.vm_repo_dir}")
        status['repo_exists'] = success
        
        # Проверка venv
        success, _, _ = self.execute_command(f"test -d {self.vm_repo_dir}/venv")
        status['venv_exists'] = success
        
        # Проверка Qdrant
        success, _, _ = self.execute_command("curl -s http://localhost:6333 >/dev/null 2>&1")
        status['qdrant_running'] = success
        
        # Проверка RAG сервиса
        success, _, _ = self.execute_command("curl -s http://localhost:8000/health >/dev/null 2>&1")
        status['rag_service_running'] = success
        
        # Показываем статус
        self._show_vm_status(status)
        
        return status
    
    def _show_vm_status(self, status: dict):
        """Отображение статуса VM"""
        table = Table(title="Статус VM")
        table.add_column("Компонент", style="cyan")
        table.add_column("Статус", style="bold")
        table.add_column("Детали")
        
        # Python версия
        python_status = "✅ Готов" if status['python_version'] else "❌ Не найден"
        table.add_row("Python", python_status, status['python_version'] or "")
        
        # Память
        memory_status = "✅ Достаточно" if status['memory_gb'] else "❌ Неизвестно"
        table.add_row("Память", memory_status, status['memory_gb'] or "")
        
        # Репозиторий
        repo_status = "✅ Клонирован" if status['repo_exists'] else "❌ Отсутствует"
        table.add_row("Репозиторий", repo_status, self.vm_repo_dir)
        
        # Venv
        venv_status = "✅ Создан" if status['venv_exists'] else "❌ Отсутствует"
        table.add_row("Virtual Env", venv_status, "")
        
        # Qdrant
        qdrant_status = "✅ Запущен" if status['qdrant_running'] else "❌ Остановлен"
        table.add_row("Qdrant", qdrant_status, "localhost:6333")
        
        # RAG сервис
        rag_status = "✅ Запущен" if status['rag_service_running'] else "❌ Остановлен"
        table.add_row("RAG Service", rag_status, "localhost:8000")
        
        self.console.print(table)
    
    def setup_repository(self) -> bool:
        """Настройка репозитория на VM"""
        self.console.print("[blue]📁 Настройка репозитория...[/blue]")
        
        # Создание рабочей папки
        success, _, _ = self.execute_command(f"mkdir -p {self.vm_work_dir}")
        if not success:
            self.console.print("[red]❌ Не удалось создать рабочую папку[/red]")
            return False
        
        # Переход в рабочую папку и клонирование
        commands = [
            f"cd {self.vm_work_dir}",
            "git clone https://github.com/Bondartsov/repo_sum.git 2>/dev/null || echo 'Репозиторий уже существует'",
            f"cd {self.vm_repo_dir}",
            "git fetch --all 2>/dev/null || true",
            "git checkout jina-embeddings-v3 2>/dev/null || echo 'Используем текущую ветку'",
            "git branch --show-current"
        ]
        
        combined_command = " && ".join(commands)
        success, output, error = self.execute_command(combined_command)
        
        if success:
            self.console.print("[green]✅ Репозиторий настроен[/green]")
            logger.info(f"Репозиторий: {output.strip()}")
        else:
            self.console.print(f"[yellow]⚠️ Частичная настройка репозитория: {error}[/yellow]")
        
        return True
    
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
        """Настройка Python окружения"""
        self.console.print("[blue]🐍 Настройка Python окружения...[/blue]")
        
        commands = [
            f"cd {self.vm_repo_dir}",
            "python3 -m venv venv",
            "source venv/bin/activate",
            "pip install --upgrade pip setuptools wheel",
            "pip install -r requirements.txt",
            "pip install sentence-transformers>=3.0 transformers>=4.35.0"
        ]
        
        combined_command = " && ".join(commands)
        success, output, error = self.execute_command(combined_command, timeout=600)  # 10 минут
        
        if success:
            self.console.print("[green]✅ Python окружение готово[/green]")
            return True
        else:
            self.console.print(f"[red]❌ Ошибка настройки Python: {error}[/red]")
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
        """Создание .env файла на VM"""
        self.console.print("[blue]⚙️ Создание .env файла...[/blue]")
        
        # Получаем OpenAI ключ из локального .env
        openai_key = os.getenv("OPENAI_API_KEY", "your_openai_key_here")
        
        env_content = f"""QDRANT_HOST=localhost
QDRANT_PORT=6333
OPENAI_API_KEY={openai_key}
"""
        
        # Создаем .env файл
        command = f"cd {self.vm_repo_dir} && cat > .env << 'EOF'\n{env_content}EOF"
        success, _, error = self.execute_command(command)
        
        if success:
            self.console.print("[green]✅ .env файл создан[/green]")
            return True
        else:
            self.console.print(f"[red]❌ Ошибка создания .env: {error}[/red]")
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
        
        success, output, error = self.execute_command(start_command)
        
        if not success:
            self.console.print(f"[red]❌ Ошибка запуска сервиса: {error}[/red]")
            # Диагностика при ошибке запуска
            self.diagnose_rag_service()
            return False
        
        # Ждем запуска сервиса с детальным прогресс-баром
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Запуск RAG сервиса...", total=60)
            
            for i in range(60):  # 60 секунд максимум
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
        
        self.console.print("[red]❌ RAG сервис не запустился в течение 60 секунд[/red]")
        
        # Полная диагностика при неудаче
        self.console.print("[blue]🔍 Запуск полной диагностики...[/blue]")
        diagnostics = self.diagnose_rag_service()
        
        # Вывод рекомендаций на основе диагностики
        if not diagnostics['python_imports_ok']:
            self.console.print("[yellow]💡 Рекомендация: Проблема с импортами Python модулей[/yellow]")
        elif diagnostics['service_logs']:
            self.console.print("[yellow]💡 Рекомендация: Проверьте логи выше для деталей ошибки[/yellow]")
        else:
            self.console.print("[yellow]💡 Рекомендация: Попробуйте ручной запуск на VM для диагностики[/yellow]")
        
        return False
    
    def test_full_system(self) -> bool:
        """Полное тестирование системы"""
        self.console.print("[blue]🧪 Тестирование полной системы...[/blue]")
        
        tests = [
            ("Health check", "curl -s http://localhost:8000/health"),
            ("Qdrant status", "curl -s http://localhost:6333"),
            ("RAG status", f"cd {self.vm_repo_dir} && source venv/bin/activate && python main.py rag status")
        ]
        
        all_passed = True
        
        for test_name, command in tests:
            self.console.print(f"[blue]🔍 {test_name}...[/blue]")
            success, output, error = self.execute_command(command, timeout=30)
            
            if success:
                self.console.print(f"[green]✅ {test_name} прошел[/green]")
            else:
                self.console.print(f"[red]❌ {test_name} не прошел: {error}[/red]")
                all_passed = False
        
        return all_passed
    
    def run_full_setup(self) -> bool:
        """Запуск полной настройки VM"""
        try:
            self.console.print(Panel.fit(
                "[bold blue]🚀 Автоматическая настройка VM для Jina v3 RAG[/bold blue]\n"
                f"VM: {self.vm_user}@{self.vm_host}:{self.vm_port}"
            ))
            
            # Подключение
            if not self.connect_ssh():
                return False
            
            # Проверка статуса
            status = self.check_vm_status()
            
            # Настройка репозитория
            if not status['repo_exists']:
                if not self.setup_repository():
                    return False
            else:
                self.console.print("[green]✅ Репозиторий уже клонирован[/green]")
            
            # Проверка критических файлов
            if not self.check_critical_files():
                return False
            
            # Настройка Python окружения
            if not status['venv_exists']:
                if not self.setup_python_environment():
                    return False
            else:
                self.console.print("[green]✅ Python окружение уже настроено[/green]")
            
            # Тест Jina v3
            if not self.test_jina_v3():
                return False
            
            # Настройка Qdrant
            if not status['qdrant_running']:
                if not self.setup_qdrant():
                    return False
            
            # Создание .env
            if not self.create_env_file():
                return False
            
            # Запуск RAG сервиса
            if not status['rag_service_running']:
                if not self.start_rag_service():
                    return False
            
            # Финальное тестирование
            if not self.test_full_system():
                return False
            
            # Успех!
            self.console.print(Panel.fit(
                "[bold green]🎉 VM настройка завершена успешно![/bold green]\n\n"
                "✅ Jina v3 загружается корректно\n"
                "✅ Qdrant работает на localhost:6333\n"
                "✅ RAG сервис доступен на localhost:8000\n"
                "✅ Система готова к использованию\n\n"
                "[bold]Для проверки локально выполните:[/bold]\n"
                "[cyan]python main.py rag status[/cyan]"
            ))
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]❌ Критическая ошибка: {e}[/red]")
            logger.error(f"Критическая ошибка настройки: {e}")
            return False
        
        finally:
            if self.ssh_client:
                self.ssh_client.close()
    
    def update_code_on_vm(self) -> bool:
        """Обновление кода на VM из репозитория"""
        self.console.print("[blue]📥 Обновление кода на VM...[/blue]")
        
        try:
            # Переход в папку и обновление кода
            commands = [
                f"cd {self.vm_repo_dir}",
                "git fetch --all",
                "git reset --hard origin/jina-embeddings-v3",
                "git branch --show-current"
            ]
            
            combined_command = " && ".join(commands)
            success, output, error = self.execute_command(combined_command)
            
            if success:
                self.console.print("[green]✅ Код на VM обновлен[/green]")
                logger.info(f"Обновление: {output.strip()}")
                return True
            else:
                self.console.print(f"[red]❌ Ошибка обновления: {error}[/red]")
                return False
                
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
    parser.add_argument("action", choices=["start", "stop", "status", "diagnose", "update"], 
                       help="Действие: start (настройка и запуск), stop (остановка), status (проверка), diagnose (диагностика), update (обновление кода)")
    
    args = parser.parse_args()
    
    try:
        manager = VMSetupManager()
        
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
                manager.update_code_on_vm()
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
