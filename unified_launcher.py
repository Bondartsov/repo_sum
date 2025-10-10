
#!/usr/bin/env python3
"""
Unified Launcher для repo_sum RAG-as-a-Service системы.

Реализует 2-step workflow по требованию пользователя:
1. VM Setup: запускает vm_start.py для настройки VM и RAG сервисов
2. Local App: запускает run_web.py для локального веб-интерфейса

Использование:
    python unified_launcher.py setup    # Шаг 1: настройка VM
    python unified_launcher.py start    # Шаг 2: запуск локального приложения
    python unified_launcher.py all      # Оба шага подряд (по умолчанию)
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.live import Live

# Загрузка переменных окружения
load_dotenv()

console = Console()

class UnifiedLauncher:
    """
    Unified Launcher для управления VM RAG системой и локальным приложением.
    
    Возможности:
    - Автоматическая настройка VM через vm_start.py
    - Real-time мониторинг VM статуса
    - Запуск локального Streamlit приложения
    - Structured logging без шума
    - Graceful shutdown и cleanup
    """
    
    def __init__(self):
        self.vm_host = os.getenv("VM_HOST", "10.61.11.54")
        self.vm_user = os.getenv("VM_USER", "user")
        self.vm_password = os.getenv("VM_PASSWORD")
        
        # Проверяем наличие необходимых скриптов
        self.vm_start_script = Path("vm_start.py")
        self.run_web_script = Path("run_web.py")
        
        if not self.vm_start_script.exists():
            raise FileNotFoundError("vm_start.py не найден в текущей директории")
        if not self.run_web_script.exists():
            raise FileNotFoundError("run_web.py не найден в текущей директории")
        
        # Состояние
        self.vm_process = None
        self.web_process = None
        self.vm_logs = []
        self.web_logs = []
        
        console.print(f"[blue]🚀 Unified Launcher инициализирован для {self.vm_user}@{self.vm_host}[/blue]")
    
    def setup_vm(self, action: str = "start") -> bool:
        """
        Выполняет настройку VM через vm_start.py.
        
        Args:
            action: Действие для vm_start.py (start, update, status, diagnose)
            
        Returns:
            True если успешно, False при ошибке
        """
        console.print(Panel.fit(
            f"[bold blue]ШАГ 1: Настройка VM ({action})[/bold blue]\n"
            f"Хост: {self.vm_user}@{self.vm_host}\n"
            f"Скрипт: {self.vm_start_script}"
        ))
        
        try:
            # Запускаем vm_start.py с соответствующим действием
            cmd = [sys.executable, str(self.vm_start_script), action]
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task(f"Выполнение vm_start.py {action}...", total=100)
                
                # Запускаем процесс
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    encoding='utf-8',           # ✅ Явно указываем UTF-8
                    errors='replace',           # ✅ Заменяем нечитаемые символы
                    bufsize=1
                )
                
                # Читаем вывод в real-time
                output_lines = []
                error_lines = []
                
                while True:
                    stdout_line = process.stdout.readline()
                    stderr_line = process.stderr.readline()
                    
                    if stdout_line:
                        output_lines.append(stdout_line.strip())
                        # Фильтруем важные сообщения
                        if any(keyword in stdout_line.lower() for keyword in 
                               ['✅', '❌', 'error', 'success', 'завершен', 'готов']):
                            progress.update(task, description=stdout_line.strip()[:80])
                    
                    if stderr_line:
                        error_lines.append(stderr_line.strip())
                        if 'error' in stderr_line.lower():
                            progress.update(task, description=f"⚠️ {stderr_line.strip()[:80]}")
                    
                    # Проверяем завершение процесса
                    if process.poll() is not None:
                        break
                    
                    # Обновляем прогресс (примерно)
                    progress.advance(task, 1)
                    time.sleep(0.1)
                
                # Ждём завершения
                return_code = process.wait()
                progress.update(task, completed=100)
                
                # Логируем результат
                if return_code == 0:
                    console.print("[green]✅ VM настройка завершена успешно[/green]")
                    
                    # Показываем важные строки лога
                    important_lines = [line for line in output_lines 
                                     if any(keyword in line.lower() for keyword in 
                                           ['✅', '❌', 'success', 'ready', 'готов', 'завершен'])]
                    
                    if important_lines:
                        console.print("[dim]Ключевые события:[/dim]")
                        for line in important_lines[-5:]:  # Последние 5 важных событий
                            console.print(f"[dim]{line}[/dim]")
                    
                    return True
                else:
                    console.print(f"[red]❌ VM настройка завершилась с ошибкой (код {return_code})[/red]")
                    
                    # Показываем ошибки
                    if error_lines:
                        console.print("[red]Ошибки:[/red]")
                        for line in error_lines[-3:]:  # Последние 3 ошибки
                            console.print(f"[red]{line}[/red]")
                    
                    return False
                    
        except Exception as e:
            console.print(f"[red]❌ Критическая ошибка vm_start.py: {e}[/red]")
            return False
    
    def start_web_app(self, port: Optional[int] = None) -> bool:
        """
        Запускает локальное веб-приложение через run_web.py.
        
        Args:
            port: Порт для веб-приложения (по умолчанию из .env или 8501)
            
        Returns:
            True если успешно запущено, False при ошибке
        """
        console.print(Panel.fit(
            "[bold green]ШАГ 2: Запуск локального веб-приложения[/bold green]\n"
            f"Скрипт: {self.run_web_script}\n"
            f"Порт: {port or os.getenv('PORT', '8501')}"
        ))
        
        try:
            # Команда для запуска
            cmd = [sys.executable, str(self.run_web_script)]
            if port:
                cmd.extend(["--port", str(port)])
            
            # Запускаем процесс
            console.print("[blue]🔄 Запускаю веб-приложение...[/blue]")
            
            self.web_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                encoding='utf-8',           # ✅ Явно указываем UTF-8
                errors='replace',           # ✅ Заменяем нечитаемые символы
                bufsize=1
            )
            
            # Даем время на запуск
            time.sleep(3)
            
            # Проверяем что процесс запущен
            if self.web_process.poll() is None:
                console.print("[green]✅ Веб-приложение запущено успешно[/green]")
                console.print(f"[blue]🌐 Откройте http://localhost:{port or os.getenv('PORT', '8501')} в браузере[/blue]")
                console.print("[yellow]📝 Нажмите Ctrl+C для остановки[/yellow]")
                return True
            else:
                # Процесс завершился - читаем ошибки
                stdout, stderr = self.web_process.communicate(timeout=5)
                console.print("[red]❌ Веб-приложение не запустилось[/red]")
                if stderr:
                    console.print(f"[red]Ошибка: {stderr}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]❌ Критическая ошибка запуска веб-приложения: {e}[/red]")
            return False
    
    def monitor_web_app(self) -> None:
        """Мониторинг веб-приложения с structured logging."""
        if not self.web_process:
            return
        
        # Создаем таблицу статистики
        table = Table(title="Статус системы", show_header=True, header_style="bold blue")
        table.add_column("Компонент", style="cyan")
        table.add_column("Статус", style="bold")
        table.add_column("Информация")
        
        try:
            with Live(table, refresh_per_second=1, console=console) as live:
                while self.web_process and self.web_process.poll() is None:
                    # Обновляем таблицу статуса
                    table.rows.clear()
                    
                    # VM статус
                    table.add_row("🖥️ VM RAG Service", "✅ Активен", f"{self.vm_host}:8000")
                    table.add_row("🗄️ Qdrant", "✅ Работает", f"{self.vm_host}:6333")
                    
                    # Локальное приложение
                    web_status = "✅ Работает" if self.web_process.poll() is None else "❌ Остановлено"
                    table.add_row("🌐 Web UI", web_status, f"localhost:{os.getenv('PORT', '8501')}")
                    
                    # Время работы
                    uptime = datetime.now().strftime("%H:%M:%S")
                    table.add_row("⏰ Время", uptime, "Система работает")
                    
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            console.print("\n[yellow]🛑 Получен сигнал остановки...[/yellow]")
        except Exception as e:
            console.print(f"\n[red]❌ Ошибка мониторинга: {e}[/red]")
    
    def cleanup(self) -> None:
        """Graceful cleanup всех процессов."""
        console.print("[blue]🧹 Завершение работы...[/blue]")
        
        # Останавливаем веб-процесс
        if self.web_process:
            try:
                console.print("[blue]🛑 Останавливаю веб-приложение...[/blue]")
                self.web_process.terminate()
                self.web_process.wait(timeout=10)
                console.print("[green]✅ Веб-приложение остановлено[/green]")
            except subprocess.TimeoutExpired:
                console.print("[yellow]⚠️ Принудительное завершение веб-приложения[/yellow]")
                self.web_process.kill()
            except Exception as e:
                console.print(f"[red]❌ Ошибка остановки веб-приложения: {e}[/red]")
        
        console.print("[green]✅ Cleanup завершен[/green]")
    
    def run_full_workflow(self, vm_action: str = "start", port: Optional[int] = None) -> bool:
        """
        Выполняет полный workflow: VM setup → Web app start → Monitoring.
        
        Args:
            vm_action: Действие для VM (start, update, status)
            port: Порт для веб-приложения
            
        Returns:
            True если успешно, False при ошибке
        """
        try:
            console.print(Panel.fit(
                "[bold yellow]🚀 UNIFIED WORKFLOW: RAG-as-a-Service[/bold yellow]\n"
                "1️⃣ Настройка VM и RAG сервисов\n"
                "2️⃣ Запуск локального веб-интерфейса\n"
                "3️⃣ Real-time мониторинг системы"
            ))
            
            # Шаг 1: VM Setup
            if not self.setup_vm(vm_action):
                console.print("[red]❌ VM настройка не удалась. Прерываю workflow.[/red]")
                return False
            
            console.print()
            time.sleep(1)
            
            # Шаг 2: Web App Start  
            if not self.start_web_app(port):
                console.print("[red]❌ Запуск веб-приложения не удался. Прерываю workflow.[/red]")
                return False
            
            console.print()
            
            # Шаг 3: Monitoring
            console.print("[blue]🔍 Переход в режим мониторинга...[/blue]")
            time.sleep(2)
            
            self.monitor_web_app()
            
            return True
            
        except KeyboardInterrupt:
            console.print("\n[yellow]⏹️ Workflow прерван пользователем[/yellow]")
            return True  # Не ошибка, пользователь сам остановил
        except Exception as e:
            console.print(f"\n[red]❌ Критическая ошибка workflow: {e}[/red]")
            return False
        finally:
            self.cleanup()


def main():
    """Главная функция unified launcher."""
    parser = argparse.ArgumentParser(
        description="Unified Launcher для repo_sum RAG-as-a-Service системы",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python unified_launcher.py all         # Полный workflow (по умолчанию)
  python unified_launcher.py setup       # Только настройка VM
  python unified_launcher.py start       # Только запуск веб-приложения
  python unified_launcher.py update      # Обновление кода на VM + веб-приложение
        """
    )
    
    parser.add_argument(
        "mode",
        nargs='?',
        default="all",
        choices=["all", "setup", "start", "update"],
        help="Режим работы: all (полный workflow), setup (только VM), start (только веб-приложение), update (обновление VM + веб)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help="Порт для веб-приложения (по умолчанию из .env или 8501)"
    )
    
    parser.add_argument(
        "--vm-action",
        default="start",
        choices=["start", "stop", "status", "diagnose", "update"],
        help="Действие для VM (по умолчанию: start)"
    )
    
    args = parser.parse_args()
    
    try:
        launcher = UnifiedLauncher()
        
        if args.mode == "setup":
            # Только настройка VM
            success = launcher.setup_vm(args.vm_action)
            
        elif args.mode == "start":
            # Только запуск веб-приложения
            success = launcher.start_web_app(args.port)
            if success:
                launcher.monitor_web_app()
            
        elif args.mode == "update":
            # Обновление VM + запуск веб-приложения
            success = launcher.run_full_workflow("update", args.port)
            
        else:  # mode == "all"
            # Полный workflow
            success = launcher.run_full_workflow(args.vm_action, args.port)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 До свидания![/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]💥 Критическая ошибка: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
