#!/usr/bin/env python3
"""
🚀 ЭТАП 1: Подготовка VM окружения для Jina v3 RAG-as-a-Service

Скрипт для автоматизации начальной настройки t-ubuntu-redis (10.61.11.54)
Выполняет подзадачи 1.1 - 1.6 из JINA_V3_VM_MIGRATION_PLAN.md
"""

import os
import sys
import subprocess
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
import paramiko

# Загружаем переменные из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VMSetupManager:
    """Менеджер настройки VM для развертывания Jina v3"""
    
    def __init__(self, vm_host: str = None, vm_user: str = None):
        # Берем параметры из .env файла если не переданы явно
        self.vm_host = vm_host or os.getenv("VM_HOST", "10.61.11.54")
        self.vm_user = vm_user or os.getenv("VM_USER", "user")
        self.vm_password = os.getenv("VM_PASSWORD")
        self.vm_connection = f"{self.vm_user}@{self.vm_host}"
        self.project_name = "repo_sum_rag"
        self.vm_project_path = f"/home/{self.vm_user}/{self.project_name}"
        
        # Статус выполнения подзадач ЭТАПА 1
        self.tasks_status = {
            "1.1": {"name": "SSH подключение", "completed": False},
            "1.2": {"name": "Проверка ресурсов", "completed": False}, 
            "1.3": {"name": "Установка Python 3.9+", "completed": False},
            "1.4": {"name": "Установка pip и virtualenv", "completed": False},
            "1.5": {"name": "Создание venv", "completed": False},
            "1.6": {"name": "Клонирование репозитория", "completed": False}
        }
    
    def print_banner(self):
        """Печатает баннер скрипта"""
        print("🚀" + "="*70)
        print("   ЭТАП 1: Подготовка VM окружения для Jina v3 RAG-as-a-Service")
        print("   VM: t-ubuntu-redis (10.61.11.54) - 32GB RAM, Xeon Gold 6248R")
        print("="*72)
        print()
    
    def ssh_execute(self, command: str) -> Tuple[bool, str, str]:
        """
        Выполняет команду через SSH на VM с автоматической аутентификацией через paramiko
        
        Args:
            command: Команда для выполнения
            
        Returns:
            Tuple[success, stdout, stderr]
        """
        if not self.vm_password:
            logger.error("❌ VM_PASSWORD не установлен в .env файле")
            return False, "", "Missing VM_PASSWORD in .env"
        
        try:
            # Создаем SSH клиент
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Подключаемся с автоматической аутентификацией
            ssh.connect(
                hostname=self.vm_host,
                username=self.vm_user, 
                password=self.vm_password,
                timeout=10,
                banner_timeout=30
            )
            
            # Выполняем команду
            stdin, stdout, stderr = ssh.exec_command(command)
            
            # Читаем результат
            output = stdout.read().decode('utf-8').strip()
            error = stderr.read().decode('utf-8').strip()
            
            # Получаем код возврата
            exit_code = stdout.channel.recv_exit_status()
            
            # Закрываем соединение
            ssh.close()
            
            return exit_code == 0, output, error
            
        except Exception as e:
            logger.error(f"SSH execute error: {e}")
            return False, "", str(e)
    
    def test_ssh_connection(self) -> bool:
        """Тестирует SSH подключение к VM (задача 1.1)"""
        logger.info("🔌 Тестируем SSH подключение к VM...")
        
        success, stdout, stderr = self.ssh_execute("echo 'SSH connection successful'")
        
        if success:
            logger.info(f"✅ SSH подключение успешно: {stdout.strip()}")
            self.tasks_status["1.1"]["completed"] = True
            return True
        else:
            logger.error(f"❌ Ошибка SSH подключения: {stderr}")
            print("\n🔧 Инструкция по настройке SSH:")
            print(f"   1. ssh-copy-id {self.vm_connection}")
            print(f"   2. ssh {self.vm_connection}")
            print("   3. Проверьте права доступа и firewall настройки")
            return False
    
    def check_vm_resources(self) -> Dict[str, str]:
        """Проверяет доступные ресурсы VM (задача 1.2)"""
        logger.info("📊 Проверяем ресурсы VM...")
        
        checks = {
            "RAM": "free -h | grep Mem",
            "Disk": "df -h /",
            "CPU": "lscpu | grep 'Model name'",
            "OS": "cat /etc/os-release | grep PRETTY_NAME",
            "Python": "python3 --version 2>/dev/null || echo 'Python not found'"
        }
        
        results = {}
        all_success = True
        
        for check_name, command in checks.items():
            success, stdout, stderr = self.ssh_execute(command)
            if success:
                results[check_name] = stdout.strip()
            else:
                results[check_name] = f"Error: {stderr}"
                all_success = False
        
        # Выводим результаты
        print("\n📋 Ресурсы VM:")
        for check_name, result in results.items():
            print(f"   {check_name}: {result}")
        
        if all_success:
            self.tasks_status["1.2"]["completed"] = True
        
        return results
    
    def install_python(self) -> bool:
        """Устанавливает Python 3.9+ если необходимо (задача 1.3)"""
        logger.info("🐍 Проверяем и устанавливаем Python 3.9+...")
        
        # Проверка текущей версии Python
        success, stdout, stderr = self.ssh_execute("python3 --version")
        if success:
            version_str = stdout.strip()
            print(f"   Найден Python: {version_str}")
            
            # Проверяем версию (нужна 3.9+)
            try:
                version = version_str.split()[1]  # "Python 3.x.x" -> "3.x.x"
                major, minor = map(int, version.split('.')[:2])
                if major >= 3 and minor >= 9:
                    logger.info("✅ Python 3.9+ уже установлен")
                    self.tasks_status["1.3"]["completed"] = True
                    return True
            except:
                pass
        
        # Установка Python 3.9+
        logger.info("📦 Устанавливаем Python 3.9+...")
        install_commands = [
            "sudo apt update -y",
            "sudo apt install -y python3.9 python3.9-dev python3.9-venv",
            "sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1"
        ]
        
        for cmd in install_commands:
            success, stdout, stderr = self.ssh_execute(cmd)
            if not success:
                logger.error(f"❌ Ошибка установки Python: {stderr}")
                return False
        
        # Проверяем установку
        success, stdout, stderr = self.ssh_execute("python3 --version")
        if success:
            logger.info(f"✅ Python успешно установлен: {stdout.strip()}")
            self.tasks_status["1.3"]["completed"] = True
            return True
        else:
            logger.error("❌ Не удалось установить Python")
            return False
    
    def install_pip_virtualenv(self) -> bool:
        """Устанавливает pip и virtualenv (задача 1.4)"""
        logger.info("📦 Устанавливаем pip и virtualenv...")
        
        install_commands = [
            "sudo apt update",
            "sudo apt install -y python3-pip python3.10-venv python3.10-distutils git",
            "python3 -m pip install --user virtualenv"
        ]
        
        for cmd in install_commands:
            success, stdout, stderr = self.ssh_execute(cmd)
            if not success:
                logger.warning(f"⚠️ Команда может уже быть выполнена: {cmd}")
        
        # Проверяем установку
        success, stdout, stderr = self.ssh_execute("python3 -m pip --version && python3 -m venv --help")
        if success:
            logger.info("✅ pip и virtualenv успешно установлены")
            self.tasks_status["1.4"]["completed"] = True
            return True
        else:
            logger.error(f"❌ Ошибка установки pip/virtualenv: {stderr}")
            return False
    
    def create_virtual_env(self) -> bool:
        """Создает виртуальное окружение (задача 1.5)"""
        logger.info("🏗️ Создаем виртуальное окружение...")
        
        venv_path = f"{self.vm_project_path}/venv"
        
        commands = [
            f"mkdir -p {self.vm_project_path}",
            f"python3 -m venv {self.vm_project_path}/venv",
            f"source {self.vm_project_path}/venv/bin/activate && pip install --upgrade pip setuptools wheel"
        ]
        
        for cmd in commands:
            success, stdout, stderr = self.ssh_execute(cmd)
            if not success:
                logger.error(f"❌ Ошибка создания venv: {stderr}")
                logger.error(f"   Команда: {cmd}")
                logger.error(f"   stdout: {stdout}")
                return False
        
        # Проверяем создание
        success, stdout, stderr = self.ssh_execute(f"test -d {venv_path} && echo 'venv exists'")
        if success and "venv exists" in stdout:
            logger.info("✅ Виртуальное окружение успешно создано")
            self.tasks_status["1.5"]["completed"] = True
            return True
        else:
            logger.error("❌ Не удалось создать виртуальное окружение")
            return False
    
    def clone_repository(self, repo_url: str = "https://github.com/Bondartsov/repo_sum.git") -> bool:
        """Клонирует репозиторий на VM (задача 1.6)"""
        logger.info("📂 Клонируем репозиторий repo_sum...")
        
        # Удаляем существующий проект если есть и клонируем заново
        commands = [
            f"rm -rf {self.vm_project_path}/src",
            f"cd {self.vm_project_path} && git clone {repo_url} src"
        ]
        
        for cmd in commands:
            success, stdout, stderr = self.ssh_execute(cmd)
            if not success and "already exists" not in stderr:
                logger.error(f"❌ Ошибка клонирования: {stderr}")
                return False
        
        # Проверяем клонирование
        success, stdout, stderr = self.ssh_execute(f"test -f {self.vm_project_path}/src/main.py && echo 'repo cloned'")
        if success and "repo cloned" in stdout:
            logger.info("✅ Репозиторий успешно клонирован")
            self.tasks_status["1.6"]["completed"] = True
            return True
        else:
            logger.error("❌ Не удалось клонировать репозиторий")
            return False
    
    def print_status_report(self):
        """Печатает отчет о выполнении задач"""
        print("\n" + "="*50)
        print("📋 ОТЧЕТ О ВЫПОЛНЕНИИ ЭТАПА 1:")
        print("="*50)
        
        completed_count = 0
        for task_id, task_info in self.tasks_status.items():
            status = "✅" if task_info["completed"] else "❌"
            print(f"   {task_id}: {status} {task_info['name']}")
            if task_info["completed"]:
                completed_count += 1
        
        print(f"\nПрогресс: {completed_count}/{len(self.tasks_status)} задач завершено")
        
        if completed_count == len(self.tasks_status):
            print("🎉 ЭТАП 1 ПОЛНОСТЬЮ ЗАВЕРШЁН!")
            print("➡️ Готов к переходу на ЭТАП 2: Установка зависимостей и тестирование Jina v3")
        else:
            print("⚠️ Есть незавершенные задачи. Проверьте ошибки выше.")
    
    def run_phase_1(self) -> bool:
        """Выполняет все задачи ЭТАПА 1"""
        self.print_banner()
        
        # Выполняем задачи по порядку
        tasks = [
            self.test_ssh_connection,
            self.check_vm_resources,
            self.install_python,
            self.install_pip_virtualenv,
            self.create_virtual_env,
            self.clone_repository
        ]
        
        success = True
        for task in tasks:
            if not task():
                success = False
                break
        
        self.print_status_report()
        return success

def main():
    """Главная функция"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("🚀 Скрипт подготовки VM для Jina v3 RAG-as-a-Service")
        print("\nИспользование:")
        print("   python3 scripts/vm_setup_phase1.py")
        print("\nПеред запуском убедитесь что:")
        print("   1. У вас есть SSH доступ к t-ubuntu-redis (10.61.11.54)")
        print("   2. SSH ключи настроены (ssh-copy-id user@10.61.11.54)")
        print("   3. У вас есть sudo права на VM")
        return
    
    # Создаем и запускаем менеджер
    manager = VMSetupManager()
    success = manager.run_phase_1()
    
    if success:
        print(f"\n🎯 Следующие шаги:")
        print(f"   1. SSH к VM: ssh user@10.61.11.54")
        print(f"   2. Активировать venv: cd {manager.vm_project_path} && source venv/bin/activate")
        print(f"   3. Запустить ЭТАП 2: python3 scripts/vm_setup_phase2.py")
        sys.exit(0)
    else:
        print(f"\n❌ ЭТАП 1 завершился с ошибками.")
        print(f"   Проверьте логи выше и устраните проблемы.")
        sys.exit(1)

if __name__ == "__main__":
    main()
