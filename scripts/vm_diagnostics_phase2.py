#!/usr/bin/env python3
"""
Диагностика VM для Фазы 2 - проверка swap, памяти и дисков
"""
import paramiko
from pathlib import Path

# Загрузка учетных данных из .env
env_path = Path(__file__).parent.parent / ".env"
vm_password = None

if env_path.exists():
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('VM_PASSWORD='):
                vm_password = line.split('=', 1)[1].strip()
                break

if not vm_password:
    print("ERROR: VM_PASSWORD не найден в .env файле")
    exit(1)

# Подключение к VM
VM_HOST = "10.61.11.54"
VM_USER = "user"

print(f"Подключение к {VM_HOST}...")
try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VM_HOST, username=VM_USER, password=vm_password, timeout=10)
    
    # Выполнение диагностических команд
    commands = {
        "ПАМЯТЬ": "free -h",
        "SWAP": "swapon --show",
        "SWAPS": "cat /proc/swaps",
        "ДИСК": "df -h /",
        "SWAPPINESS": "cat /proc/sys/vm/swappiness"
    }
    
    print("\n" + "="*60)
    print("ДИАГНОСТИКА VM - ФАЗА 2")
    print("="*60)
    
    for section, cmd in commands.items():
        print(f"\n[{section}]")
        print("-" * 40)
        stdin, stdout, stderr = ssh.exec_command(cmd)
        output = stdout.read().decode('utf-8').strip()
        error = stderr.read().decode('utf-8').strip()
        
        if output:
            print(output)
        if error and "No such file or directory" not in error:
            print(f"ERROR: {error}")
        if not output and not error:
            print("(нет данных)")
    
    ssh.close()
    print("\n" + "="*60)
    print("Диагностика завершена")
    print("="*60)
    
except Exception as e:
    print(f"ERROR: Не удалось подключиться к VM: {e}")
    exit(1)