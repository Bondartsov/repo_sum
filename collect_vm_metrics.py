#!/usr/bin/env python3
"""Сбор метрик с VM для отчёта OOM тестирования"""
import paramiko
import os
from dotenv import load_dotenv

load_dotenv()

def collect_metrics():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect('10.61.11.54', username='user', password=os.getenv('VM_PASSWORD'))
        
        # Память и swap
        print("=" * 60)
        print("ПАМЯТЬ И SWAP")
        print("=" * 60)
        stdin, stdout, stderr = ssh.exec_command('free -h')
        print(stdout.read().decode())
        
        stdin, stdout, stderr = ssh.exec_command('swapon --show')
        swap_output = stdout.read().decode()
        print("\nSwap:")
        print(swap_output if swap_output.strip() else "Swap не используется")
        
        # Процесс vm_rag_service
        print("\n" + "=" * 60)
        print("ПРОЦЕСС VM_RAG_SERVICE")
        print("=" * 60)
        stdin, stdout, stderr = ssh.exec_command('ps aux | grep vm_rag_service | grep -v grep')
        print(stdout.read().decode())
        
        # OOM события
        print("\n" + "=" * 60)
        print("OOM СОБЫТИЯ (последние 20)")
        print("=" * 60)
        stdin, stdout, stderr = ssh.exec_command('sudo dmesg -T | grep -i "killed\\|oom" | tail -20')
        oom_output = stdout.read().decode()
        if oom_output.strip():
            print(oom_output)
        else:
            print("✅ OOM события отсутствуют")
        
        # Информация о коллекции
        print("\n" + "=" * 60)
        print("ИНФОРМАЦИЯ О КОЛЛЕКЦИИ")
        print("=" * 60)
        stdin, stdout, stderr = ssh.exec_command('curl -s localhost:8000/collection_info')
        print(stdout.read().decode())
        
        # Логи RAG сервиса (последние 30 строк с метриками)
        print("\n" + "=" * 60)
        print("ЛОГИ RAG СЕРВИСА (метрики чанков)")
        print("=" * 60)
        stdin, stdout, stderr = ssh.exec_command('tail -100 ~/repo_sum_rag/repo_sum/rag_service.log | grep -E "Метрики чанков|Память:" || echo "Метрики не найдены"')
        print(stdout.read().decode())
        
    finally:
        ssh.close()

if __name__ == '__main__':
    collect_metrics()