#!/usr/bin/env python3
"""Добавление FORCE_LOCAL_VECTOR_STORE=true в .env на VM"""
import paramiko
import os
from dotenv import load_dotenv

load_dotenv()

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(
        hostname=os.getenv('VM_HOST', '10.61.11.54'),
        username=os.getenv('VM_USER', 'user'),
        password=os.getenv('VM_PASSWORD')
    )
    
    # Проверяем существует ли переменная
    stdin, stdout, stderr = ssh.exec_command(
        'cd ~/repo_sum_rag/repo_sum && grep "FORCE_LOCAL_VECTOR_STORE" .env'
    )
    existing = stdout.read().decode().strip()
    
    if existing:
        print(f"✅ Переменная уже существует:\n{existing}")
    else:
        # Добавляем переменную
        stdin, stdout, stderr = ssh.exec_command(
            'cd ~/repo_sum_rag/repo_sum && echo "FORCE_LOCAL_VECTOR_STORE=true" >> .env'
        )
        stdout.channel.recv_exit_status()
        
        # Проверяем добавление
        stdin, stdout, stderr = ssh.exec_command(
            'cd ~/repo_sum_rag/repo_sum && grep "FORCE_LOCAL_VECTOR_STORE" .env'
        )
        result = stdout.read().decode().strip()
        print(f"✅ Переменная добавлена:\n{result}")
    
    # Также добавим EMBEDDING_PROVIDER=local если нужно
    stdin, stdout, stderr = ssh.exec_command(
        'cd ~/repo_sum_rag/repo_sum && grep "EMBEDDING_PROVIDER" .env'
    )
    emb_prov = stdout.read().decode().strip()
    
    if 'EMBEDDING_PROVIDER=local' not in emb_prov:
        print("\n📝 Обновляю EMBEDDING_PROVIDER на local...")
        stdin, stdout, stderr = ssh.exec_command(
            'cd ~/repo_sum_rag/repo_sum && sed -i "s/EMBEDDING_PROVIDER=.*/EMBEDDING_PROVIDER=local/" .env'
        )
        stdout.channel.recv_exit_status()
        print("✅ EMBEDDING_PROVIDER обновлён")
    
finally:
    ssh.close()