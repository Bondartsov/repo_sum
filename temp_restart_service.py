#!/usr/bin/env python3
"""Перезапуск RAG сервиса на VM (без устаревших флагов FORCE_LOCAL_VECTOR_STORE/EMBEDDING_PROVIDER=local)"""
import paramiko
import os
import time
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
    
    print("🛑 Останавливаю RAG сервис...")
    # Остановка через PID файл
    stdin, stdout, stderr = ssh.exec_command(
        'cd ~/repo_sum_rag/repo_sum && '
        'kill $(cat rag_service.pid 2>/dev/null) 2>/dev/null || true'
    )
    stdout.channel.recv_exit_status()
    time.sleep(2)
    
    # Проверка что остановился
    stdin, stdout, stderr = ssh.exec_command(
        'ps aux | grep vm_rag_service | grep -v grep'
    )
    running = stdout.read().decode().strip()
    
    if running:
        print("⚠️ Процесс всё ещё запущен, пробую kill -9...")
        stdin, stdout, stderr = ssh.exec_command(
            'pkill -9 -f vm_rag_service.py'
        )
        stdout.channel.recv_exit_status()
        time.sleep(1)
    
    print("✅ Сервис остановлен")
    
    print("\n🚀 Запускаю RAG сервис (без скрытых флагов, только .env/фабрика)...")
    # Запуск без устаревших флагов, с загрузкой переменных из .env
    # Фабрика [RAGFactory.create_vector_store()](rag/factory.py:146) сама выбирает локальную/удалённую реализацию.
    stdin, stdout, stderr = ssh.exec_command(
        'cd ~/repo_sum_rag/repo_sum && '
        'source venv/bin/activate && '
        'unset FORCE_LOCAL_VECTOR_STORE || true && '
        'set -a; [ -f .env ] && . ./.env; set +a; '
        'nohup python vm_rag_service.py > rag_service.log 2>&1 & '
        'echo $! > rag_service.pid && '
        'echo "STARTED"'
    )
    result = stdout.read().decode()
    
    if 'STARTED' in result:
        print("✅ Команда запуска выполнена")
        
        # Получаем PID
        time.sleep(2)
        stdin, stdout, stderr = ssh.exec_command(
            'cd ~/repo_sum_rag/repo_sum && cat rag_service.pid'
        )
        pid = stdout.read().decode().strip()
        print(f"📌 PID: {pid}")
        
        print("\n⏳ Ожидаю запуска сервиса (30 сек)...")
        for i in range(30):
            time.sleep(1)
            stdin, stdout, stderr = ssh.exec_command(
                'curl -s http://localhost:8000/health >/dev/null 2>&1 && echo "OK"'
            )
            health = stdout.read().decode().strip()
            
            if health == 'OK':
                print("\n✅ Сервис запущен и отвечает на health check!")
                break
            
            if (i + 1) % 10 == 0:
                print(f"   {i+1}/30 сек...")
        else:
            print("\n⚠️ Сервис не ответил на health check за 30 секунд")
            print("💡 Проверяем логи...")
            
            stdin, stdout, stderr = ssh.exec_command(
                'cd ~/repo_sum_rag/repo_sum && tail -10 rag_service.log'
            )
            logs = stdout.read().decode()
            print(f"\n📄 Последние строки логов:\n{logs}")
    else:
        print(f"❌ Ошибка запуска: {result}")
        stderr_text = stderr.read().decode()
        if stderr_text:
            print(f"STDERR: {stderr_text}")
    
finally:
    ssh.close()