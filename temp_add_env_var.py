#!/usr/bin/env python3
"""
Удаление устаревшего флага FORCE_LOCAL_VECTOR_STORE на VM (DEPRECATED).

Этот скрипт предназначен для безопасного удаления временного workaround
FORCE_LOCAL_VECTOR_STORE из окружения VM в рамках Фазы 1.2 (нормализация фабрик).
Фабрика [RAGFactory.create_vector_store()](rag/factory.py:146) автоматически выбирает
локальную/удалённую реализацию по контексту, поэтому ручной флаг больше не нужен.
"""
import paramiko
import os
from dotenv import load_dotenv

load_dotenv()

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

def _exec(ssh, cmd: str) -> str:
    stdin, stdout, stderr = ssh.exec_command(cmd)
    stdout.channel.recv_exit_status()
    out = stdout.read().decode(errors="replace").strip()
    err = stderr.read().decode(errors="replace").strip()
    if err:
        # Не падаем, но логируем stderr для диагностики
        print(f"[diag] {cmd} -> {err}")
    return out

try:
    ssh.connect(
        hostname=os.getenv('VM_HOST', '10.61.11.54'),
        username=os.getenv('VM_USER', 'user'),
        password=os.getenv('VM_PASSWORD')
    )

    print("🔍 Проверяю наличие FORCE_LOCAL_VECTOR_STORE в .env ...")
    out = _exec(ssh, 'cd ~/repo_sum_rag/repo_sum && grep -n "FORCE_LOCAL_VECTOR_STORE" .env || true')
    if out:
        print(f"⚠️ Найдены записи:\n{out}\n→ Удаляю из .env ...")
        _exec(ssh, 'cd ~/repo_sum_rag/repo_sum && sed -i "/FORCE_LOCAL_VECTOR_STORE/d" .env')
        verify = _exec(ssh, 'cd ~/repo_sum_rag/repo_sum && grep -n "FORCE_LOCAL_VECTOR_STORE" .env || true')
        if not verify:
            print("✅ Удалено из .env")
        else:
            print("❌ Не удалось удалить все вхождения из .env, проверьте файл вручную.")
    else:
        print("✅ В .env нет записей FORCE_LOCAL_VECTOR_STORE")

    # Чистим из пользовательских профилей
    print("\n🧹 Чищу ~/.bashrc и ~/.profile от FORCE_LOCAL_VECTOR_STORE ...")
    _exec(ssh, 'sed -i "/FORCE_LOCAL_VECTOR_STORE/d" ~/.bashrc || true')
    _exec(ssh, 'sed -i "/FORCE_LOCAL_VECTOR_STORE/d" ~/.profile || true')
    print("✅ Очистка профилей завершена")

    # Никаких автоподмен EMBEDDING_PROVIDER больше не выполняется этим скриптом.
    # Оставляем текущее значение нетронутым.

    print("\nℹ️ Завершено. Фабрика выбирает реализацию автоматически:")
    print("   - VM контекст → QdrantVectorStore (локально)")
    print("   - CLIENT контекст → RemoteVMVectorStore (HTTP)")
    print("   Проверка: [RAGFactory.get_factory_info()](rag/factory.py:235)")

except Exception as e:
    print(f"❌ Ошибка при выполнении операции: {e}")
finally:
    ssh.close()