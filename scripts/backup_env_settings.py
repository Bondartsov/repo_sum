#!/usr/bin/env python3
"""
Backup Environment Settings для Jina v3 Migration
Создает резервную копию текущих настроек перед миграцией
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class EnvironmentBackup:
    """Класс для создания backup настроек окружения"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.backup_dir = self.base_dir / "backups"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_backup_directory(self):
        """Создать директорию для backup"""
        self.backup_dir.mkdir(exist_ok=True)
        current_backup = self.backup_dir / f"migration_backup_{self.timestamp}"
        current_backup.mkdir(exist_ok=True)
        return current_backup
        
    def backup_env_files(self):
        """Создать backup .env файлов"""
        backup_dir = self.create_backup_directory()
        
        # Список файлов для backup
        env_files = [".env", ".env.example", "settings.json"]
        
        backed_up_files = []
        for env_file in env_files:
            source_path = self.base_dir / env_file
            if source_path.exists():
                destination = backup_dir / env_file
                shutil.copy2(source_path, destination)
                backed_up_files.append(env_file)
                logger.info(f"Backed up {env_file} to {destination}")
        
        return backed_up_files, backup_dir
    
    def backup_qdrant_settings(self):
        """Создать backup Qdrant конфигурации"""
        backup_dir = self.create_backup_directory()
        
        # Создать snapshot текущих настроек Qdrant
        current_settings = {
            "timestamp": self.timestamp,
            "migration": "jina_v3",
            "from_model": "BAAI/bge-small-en-v1.5",
            "to_model": "jinaai/jina-embeddings-v3",
            "from_dimension": 384,
            "to_dimension": 1024,
            "from_collection": "code_chunks",
            "to_collection": "repo_sum_v3",
            "environment_variables": self._get_current_env_vars()
        }
        
        settings_file = backup_dir / "migration_settings.json"
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(current_settings, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved migration settings to {settings_file}")
        return settings_file
    
    def _get_current_env_vars(self):
        """Получить текущие переменные окружения, связанные с RAG"""
        rag_vars = {}
        rag_prefixes = ["QDRANT_", "EMB_", "FASTEMBED_", "EMBEDDING_", "RAG_", "TORCH_", "OMP_", "MKL_"]
        
        for key, value in os.environ.items():
            for prefix in rag_prefixes:
                if key.startswith(prefix):
                    rag_vars[key] = value
                    break
        
        return rag_vars
    
    def create_rollback_script(self):
        """Создать скрипт для rollback настроек"""
        backup_dir = self.create_backup_directory()
        
        rollback_script = f'''#!/bin/bash
# Rollback Script для Jina v3 Migration
# Создан: {datetime.now().isoformat()}

echo "🔄 Rollback Jina v3 Migration to BGE-small..."

# Backup текущих настроек
mv .env .env.jina_v3_backup_{self.timestamp} 2>/dev/null || true
mv settings.json settings.json.jina_v3_backup_{self.timestamp} 2>/dev/null || true

# Восстановление старых настроек  
cp backups/migration_backup_{self.timestamp}/.env . 2>/dev/null || echo "⚠️  .env backup not found"
cp backups/migration_backup_{self.timestamp}/settings.json . 2>/dev/null || echo "⚠️  settings.json backup not found"

echo "✅ Environment files restored"

# Qdrant коллекция rollback (требует подтверждения)
echo "⚠️  Manual step: Delete repo_sum_v3 collection and restore code_chunks"
echo "   Use: python scripts/database_migration_jina_v3.py --rollback"

echo "🎉 Rollback completed. Restart application to use BGE-small model."
'''
        
        rollback_file = backup_dir / "rollback_migration.sh"
        with open(rollback_file, 'w', encoding='utf-8') as f:
            f.write(rollback_script)
        
        # Make executable
        os.chmod(rollback_file, 0o755)
        
        logger.info(f"Created rollback script: {rollback_file}")
        return rollback_file

def main():
    """Main function для создания backup"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🔧 Creating backup for Jina v3 migration...")
    
    backup = EnvironmentBackup()
    
    try:
        # Backup environment files
        backed_files, backup_dir = backup.backup_env_files()
        print(f"✅ Backed up files: {', '.join(backed_files)}")
        
        # Backup Qdrant settings
        settings_file = backup.backup_qdrant_settings()
        print(f"✅ Saved migration settings: {settings_file}")
        
        # Create rollback script
        rollback_script = backup.create_rollback_script()
        print(f"✅ Created rollback script: {rollback_script}")
        
        print(f"\n🎉 Backup completed successfully!")
        print(f"📁 Backup location: {backup_dir}")
        print(f"🔄 Rollback command: bash {rollback_script}")
        
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
