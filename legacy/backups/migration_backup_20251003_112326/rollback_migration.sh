#!/bin/bash
# Rollback Script для Jina v3 Migration
# Создан: 2025-10-03T11:23:26.581497

echo "🔄 Rollback Jina v3 Migration to BGE-small..."

# Backup текущих настроек
mv .env .env.jina_v3_backup_20251003_112326 2>/dev/null || true
mv settings.json settings.json.jina_v3_backup_20251003_112326 2>/dev/null || true

# Восстановление старых настроек  
cp backups/migration_backup_20251003_112326/.env . 2>/dev/null || echo "⚠️  .env backup not found"
cp backups/migration_backup_20251003_112326/settings.json . 2>/dev/null || echo "⚠️  settings.json backup not found"

echo "✅ Environment files restored"

# Qdrant коллекция rollback (требует подтверждения)
echo "⚠️  Manual step: Delete repo_sum_v3 collection and restore code_chunks"
echo "   Use: python scripts/database_migration_jina_v3.py --rollback"

echo "🎉 Rollback completed. Restart application to use BGE-small model."
