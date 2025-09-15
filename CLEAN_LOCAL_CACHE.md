# 🧹 Очистка локальных кешей моделей

**Выполните команды по порядку в терминале:**

## 1. Прервите текущую команду
```
Ctrl+C
```

## 2. Очистите HuggingFace кеш (PowerShell)
```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface" -ErrorAction SilentlyContinue
```

## 3. Очистите Torch кеш (PowerShell)  
```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\torch" -ErrorAction SilentlyContinue
```

## 4. Очистите FastEmbed кеш (PowerShell)
```powershell  
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\fastembed" -ErrorAction SilentlyContinue
```

## 5. Очистите pip кеш
```powershell
pip cache purge
```

## 6. Проверьте статус после очистки
```powershell
python main.py rag status
```

---

**После выполнения всех команд сообщите результат!**
