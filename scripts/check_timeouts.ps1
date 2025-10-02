# PowerShell скрипт для диагностики timeout параметров на Windows

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "🔍 ДИАГНОСТИКА TIMEOUT И RETRY ПАРАМЕТРОВ (WINDOWS)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$projectRoot = "D:\Scripts_Python\repo_sum"
Set-Location $projectRoot

# 1. Environment Variables
Write-Host "`n================================================================================" -ForegroundColor Yellow
Write-Host "1. Environment Variables" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow

$envVars = @(
    "RAG_TIMEOUT_SECONDS",
    "RAG_MAX_RETRIES",
    "RAG_RETRY_DELAY",
    "OPENAI_TIMEOUT",
    "OPENAI_RETRY_ATTEMPTS",
    "OPENAI_RETRY_DELAY"
)

$foundVars = $false
foreach ($var in $envVars) {
    $value = [Environment]::GetEnvironmentVariable($var)
    if ($value) {
        Write-Host "  $var = $value" -ForegroundColor Green
        $foundVars = $true
    }
}

if (-not $foundVars) {
    Write-Host "  ❌ Нет установленных environment variables для timeout/retry" -ForegroundColor Red
}

# 2. Config.py
Write-Host "`n================================================================================" -ForegroundColor Yellow
Write-Host "2. config.py - RemoteServiceConfig" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow

if (Test-Path "config.py") {
    $configContent = Get-Content "config.py" -Raw

    # Ищем RemoteServiceConfig
    if ($configContent -match 'class RemoteServiceConfig.*?timeout_seconds:\s*int\s*=\s*(\d+)') {
        $timeout = $matches[1]
        Write-Host "  timeout_seconds: $timeout" -ForegroundColor $(if ($timeout -eq "600") { "Green" } else { "Red" })
    }

    if ($configContent -match 'max_retries:\s*int\s*=\s*(\d+)') {
        $retries = $matches[1]
        Write-Host "  max_retries: $retries" -ForegroundColor Green
    }

    if ($configContent -match 'retry_delay:\s*float\s*=\s*([\d.]+)') {
        $delay = $matches[1]
        Write-Host "  retry_delay: $delay" -ForegroundColor Green
    }
} else {
    Write-Host "  ❌ config.py не найден" -ForegroundColor Red
}

# 3. settings.json
Write-Host "`n================================================================================" -ForegroundColor Yellow
Write-Host "3. settings.json - remote_service" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow

if (Test-Path "settings.json") {
    try {
        $settings = Get-Content "settings.json" | ConvertFrom-Json

        if ($settings.rag.remote_service) {
            $rs = $settings.rag.remote_service
            Write-Host "  timeout_seconds: $($rs.timeout_seconds)" -ForegroundColor Green
            Write-Host "  max_retries: $($rs.max_retries)" -ForegroundColor Green
            Write-Host "  retry_delay: $($rs.retry_delay)" -ForegroundColor Green
        } else {
            Write-Host "  ⚠️ remote_service секция не найдена" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  ❌ Ошибка чтения settings.json: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "  ❌ settings.json не найден" -ForegroundColor Red
}

# 4. Retry Policy файл
Write-Host "`n================================================================================" -ForegroundColor Yellow
Write-Host "4. rag/retry_policy.py - RetryConfig" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow

if (Test-Path "rag\retry_policy.py") {
    $retryContent = Get-Content "rag\retry_policy.py" -Raw

    if ($retryContent -match 'max_attempts:\s*int\s*=\s*(\d+)') {
        Write-Host "  max_attempts: $($matches[1])" -ForegroundColor Green
    }

    if ($retryContent -match 'base_delay:\s*float\s*=\s*([\d.]+)') {
        Write-Host "  base_delay: $($matches[1])" -ForegroundColor Green
    }

    if ($retryContent -match 'max_delay:\s*float\s*=\s*([\d.]+)') {
        Write-Host "  max_delay: $($matches[1])" -ForegroundColor Green
    }

    if ($retryContent -match 'timeout_seconds:\s*float\s*=\s*([\d.]+)') {
        $timeout = $matches[1]
        Write-Host "  timeout_seconds: $timeout" -ForegroundColor $(if ($timeout -eq "600.0") { "Green" } else { "Red" })
    }
} else {
    Write-Host "  ❌ rag\retry_policy.py не найден" -ForegroundColor Red
}

# 5. Circuit Breaker
Write-Host "`n================================================================================" -ForegroundColor Yellow
Write-Host "5. rag/remote_embedder.py - CircuitBreakerConfig" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow

if (Test-Path "rag\remote_embedder.py") {
    $embedderContent = Get-Content "rag\remote_embedder.py" -Raw

    if ($embedderContent -match 'failure_threshold=(\d+)') {
        Write-Host "  failure_threshold: $($matches[1])" -ForegroundColor Green
    }

    if ($embedderContent -match 'timeout_seconds=([\d.]+)') {
        $cbTimeout = $matches[1]
        Write-Host "  timeout_seconds: $cbTimeout" -ForegroundColor $(if ($cbTimeout -eq "300.0") { "Green" } else { "Red" })
    }
} else {
    Write-Host "  ❌ rag\remote_embedder.py не найден" -ForegroundColor Red
}

# 6. Event Loop Manager
Write-Host "`n================================================================================" -ForegroundColor Yellow
Write-Host "6. rag/event_loop_manager.py - ClientTimeout" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow

if (Test-Path "rag\event_loop_manager.py") {
    $loopContent = Get-Content "rag\event_loop_manager.py" -Raw

    if ($loopContent -match 'ClientTimeout\([^)]+total=(\d+)') {
        $total = $matches[1]
        Write-Host "  total: $total" -ForegroundColor $(if ($total -eq "600") { "Green" } else { "Red" })
    }

    if ($loopContent -match 'sock_read=(\d+)') {
        Write-Host "  sock_read: $($matches[1])" -ForegroundColor Green
    }
} else {
    Write-Host "  ❌ rag\event_loop_manager.py не найден" -ForegroundColor Red
}

# 7. Remote Vector Store
Write-Host "`n================================================================================" -ForegroundColor Yellow
Write-Host "7. rag/remote_vector_store.py - timeouts" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow

if (Test-Path "rag\remote_vector_store.py") {
    $vsContent = Get-Content "rag\remote_vector_store.py"

    Write-Host "  Найденные timeout параметры:"
    $vsContent | Select-String "timeout=\d+" | ForEach-Object {
        Write-Host "    Line $($_.LineNumber): $($_.Line.Trim())" -ForegroundColor Cyan
    } | Select-Object -First 10
} else {
    Write-Host "  ❌ rag\remote_vector_store.py не найден" -ForegroundColor Red
}

# 8. Running Processes
Write-Host "`n================================================================================" -ForegroundColor Yellow
Write-Host "8. Running Python Processes" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow

$pythonProcesses = Get-Process python* -ErrorAction SilentlyContinue

if ($pythonProcesses) {
    Write-Host "  Найдено Python процессов: $($pythonProcesses.Count)" -ForegroundColor Green
    $pythonProcesses | ForEach-Object {
        Write-Host "    PID: $($_.Id), Start: $($_.StartTime), Memory: $([math]::Round($_.WorkingSet64/1MB, 2)) MB" -ForegroundColor Cyan
    }
} else {
    Write-Host "  ⚠️ Нет запущенных Python процессов" -ForegroundColor Yellow
}

# 9. Recommendations
Write-Host "`n================================================================================" -ForegroundColor Yellow
Write-Host "9. Рекомендации" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow

# Проверяем config.py
$configContent = Get-Content "config.py" -Raw
if ($configContent -match 'timeout_seconds:\s*int\s*=\s*(\d+)') {
    $timeout = $matches[1]

    if ($timeout -eq "600") {
        Write-Host "  ✅ HOTFIX применён корректно (timeout_seconds = 600)" -ForegroundColor Green
        Write-Host "  ⚠️  Требуется ПЕРЕЗАПУСК приложения для применения!" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  Команды для перезапуска:" -ForegroundColor Cyan
        Write-Host "    1. Остановите текущий web_ui (Ctrl+C)" -ForegroundColor White
        Write-Host "    2. Запустите заново: python run_web.py" -ForegroundColor White
    } elseif ($timeout -eq "60") {
        Write-Host "  ❌ HOTFIX НЕ применён! Используются старые значения (60s)" -ForegroundColor Red
        Write-Host "  Рекомендация: Проверьте изменения в config.py" -ForegroundColor Yellow
    } else {
        Write-Host "  ⚠️ Неожиданное значение: $timeout" -ForegroundColor Yellow
    }
}

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host "Диагностика завершена!" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

Write-Host "`nДля запуска диагностики runtime значений (требует импорта модулей):" -ForegroundColor Yellow
Write-Host "  python scripts\check_timeouts.py" -ForegroundColor White
