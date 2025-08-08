# 🤖 Python Анализатор Репозиториев с OpenAI GPT

Автоматический анализатор кода репозиториев с генерацией детальной Markdown документации через OpenAI GPT. Создаёт подробные отчёты с анализом логики, компонентов, входов/выходов и итоговой сводкой для каждого файла.

## ✨ Ключевые возможности

- 🔍 **Рекурсивное сканирование** репозиториев с поддержкой 9+ языков программирования
- 🧠 **ИИ-анализ кода** через OpenAI GPT с настраиваемыми промптами
- ⚡ **Батчевая обработка** - параллельный анализ файлов с адаптивной оптимизацией
- 📝 **Структурированные отчеты** с анализом логики, компонентов, входов/выходов
- 🗂️ **Сохранение структуры** - иерархия папок воссоздается в документации
- 🛡️ **Безопасная загрузка** файлов с валидацией и защитой от path traversal
- 💾 **Умное кэширование** результатов для экономии API вызовов
- 🌐 **Веб-интерфейс** на Streamlit и CLI с прогресс-барами
- 🔒 **Защита API ключей** - логирование только метаданных без конфиденциальной информации

## 🆕 Что нового (ветка feature/next-gen)

- ⚡ Инкрементальный анализ: анализируются только изменённые файлы, ведётся индекс `./.repo_sum/index.json`.
- 🔒 Санитайзинг секретов: маскирование чувствительных данных по настраиваемым regex-паттернам перед отправкой в LLM.
- 🔁 Повторы вызовов OpenAI (retries): устойчивость к временным ошибкам API (настраивается в `settings.json`).
- 🧹 Точный фильтр библиотечных файлов: исключаются только явные вендорные/сборочные каталоги, меньше пропусков реального кода.
- 📁 Корректная директория вывода: генерация в `--output/ SUMMARY_REPORT_<repo>` (единое поведение CLI и веб).

## 🛠 Поддерживаемые языки

- **Python** (.py) - полный AST анализ
- **JavaScript/TypeScript** (.js, .ts, .jsx, .tsx)  
- **Java** (.java)
- **C++** (.cpp, .cc, .cxx, .h, .hpp)
- **C#** (.cs)
- **Go** (.go)
- **Rust** (.rs)
- **PHP** (.php)
- **Ruby** (.rb)

## 📦 Установка и настройка

### 1. Клонирование и установка зависимостей

```bash
git clone <repository-url>
cd repo_sum
pip install -r requirements.txt
```

### 2. Настройка OpenAI API

1. Получите API ключ на [OpenAI Platform](https://platform.openai.com/api-keys)
2. Настройте переменную окружения:

```bash
# Linux/macOS
export OPENAI_API_KEY="sk-your-api-key-here"

# Windows
set OPENAI_API_KEY=sk-your-api-key-here
```

Или создайте файл `.env`:
```
OPENAI_API_KEY=sk-your-api-key-here
```

## 🚀 Использование

### Веб-интерфейс (рекомендуется)

```bash
python run_web.py
# или напрямую:
streamlit run web_ui.py
```

Откройте http://localhost:8501 в браузере:
1. 🔑 Введите API ключ в боковой панели
2. 📁 Выберите локальную папку или загрузите ZIP архив
3. 🚀 Нажмите "Начать анализ"
4. 📥 Скачайте готовую документацию

**Возможности веб-интерфейса:**
- ✅ Безопасная валидация загружаемых файлов (до 100MB)
- ✅ Предварительный просмотр статистики проекта
- ✅ Отслеживание прогресса батчевой обработки
- ✅ Защита от небезопасных архивов

### CLI команды

```bash
# Анализ репозитория (инкрементальный)
python main.py analyze /path/to/repository -o ./documentation --incremental

# Статистика без анализа
python main.py stats /path/to/repository

# Управление кэшем
python main.py clear-cache
python main.py token-stats
```

## ⚙️ Конфигурация

Настройки в файле `settings.json`:

```json
{
  "openai": {
    "max_tokens_per_chunk": 4000,
    "max_response_tokens": 5000,
    "temperature": 0.1,
    "retry_attempts": 3,
    "retry_delay": 1.0
  },
  "token_management": {
    "enable_caching": true,
    "cache_expiry_days": 7
  },
  "analysis": {
    "chunk_strategy": "logical",
    "min_chunk_size": 100,
    "languages_priority": ["python", "javascript", "java"],
    "enable_advanced_scoring": false,
    "sanitize_enabled": false,
    "sanitize_patterns": [
      "(?i)api_key\\s*[:=]\\s*['\"][^'\"]+['\"]",
      "(?i)password\\s*[:=]\\s*['\"][^'\"]+['\"]"
    ]
  },
  "file_scanner": {
    "max_file_size": 10485760,
    "excluded_directories": [".git", "node_modules", "__pycache__"],
    "supported_extensions": {
      ".py": "python",
      ".js": "javascript",
      ".ts": "typescript"
    }
  },
  "output": {
    "default_output_dir": "./docs",
    "file_template": "minimal_file.md",
    "index_template": "index_template.md",
    "format": "markdown",
    "templates_dir": "report_templates"
  },
  "prompts": {
    "code_analysis_prompt_file": "prompts/code_analysis_prompt.md"
  }
}
```

### Ключевые параметры (коротко):
- `analysis.sanitize_enabled` — включить маскирование секретов перед отправкой в LLM.
- `analysis.sanitize_patterns` — список regex-паттернов для маскировки (см. пример выше).
- `openai.retry_attempts` / `openai.retry_delay` — повторы и задержка при ошибках API.
- `output.default_output_dir` — базовая директория вывода для CLI/веб.
- `output.format`/`output.templates_dir` — подготовка к HTML/PDF (пока формат — markdown).
- `analysis.chunk_strategy` — стратегия разбивки кода (logical/size/lines).
- `prompts.code_analysis_prompt_file` — путь к промпту.

## ⚡ Инкрементальный анализ

При включённом `--incremental` анализируются только изменённые файлы относительно индекса `./.repo_sum/index.json`.

- Как отключить: добавьте флаг `--no-incremental` в CLI.
- Как сбросить состояние: удалите `./.repo_sum/index.json` в корне анализируемого репозитория.

## 🔒 Санитайзинг секретов

Включите `analysis.sanitize_enabled` и задайте `analysis.sanitize_patterns`, чтобы маскировать чувствительные данные (например ключи/пароли) в коде перед отправкой в LLM. Это снижает риск утечки секретов.

## 🔁 Повторы вызовов OpenAI (retries)

Если API временно недоступен или исчерпана квота, выполняются повторы с задержкой:

- Настройка в `settings.json`: `openai.retry_attempts`, `openai.retry_delay`.
- При исчерпании всех попыток — ошибка фиксируется в отчёте файла.

## 📁 Путь вывода результатов

CLI/веб сохраняют отчёты в: `--output/ SUMMARY_REPORT_<repo_name>`. Главный индекс — `README.md` внутри этой папки.

## 📁 Структура вывода

```
SUMMARY_REPORT_<repo_name>/
├── README.md                           # Главный индексный файл
├── report_main.py.md                   # Файлы из корня
├── report_config.py.md
├── src/                                # Подкаталоги сохраняются
│   ├── report_src_app.py.md           # Файлы из подпапок
│   └── models/
│       └── report_src_models_user.py.md
└── tests/
    └── report_tests_test_main.py.md
```

### Формат отчёта для каждого файла:

```markdown
# Audit Report: filename.py

## 🔍 1. Краткий обзор (что делает файл?)
- **Назначение файла** — описание главной задачи
- **Последовательность операций** — пошаговый процесс

## ⚙️ 2. Подробности реализации (как работает?)
#### Входные данные
#### Обработка данных  
#### Выходные данные

## 🧩 3. Структура кода
Функции, методы и классы с описанием взаимосвязей

## 📌 4. Общий поток данных
Цепочка обработки от входа до выхода

## 🛑 5. Ограничения анализа
Что не включено в анализ и почему
```

## 🎨 Кастомизация промптов

Промпты хранятся в отдельных файлах для удобного редактирования:

1. **Редактирование текущего промпта:**
   ```bash
   nano prompts/code_analysis_prompt.md
   ```

2. **Создание собственного промпта:**
   ```bash
   # Создайте новый файл
   cp prompts/code_analysis_prompt.md prompts/my_prompt.md
   
   # Обновите конфигурацию
   # settings.json -> "prompts.code_analysis_prompt_file": "prompts/my_prompt.md"
   ```

3. **Перезапустите приложение** для применения изменений

## ⚡ Инкрементальный анализ

При включённом `--incremental` анализируются только изменённые файлы относительно индекса `./.repo_sum/index.json`. Индекс обновляется после успешной генерации отчётов.

## 🔒 Санитайзинг секретов

Включите `analysis.sanitize_enabled` и задайте `analysis.sanitize_patterns` (regex), чтобы маскировать чувствительные данные перед отправкой к LLM.

## 🌍 Развёртывание

### Локальная сеть
```bash
streamlit run web_ui.py --server.address 0.0.0.0 --server.port 8501
# Доступ: http://<your-ip>:8501
```

### Облачное развёртывание
1. Разверните на VPS (DigitalOcean, AWS, Yandex Cloud)
2. Установите зависимости и откройте порт 8501
3. Запустите с публичным доступом:
   ```bash
   streamlit run web_ui.py --server.address 0.0.0.0 --server.port 8501
   ```

### Автозапуск через systemd
```ini
[Unit]
Description=Repository Analyzer
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/repo_sum
Environment=OPENAI_API_KEY=your-key
ExecStart=/usr/bin/python3 -m streamlit run web_ui.py --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

## 🔧 Производительность и оптимизация

### Батчевая обработка
- **Малые проекты** (≤10 файлов): батчи по 2 файла
- **Средние проекты** (11-50 файлов): батчи по 3 файла  
- **Большие проекты** (51-200 файлов): батчи по 5 файлов
- **Очень большие** (200+ файлов): батчи по 8 файлов

### Рекомендации по времени анализа:
- **Малые проекты** (< 50 файлов): 5-10 минут
- **Средние проекты** (50-200 файлов): 15-30 минут  
- **Большие проекты** (200+ файлов): 45+ минут

### Оптимизация для больших репозиториев:
1. Настройте `excluded_directories` для исключения ненужных папок
2. Уменьшите `max_file_size` для пропуска больших файлов
3. Используйте кэширование (`enable_caching: true`)
4. Периодически очищайте кэш: `python main.py clear-cache`

## 💰 Стоимость использования OpenAI API

### Примерные расходы (gpt-4o-mini):
- **Один файл**: ~$0.0001-0.0002
- **Малый проект** (20 файлов): ~$0.002-0.004
- **Средний проект** (100 файлов): ~$0.01-0.02
- **Большой проект** (500 файлов): ~$0.05-0.10

**Кэширование значительно снижает затраты при повторном анализе.**

## 🐛 Устранение проблем

### Частые ошибки

**Ошибка аутентификации OpenAI:**
```
Проверьте правильность API ключа в переменной OPENAI_API_KEY
```

**"Файл слишком большой" (>100MB):**
```
Используйте файлы меньшего размера или увеличьте MAX_FILE_SIZE в web_ui.py
```

**"Файл промпта не найден":**
```
Убедитесь что prompts/code_analysis_prompt.md существует
Проверьте путь в settings.json
```

**Файлы не найдены для анализа:**
```
Проверьте supported_extensions в settings.json
Убедитесь что папки не исключены в excluded_directories
```

### Логи и отладка

```bash
# Подробные логи
python main.py -v analyze /path/to/repo

# Только ошибки  
python main.py -q analyze /path/to/repo

# Логи в файл
python main.py analyze /path/to/repo 2> debug.log
```

## 🏗 Архитектура проекта

```
repo_sum/
├── main.py                 # CLI интерфейс и RepositoryAnalyzer
├── web_ui.py              # Streamlit веб-интерфейс
├── config.py              # Система конфигурации
├── file_scanner.py        # Сканирование файлов репозитория
├── openai_integration.py  # Интеграция с OpenAI API
├── doc_generator.py       # Генерация MD документации
├── code_chunker.py        # Разбивка кода на логические части
├── utils.py               # Утилиты и структуры данных
├── parsers/               # Парсеры для разных языков
│   ├── base_parser.py     # Базовый класс парсера
│   ├── python_parser.py   # Python AST парсер
│   └── [language]_parser.py
├── prompts/               # Внешние промпты для анализа
│   └── code_analysis_prompt.md
├── settings.json          # Конфигурация приложения
├── requirements.txt       # Python зависимости
└── tests/                 # Тесты
```

### Ключевые компоненты:
- **RepositoryAnalyzer** - основной класс координации анализа
- **FileScanner** - поиск и фильтрация файлов
- **ParserRegistry** - выбор парсера по типу файла
- **CodeChunker** - интеллектуальная разбивка кода
- **OpenAIManager** - работа с API, кэширование, retry-логика
- **DocumentationGenerator** - создание финальных MD отчетов

## 📄 Лицензия

MIT License - подробности в файле LICENSE.

## 🤝 Разработка и вклад

1. Форкните репозиторий
2. Создайте feature-ветку (`git checkout -b feature/new-feature`)
3. Внесите изменения и добавьте тесты
4. Создайте коммит (`git commit -m 'Add new feature'`)
5. Отправьте в ветку (`git push origin feature/new-feature`)
6. Создайте Pull Request

## 📧 Поддержка

При возникновении проблем:
- Создайте Issue в GitHub с подробным описанием
- Приложите логи с флагом `-v` (verbose mode)
- Укажите версию Python и операционную систему
- Приложите содержимое `settings.json` (без API ключа)

---

**Создано для автоматизации документирования кода с ❤️ и ИИ**