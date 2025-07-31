# 🤖 Python Анализатор Репозиториев с OpenAI GPT

Автоматический анализатор кода репозиториев с генерацией детальной Markdown документации через OpenAI GPT. Создаёт подробные отчёты с анализом логики, компонентов, входов/выходов и итоговой сводкой для каждого файла.

## ✨ Возможности

- 🔍 **Рекурсивное сканирование** репозиториев с поддержкой множества языков программирования
- 🧠 **Детальный ИИ-анализ кода** через OpenAI GPT с полноформатными отчётами
- 📝 **Структурированные MD отчеты** с анализом логики, компонентов, входов/выходов и сводкой
- 🗂️ **Сохранение структуры папок** — подкаталоги исходного проекта воссоздаются в отчётах
- 🎯 **Умная разбивка кода** на логические части для оптимизации токенов
- 💾 **Кэширование результатов** для экономии API вызовов
- 📊 **Детальная статистика** использования токенов и результатов анализа
- 🚀 **CLI интерфейс** с прогресс-барами и цветным выводом
- 🌐 **Веб-интерфейс** на Streamlit для удобного использования
- 🌍 **Возможность развёртывания** на облачных серверах для публичного доступа

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

## 📦 Установка

1. **Клонируйте репозиторий:**
   ```bash
   git clone <repository-url>
   cd repo_sum
   ```

2. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Получите OpenAI API ключ:**
   - Зарегистрируйтесь на [OpenAI Platform](https://platform.openai.com/)
   - Перейдите в раздел [API Keys](https://platform.openai.com/api-keys)
   - Нажмите "Create new secret key"
   - Скопируйте созданный ключ (начинается с `sk-`)

4. **Настройте API ключ:**
   
   **Способ 1 - Через веб-интерфейс (рекомендуется):**
   - Запустите `python run_web.py`
   - Введите API ключ в боковой панели веб-интерфейса
   
   **Способ 2 - Через переменные окружения:**
   ```bash
   cp .env.example .env
   # Отредактируйте .env и добавьте ваш OpenAI API ключ
   export OPENAI_API_KEY="sk-your-api-key-here"
   ```
   
   **Способ 3 - Для Windows:**
   ```cmd
   set OPENAI_API_KEY=sk-your-api-key-here
   ```

## 🚀 Быстрый старт

### 🌐 Веб-интерфейс (рекомендуется)

Самый простой способ использования - через веб-интерфейс:

```bash
# Запуск веб-интерфейса
python run_web.py

# Или напрямую через streamlit
streamlit run web_ui.py
```

После запуска:
1. 📱 Откройте браузер: http://localhost:8501
2. 🔑 Введите ваш OpenAI API ключ в боковой панели
3. 📁 Выберите репозиторий (путь к папке или ZIP архив)
4. 🚀 Нажмите "Начать анализ"
5. 📥 Скачайте готовую документацию

**Преимущества веб-интерфейса:**
- ✅ Удобный графический интерфейс
- ✅ Загрузка ZIP архивов
- ✅ Предварительный просмотр статистики
- ✅ Отслеживание прогресса в реальном времени
- ✅ Скачивание результатов одним кликом

#### 🌍 Доступ с других устройств

**Для локальной сети:**
```bash
# Запуск с доступом в локальной сети
python -m streamlit run web_ui.py --server.address 0.0.0.0 --server.port 8501

# Затем другие устройства смогут зайти по адресу:
# http://<ваш_IP_в_сети>:8501
```

**Для публичного доступа:**
- Развернуть на облачном сервере (VPS/облако)
- Использовать туннели: ngrok, localtunnel, cloudflared
- Настроить проброс портов на роутере (небезопасно)

### 💻 CLI команды

Для продвинутых пользователей доступны CLI команды:

```bash
# Анализ репозитория
python main.py analyze /path/to/repository

# Анализ с указанием выходной директории
python main.py analyze /path/to/repository -o ./documentation

# Получить статистику репозитория без анализа
python main.py stats /path/to/repository

# Показать статистику использования токенов
python main.py token-stats

# Очистить кэш
python main.py clear-cache
```

### Пример использования

```bash
# Анализируем Python проект
python main.py analyze ~/my-python-project -o ./docs

# Результат:
# ✓ Найдено 25 файлов для анализа
# ✓ Успешно проанализировано: 23 файла
# ✓ Документация сохранена в: ./docs
# ✓ Главный файл: ./docs/README.md
```

## 📁 Структура вывода

После анализа создается следующая структура с сохранением иерархии исходного проекта:

```
SUMMARY_REPORT_<repo_name>/
├── README.md                           # Главный индексный файл
├── report_main.py.md                   # Файлы из корня
├── report_config.py.md
├── src/                                # Подкаталоги сохраняются
│   ├── report_src_app.py.md           # Файлы из подпапок
│   ├── report_src_utils.py.md
│   └── models/
│       └── report_src_models_user.py.md
└── tests/
    └── report_tests_test_main.py.md
```

### Именование файлов отчётов

- **Корневые файлы:** `report_<filename>.md`
- **Файлы из подпапок:** `report_<top-folder>_<filename>.md`
- **Вложенные подпапки:** `report_<folder1>_<folder2>_<filename>.md`

### Формат документации файла

Каждый файл содержит полный анализ в структурированном формате:

```markdown
# Audit Report: filename.py

## 🔍 Общее описание логики
Подробное описание назначения файла, его роли в проекте и основной логики работы.

## 🧩 Составные элементы
- **Класс DatabaseManager**: Управляет подключениями к базе данных
- **Функция connect()**: Устанавливает соединение с БД
- **Метод execute_query()**: Выполняет SQL запросы
- **Константа DB_CONFIG**: Конфигурация подключения

## 📥 Входы и 📤 Выходы
**Входные параметры:**
- config_file: строка пути к файлу конфигурации
- connection_string: строка подключения к БД

**Возвращаемые значения:**
- DatabaseManager: инициализированный объект менеджера
- query_result: результат выполнения запроса

## 📝 Итоговая сводка
Файл реализует паттерн менеджера подключений к базе данных с поддержкой 
пула соединений и автоматического переподключения. Используется как основной
интерфейс для всех операций с БД в приложении.

---
*Документация сгенерирована автоматически 2025-01-24 17:39:00*
```

## ⚙️ Конфигурация

Настройки находятся в файле `settings.json`:

```json
{
  "openai": {
    "api_key_env_var": "OPENAI_API_KEY",
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
    "enable_fallback": true,
    "languages_priority": ["python", "javascript", "java"]
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
    "index_template": "index_template.md"
  },
  "prompts": {
    "code_analysis_prompt_file": "prompts/code_analysis_prompt.md"
  }
}
```

### Новые параметры v2.1:
- `max_response_tokens` - максимум токенов в ответе GPT (по умолчанию 5000)
- `retry_attempts` - количество попыток при ошибках API (по умолчанию 3)
- `retry_delay` - задержка между попытками в секундах (по умолчанию 1.0)
- `prompts.code_analysis_prompt_file` - путь к файлу с промптом для анализа

### Переменные окружения

Создайте файл `.env` на основе `.env.example`:

```bash
# OpenAI API ключ (обязательно)
OPENAI_API_KEY=your-openai-api-key-here

# Дополнительные настройки (опционально)
LOG_LEVEL=INFO
CACHE_DIR=./cache
```

## 🌍 Развёртывание на сервере

### Для публичного доступа через облачный сервер:

1. **Арендуйте VPS/облачный сервер** (Yandex Cloud, DigitalOcean, AWS EC2, и т.д.)

2. **Настройте сервер:**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3 python3-pip git
   
   # Клонируйте проект
   git clone <your-repo-url>
   cd repo_sum
   
   # Установите зависимости
   pip3 install -r requirements.txt
   ```

3. **Откройте порт в firewall:**
   ```bash
   # Ubuntu/Debian
   sudo ufw allow 8501/tcp
   
   # CentOS/RHEL
   sudo firewall-cmd --permanent --add-port=8501/tcp
   sudo firewall-cmd --reload
   ```

4. **Запустите сервис:**
   ```bash
   # Запуск с публичным доступом
   python3 -m streamlit run web_ui.py --server.address 0.0.0.0 --server.port 8501
   ```

5. **Доступ к приложению:**
   ```
   http://<публичный_IP_сервера>:8501
   ```

### Автоматический запуск через systemd:

Создайте сервис для автозапуска:

```bash
# Создайте файл сервиса
sudo nano /etc/systemd/system/repo-analyzer.service
```

```ini
[Unit]
Description=Repository Analyzer Streamlit App
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/repo_sum
Environment=PATH=/usr/bin:/usr/local/bin
Environment=OPENAI_API_KEY=your-api-key-here
ExecStart=/usr/bin/python3 -m streamlit run web_ui.py --server.address 0.0.0.0 --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Активируйте сервис
sudo systemctl daemon-reload
sudo systemctl enable repo-analyzer
sudo systemctl start repo-analyzer
```

## 📊 Статистика и мониторинг

### Просмотр статистики репозитория

```bash
python main.py stats /path/to/repo
```

Показывает:
- Количество файлов по языкам
- Общий размер кода
- Самые большие файлы
- Распределение по типам файлов

### Мониторинг токенов OpenAI

```bash
python main.py token-stats
```

Отображает:
- Использовано токенов сегодня
- Средний расход на файл
- Количество запросов

## 🎯 Оптимизация токенов

Программа автоматически оптимизирует использование токенов:

- **Умная разбивка**: код разделяется на логические части (классы, функции)
- **Кэширование**: результаты сохраняются для избежания повторных запросов
- **Лимиты размера**: слишком большие файлы анализируются частично
- **Увеличенный лимит**: до 2048 токенов на ответ для полных отчётов
- **Fallback режим**: при ошибках API создается базовый анализ

## 🔧 Расширение функциональности

### Добавление нового парсера

1. Создайте класс наследник от `BaseParser`:

```python
# parsers/your_language_parser.py
from .base_parser import BaseParser

class YourLanguageParser(BaseParser):
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.ext']
        
    def can_parse(self, file_path: str) -> bool:
        return file_path.endswith('.ext')
        
    # Реализуйте абстрактные методы...
```

2. Зарегистрируйте в `ParserRegistry`:

```python
# parsers/base_parser.py
def _initialize_parsers(self):
    # ... существующие парсеры
    try:
        from .your_language_parser import YourLanguageParser
        self.parsers.append(YourLanguageParser())
    except ImportError:
        pass
```

### Настройка промптов

**Новое в v2.1:** Промпты теперь хранятся в отдельных файлах для удобства редактирования.

Отредактируйте файл `prompts/code_analysis_prompt.md`:

```markdown
# Цель
Главная цель — исследовать каждый файл с кодом и точно описать...

# Правила анализа
1. Ты — аналитик архитектуры кода...
2. Запрещено:
   - Оценивать или упоминать проблемы/баги...

# Шаблон отчета
---
Audit Report: {filename}
---

### 🔍 1. Краткий обзор (что делает файл?)
...
```

**Создание собственных промптов:**
1. Создайте новый `.md` файл в папке `prompts/`
2. Обновите `settings.json`:
```json
{
  "prompts": {
    "code_analysis_prompt_file": "prompts/my_custom_prompt.md"
  }
}
```
3. Перезапустите приложение

## 🐛 Устранение проблем

### Частые ошибки

**Ошибка аутентификации OpenAI:**
```
Проверьте правильность API ключа в переменной OPENAI_API_KEY
```

**Файлы не найдены для анализа:**
```
Проверьте поддерживаемые расширения в settings.json
Убедитесь, что директории не исключены в excluded_directories
```

**Ошибка загрузки файла "Файл слишком большой":**
```
Размер файла превышает 100MB. Используйте файлы меньшего размера
или увеличьте MAX_FILE_SIZE в web_ui.py (не рекомендуется)
```

**Ошибка конфигурации "Файл промпта не найден":**
```
Убедитесь что файл prompts/code_analysis_prompt.md существует
Проверьте путь в settings.json -> prompts.code_analysis_prompt_file
```

**Предупреждения о "подозрительных файлах" в архиве:**
```
Это нормально - система безопасности обнаружила исполняемые файлы
Они не блокируют анализ, только предупреждают в логах
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

## 📈 Производительность

### Рекомендации по использованию

- **Малые проекты (< 50 файлов)**: ~5-10 минут анализа
- **Средние проекты (50-200 файлов)**: ~15-30 минут  
- **Большие проекты (200+ файлов)**: ~45+ минут

### Оптимизация для больших репозиториев

1. Исключите ненужные директории через `excluded_directories`
2. Уменьшите `max_file_size` для пропуска больших файлов
3. Используйте кэширование (`enable_caching: true`)
4. Очищайте кэш периодически: `python main.py clear-cache`

## 💰 Стоимость использования

### Примерные расходы OpenAI API:

- **gpt-4o-mini**: ~$0.0001-0.0002 за файл
- **Малый проект** (20 файлов): ~$0.002-0.004
- **Средний проект** (100 файлов): ~$0.01-0.02
- **Большой проект** (500 файлов): ~$0.05-0.10

Кэширование значительно снижает затраты при повторном анализе.

## 📁 Файловая структура проекта

```
repo_sum/
├── .env.example              # Пример переменных окружения
├── .gitignore               # Игнорируемые файлы
├── README.md                # Документация
├── requirements.txt         # Зависимости Python
├── settings.json           # Конфигурация
├── main.py                 # CLI интерфейс
├── run_web.py             # Запуск веб-интерфейса
├── web_ui.py              # Streamlit приложение
├── config.py              # Управление конфигурацией
├── file_scanner.py        # Сканирование файлов
├── code_chunker.py        # Разбивка кода на части
├── openai_integration.py  # Работа с OpenAI API
├── doc_generator.py       # Генерация документации
├── utils.py               # Утилиты и структуры данных
├── parsers/               # Парсеры языков программирования
│   ├── __init__.py
│   ├── base_parser.py     # Базовый класс парсера
│   ├── python_parser.py   # Python парсер
│   └── cpp_parser.py      # C++ парсер
├── cache/                 # Кэш результатов (в .gitignore)
└── logs/                  # Логи приложения (в .gitignore)
```

## 📝 Лицензия

MIT License - смотрите файл LICENSE для деталей.

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте feature ветку (`git checkout -b feature/amazing-feature`)
3. Закоммитьте изменения (`git commit -m 'Add amazing feature'`)  
4. Запушьте в ветку (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## 📧 Поддержка

При возникновении проблем:
- Создайте Issue в GitHub репозитории
- Приложите логи с флагом `-v` (verbose)
- Укажите версию Python и операционную систему
- Приложите содержимое файла `settings.json`

## 🆕 Последние обновления

### v2.1 (Июль 2025) - Улучшения безопасности и производительности
- 🔒 **Полная защита API ключей** - убрано логирование конфиденциальных данных
- 🛡️ **Безопасная загрузка файлов** - валидация размера, типа, защита от path traversal
- ⚡ **Батчевая обработка** - параллельный анализ файлов с адаптивными размерами батчей (2-8 файлов)
- 📄 **Внешние промпты** - вынесение промптов в `prompts/code_analysis_prompt.md` для лёгкого редактирования
- ✅ **Комплексная валидация** - детальная проверка всех параметров конфигурации
- 🧰 **Улучшенная обработка ошибок** - централизованные утилиты, graceful degradation
- ⚙️ **Расширенная конфигурация** - новые параметры retry_attempts, retry_delay, max_response_tokens

### v2.0 (Январь 2025)
- ✅ **Исправлена структура папок** - теперь сохраняется иерархия исходного проекта
- ✅ **Восстановлены полные отчёты GPT** - убрана обрезка, увеличен лимит токенов до 2048
- ✅ **Новый формат отчётов** - структурированный анализ с логикой, компонентами, входами/выходами
- ✅ **Исправлены ошибки веб-интерфейса** - убраны несуществующие параметры
- ✅ **Добавлена поддержка развёртывания** - инструкции для облачных серверов
- ✅ **Обновлена документация** - полное описание всех возможностей

---

**Создано с ❤️ для автоматизации документирования кода**
