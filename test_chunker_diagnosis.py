#!/usr/bin/env python3
"""Диагностический тест для code_chunker.py"""

import logging

# Настройка логирования для диагностики
logging.basicConfig(
    level=logging.ERROR,
    format='%(levelname)s: %(message)s'
)

from file_scanner import FileScanner
from parsers.base_parser import ParserRegistry
from code_chunker import CodeChunker

def test_single_file():
    """Тест на одном файле config.py"""
    print("=" * 80)
    print("ДИАГНОСТИЧЕСКИЙ ТЕСТ: code_chunker.py")
    print("=" * 80)
    
    # 1. Сканируем файл
    scanner = FileScanner()
    files = list(scanner.scan_repository('.'))
    
    # Находим config.py
    config_file = None
    for f in files:
        if f.name == 'config.py':
            config_file = f
            break
    
    if not config_file:
        print("❌ Файл config.py не найден!")
        return
    
    print(f"\n✅ Найден файл: {config_file.path}")
    print(f"   Тип path: {type(config_file.path)}")
    print(f"   Кодировка: {config_file.encoding}")
    
    # 2. Парсим файл
    parser_registry = ParserRegistry()
    parser = parser_registry.get_parser(config_file.path)
    
    if not parser:
        print("❌ Парсер не найден!")
        return
    
    print(f"\n✅ Парсер: {type(parser).__name__}")
    
    try:
        parsed_file = parser.safe_parse(config_file)
        print("✅ Файл успешно распарсен")
        print(f"   Элементов: {len(parsed_file.elements)}")
    except Exception as e:
        print(f"❌ Ошибка парсинга: {e}")
        return
    
    # 3. Разбиваем на чанки - здесь должна быть ошибка
    print("\n🔍 НАЧИНАЕМ CHUNK_PARSED_FILE...")
    print(f"   parsed_file.file_info: {type(parsed_file.file_info)}")
    print(f"   parsed_file.file_info.path: {type(parsed_file.file_info.path)}")
    
    chunker = CodeChunker()
    
    try:
        chunks = chunker.chunk_parsed_file(parsed_file)
        print(f"\n✅ УСПЕХ! Создано {len(chunks)} чанков")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"   Чанк {i}: {chunk.name} ({chunk.tokens_estimate} токенов)")
    except Exception as e:
        print(f"\n❌ ОШИБКА: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("ТЕСТ ЗАВЕРШЁН")
    print("=" * 80)

if __name__ == '__main__':
    test_single_file()