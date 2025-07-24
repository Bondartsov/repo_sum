"""
Парсер для Python кода с использованием встроенного модуля AST.
"""

import ast
import re
from typing import List, Optional

from .base_parser import BaseParser
from utils import FileInfo, ParsedFile, ParsedElement, ParsingError


class PythonParser(BaseParser):
    """Парсер для Python файлов с использованием AST"""
    
    def __init__(self):
        super().__init__()
        self.comment_patterns = [
            r'#\s*(.+)',  # Однострочные комментарии
            r'"""([^"]*?)"""',  # Многострочные комментарии с """
            r"'''([^']*?)'''",  # Многострочные комментарии с '''
        ]
        self.supported_extensions = ['.py']
        
    def can_parse(self, file_path: str) -> bool:
        """Проверяет, может ли парсер обработать файл"""
        return any(file_path.lower().endswith(ext) for ext in self.supported_extensions)
        
    def parse_file(self, file_info: FileInfo) -> ParsedFile:
        """Основной метод парсинга Python файла"""
        try:
            source_code = self._read_file_content(file_info)
            
            # Парсим код с помощью AST
            tree = ast.parse(source_code, filename=file_info.path)
            
            # Извлекаем элементы
            elements = []
            elements.extend(self._extract_classes_from_ast(tree, source_code))
            elements.extend(self._extract_functions_from_ast(tree, source_code))
            elements.extend(self._extract_assignments_from_ast(tree, source_code))
            
            # Извлекаем импорты
            imports = self._extract_imports_from_ast(tree)
            
            # Извлекаем комментарии
            global_comments = self.extract_comments(source_code)
            
            return ParsedFile(
                file_info=file_info,
                elements=elements,
                imports=imports,
                global_comments=global_comments,
                parse_errors=[]
            )
            
        except SyntaxError as e:
            self.logger.error(f"Синтаксическая ошибка в {file_info.path}: {e}")
            return ParsedFile(
                file_info=file_info,
                elements=[],
                imports=[],
                global_comments=[],
                parse_errors=[f"Синтаксическая ошибка: {e}"]
            )
        except Exception as e:
            raise ParsingError(f"Ошибка парсинга {file_info.path}: {e}")
    
    def extract_imports(self, source_code: str) -> List[str]:
        """Извлекает импорты из Python кода"""
        try:
            tree = ast.parse(source_code)
            return self._extract_imports_from_ast(tree)
        except:
            return []
    
    def extract_classes(self, source_code: str) -> List[ParsedElement]:
        """Извлекает классы из Python кода"""
        try:
            tree = ast.parse(source_code)
            return self._extract_classes_from_ast(tree, source_code)
        except:
            return []
    
    def extract_functions(self, source_code: str) -> List[ParsedElement]:
        """Извлекает функции из Python кода"""
        try:
            tree = ast.parse(source_code)
            return self._extract_functions_from_ast(tree, source_code)
        except:
            return []
    
    def _extract_imports_from_ast(self, tree: ast.AST) -> List[str]:
        """Извлекает импорты из AST дерева"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_name = alias.name
                    if alias.asname:
                        import_name += f" as {alias.asname}"
                    imports.append(f"import {import_name}")
                    
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    import_name = alias.name
                    if alias.asname:
                        import_name += f" as {alias.asname}"
                    imports.append(f"from {module} import {import_name}")
        
        return imports
    
    def _extract_classes_from_ast(self, tree: ast.AST, source_code: str) -> List[ParsedElement]:
        """Извлекает классы из AST дерева"""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Извлекаем докстринг класса
                docstring = ast.get_docstring(node)
                
                # Создаем сигнатуру класса
                bases = [self._get_node_name(base) for base in node.bases]
                signature = f"class {node.name}"
                if bases:
                    signature += f"({', '.join(bases)})"
                
                classes.append(ParsedElement(
                    name=node.name,
                    type="class",
                    line_number=node.lineno,
                    docstring=self._clean_docstring(docstring) if docstring else None,
                    signature=signature,
                    comments=[]
                ))
        
        return classes
    
    def _extract_functions_from_ast(self, tree: ast.AST, source_code: str) -> List[ParsedElement]:
        """Извлекает функции и методы из AST дерева"""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Определяем тип - функция или метод
                element_type = "method" if self._is_method(node, tree) else "function"
                
                # Извлекаем докстринг
                docstring = ast.get_docstring(node)
                
                # Создаем сигнатуру функции
                signature = self._create_function_signature(node)
                
                functions.append(ParsedElement(
                    name=node.name,
                    type=element_type,
                    line_number=node.lineno,
                    docstring=self._clean_docstring(docstring) if docstring else None,
                    signature=signature,
                    comments=[]
                ))
        
        return functions
    
    def _extract_assignments_from_ast(self, tree: ast.AST, source_code: str) -> List[ParsedElement]:
        """Извлекает глобальные переменные и константы"""
        assignments = []
        
        # Получаем только присваивания на верхнем уровне
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Определяем тип - константа или переменная
                        var_type = "constant" if target.id.isupper() else "variable"
                        
                        assignments.append(ParsedElement(
                            name=target.id,
                            type=var_type,
                            line_number=node.lineno,
                            signature=f"{target.id} = ...",
                            comments=[]
                        ))
        
        return assignments
    
    def _is_method(self, func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Проверяет, является ли функция методом класса"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for class_item in node.body:
                    if class_item == func_node:
                        return True
        return False
    
    def _create_function_signature(self, func_node: ast.FunctionDef) -> str:
        """Создает сигнатуру функции"""
        args = []
        
        # Обычные аргументы
        for arg in func_node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_node_name(arg.annotation)}"
            args.append(arg_str)
        
        # Аргумент *args
        if func_node.args.vararg:
            vararg_str = f"*{func_node.args.vararg.arg}"
            if func_node.args.vararg.annotation:
                vararg_str += f": {self._get_node_name(func_node.args.vararg.annotation)}"
            args.append(vararg_str)
        
        # Аргумент **kwargs
        if func_node.args.kwarg:
            kwarg_str = f"**{func_node.args.kwarg.arg}"
            if func_node.args.kwarg.annotation:
                kwarg_str += f": {self._get_node_name(func_node.args.kwarg.annotation)}"
            args.append(kwarg_str)
        
        signature = f"def {func_node.name}({', '.join(args)})"
        
        # Добавляем аннотацию возвращаемого типа
        if func_node.returns:
            signature += f" -> {self._get_node_name(func_node.returns)}"
        
        return signature
    
    def _get_node_name(self, node: ast.AST) -> str:
        """Получает имя узла AST как строку"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_node_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._get_node_name(node.value)}[{self._get_node_name(node.slice)}]"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return ast.unparse(node) if hasattr(ast, 'unparse') else str(node)