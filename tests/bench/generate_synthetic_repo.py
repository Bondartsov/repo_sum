    import argparse, os, random, string, pathlib, math

    TEMPLATE_HEADER = '''

    import math
    import sys
    import typing

    '''

    CLASS_TEMPLATE = '''
    class {class_name}:
        """Synthetic class {class_name}. Marker: {marker}.
        {doc}
        """
{methods}
    '''

    METHOD_TEMPLATE = '''
        def {method_name}(self, x: int) -> int:
            """Method {method_name}. Marker: {marker}. {doc}"""
{body}
            return x
    '''

    FUNC_TEMPLATE = '''
    def {func_name}(x: int, y: int) -> int:
        """Function {func_name}. Marker: {marker}. {doc}"""
{body}
        return x + y
    '''

    def random_identifier(prefix: str, n: int = 6) -> str:
        return prefix + '_' + ''.join(random.choices(string.ascii_lowercase, k=n))

    def gen_long_body(lines: int, indent: int = 12) -> str:
        pad = ' ' * indent
        parts = []
        for i in range(lines):
            parts.append(f"{pad}# line {i}: heavy computation and text marker lorem ipsum dolor sit amet\n")
            parts.append(f"{pad}value_{i} = {i} * {i} + {i}\n")
        return ''.join(parts)

    def gen_doc(extra_lines: int) -> str:
        return "\n".join([f"Doc line {i} with context and keywords." for i in range(extra_lines)])

    def write_file(path: pathlib.Path, text: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)

    def build_python_file(module_name: str, large_classes: int, large_class_lines: int, small_funcs: int, seed: int):
        random.seed(seed + hash(module_name) % 9973)
        content = [TEMPLATE_HEADER.format(module_name=module_name)]
        # Large classes
        for i in range(large_classes):
            cls = random_identifier("BigClass")
            marker = f"NEEDLE_{module_name}_{i}"
            methods = []
            # a few large methods
            for m in range(3):
                mname = random_identifier("method")
                body = gen_long_body(max(10, large_class_lines // 3))
                doc = gen_doc(2)
                methods.append(METHOD_TEMPLATE.format(method_name=mname, marker=marker, doc=doc, body=body))
            content.append(CLASS_TEMPLATE.format(class_name=cls, marker=marker, doc=gen_doc(3), methods=''.join(methods)))
        # Small functions
        for i in range(small_funcs):
            fname = random_identifier("func")
            marker = f"NEEDLE_{module_name}_f{i}"
            body = gen_long_body(3, indent=8)
            content.append(FUNC_TEMPLATE.format(func_name=fname, marker=marker, doc=gen_doc(1), body=body))
        return ''.join(content)

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument('--out-dir', required=True)
        ap.add_argument('--files', type=int, default=200)
        ap.add_argument('--large-classes', type=int, default=5)
        ap.add_argument('--large-class-lines', type=int, default=1500)
        ap.add_argument('--small-func-files', type=int, default=50, help='сколько файлов с мелкими функциями (каждый по 20 функций)')
        ap.add_argument('--seed', type=int, default=42)
        args = ap.parse_args()

        out = pathlib.Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # mix of files: some with big classes, some with many small funcs
        for i in range(args.files):
            module = f"mod_{i}"
            text = build_python_file(module, args.large_classes, args.large_class_lines, small_funcs=4, seed=args.seed)
            write_file(out / f"{module}.py", text)

        for j in range(args.small_func_files):
            module = f"funcpack_{j}"
            text = build_python_file(module, large_classes=0, large_class_lines=0, small_funcs=20, seed=args.seed + 1000)
            write_file(out / f"{module}.py", text)

        print(f"Generated synthetic repo at: {out}")

    if __name__ == "__main__":
        main()
