#!/usr/bin/env python3
# File: scripts/add_headers_simple.py
# Author: Denis Kurka
# Year: 2025
# License: CC0

"""
add_headers_simple.py
Prepend a simple standardized header to source files (.py, .sh, .c, .h).
Behavior:
- Does NOT parse or extract docstrings/comments from files — it simply prepends the header.
- Preserves shebang if present by inserting the header after the shebang line.
- Skips files that already contain the exact skip marker in the first 512 bytes to avoid duplication.

Usage: run from repository root: python scripts/add_headers_simple.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTS = ['.py', '.sh', '.c', '.h']
AUTHOR = 'Denis Kurka'
YEAR = '2025'
LICENSE = 'CC0'
SKIP_MARKER = f'Author: {AUTHOR}'

HEADER_TMPL_PY_SH = """# File: {file}
# Author: {author}
# Year: {year}
# License: {license}

"""
HEADER_TMPL_C = """/*
 * File: {file}
 * Author: {author}
 * Year: {year}
 * License: {license}
 */

"""


def make_header(relpath, ext):
    if ext in ('.py', '.sh'):
        return HEADER_TMPL_PY_SH.format(file=relpath, author=AUTHOR, year=YEAR, license=LICENSE).encode('utf-8')
    else:
        return HEADER_TMPL_C.format(file=relpath, author=AUTHOR, year=YEAR, license=LICENSE).encode('utf-8')


def process_file(path: Path) -> bool:
    ext = path.suffix
    if ext not in EXTS:
        return False
    data = path.read_bytes()
    # avoid duplicating headers: check first 512 bytes
    if SKIP_MARKER.encode('utf-8') in data[:512]:
        return False
    header = make_header(str(path.relative_to(ROOT)).replace('\\', '/'), ext)
    # preserve shebang if present
    if data.startswith(b'#!'):
        # split first line
        parts = data.split(b"\n", 1)
        shebang = parts[0] + b"\n"
        rest = parts[1] if len(parts) > 1 else b""
        new = shebang + header + rest
    else:
        new = header + data
    path.write_bytes(new)
    print(f'Updated: {path.relative_to(ROOT)}')
    return True


def main():
    changed = 0
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix in EXTS:
            try:
                if process_file(p):
                    changed += 1
            except Exception as e:
                print(f'Error processing {p}: {e}', file=sys.stderr)
    print(f'Headers added to {changed} files.')


if __name__ == '__main__':
    main()
