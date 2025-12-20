#!/usr/bin/env python3
"""
add_headers.py
Add standardized file headers to source files in the repo.
Run without arguments to update files in-place.
"""

import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EXTS = {
    '.py': '#',
    '.sh': '#',
    '.c': '/*',
    '.h': '/*',
}

AUTHOR = 'Denis Kurka'
YEAR = '2025'
LICENSE = 'CC0'

SKIP_MARKER = 'Author: Denis Kurka'


def brief_from_file(path, text, ext):
    # Try to extract a short description from the file's top comments or docstring
    lines = text.splitlines()
    # skip shebang and encoding lines
    idx = 0
    if lines and lines[0].startswith('#!'):
        idx = 1
    if idx < len(lines) and re.match(r"#.*coding[:=]", lines[idx]):
        idx += 1
    # For Python, check for triple-quoted module docstring
    if ext == '.py':
        joined = '\n'.join(lines[idx:idx+10])
        m = re.search(r"^[ \t]*(['\"][\"'])([\s\S]*?)\1", joined)
        if m:
            first = m.group(2).strip().splitlines()[0]
            return first if len(first) < 80 else first[:77] + '...'
    # Fallback: find first comment line
    for ln in lines[idx:idx+8]:
        ln = ln.strip()
        if ln.startswith('#'):
            text = ln.lstrip('#').strip()
            if text:
                return text if len(text) < 80 else text[:77] + '...'
        if ln.startswith('//'):
            text = ln.lstrip('/').strip()
            if text:
                return text
        if ln.startswith('/*'):
            ln = ln.lstrip('/*').strip()
            if ln:
                return ln
    # Otherwise use filename
    return path.name


def make_header(ext, brief):
    if ext in ('.py', '.sh'):
        lines = [f"# File: {brief}", f"# Author: {AUTHOR}", f"# Year: {YEAR}", f"# License: {LICENSE}", ""]
        return '\n'.join(lines) + '\n'
    else:
        lines = ["/*", f" * File: {brief}", f" * Author: {AUTHOR}", f" * Year: {YEAR}", f" * License: {LICENSE}", " */", ""]
        return '\n'.join(lines) + '\n'


def process_file(path: Path):
    ext = path.suffix
    if ext not in EXTS:
        return False
    text = path.read_text(encoding='utf-8')
    if SKIP_MARKER in text.splitlines()[:20]:
        return False
    brief = brief_from_file(path, text, ext)
    header = make_header(ext, brief)

    # For scripts with shebang, preserve shebang on top
    lines = text.splitlines(True)
    insert_at = 0
    if lines and lines[0].startswith('#!'):
        insert_at = 1
    # After shebang, also preserve encoding line for python (# -*- coding: utf-8 -*-)
    if insert_at < len(lines) and re.match(r"#.*coding[:=]", lines[insert_at]):
        insert_at += 1

    new_text = ''.join(lines[:insert_at]) + header + ''.join(lines[insert_at:])
    path.write_text(new_text, encoding='utf-8')
    print(f'Updated: {path.relative_to(ROOT)}')
    return True


def main():
    updated = 0
    for p in ROOT.rglob('*'):
        if p.is_file() and p.suffix in EXTS:
            try:
                if process_file(p):
                    updated += 1
            except Exception as e:
                print(f'Error processing {p}: {e}')
    print(f'Headers added to {updated} files.')


if __name__ == '__main__':
    main()
