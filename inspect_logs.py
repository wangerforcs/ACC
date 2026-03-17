#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
from collections import Counter
from pathlib import Path

# 匹配：
# Attention-based chunk size: 25
PATTERN = re.compile(r"chunk size:\s*(\d+)")

def extract_chunk_sizes(text: str):
    return [int(x) for x in PATTERN.findall(text)]

def analyze_files(file_paths):
    counter = Counter()
    total = 0

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"[WARN] 文件不存在: {file_path}", file=sys.stderr)
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] 读取失败: {file_path}, error={e}", file=sys.stderr)
            continue

        values = extract_chunk_sizes(content)
        counter.update(values)
        total += len(values)

    return counter, total

def print_report(counter, total):
    if total == 0:
        print("未找到任何 'Attention-based chunk size' 字段。")
        return

    print(f"总计提取到 {total} 条 chunk size 记录\n")
    print(f"{'chunk_size':>12} {'count':>10} {'ratio':>10}")
    print("-" * 34)

    for chunk_size in sorted(counter):
        count = counter[chunk_size]
        ratio = count / total * 100
        print(f"{chunk_size:>12} {count:>10} {ratio:>9.2f}%")

def main():
    if len(sys.argv) < 2:
        print("用法:")
        print(f"  {sys.argv[0]} log1.txt [log2.txt ...]")
        print()
        print("示例:")
        print(f"  {sys.argv[0]} train.log")
        sys.exit(1)

    file_paths = sys.argv[1:]
    counter, total = analyze_files(file_paths)
    print_report(counter, total)

if __name__ == "__main__":
    main()