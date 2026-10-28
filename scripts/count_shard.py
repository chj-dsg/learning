import sys
import tarfile
from collections import Counter

for path in sys.argv[1:]:
    keys = set()
    exts = Counter()
    with tarfile.open(path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            key, _, ext = member.name.partition(".")
            keys.add(key)
            exts[ext] += 1
    print(f"{path}: 样本数 {len(keys)}，文件类型 {dict(exts)}")
