"""去重：找出完全相同的图片、看起来一样的图片、一样的文字。不删数据，只打标记。"""
import glob
import hashlib
import io
import os

import numpy as np
import pandas as pd
import webdataset as wds
from PIL import Image


def dhash(img_bytes, size=8):
    """感知哈希：缩成 9×8 灰度小图，比较每个像素和右边邻居谁更亮，得到 64 位指纹。"""
    img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((size + 1, size))
    px = np.asarray(img, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


shards = sorted(glob.glob(os.path.expanduser("~/work/cc12m/*.tar")))
rows = []
for sample in wds.WebDataset(shards, shardshuffle=False):
    rows.append({
        "key": sample["__key__"],
        "md5": hashlib.md5(sample["jpg"]).hexdigest(),
        "dhash": dhash(sample["jpg"]),
    })
    if len(rows) % 2000 == 0:
        print(f"已处理 {len(rows)} 条")

hashes = pd.DataFrame(rows)
df = pd.read_csv(os.path.expanduser("~/work/meta/meta_filtered.csv"), dtype={"key": str})
df = df.merge(hashes, on="key")

# 文字规范化：转小写、合并多余空格，再比较
df["caption_norm"] = df["caption"].fillna("").str.lower().str.split().str.join(" ")

# 每组重复里保留第一条，其余标记为重复
df["dup_exact_image"] = df.duplicated("md5", keep="first")
df["dup_similar_image"] = df.duplicated("dhash", keep="first")
df["dup_text"] = df.duplicated("caption_norm", keep="first")

total = len(df)
print(f"\n总条数：{total}")
for col, name in [("dup_exact_image", "完全相同的图片"),
                  ("dup_similar_image", "看起来一样的图片"),
                  ("dup_text", "完全一样的文字")]:
    n = df[col].sum()
    print(f"{name}：多出来的重复 {n} 条（{n / total:.2%}）")

print("\n【看起来一样的图片：最大的 10 个重复组】")
sizes = df.groupby("dhash")["key"].count().sort_values(ascending=False)
for h, n in sizes.head(10).items():
    if n < 2:
        break
    example = str(df.loc[df["dhash"] == h, "caption"].iloc[0])
    print(f"{n} 张  例：{example[:60]}")

out = os.path.expanduser("~/work/meta/meta_dedup.csv")
df.to_csv(out, index=False)
print(f"\n已保存到 {out}")
