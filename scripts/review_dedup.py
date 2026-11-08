"""查看最大的几个重复组：指纹一样的，是不是真的是同一张图？"""
import os
import pandas as pd
from review_lib import build_review

df = pd.read_csv(os.path.expanduser("~/work/meta/meta_dedup.csv"),
                 dtype={"key": str, "md5": str, "dhash": str})

groups = {}
sizes = df.groupby("dhash")["key"].count().sort_values(ascending=False)
for i, (h, n) in enumerate(sizes[sizes >= 2].head(8).items(), 1):
    groups[f"图片重复组 {i}：共 {n} 张（只显示前 6 张）"] = df[df["dhash"] == h].head(6)

text_sizes = df.groupby("caption_norm")["key"].count().sort_values(ascending=False)
for i, (t, n) in enumerate(text_sizes[text_sizes >= 2].head(3).items(), 1):
    groups[f"文字重复组 {i}：共 {n} 条（只显示前 6 条）"] = df[df["caption_norm"] == t].head(6)

build_review(groups, "review_dedup.html")
