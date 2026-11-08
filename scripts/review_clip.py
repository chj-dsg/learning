"""CLIP 分数段抽样：在低分区域细分，每段随机抽 20 条，数一数图文不配的比例。"""
import os
import pandas as pd
from review_lib import build_review

df = pd.read_csv(os.path.expanduser("~/work/meta/meta_clip.csv"),
                 dtype={"key": str, "md5": str, "dhash": str})
kept = df[df["keep"]]
N = 20

print("规则和去重保留的数据里，分数低于某个门槛的有多少：")
for t in [0.18, 0.20, 0.22, 0.24]:
    n = (kept["clip_score"] < t).sum()
    print(f"  低于 {t}：{n} 条（{n / len(kept):.2%}）")


def band(lo, hi):
    g = kept[(kept["clip_score"] >= lo) & (kept["clip_score"] < hi)]
    return g.sample(min(N, len(g)), random_state=0).sort_values("clip_score")


groups = {
    "低于 0.18": band(0.0, 0.18),
    "0.18 – 0.20": band(0.18, 0.20),
    "0.20 – 0.22": band(0.20, 0.22),
    "0.22 – 0.24": band(0.22, 0.24),
    "0.24 – 0.26": band(0.24, 0.26),
}
build_review(groups, "review_clip.html")
