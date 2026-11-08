"""最终过滤：规则 + 去重 + CLIP 门槛。输出 Markdown 格式，可以直接追加进笔记。"""
import os
import pandas as pd

MIN_CLIP = 0.20

# 人工检查结果：每个分数段随机抽样，数出的图文不配条数
manual = {(0.00, 0.18): 15, (0.18, 0.20): 9, (0.20, 0.22): 3, (0.22, 0.24): 3, (0.24, 0.26): 1}

df = pd.read_csv(os.path.expanduser("~/work/meta/meta_clip.csv"),
                 dtype={"key": str, "md5": str, "dhash": str})
rules = ["rule_small_image", "rule_tiny_file", "rule_few_words"]
dedup = ["rule_placeholder_image", "rule_boilerplate_text", "rule_dup_exact"]
before_clip = ~df[rules + dedup].any(axis=1)

print()
print("| 分数段 | 该段共有 | 抽样 | 不配 | 不配比例 |")
print("|---|---|---|---|---|")
for (lo, hi), bad in manual.items():
    n = (before_clip & (df["clip_score"] >= lo) & (df["clip_score"] < hi)).sum()
    k = min(20, n)
    print(f"| {lo:.2f}–{hi:.2f} | {n} | {k} | {bad} | {bad / k:.0%} |")
print()
print(f"- 门槛定为 {MIN_CLIP}：不配比例在这里明显下降。每段只抽了约 20 条，比例是趋势，不是精确值")

df["rule_low_clip"] = df["clip_score"] < MIN_CLIP
df["keep_final"] = before_clip & ~df["rule_low_clip"]

n0 = len(df)
n1 = (~df[rules].any(axis=1)).sum()
n2 = before_clip.sum()
n3 = df["keep_final"].sum()
print("- 留存率漏斗：")
print(f"  - 原始 {n0} 条")
print(f"  - 规则过滤后 {n1} 条（{n1 / n0:.2%}）")
print(f"  - 去重后 {n2} 条（{n2 / n0:.2%}）")
print(f"  - CLIP ≥ {MIN_CLIP} 后 {n3} 条（{n3 / n0:.2%}）")

df.to_csv(os.path.expanduser("~/work/meta/meta_final.csv"), index=False)
