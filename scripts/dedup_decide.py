"""根据检查结果，决定哪些重复要删，并和第一轮的过滤规则合并。"""
import os
import pandas as pd
from review_lib import build_review

MIN_REPEAT = 3  # 同一张图配了至少 3 种文字 → 占位图；同一句话配了至少 3 张图 → 套话

df = pd.read_csv(os.path.expanduser("~/work/meta/meta_dedup.csv"),
                 dtype={"key": str, "md5": str, "dhash": str})
total = len(df)

# 同图异文：数一数每张图（按 md5）配了几种不同的文字
caps_per_img = df.groupby("md5")["caption_norm"].transform("nunique")
df["rule_placeholder_image"] = caps_per_img >= MIN_REPEAT

# 同文异图：数一数每句话配了几张不同的图
imgs_per_cap = df.groupby("caption_norm")["md5"].transform("nunique")
df["rule_boilerplate_text"] = imgs_per_cap >= MIN_REPEAT

# 完全相同的文件：每组保留第一条
df["rule_dup_exact"] = df["dup_exact_image"]

dedup_cols = ["rule_placeholder_image", "rule_boilerplate_text", "rule_dup_exact"]
print("【去重规则】")
for col in dedup_cols:
    print(f"{col}：{df[col].sum()} 条")

all_rules = ["rule_small_image", "rule_tiny_file", "rule_few_words"] + dedup_cols
df["keep"] = ~df[all_rules].any(axis=1)
removed = (~df["keep"]).sum()
print(f"\n【合计】规则过滤 + 去重，删除 {removed} 条（{removed / total:.2%}），保留 {df['keep'].sum()} 条")

df.to_csv(os.path.expanduser("~/work/meta/meta_stage1.csv"), index=False)

# 检查网页：被这两条新规则删掉的，是不是都是垃圾？
build_review({
    "【被删】同图异文（占位图）": df[df["rule_placeholder_image"]],
    "【被删】同文异图（套话）": df[df["rule_boilerplate_text"]].sort_values("caption_norm"),
}, "review_stage1.html")
