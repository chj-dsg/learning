"""第一版过滤规则：不删数据，只给每条数据打标记，记录它违反了哪些规则。"""
import os
import pandas as pd

df = pd.read_csv(os.path.expanduser("~/work/meta/meta.csv"), dtype={"key": str})
total = len(df)

# ===== 规则参数：想调整门槛，只改这里 =====
MIN_SHORT_SIDE = 224   # 参照 CLIP 的输入分辨率 224×224
MIN_FILE_KB = 2        # 小于 2 KB 的基本都是占位图
MIN_WORDS = 3          # 检查后很准确，保持不变

# ===== 计算规则要用的指标 =====
df["short_side"] = df[["width", "height"]].min(axis=1)
df["n_words_clean"] = (
    df["caption"].fillna("")
    .str.replace("<PERSON>", " ", regex=False)
    .str.split()
    .str.len()
)

# ===== 每条规则：True 表示违反了这条规则 =====
rules = {
    "rule_small_image": df["short_side"] < MIN_SHORT_SIDE,
    "rule_tiny_file": df["file_kb"] < MIN_FILE_KB,
    "rule_few_words": df["n_words_clean"] < MIN_WORDS,
}
for name, hit in rules.items():
    df[name] = hit

rule_cols = list(rules)
df["n_rules_hit"] = df[rule_cols].sum(axis=1)
df["keep"] = df["n_rules_hit"] == 0

# ===== 统计 =====
print(f"总条数：{total}\n")
print("【每条规则命中多少】")
for name in rule_cols:
    hit = df[name]
    only = hit & (df["n_rules_hit"] == 1)
    print(f"{name}：命中 {hit.sum()} 条（{hit.mean():.2%}），其中只违反这一条的 {only.sum()} 条")

removed = (~df["keep"]).sum()
print(f"\n【合计】删除 {removed} 条（{removed / total:.2%}），保留 {df['keep'].sum()} 条")

out = os.path.expanduser("~/work/meta/meta_filtered.csv")
df.to_csv(out, index=False)
print(f"\n带标记的表已保存到 {out}")
