"""给 10082 条数据做一份整体体检报告。"""
import os
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

df = pd.read_csv(os.path.expanduser("~/work/meta/meta.csv"), dtype={"key": str})
total = len(df)
print(f"总条数：{total}\n")

# 一、数值指标：看整体分布
num_cols = ["n_chars", "n_words", "lang_score", "width", "height", "aspect", "file_kb"]
print("【一、数值指标】")
print(df[num_cols].describe().T.round(2))

# 二、类别指标：每个类别各有多少条
print("\n【二、语言 Top 10】")
print(df["lang"].value_counts().head(10))
print("\n【三、颜色模式】")
print(df["mode"].value_counts())
print("\n【四、来源网站 Top 10】")
print(df["domain"].value_counts().head(10))

# 三、可疑数据：按几条粗略的规则，各数一数有多少
print("\n【五、可疑数据统计】")
checks = {
    "图片损坏": df["broken"],
    "非英文": df["lang"] != "en",
    "语言把握程度低于 0.5": df["lang_score"] < 0.5,
    "文字少于 3 个词": df["n_words"] < 3,
    "文字超过 50 个词": df["n_words"] > 50,
    "图片短边小于 200 像素": df[["width", "height"]].min(axis=1) < 200,
    "宽高比极端（大于 3 或小于 1/3）": (df["aspect"] > 3) | (df["aspect"] < 1/3),
}
for name, mask in checks.items():
    n = int(mask.sum())
    print(f"{name}：{n} 条（{n / total:.2%}）")
