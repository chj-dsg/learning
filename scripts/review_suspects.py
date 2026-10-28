"""把几类可疑数据挑出来，生成一个网页，方便逐条查看。"""
import glob
import html
import json
import os

import pandas as pd
import webdataset as wds

df = pd.read_csv(os.path.expanduser("~/work/meta/meta.csv"), dtype={"key": str})
df["short_side"] = df[["width", "height"]].min(axis=1)

N = 20
groups = {
    "随机抽样（对照组：正常数据长什么样）": df.sample(N, random_state=0),
    "短边小于 400 像素": df[df["short_side"] < 400],
    "文件最小的 20 张": df.nsmallest(N, "file_kb"),
    "语言分数最低的 20 条": df.nsmallest(N, "lang_score"),
    "文字少于 3 个词": df[df["n_words"] < 3],
    "文字最长的 20 条": df.nlargest(N, "n_words"),
}

wanted = set()
for g in groups.values():
    wanted.update(g["key"])

out_dir = os.path.expanduser("~/work/review")
img_dir = os.path.join(out_dir, "images")
os.makedirs(img_dir, exist_ok=True)

# 遍历一遍数据，把需要的图片存出来，顺便记下原图尺寸
extra = {}
shards = sorted(glob.glob(os.path.expanduser("~/work/cc12m/*.tar")))
for sample in wds.WebDataset(shards, shardshuffle=False):
    key = sample["__key__"]
    if key in wanted:
        with open(os.path.join(img_dir, key + ".jpg"), "wb") as f:
            f.write(sample["jpg"])
        meta = json.loads(sample["json"])
        extra[key] = (meta.get("original_width"), meta.get("original_height"))

# 生成网页
parts = [
    "<html><head><meta charset='utf-8'><title>可疑数据检查</title><style>",
    "body{font-family:sans-serif;margin:20px}",
    ".grid{display:flex;flex-wrap:wrap;gap:12px}",
    ".card{width:260px;border:1px solid #ccc;padding:8px;font-size:12px}",
    ".card img{width:100%;height:200px;object-fit:contain;background:#eee}",
    ".cap{margin:6px 0;font-size:13px}",
    "</style></head><body>",
]
for name, g in groups.items():
    parts.append(f"<h2>{html.escape(name)}（{len(g)} 条）</h2><div class='grid'>")
    for _, r in g.iterrows():
        ow, oh = extra.get(r["key"], (None, None))
        parts.append(
            "<div class='card'>"
            f"<img src='images/{r['key']}.jpg'>"
            f"<div class='cap'>{html.escape(str(r['caption']))}</div>"
            f"编号 {r['key']} ｜ {r['n_words']} 词 ｜ {r['lang']} {r['lang_score']:.2f}<br>"
            f"尺寸 {r['width']:.0f}×{r['height']:.0f}（原图 {ow}×{oh}）｜ {r['file_kb']:.1f} KB<br>"
            f"来源 {html.escape(str(r['domain']))}"
            "</div>"
        )
    parts.append("</div>")
parts.append("</body></html>")

out = os.path.join(out_dir, "review.html")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(parts))
print(f"已生成 {out}，共 {len(wanted)} 张图片")
