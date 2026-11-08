"""生成检查网页的通用工具：给几组数据，生成一个能在浏览器里翻看的网页。"""
import glob
import html
import json
import os

import webdataset as wds

SHARDS = os.path.expanduser("~/work/cc12m/*.tar")


def build_review(groups, out_name):
    """groups：{组名: 表格}，out_name：网页的文件名。"""
    out_dir = os.path.expanduser("~/work/review")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    wanted = set()
    for g in groups.values():
        wanted.update(g["key"])

    extra = {}
    for sample in wds.WebDataset(sorted(glob.glob(SHARDS)), shardshuffle=False):
        key = sample["__key__"]
        if key in wanted:
            with open(os.path.join(img_dir, key + ".jpg"), "wb") as f:
                f.write(sample["jpg"])
            meta = json.loads(sample["json"])
            extra[key] = (meta.get("original_width"), meta.get("original_height"))

    parts = [
        "<html><head><meta charset='utf-8'><title>数据检查</title><style>",
        "body{font-family:sans-serif;margin:20px}",
        ".grid{display:flex;flex-wrap:wrap;gap:12px}",
        ".card{width:260px;border:1px solid #ccc;padding:8px;font-size:12px}",
        ".card img{width:100%;height:200px;object-fit:contain;background:#eee}",
        ".cap{margin:6px 0;font-size:13px}",
        ".score{color:#c00;font-weight:bold}",
        "</style></head><body>",
    ]
    for name, g in groups.items():
        parts.append(f"<h2>{html.escape(name)}（{len(g)} 条）</h2><div class='grid'>")
        for _, r in g.iterrows():
            ow, oh = extra.get(r["key"], (None, None))
            card = [
                "<div class='card'>",
                f"<img src='images/{r['key']}.jpg'>",
                f"<div class='cap'>{html.escape(str(r['caption']))}</div>",
            ]
            if "clip_score" in r.index:
                card.append(f"<div class='score'>CLIP 分数 {r['clip_score']:.3f}</div>")
            card.append(
                f"编号 {r['key']} ｜ {r['n_words']} 词 ｜ {r['lang']} {r['lang_score']:.2f}<br>"
                f"尺寸 {r['width']:.0f}×{r['height']:.0f}（原图 {ow}×{oh}）｜ {r['file_kb']:.1f} KB<br>"
                f"来源 {html.escape(str(r['domain']))}</div>"
            )
            parts.append("".join(card))
        parts.append("</div>")
    parts.append("</body></html>")

    out = os.path.join(out_dir, out_name)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"已生成 {out}，共 {len(wanted)} 张图片")
