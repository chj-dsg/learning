"""把每条数据的信息提取成一行，保存成表格（CSV），方便后面统计。"""
import glob
import io
import json
import os
from urllib.parse import urlparse

import fasttext
import pandas as pd
import webdataset as wds
from PIL import Image

DATA = os.path.expanduser("~/work/cc12m/*.tar")
MODEL = os.path.expanduser("~/work/models/lid.176.bin")
OUT_DIR = os.path.expanduser("~/work/meta")
os.makedirs(OUT_DIR, exist_ok=True)

lid = fasttext.load_model(MODEL)
shards = sorted(glob.glob(DATA))
print("找到分片：", shards)

rows = []
for sample in wds.WebDataset(shards, shardshuffle=False):
    caption = sample["txt"].decode("utf-8", errors="replace")
    meta = json.loads(sample["json"])

    # 语言识别：fastText 不接受换行，先换成空格
    labels, scores = lid.predict(caption.replace("\n", " "))

    # 图片：尝试打开，读出真实尺寸；打不开就记为损坏
    try:
        img = Image.open(io.BytesIO(sample["jpg"]))
        img_w, img_h = img.size
        img_mode = img.mode
        broken = False
    except Exception:
        img_w = img_h = None
        img_mode = None
        broken = True

    rows.append({
        "key": sample["__key__"],
        "caption": caption,
        "n_chars": len(caption),
        "n_words": len(caption.split()),
        "lang": labels[0].replace("__label__", ""),
        "lang_score": float(scores[0]),
        "width": img_w,
        "height": img_h,
        "aspect": img_w / img_h if img_w and img_h else None,
        "mode": img_mode,
        "broken": broken,
        "file_kb": len(sample["jpg"]) / 1024,
        "domain": urlparse(meta.get("url") or "").netloc,
    })
    if len(rows) % 1000 == 0:
        print(f"已处理 {len(rows)} 条")

df = pd.DataFrame(rows)
out = os.path.join(OUT_DIR, "meta.csv")
df.to_csv(out, index=False)
print(f"完成！共 {len(df)} 条，已保存到 {out}")
