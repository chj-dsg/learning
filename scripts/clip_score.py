"""用 CLIP 给每一对图文打分：分数越高，图和文字越匹配。"""
import glob
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import open_clip
import pandas as pd
import torch
import webdataset as wds
from PIL import Image

BATCH = 256
device = "cuda"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu", pretrained="openai", device=device
)
tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
model.eval()


def score_batch(images, texts):
    """一批图文一起送进显卡，每张图只和它自己的文字比较。"""
    imgs = torch.stack(images).to(device)
    toks = tokenizer(texts).to(device)
    with torch.no_grad():
        iv = model.encode_image(imgs)
        tv = model.encode_text(toks)
    iv = iv / iv.norm(dim=-1, keepdim=True)
    tv = tv / tv.norm(dim=-1, keepdim=True)
    return (iv * tv).sum(dim=-1).tolist()


shards = sorted(glob.glob(os.path.expanduser("~/work/cc12m/*.tar")))
keys, scores = [], []
batch_keys, batch_imgs, batch_txts = [], [], []
for sample in wds.WebDataset(shards, shardshuffle=False):
    batch_keys.append(sample["__key__"])
    batch_imgs.append(preprocess(Image.open(io.BytesIO(sample["jpg"])).convert("RGB")))
    batch_txts.append(sample["txt"].decode("utf-8", errors="replace"))
    if len(batch_keys) == BATCH:
        scores += score_batch(batch_imgs, batch_txts)
        keys += batch_keys
        batch_keys, batch_imgs, batch_txts = [], [], []
        print(f"已打分 {len(keys)} 条")
if batch_keys:
    scores += score_batch(batch_imgs, batch_txts)
    keys += batch_keys

clip = pd.DataFrame({"key": keys, "clip_score": scores})
df = pd.read_csv(os.path.expanduser("~/work/meta/meta_stage1.csv"),
                 dtype={"key": str, "md5": str, "dhash": str})
df = df.merge(clip, on="key")
df.to_csv(os.path.expanduser("~/work/meta/meta_clip.csv"), index=False)

s = df["clip_score"]
print("\n【CLIP 分数的整体分布】")
print(s.describe().round(3))
print(f"\n分数 ≥ 0.28（LAION 的门槛）的比例：{(s >= 0.28).mean():.2%}")
print(f"如果只保留分数最高的 30%，门槛是：{s.quantile(0.7):.3f}")
print(f"\n规则和去重【保留】的数据，平均分：{df.loc[df['keep'], 'clip_score'].mean():.3f}")
print(f"规则和去重【删掉】的数据，平均分：{df.loc[~df['keep'], 'clip_score'].mean():.3f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df.loc[df["keep"], "clip_score"], bins=60, alpha=0.6, density=True, label="kept by rules")
ax.hist(df.loc[~df["keep"], "clip_score"], bins=30, alpha=0.6, density=True, label="removed by rules")
ax.axvline(0.28, color="red", linestyle="--", label="LAION threshold 0.28")
ax.axvline(s.quantile(0.7), color="green", linestyle="--", label="top 30% cutoff")
ax.set_title("CLIP score distribution")
ax.set_xlabel("CLIP score")
ax.set_ylabel("density")
ax.legend()
fig.savefig(os.path.expanduser("~/work/learning/figures/clip_score.png"), dpi=100)
print("\n分布图已保存到 figures/clip_score.png")
