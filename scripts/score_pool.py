"""给训练池（分片 0000–0097）里的每一对图文打 CLIP 分数。用多个进程并行读数据，加快速度。"""
import functools
import io
import os
import time

import open_clip
import pandas as pd
import torch
import webdataset as wds
from PIL import Image

SHARDS = os.path.expanduser("~/work/cc12m/cc12m-train-{0000..0097}.tar")
OUT = os.path.expanduser("~/work/meta/pool_clip.csv")


def to_item(sample, preprocess):
    """在读数据的进程里完成：解码图片、缩放成模型要的尺寸、解码文字。"""
    image = preprocess(Image.open(io.BytesIO(sample["jpg"])).convert("RGB"))
    text = sample["txt"].decode("utf-8", errors="replace")
    return sample["__key__"], image, text


def main():
    device = "cuda"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    model.eval()

    dataset = wds.WebDataset(SHARDS, shardshuffle=False).map(
        functools.partial(to_item, preprocess=preprocess)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=6)

    keys, scores = [], []
    start = time.time()
    with torch.no_grad():
        for batch_keys, images, texts in loader:
            iv = model.encode_image(images.to(device))
            tv = model.encode_text(tokenizer(list(texts)).to(device))
            iv = iv / iv.norm(dim=-1, keepdim=True)
            tv = tv / tv.norm(dim=-1, keepdim=True)
            scores += (iv * tv).sum(dim=-1).tolist()
            keys += list(batch_keys)
            if len(keys) % (256 * 40) == 0:
                speed = len(keys) / (time.time() - start)
                print(f"已打分 {len(keys)} 条，每秒 {speed:.0f} 条")

    print(f"\n一共 {len(keys)} 条，其中不重复的编号 {len(set(keys))} 个")
    pd.DataFrame({"key": keys, "clip_score": scores}).to_csv(OUT, index=False)
    print(f"已保存到 {OUT}")


if __name__ == "__main__":
    main()
