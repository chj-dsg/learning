"""从训练池里挑出 B 组（随机 30%）和 C 组（CLIP 分数最高的 30%），各自打包成新的分片。A 组直接用原始分片。"""
import os
import pandas as pd
import webdataset as wds

POOL = os.path.expanduser("~/work/cc12m/cc12m-train-{0000..0097}.tar")
OUT = os.path.expanduser("~/work/sets")
FRACTION = 0.3

df = pd.read_csv(os.path.expanduser("~/work/meta/pool_clip.csv"), dtype={"key": str})
n = int(len(df) * FRACTION)
random_keys = set(df.sample(n, random_state=0)["key"])
top = df.nlargest(n, "clip_score")
top_keys = set(top["key"])
print(f"训练池 {len(df)} 条，每组挑 {n} 条")
print(f"C 组的门槛（入选的最低分）：{top['clip_score'].min():.3f}")
print(f"B、C 两组重叠：{len(random_keys & top_keys)} 条")

os.makedirs(f"{OUT}/random30", exist_ok=True)
os.makedirs(f"{OUT}/clip30", exist_ok=True)
writer_b = wds.ShardWriter(f"{OUT}/random30/%04d.tar", maxcount=1000, verbose=0)
writer_c = wds.ShardWriter(f"{OUT}/clip30/%04d.tar", maxcount=1000, verbose=0)

count = 0
for sample in wds.WebDataset(POOL, shardshuffle=False):
    key = sample["__key__"]
    item = {"__key__": key, "jpg": sample["jpg"], "txt": sample["txt"], "json": sample["json"]}
    if key in random_keys:
        writer_b.write(item)
    if key in top_keys:
        writer_c.write(item)
    count += 1
    if count % 50000 == 0:
        print(f"已处理 {count} 条")

writer_b.close()
writer_c.close()
print("导出完成")
