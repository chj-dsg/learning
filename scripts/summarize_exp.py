"""汇总三组实验、多个随机种子的结果：算平均值和标准差，输出 Markdown 表格。"""
import glob
import os
import re

import pandas as pd

rows = []
for log in glob.glob(os.path.expanduser("~/work/runs/exp_*_s*/out.log")):
    name = os.path.basename(os.path.dirname(log))      # 例如 exp_C_clip30_s1
    group, seed = name.rsplit("_s", 1)
    with open(log) as f:
        line = [l for l in f if "image_to_text_mean_rank" in l][-1]
    metrics = dict(re.findall(r"(\S+): ([\d.]+)", line.split("Eval Epoch: 1")[1]))
    rows.append({"group": group, "seed": int(seed), **{k: float(v) for k, v in metrics.items()}})
df = pd.DataFrame(rows)

groups = {"exp_A_full": "A 全部", "exp_B_random30": "B 随机 30%", "exp_C_clip30": "C CLIP 前 30%"}
metrics = {
    "image_to_text_R@10": ("以图找文 R@10（%）", 100),
    "image_to_text_R@5": ("以图找文 R@5（%）", 100),
    "text_to_image_R@10": ("以文找图 R@10（%）", 100),
    "image_to_text_median_rank": ("以图找文 中位排名", 1),
    "text_to_image_median_rank": ("以文找图 中位排名", 1),
}

print(f"\n每组 {df.groupby('group').size().min()} 个随机种子，平均值 ± 标准差：\n")
print("| 指标 | " + " | ".join(groups.values()) + " |")
print("|---|" + "---|" * len(groups))
for col, (label, scale) in metrics.items():
    fmt = "{:.2f}" if scale == 100 else "{:.0f}"
    cells = []
    for g in groups:
        v = df.loc[df["group"] == g, col] * scale
        cells.append(f"{fmt.format(v.mean())} ± {fmt.format(v.std())}")
    print(f"| {label} | " + " | ".join(cells) + " |")

# 顺便算出 C 组的门槛、B 和 C 的重叠，和导出时的挑法完全一致
pool = pd.read_csv(os.path.expanduser("~/work/meta/pool_clip.csv"), dtype={"key": str})
n = int(len(pool) * 0.3)
top = pool.nlargest(n, "clip_score")
overlap = len(set(pool.sample(n, random_state=0)["key"]) & set(top["key"]))
print(f"\n- C 组门槛（入选的最低 CLIP 分数）：{top['clip_score'].min():.3f}")
print(f"- B、C 两组重叠：{overlap} 条（{overlap / n:.1%}）")
