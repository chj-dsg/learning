"""把各项指标的分布画成图，保存到 figures/ 文件夹。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(os.path.expanduser("~/work/meta/meta.csv"), dtype={"key": str})
df["short_side"] = df[["width", "height"]].min(axis=1)

out_dir = os.path.expanduser("~/work/learning/figures")
os.makedirs(out_dir, exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. 文字长度（词数）
ax = axes[0, 0]
ax.hist(df["n_words"], bins=80)
median = df["n_words"].median()
ax.axvline(median, color="red", linestyle="--", label=f"median = {median:.0f}")
ax.set_title("Caption length (words)")
ax.set_xlabel("number of words")
ax.set_ylabel("number of samples")
ax.legend()

# 2. 语言把握程度
ax = axes[0, 1]
ax.hist(df["lang_score"], bins=50)
ax.axvline(0.5, color="red", linestyle="--", label="threshold 0.5")
ax.set_title("Language ID confidence")
ax.set_xlabel("lang_score")
ax.legend()

# 3. 文字长度和语言把握程度的关系
ax = axes[0, 2]
ax.scatter(df["n_words"], df["lang_score"], s=3, alpha=0.3)
ax.set_xscale("log")
ax.set_title("Caption length vs. language confidence")
ax.set_xlabel("number of words (log scale)")
ax.set_ylabel("lang_score")

# 4. 图片短边
ax = axes[1, 0]
ax.hist(df["short_side"], bins=60)
ax.set_yscale("log")
ax.set_title("Image short side (pixels)")
ax.set_xlabel("pixels")
ax.set_ylabel("number of samples (log scale)")

# 5. 宽高比
ax = axes[1, 1]
ax.hist(df["aspect"], bins=100)
for ratio, name in [(1.0, "1:1"), (4 / 3, "4:3"), (1.5, "3:2")]:
    ax.axvline(ratio, color="red", linestyle=":", alpha=0.7, label=name)
ax.set_title("Aspect ratio (width / height)")
ax.set_xlabel("aspect")
ax.legend()

# 6. 文件大小
ax = axes[1, 2]
ax.hist(df["file_kb"], bins=100, range=(0, 400))
ax.set_title("File size (KB, only 0-400 shown)")
ax.set_xlabel("KB")

fig.tight_layout()
out = os.path.join(out_dir, "overview.png")
fig.savefig(out, dpi=100)
print("图已保存到", out)
