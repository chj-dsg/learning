"""边界检查：被删的有没有误删？差一点被删的有没有漏网？"""
import os
import pandas as pd
from review_lib import build_review
from filter_rules import MIN_SHORT_SIDE, MIN_FILE_KB, MIN_WORDS

df = pd.read_csv(os.path.expanduser("~/work/meta/meta_filtered.csv"), dtype={"key": str})
N = 20

groups = {
    f"【被删】短边小于 {MIN_SHORT_SIDE}": df[df["rule_small_image"]],
    f"【差一点被删】短边刚过 {MIN_SHORT_SIDE}": df[~df["rule_small_image"]].nsmallest(N, "short_side"),
    f"【被删】文件小于 {MIN_FILE_KB} KB": df[df["rule_tiny_file"]].sort_values("file_kb"),
    f"【差一点被删】文件刚过 {MIN_FILE_KB} KB": df[~df["rule_tiny_file"]].nsmallest(N, "file_kb"),
    f"【被删】去掉 <PERSON> 后少于 {MIN_WORDS} 个词": df[df["rule_few_words"]],
    f"【差一点被删】去掉 <PERSON> 后正好 {MIN_WORDS} 个词": df[df["n_words_clean"] == MIN_WORDS].head(N),
}
build_review(groups, "review_filter.html")
