import pandas as pd
import numpy as np
from pathlib import Path
import os


base_path = os.getcwd()  # /workspace
src_path = Path(base_path) / "data/cityA-dataset.csv"
out_path = Path(base_path) / "data/cityA-dataset_aug1.csv"


df = pd.read_csv(src_path)

df.reset_index(drop=True, inplace=True)

orig_uid_cnt = df["uid"].nunique()


def mirror_x(sub):
    out = sub.copy()
    out.loc[:, "x"] = 201 - out["x"]
    return out

def mirror_y(sub):
    out = sub.copy()
    out.loc[:, "y"] = 201 - out["y"]
    return out

def rotate_180(sub):
    out = sub.copy()
    out.loc[:, "x"] = 201 - out["x"]
    out.loc[:, "y"] = 201 - out["y"]
    return out

transforms = [mirror_x, mirror_y, rotate_180]


np.random.seed(42)
n = len(df)
choices = np.array([np.random.choice(3, 2, replace=False) for _ in range(n)])


aug1 = df.copy(deep=True)
aug1["uid"] += 100000
for i in range(3):
    m = choices[:, 0] == i
    if m.any():
        xy = transforms[i](aug1.loc[m, ["x", "y"]]).values
        aug1.loc[m, ["x", "y"]] = xy


aug2 = df.copy(deep=True)
aug2["uid"] += 200000
for i in range(3):
    m = choices[:, 1] == i
    if m.any():
        xy = transforms[i](aug2.loc[m, ["x", "y"]]).values
        aug2.loc[m, ["x", "y"]] = xy


final_df = (
    pd.concat([df, aug1, aug2], ignore_index=True)
      .drop_duplicates(subset=["uid", "d", "t", "x", "y"])
      .sort_values(["uid", "d", "t"], ignore_index=True)
)

assert final_df["uid"].nunique() == orig_uid_cnt * 3, \
       f"#UID are not triple: {final_df['uid'].nunique()} vs {orig_uid_cnt*3}"

final_df.to_csv(out_path, index=False, encoding="utf-8")
print(f"file generated: {out_path.resolve()}")
