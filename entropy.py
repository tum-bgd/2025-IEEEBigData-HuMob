import pandas as pd
import numpy as np
import math
import random
import csv
from collections import defaultdict
from multiprocessing import Pool
import warnings
import os


def lempel_ziv_entropy(seq):
    n = len(seq)
    if n < 2:
        return 0.0

    seq_str = list(map(str, seq))
    buffer = set()
    i = 0
    cnt = 0
    length_sum = 0

    while i < n:
        j = 1
        while i + j <= n and "".join(seq_str[i:i + j]) in buffer:
            j += 1
        substring = "".join(seq_str[i:i + j])
        buffer.add(substring)
        cnt += 1
        length_sum += j
        i += j

    c = length_sum / cnt
    return (math.log(n) / c) / math.log(2)


def compute_lz_spatial_entropy(input_csv, output_txt):
    df = pd.read_csv(input_csv)
    max_y = df['y'].max() + 1
    records = []
    for uid, traj in df.groupby('uid'):
        traj = traj[traj['d'] > 59]
        cells = (traj['x'] * max_y + traj['y']).tolist()
        N = len(cells)
        lz_entropy = lempel_ziv_entropy(cells)
        if N > 1:
            S_norm = lz_entropy / math.log2(N)
        else:
            S_norm = 0.0
        records.append({
            'uid': uid,
            'N': N,
            'lz_spatial_entropy': lz_entropy,
            'S_norm': S_norm
        })
    entropy_df = pd.DataFrame(records)
    entropy_df.to_csv(output_txt, sep='\t', index=False)

    print(f"computation completed, file saved at: {output_txt}")


if __name__ == '__main__':
    base_path = os.getcwd()
    DATA_PATH = os.path.join(base_path, 'data/cityA-dataset.csv')
    output_txt = 'lz_spatial_entropy_normA.csv'
    compute_lz_spatial_entropy(DATA_PATH, output_txt)
