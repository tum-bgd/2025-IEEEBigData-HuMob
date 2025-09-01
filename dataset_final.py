import copy
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


import copy, numpy as np, pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

class HuMobDatasetTrain(Dataset):
    MASK_ID     = 201
    BLOCK_LEN   = 15
    TOTAL_DAYS  = 75
    def __init__(self, csv_path, seed: int = 17):
        self.seed = seed
        self.df   = pd.read_csv(csv_path)

        (self.d_arr, self.t_arr,  self.inx_arr, self.iny_arr,
         self.td_arr, self.lbx_arr, self.lby_arr,
         self.dx_sign_arr, self.dy_sign_arr,
         self.dx_cat_arr,  self.dy_cat_arr,
         self.len_arr) = ([], [], [], [], [], [], [], [], [], [], [], [])

        for _, traj in tqdm(self.df.groupby('uid')):
            d  = traj['d'].to_numpy()
            t  = traj['t'].to_numpy()
            x  = traj['x'].to_numpy()
            y  = traj['y'].to_numpy()

            td = np.insert(
                (d[1:]*48 + t[1:]) - (d[:-1]*48 + t[:-1]), 0, 0)
            td[td > 47] = 47

            dx_sign = traj['dx_sign_cat'].to_numpy()
            dy_sign = traj['dy_sign_cat'].to_numpy()
            dx_cat  = traj['abs_dx_cat'].to_numpy()
            dy_cat  = traj['abs_dy_cat'].to_numpy()

            self.d_arr.append(d + 1)
            self.t_arr.append(t + 1)
            self.inx_arr.append(x.copy())
            self.iny_arr.append(y.copy())
            self.td_arr.append(td)
            self.lbx_arr.append(x - 1)
            self.lby_arr.append(y - 1)
            self.dx_sign_arr.append(dx_sign)
            self.dy_sign_arr.append(dy_sign)
            self.dx_cat_arr.append(dx_cat)
            self.dy_cat_arr.append(dy_cat)
            self.len_arr.append(len(d))

        self.len_arr   = np.asarray(self.len_arr, dtype=np.int64)
        self.base_len  = len(self.d_arr)

    # --------------------------------------------------------
    def __len__(self):
        return self.base_len * 2

    # --------------------------------------------------------
    def __getitem__(self, idx):
        real = idx % self.base_len

        d   = torch.tensor(self.d_arr[real],  dtype=torch.long)
        t   = torch.tensor(self.t_arr[real],  dtype=torch.long)
        x   = torch.tensor(self.inx_arr[real].copy(), dtype=torch.long)
        y   = torch.tensor(self.iny_arr[real].copy(), dtype=torch.long)
        td  = torch.tensor(self.td_arr[real], dtype=torch.long)

        lbx = torch.tensor(self.lbx_arr[real], dtype=torch.long)
        lby = torch.tensor(self.lby_arr[real], dtype=torch.long)

        dxs = torch.tensor(self.dx_sign_arr[real], dtype=torch.long)
        dys = torch.tensor(self.dy_sign_arr[real], dtype=torch.long)
        dxc = torch.tensor(self.dx_cat_arr[real],  dtype=torch.long)
        dyc = torch.tensor(self.dy_cat_arr[real],  dtype=torch.long)

        seq_len = torch.tensor(self.len_arr[real], dtype=torch.long)

        rng   = np.random.RandomState(self.seed + idx)
        if idx < self.base_len:
            start = 60
        else:
            start = rng.randint(0, self.TOTAL_DAYS - self.BLOCK_LEN + 1)
        end = start + self.BLOCK_LEN - 1
        if (end>26)&(start<26):
            end=end+1

        mask = ((d.numpy() - 1) >= start) & ((d.numpy() - 1) <= end)
        x[mask] = self.MASK_ID
        y[mask] = self.MASK_ID

        return {
            'd': d, 't': t,
            'input_x': x, 'input_y': y,
            'time_delta': td,
            'label_x': lbx, 'label_y': lby,
            'dx_sign_cat': dxs, 'dy_sign_cat': dys,
            'dx_cat': dxc,  'dy_cat': dyc,
            'len': seq_len
        }


class HuMobDatasetVal(Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path)

        self.d_array = []
        self.t_array = []
        self.input_x_array = []
        self.input_y_array = []
        self.time_delta_array = []
        self.label_x_array = []
        self.label_y_array = []
        self.len_array = []

        self.dx_sign_cat_array = []
        self.dy_sign_cat_array = []
        self.dx_cat_array = []
        self.dy_cat_array = []

        for uid, traj in tqdm(self.df.groupby('uid')):
            d = traj['d'].to_numpy()
            t = traj['t'].to_numpy()
            input_x = copy.deepcopy(traj['x'].to_numpy())
            input_y = copy.deepcopy(traj['y'].to_numpy())
            time_delta = np.insert((traj['d'].to_numpy()[1:] * 48 + traj['t'].to_numpy()[1:]) - (traj['d'].to_numpy()[:-1] * 48 + traj['t'].to_numpy()[:-1]), 0, 0)
            time_delta[time_delta > 47] = 47
            label_x = traj['x'].to_numpy()
            label_y = traj['y'].to_numpy()
            dx_sign_cat = traj['dx_sign_cat'].to_numpy()
            dy_sign_cat = traj['dy_sign_cat'].to_numpy()
            dx_cat = traj['abs_dx_cat'].to_numpy()
            dy_cat = traj['abs_dy_cat'].to_numpy()

            mask_d_start = 60
            mask_d_end = 74
            need_mask_idx = np.where((d >= mask_d_start) & (d <= mask_d_end))
            input_x[need_mask_idx] = 201
            input_y[need_mask_idx] = 201

            self.d_array.append(d + 1)
            self.t_array.append(t + 1)
            self.input_x_array.append(input_x)
            self.input_y_array.append(input_y)
            self.time_delta_array.append(time_delta)
            self.label_x_array.append(label_x - 1)
            self.label_y_array.append(label_y - 1)
            self.dx_sign_cat_array.append(dx_sign_cat)
            self.dy_sign_cat_array.append(dy_sign_cat)
            self.dx_cat_array.append(dx_cat)
            self.dy_cat_array.append(dy_cat)
            self.len_array.append(len(d))

        self.len_array = np.array(self.len_array, dtype=np.int64)

    def __len__(self):
        return len(self.d_array)

    def __getitem__(self, index):
        d = torch.tensor(self.d_array[index])
        t = torch.tensor(self.t_array[index])
        input_x = torch.tensor(self.input_x_array[index])
        input_y = torch.tensor(self.input_y_array[index])
        time_delta = torch.tensor(self.time_delta_array[index])
        label_x = torch.tensor(self.label_x_array[index])
        label_y = torch.tensor(self.label_y_array[index])
        len = torch.tensor(self.len_array[index])
        dx_sign_cat = torch.tensor(self.dx_sign_cat_array[index])
        dy_sign_cat = torch.tensor(self.dy_sign_cat_array[index])
        dx_cat = torch.tensor(self.dx_cat_array[index])
        dy_cat = torch.tensor(self.dy_cat_array[index])


        return {
            'd': d,
            't': t,
            'input_x': input_x,
            'input_y': input_y,
            'time_delta': time_delta,
            'label_x': label_x,
            'label_y': label_y,
            'dx_sign_cat': dx_sign_cat,
            'dy_sign_cat': dy_sign_cat,
            'dx_cat':dx_cat,
            'dy_cat':dy_cat,
            'len': len
        }
