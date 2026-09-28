"""把特徵表切成滑動視窗（每杆、每個切分各自切，不跨切分邊界）。

每個視窗回傳：
    xp, mp   杆體特徵與遮罩     (T, n_pole)
    xc, mc   迴路特徵與遮罩     (T, n_circ)；B 期遮罩全 0
    cond     條件輸入           (T, n_cond)
    pid      杆編號
    row      視窗內每個時段在特徵表中的列號（打分數後對回時段用）

訓練時 loss 遮罩 = 特徵遮罩 × 該時段為乾淨正常（label == normal）；
評估時 loss 遮罩只用特徵遮罩，異常時段照樣打分數。
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config
import features as F


class WindowSet(Dataset):
    def __init__(self, feats: pd.DataFrame, splits, train: bool, window: int = config.WINDOW,
                 stride: int = config.STRIDE, min_normal: float = 0.8):
        feats = feats[feats.split.isin(splits)].sort_values(["split", "polename", "ts"])
        if "label" not in feats:  # 推論用的新資料沒有標籤
            feats = feats.assign(label="normal")
        self.feats = feats
        pole_cols, circ_cols = F.POLE_FEATURES, F.CIRCUIT_FEATURES
        self.xp = feats[pole_cols].to_numpy(np.float32, copy=True)
        self.mp = feats[[f"m_{c}" for c in pole_cols]].to_numpy(np.float32, copy=True)
        self.xc = feats[circ_cols].to_numpy(np.float32, copy=True)
        self.mc = feats[[f"m_{c}" for c in circ_cols]].to_numpy(np.float32, copy=True)
        self.cond = feats[F.COND_FEATURES].to_numpy(np.float32, copy=True)
        self.pid = feats.pole_id.to_numpy(np.int64)
        normal = (feats.label == "normal").to_numpy(np.float32)[:, None]
        if train:  # 只從乾淨正常時段學
            self.mp, self.mc = self.mp * normal, self.mc * normal

        starts = []
        pos = np.arange(len(feats))
        for _, g in feats.groupby(["split", "polename"], sort=False):
            idx = pos[feats.index.get_indexer(g.index)]
            ok = (g.label == "normal").to_numpy()
            offsets = list(range(0, len(g) - window + 1, stride))
            if not train and offsets and offsets[-1] != len(g) - window:
                offsets.append(len(g) - window)  # 評估 / 推論時補最後一個視窗，讓每個時段都有分數
            for s in offsets:
                if not train or ok[s:s + window].mean() >= min_normal:
                    starts.append(idx[s])
        self.starts = np.array(starts)
        self.window = window

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        sl = slice(s, s + self.window)
        return {
            "xp": torch.from_numpy(self.xp[sl]), "mp": torch.from_numpy(self.mp[sl]),
            "xc": torch.from_numpy(self.xc[sl]), "mc": torch.from_numpy(self.mc[sl]),
            "cond": torch.from_numpy(self.cond[sl]), "pid": torch.tensor(self.pid[s]),
            "row": torch.arange(s, s + self.window),
        }
