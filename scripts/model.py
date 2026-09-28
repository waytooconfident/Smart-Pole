"""雙分支 LSTM-Autoencoder。

    迴路分支（circuit）：迴路特徵 → enc_c → z_c → dec_c → 重建迴路特徵
    杆體分支（pole）：   杆體特徵 → enc_p → z_p → dec_p → 重建杆體特徵
                                         z_p → dec_c_from_p → 推估迴路特徵（知識轉移）

- 兩個解碼器每一步都吃條件輸入（時間、節日、路燈亮度），模型知道「此刻該不該亮燈」。
- 兩分支各有自己的 pole embedding，凍結迴路分支時不受杆體分支更新影響。
- 階段一（A_train）：三個輸出一起訓練。
- 階段二（B_adapt）：凍結迴路分支（enc_c / dec_c / emb_c）與 dec_c_from_p，只 fine-tune 杆體分支。
  B 期沒有迴路資料，推論時只用杆體分支；dec_c_from_p 仍可給出各迴路的推估值。
"""

import torch
from torch import nn


class SeqEncoder(nn.Module):
    def __init__(self, n_in: int, hidden: int, z_dim: int, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.to_z = nn.Linear(hidden, z_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.to_z(h[-1])


class SeqDecoder(nn.Module):
    def __init__(self, z_dim: int, n_cond: int, emb_dim: int, hidden: int, n_out: int, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(z_dim + n_cond + emb_dim, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, n_out)

    def forward(self, z, cond, emb):
        T = cond.shape[1]
        rep = torch.cat([z, emb], dim=-1).unsqueeze(1).expand(-1, T, -1)
        h, _ = self.lstm(torch.cat([rep, cond], dim=-1))
        return self.out(h)


class TfEncoder(nn.Module):
    """Transformer 編碼器：逐步投影 + 位置編碼 → self-attention → 時間平均 → z（與 LSTM 版相同的瓶頸）。"""
    def __init__(self, n_in: int, d: int, z_dim: int, layers: int = 2, heads: int = 4, dropout: float = 0.1,
                 max_len: int = 512):
        super().__init__()
        self.inp = nn.Linear(n_in, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.to_z = nn.Linear(d, z_dim)

    def forward(self, x):
        T = x.shape[1]
        h = self.inp(x) + self.pos(torch.arange(T, device=x.device))
        return self.to_z(self.enc(h).mean(dim=1))


class TfDecoder(nn.Module):
    """Transformer 解碼器：每一步輸入 [z, embedding, 條件] → self-attention → 輸出。只能從 z 取得該視窗的資訊。"""
    def __init__(self, z_dim: int, n_cond: int, emb_dim: int, d: int, n_out: int, layers: int = 2, heads: int = 4,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.inp = nn.Linear(z_dim + n_cond + emb_dim, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, n_out)

    def forward(self, z, cond, emb):
        T = cond.shape[1]
        rep = torch.cat([z, emb], dim=-1).unsqueeze(1).expand(-1, T, -1)
        h = self.inp(torch.cat([rep, cond], dim=-1)) + self.pos(torch.arange(T, device=cond.device))
        return self.out(self.enc(h))


class TwoBranchAE(nn.Module):
    def __init__(self, n_pole: int, n_circ: int, n_cond: int, n_poles: int,
                 hidden: int = 64, z_dim: int = 16, emb_dim: int = 4, backbone: str = "lstm",
                 cond_keep=None):
        super().__init__()
        self.emb_p = nn.Embedding(n_poles, emb_dim)
        self.emb_c = nn.Embedding(n_poles, emb_dim)
        # 條件輸入遮罩（ablation 用）：1 = 使用，0 = 拿掉；不存入 checkpoint
        keep = torch.ones(n_cond) if cond_keep is None else torch.as_tensor(cond_keep, dtype=torch.float32)
        self.register_buffer("cond_keep", keep, persistent=False)
        if backbone == "lstm":
            enc = lambda n_in: SeqEncoder(n_in, hidden, z_dim)
            dec = lambda n_out: SeqDecoder(z_dim, n_cond, emb_dim, hidden, n_out)
        elif backbone == "transformer":
            enc = lambda n_in: TfEncoder(n_in, hidden, z_dim)
            dec = lambda n_out: TfDecoder(z_dim, n_cond, emb_dim, hidden, n_out)
        else:
            raise ValueError(backbone)
        # 編碼器輸入 = 特徵 + 遮罩 + 條件 + embedding
        self.enc_p = enc(2 * n_pole + n_cond + emb_dim)
        self.enc_c = enc(2 * n_circ + n_cond + emb_dim)
        self.dec_p = dec(n_pole)
        self.dec_c = dec(n_circ)
        self.dec_c_from_p = dec(n_circ)

    def pole_modules(self):
        return [self.emb_p, self.enc_p, self.dec_p]

    def circuit_modules(self):
        return [self.emb_c, self.enc_c, self.dec_c, self.dec_c_from_p]

    def forward(self, xp, mp, cond, pid, xc=None, mc=None):
        T = cond.shape[1]
        cond = cond * self.cond_keep
        ep = self.emb_p(pid)
        zp = self.enc_p(torch.cat([xp * mp, mp, cond, ep.unsqueeze(1).expand(-1, T, -1)], dim=-1))
        out = {"pole": self.dec_p(zp, cond, ep), "circ_from_pole": self.dec_c_from_p(zp, cond, ep), "z_p": zp}
        if xc is not None:
            ec = self.emb_c(pid)
            zc = self.enc_c(torch.cat([xc * mc, mc, cond, ec.unsqueeze(1).expand(-1, T, -1)], dim=-1))
            out["circ"] = self.dec_c(zc, cond, ec)
        return out


def set_frozen(modules, frozen: bool) -> None:
    for m in modules:
        for p in m.parameters():
            p.requires_grad = not frozen


def masked_mse(pred, target, mask, reduce: bool = True):
    err = (pred - target) ** 2 * mask
    if reduce:
        return err.sum() / mask.sum().clamp_min(1.0)
    return err  # 逐元素誤差（打分數用）
