"""第 5 步：兩階段訓練。

    階段一：uv run python scripts/train.py --stage 1
        A_train 訓練整個模型（杆體重建 + 迴路重建 + 從杆體推估迴路），A_cal 做 early stopping。
    階段二：uv run python scripts/train.py --stage 2
        載入階段一，凍結迴路分支，只用 B_adapt fine-tune 杆體分支，B_cal 做 early stopping。

輸出：outputs/models/stage{1,2}.pt、stage{1,2}_log.json、stage{1,2}_curve.png
"""

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
import features as F
from model import TwoBranchAE, masked_mse, set_frozen
from windows import WindowSet

MODEL_DIR = config.OUT_DIR / "models"
HPARAMS = dict(hidden=64, z_dim=16, emb_dim=4)
LAMBDA_TRANSFER = 0.5  # 從杆體推估迴路的 loss 權重
STAGE_CFG = {
    1: dict(train=["A_train"], val=["A_cal"], lr=1e-3, epochs=80, patience=8),
    2: dict(train=["B_adapt"], val=["B_cal"], lr=3e-4, epochs=40, patience=6),
}
SEED = 42


def batch_losses(model, b, device, stage):
    b = {k: v.to(device) for k, v in b.items()}
    use_circ = stage == 1
    out = model(b["xp"], b["mp"], b["cond"], b["pid"],
                b["xc"] if use_circ else None, b["mc"] if use_circ else None)
    losses = {"pole": masked_mse(out["pole"], b["xp"], b["mp"])}
    if use_circ:
        losses["circ"] = masked_mse(out["circ"], b["xc"], b["mc"])
        losses["transfer"] = masked_mse(out["circ_from_pole"], b["xc"], b["mc"])
        losses["total"] = losses["pole"] + losses["circ"] + LAMBDA_TRANSFER * losses["transfer"]
    else:
        losses["total"] = losses["pole"]
    return losses


def run_epoch(model, loader, device, stage, opt=None):
    model.train(opt is not None)
    sums, n = {}, 0
    with torch.set_grad_enabled(opt is not None):
        for b in loader:
            losses = batch_losses(model, b, device, stage)
            if opt is not None:
                opt.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            bs = len(b["pid"])
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item() * bs
            n += bs
    return {k: v / n for k, v in sums.items()}


def evaluate_a_cal(model, feats, device):
    """A_cal 上的杆體 / 迴路重建誤差，用來檢查階段二有沒有造成遺忘。"""
    ds = WindowSet(feats, ["A_cal"], train=True)
    return run_epoch(model, DataLoader(ds, batch_size=256), device, stage=1)


def plot_curve(log, stage, path):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ep = [r["epoch"] for r in log]
    ax.plot(ep, [r["train"]["total"] for r in log], color="#2a78d6", label="訓練")
    ax.plot(ep, [r["val"]["total"] for r in log], color="#eb6834", label="驗證")
    ax.set_xlabel("epoch"); ax.set_ylabel("masked MSE")
    ax.set_title(f"階段 {stage} 訓練曲線", loc="left", fontweight="bold")
    ax.legend(frameon=False); ax.grid(color="#e1e0d9", lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    args = ap.parse_args()
    stage, cfg = args.stage, STAGE_CFG[args.stage]
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    feats = pd.read_parquet(config.OUT_DIR / "features" / "features.parquet")

    model = TwoBranchAE(len(F.POLE_FEATURES), len(F.CIRCUIT_FEATURES), len(F.COND_FEATURES),
                        len(config.POLES), **HPARAMS).to(device)
    before = None
    if stage == 2:
        model.load_state_dict(torch.load(MODEL_DIR / "stage1.pt", map_location=device)["state_dict"])
        before = evaluate_a_cal(model, feats, device)
        set_frozen(model.circuit_modules(), True)  # 凍結迴路層級
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in model.parameters())

    tr = WindowSet(feats, cfg["train"], train=True)
    va = WindowSet(feats, cfg["val"], train=True)
    tl = DataLoader(tr, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    vl = DataLoader(va, batch_size=256)
    print(f"stage {stage} | device={device} | 訓練視窗 {len(tr)} | 驗證視窗 {len(va)} | "
          f"可訓練參數 {n_train:,} / {n_all:,}")

    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=1e-4)
    best, best_ep, log, t0 = float("inf"), 0, [], time.time()
    for ep in range(1, cfg["epochs"] + 1):
        trl = run_epoch(model, tl, device, stage, opt)
        val = run_epoch(model, vl, device, stage)
        log.append({"epoch": ep, "train": trl, "val": val})
        flag = ""
        if val["total"] < best - 1e-5:
            best, best_ep, flag = val["total"], ep, " *"
            torch.save({"state_dict": model.state_dict(), "hparams": HPARAMS, "stage": stage,
                        "features": {"pole": F.POLE_FEATURES, "circuit": F.CIRCUIT_FEATURES, "cond": F.COND_FEATURES}},
                       MODEL_DIR / f"stage{stage}.pt")
        print(f"ep {ep:3d} | train " + " ".join(f"{k}={v:.4f}" for k, v in trl.items()) +
              f" | val {val['total']:.4f}{flag}")
        if ep - best_ep >= cfg["patience"]:
            break

    model.load_state_dict(torch.load(MODEL_DIR / f"stage{stage}.pt", map_location=device)["state_dict"])
    summary = {"stage": stage, "best_epoch": best_ep, "best_val": best, "seconds": round(time.time() - t0, 1),
               "train_windows": len(tr), "val_windows": len(va), "trainable_params": n_train, "log": log}
    if stage == 2:
        after = evaluate_a_cal(model, feats, device)
        summary["a_cal_before"], summary["a_cal_after"] = before, after
        print("A_cal 重建誤差（遺忘檢查） before:", {k: round(v, 4) for k, v in before.items()})
        print("                           after: ", {k: round(v, 4) for k, v in after.items()})
    (MODEL_DIR / f"stage{stage}_log.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    plot_curve(log, stage, MODEL_DIR / f"stage{stage}_curve.png")
    print(f"best epoch {best_ep}, val {best:.4f}, {summary['seconds']}s")


if __name__ == "__main__":
    main()
