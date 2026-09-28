"""Ablation study：LSTM-AE 與 Transformer-AE，各組實驗回答一個問題。

    G0 統計基準    深度模型有沒有比簡單方法好？          每杆×每小時×休息日 中位數/MAD、PCA 重建誤差（線性 AE）
                   （原訂 Isolation Forest，但本機應用程式控制原則封鎖了 scikit-learn 的 neighbors DLL，改用 numpy 實作的 PCA）
    G1 骨幹        LSTM vs Transformer                   其他條件相同（以 G2 的 ft_pole 比較）
    G2 訓練策略    兩階段 fine-tune 有沒有必要？          no_update / b_only / joint / ft_pole / ft_decoder
    G3 迴路知識    階段一學迴路組成，對 B 期有沒有幫助？  λ（知識轉移權重）= 0 / 0.5 / 1.0
    G4 條件輸入    讓模型知道路燈排程有沒有用？           full / no_light / none
    G5 視窗長度    要看多長的歷史？                       12 / 24 / 48 小時
    G6 混合版本    統計基準 + 深度模型能否勝過兩者單獨？  H1 殘差輸入（feat=resid）、H2 分數融合（fusion）

注意：目前架構兩分支不共用參數，第二階段只算杆體 loss，迴路分支本來就收不到梯度，
「凍結迴路分支」與「不凍結」結果相同，因此 G2 改比較 fine-tune 杆體分支的哪個部分。

評估：閾值 = 校準段乾淨時段第 99 百分位（max 分數，同 evaluate.py）。
    *_val  ：校準段注入模擬異常（選設定看這個）
    *_test ：測試段注入模擬異常（最後確認）
每個設定 3 個種子。結果逐次寫入 outputs/ablation/results.csv，可中斷後接續。

用法：
    uv run python scripts/ablation.py            # 跑全部（已完成的會跳過）
    uv run python scripts/ablation.py --summary  # 只彙整結果
"""

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
import evaluate as E
import features as F
from model import TwoBranchAE, masked_mse, set_frozen
from windows import WindowSet

AB_DIR = config.OUT_DIR / "ablation"
RESULTS = AB_DIR / "results.csv"
FEAT_DIR = config.OUT_DIR / "features"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2]
STAGE1 = dict(lr=1e-3, epochs=80, patience=8)
STAGE2 = dict(lr=3e-4, epochs=40, patience=6)
COND_KEEP = {
    "full": [1] * len(F.COND_FEATURES),
    "no_light": [0 if c in ("light_level_f", "m_light_level") else 1 for c in F.COND_FEATURES],
    "none": [0] * len(F.COND_FEATURES),
}


@dataclass(frozen=True)
class Cfg:
    group: str
    backbone: str            # lstm / transformer / stat / iforest
    strategy: str = "ft_pole"
    lam: float = 0.5
    cond: str = "full"
    window: int = 96
    feat: str = "raw"        # raw = 逐杆 robust scaling；resid = 統計基準殘差（H1）
    fusion: bool = False     # True = 與統計基準分數取最大值（H2）

    @property
    def name(self):
        if self.backbone in ("stat", "pca"):
            return self.backbone
        base = f"{self.backbone}|{self.strategy}|lam{self.lam}|{self.cond}|w{self.window}"
        return base + ("|resid" if self.feat == "resid" else "") + ("|fusion" if self.fusion else "")


def configs() -> list[Cfg]:
    out = [Cfg("G0", "stat"), Cfg("G0", "pca")]
    for bb in ["lstm", "transformer"]:
        for st in ["no_update", "b_only", "joint", "ft_pole", "ft_decoder"]:
            out.append(Cfg("G2", bb, strategy=st))
        out += [Cfg("G3", bb, lam=0.0), Cfg("G3", bb, lam=1.0)]
        out += [Cfg("G4", bb, cond="no_light"), Cfg("G4", bb, cond="none")]
        out += [Cfg("G5", bb, window=48), Cfg("G5", bb, window=192)]
    # G6 混合版本：只用 LSTM（使用者決定不再使用 Transformer）；視窗 12 小時（G5 在驗證集上最佳），另留 24 小時對照
    out += [Cfg("G6", "lstm", window=48, feat="resid"), Cfg("G6", "lstm", window=96, feat="resid"),
            Cfg("G6", "lstm", window=48, fusion=True), Cfg("G6", "lstm", window=48, feat="resid", fusion=True)]
    return out


# ---------------------------------------------------------------------------- 訓練
def new_model(cfg: Cfg) -> TwoBranchAE:
    return TwoBranchAE(len(F.POLE_FEATURES), len(F.CIRCUIT_FEATURES), len(F.COND_FEATURES), len(config.POLES),
                       hidden=64, z_dim=16, emb_dim=4, backbone=cfg.backbone, cond_keep=COND_KEEP[cfg.cond]).to(DEVICE)


def run_epoch(model, loader, use_circ, lam, opt=None):
    model.train(opt is not None)
    tot, n = 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for b in loader:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            out = model(b["xp"], b["mp"], b["cond"], b["pid"], b["xc"] if use_circ else None, b["mc"] if use_circ else None)
            loss = masked_mse(out["pole"], b["xp"], b["mp"])
            if use_circ:
                loss = loss + masked_mse(out["circ"], b["xc"], b["mc"])
                if lam > 0:
                    loss = loss + lam * masked_mse(out["circ_from_pole"], b["xc"], b["mc"])
            if opt is not None:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(b["pid"]); n += len(b["pid"])
    return tot / max(n, 1)


def train(model, feats, tr_splits, va_splits, use_circ, lam, window, seed, lr, epochs, patience):
    tr = WindowSet(feats, tr_splits, train=True, window=window)
    va = WindowSet(feats, va_splits, train=True, window=window)
    tl = DataLoader(tr, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed))
    vl = DataLoader(va, batch_size=256)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    best, best_state, best_ep = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        run_epoch(model, tl, use_circ, lam, opt)
        v = run_epoch(model, vl, use_circ, lam)
        if v < best - 1e-5:
            best, best_ep, best_state = v, ep, copy.deepcopy(model.state_dict())
        if ep - best_ep >= patience:
            break
    model.load_state_dict(best_state)
    return best_ep


_stage1_cache: dict = {}


def build_model(cfg: Cfg, seed: int, feats) -> tuple[TwoBranchAE, dict]:
    torch.manual_seed(seed); np.random.seed(seed)
    info = {}
    if cfg.strategy == "b_only":
        m = new_model(cfg)
        info["ep1"] = train(m, feats, ["B_adapt"], ["B_cal"], False, 0, cfg.window, seed, **STAGE1)
        return m, info
    if cfg.strategy == "joint":
        m = new_model(cfg)
        info["ep1"] = train(m, feats, ["A_train", "B_adapt"], ["A_cal", "B_cal"], True, cfg.lam, cfg.window, seed, **STAGE1)
        return m, info
    key = (cfg.backbone, cfg.lam, cfg.cond, cfg.window, cfg.feat, seed)  # 三種策略共用同一個階段一
    if key not in _stage1_cache:
        m = new_model(cfg)
        ep = train(m, feats, ["A_train"], ["A_cal"], True, cfg.lam, cfg.window, seed, **STAGE1)
        _stage1_cache[key] = (copy.deepcopy(m.state_dict()), ep)
    m = new_model(cfg)
    state, info["ep1"] = _stage1_cache[key]
    m.load_state_dict(state)
    if cfg.strategy == "no_update":
        return m, info
    set_frozen(m.circuit_modules(), True)
    if cfg.strategy == "ft_decoder":  # 只調解碼器：編碼器與 embedding 也凍結，測試 A 期學到的表示能否直接沿用
        set_frozen([m.emb_p, m.enc_p], True)
    torch.manual_seed(seed)
    info["ep2"] = train(m, feats, ["B_adapt"], ["B_cal"], False, 0, cfg.window, seed, **STAGE2)
    return m, info


# ---------------------------------------------------------------------------- 評估
def eval_sets(feat: str = "raw"):
    """feat = raw：原本的特徵檔；resid：統計基準殘差特徵檔（H1）。注入位置與事件清單完全相同。"""
    ev = pd.read_csv(config.OUT_DIR / "injected_events.csv", parse_dates=["start", "end"])
    tag = "" if feat == "raw" else "_resid"
    base = pd.read_parquet(FEAT_DIR / f"features{tag}.parquet")
    sets = {}
    for p in "AB":
        sets[p] = {
            "cal": base,
            f"{p}_val": (pd.read_parquet(FEAT_DIR / f"features{tag}_val_injected_{p}.parquet"), f"{p}_cal", ev[ev.split == f"{p}_cal"]),
            f"{p}_test": (pd.read_parquet(FEAT_DIR / f"features{tag}_injected_{p}.parquet"), f"{p}_test", ev[ev.split == f"{p}_test"]),
        }
    return sets


def summarize(test: pd.DataFrame, thr: float, events: pd.DataFrame, col: str) -> dict:
    m, evt = E.metrics(test, thr, events, col)
    rec = evt.groupby(["kind", "magnitude"]).detected.mean().round(3)
    return {"auroc": m["auroc"], "event_recall": m["event_recall"], "bin_recall": m["bin_recall"], "fpr": m["fpr"],
            "fa_per_pole_week": m["false_alarm_events_per_pole_week"],
            "recall_by_kind": json.dumps({f"{k}|{g}": v for (k, g), v in rec.items()})}


def fuse(model_df: pd.DataFrame, stat_df: pd.DataFrame) -> pd.DataFrame:
    """H2：兩邊各自正規化（÷ 該杆該特徵在校準段的第 99 百分位）後的分數取最大值。"""
    m = model_df.merge(stat_df[["polename", "ts", "s"]].rename(columns={"s": "s_stat"}), on=["polename", "ts"], how="left")
    m["s"] = np.fmax(m["s"], m["s_stat"])
    return m


def eval_model(model, cfg: Cfg, sets, raw_sets=None) -> dict:
    model.eval()
    out = {}
    for p in "AB":
        if p == "A" and cfg.strategy == "b_only":
            continue  # 迴路分支沒訓練過，A 期結果沒有意義
        use_circ = p == "A"
        cal = E.score(model, sets[p]["cal"], [f"{p}_cal"], use_circ, window=cfg.window)
        norm = E.fit_norm(cal)
        cal["s"] = E.apply_norm(cal, norm)
        if cfg.fusion:
            fitted = {}
            stat_cal = baseline_scores("stat", p, raw_sets[p]["cal"], f"{p}_cal", fitted, 0)
            stat_norm = E.fit_norm(stat_cal)
            stat_cal["s"] = E.apply_norm(stat_cal, stat_norm)
            cal = fuse(cal, stat_cal)
        thr = float(cal.loc[cal.label == "normal", "s"].quantile(E.THRESH_Q))
        for name in (f"{p}_val", f"{p}_test"):
            feats, split, events = sets[p][name]
            t = E.score(model, feats, [split], use_circ, window=cfg.window)
            t["s"] = E.apply_norm(t, norm)
            if cfg.fusion:
                st = baseline_scores("stat", p, raw_sets[p][name][0], split, fitted, 0)
                st["s"] = E.apply_norm(st, stat_norm)
                t = fuse(t, st)
            out[name] = summarize(t, thr, events, "s")
    return out


def baseline_scores(kind: str, p: str, feats: pd.DataFrame, split: str, fitted: dict, seed: int) -> pd.DataFrame:
    """統計基準 / Isolation Forest：輸出與 E.score 相同格式（err_* 欄位 + score_pole），可共用正規化與閾值流程。"""
    use = ["w_total_s"] + (F.CIRCUIT_FEATURES if p == "A" else [])
    d = feats[feats.split == split].copy()
    d["hour"] = d.ts.dt.hour
    res = d[["polename", "ts", "split", "label"] + (["inj_event_id"] if "inj_event_id" in d else [])].copy()
    if kind == "stat":
        if "stat" not in fitted:
            tr = feats[(feats.split == config.TRAIN_SPLITS[p]) & (feats.label == "normal")].assign(hour=lambda x: x.ts.dt.hour)
            g = tr.groupby(["polename", "hour", "offday_f"])
            med = g[use].median()
            mad = g[use].agg(lambda s: (s - s.median()).abs().median()) * 1.4826
            fitted["stat"] = (med, mad.clip(lower=0.1))
        med, mad = fitted["stat"]
        key = pd.MultiIndex.from_frame(d[["polename", "hour", "offday_f"]])
        for c in use:
            z = (d[c].values - med[c].reindex(key).values) / mad[c].reindex(key).values
            res[f"err_{c}"] = np.where(d[f"m_{c}"].values > 0, z ** 2, np.nan)
        res["score_pole"] = np.where(d["m_w_total_s"].values > 0, res["err_w_total_s"], np.nan)
    else:  # pca：線性 autoencoder。輸入 = 前後 1 小時（5 個時段）的特徵 + 條件 + 杆 one-hot，保留 90% 變異
        lags = [-2, -1, 0, 1, 2]
        def X(df):
            df = df.sort_values(["polename", "ts"])
            parts = [df.groupby("polename")[use].shift(-l).fillna(0).to_numpy() for l in lags]
            return df, np.hstack(parts + [df[F.COND_FEATURES].to_numpy(), np.eye(len(config.POLES))[df.pole_id.to_numpy()]])
        if "pca" not in fitted:
            tr = feats[(feats.split == config.TRAIN_SPLITS[p]) & (feats.label == "normal") & (feats.m_w_total_s > 0)]
            _, Xt = X(tr)
            mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
            _, S, Vt = np.linalg.svd((Xt - mu) / sd, full_matrices=False)
            k = int(np.searchsorted(np.cumsum(S ** 2) / np.sum(S ** 2), 0.90)) + 1
            fitted["pca"] = (mu, sd, Vt[:k])
        mu, sd, V = fitted["pca"]
        d, Xd = X(d)
        Z = (Xd - mu) / sd
        err = (Z - Z @ V.T @ V) ** 2
        res = d[res.columns.intersection(d.columns)].copy()
        center = lags.index(0) * len(use)
        for j, c in enumerate(use):
            res[f"err_{c}"] = np.where(d[f"m_{c}"].values > 0, err[:, center + j], np.nan)
        res["score_pole"] = np.where(d["m_w_total_s"].values > 0, res["err_w_total_s"], np.nan)
    return res.reset_index(drop=True)


def eval_baseline(cfg: Cfg, sets, seed) -> dict:
    out = {}
    for p in "AB":
        fitted = {}
        cal = baseline_scores(cfg.backbone, p, sets[p]["cal"], f"{p}_cal", fitted, seed)
        norm = E.fit_norm(cal)
        cal["s"] = E.apply_norm(cal, norm)
        thr = float(cal.loc[cal.label == "normal", "s"].quantile(E.THRESH_Q))
        for name in (f"{p}_val", f"{p}_test"):
            feats, split, events = sets[p][name]
            t = baseline_scores(cfg.backbone, p, feats, split, fitted, seed)
            t["s"] = E.apply_norm(t, norm)
            out[name] = summarize(t, thr, events, "s")
    return out


# ---------------------------------------------------------------------------- 主流程
def load_results() -> pd.DataFrame:
    """讀 results.csv。早期版本在沒有 best_ep1 / best_ep2 時少寫欄位，依欄位數補回 NaN 對齊。"""
    import csv
    with open(RESULTS, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    i1 = header.index("best_ep1")
    fixed = []
    for r in body:
        missing = len(header) - len(r)
        if missing == 1:    # 缺 best_ep2
            r = r[:i1 + 1] + [""] + r[i1 + 1:]
        elif missing == 2:  # 兩個都缺
            r = r[:i1] + ["", ""] + r[i1:]
        fixed.append(r)
    df = pd.DataFrame(fixed, columns=header)
    for c in ["lam", "window", "seed", "seconds", "best_ep1", "best_ep2", "auroc", "event_recall", "bin_recall", "fpr",
              "fa_per_pole_week"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def done_keys() -> set:
    if not RESULTS.exists():
        return set()
    r = load_results()
    return set(zip(r.config, r.seed))


def run_all():
    AB_DIR.mkdir(parents=True, exist_ok=True)
    feats = {"raw": pd.read_parquet(FEAT_DIR / "features.parquet"),
             "resid": pd.read_parquet(FEAT_DIR / "features_resid.parquet")}
    sets = {"raw": eval_sets("raw"), "resid": eval_sets("resid")}
    done = done_keys()
    todo = [(c, s) for c in configs() for s in SEEDS if (c.name, s) not in done]
    # 依 (骨幹, λ, 條件, 視窗, 種子) 排序，讓共用階段一的策略連續執行
    todo.sort(key=lambda cs: (cs[0].backbone, cs[0].lam, cs[0].cond, cs[0].window, cs[0].feat, cs[1], cs[0].strategy))
    print(f"待跑 {len(todo)} 組（已完成 {len(done)}）", flush=True)
    for i, (cfg, seed) in enumerate(todo, 1):
        t0 = time.time()
        if cfg.backbone in ("stat", "pca"):
            res, info = eval_baseline(cfg, sets["raw"], seed), {}
        else:
            model, info = build_model(cfg, seed, feats[cfg.feat])
            res = eval_model(model, cfg, sets[cfg.feat], sets["raw"])
        secs = round(time.time() - t0, 1)
        rows = [{**asdict(cfg), "config": cfg.name, "seed": seed, "eval_set": k, "seconds": secs,
                 "best_ep1": info.get("ep1"), "best_ep2": info.get("ep2"), **v} for k, v in res.items()]
        pd.DataFrame(rows).to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
        b = res.get("B_val", {})
        print(f"[{i}/{len(todo)}] {cfg.group} {cfg.name} seed={seed} {secs}s | B_val AUROC={b.get('auroc', float('nan')):.3f} "
              f"recall={b.get('event_recall', float('nan')):.2f}", flush=True)


def summary():
    r = load_results()
    metrics = ["auroc", "event_recall", "fpr", "fa_per_pole_week"]
    agg = r.groupby(["group", "config", "eval_set"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(AB_DIR / "summary.csv", index=False)
    fmt = lambda row, m: f"{row[m + '_mean']:.3f}±{0 if pd.isna(row[m + '_std']) else row[m + '_std']:.3f}"
    for es in ["B_val", "B_test", "A_val", "A_test"]:
        s = agg[agg.eval_set == es].copy()
        if s.empty:
            continue
        s["AUROC"] = s.apply(lambda x: fmt(x, "auroc"), axis=1)
        s["recall"] = s.apply(lambda x: fmt(x, "event_recall"), axis=1)
        s["FPR"] = s.apply(lambda x: fmt(x, "fpr"), axis=1)
        print(f"\n===== {es}")
        print(s.sort_values(["group", "config"])[["group", "config", "AUROC", "recall", "FPR"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if not a.summary:
        run_all()
    summary()
