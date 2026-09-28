"""第 6 步：打分數、定閾值、在測試集上評估。

異常分數（每杆每 15 分鐘）：先取所有涵蓋該時段的視窗、把各特徵重建誤差平均，再用兩種方式合併：
    mean（v1）：對有觀測的特徵取平均；A 期 = 杆體分數 + 迴路分數。
                單一特徵的異常會被其他特徵稀釋（例如夜間熄燈只反映在路燈迴路）。
    max（v2，預設）：每個特徵誤差除以「該杆、該特徵在 cal 段正常時段的第 99 百分位誤差」，
                再取最大值。單一特徵異常不被稀釋，各杆也有各自的誤差尺度
                （SP-02、SP-07 看板開關時間不規律，平常誤差就大）。
    B 期只有杆體特徵（沒有迴路資料）。
閾值：cal 段（A_cal / B_cal）乾淨正常時段分數的第 THRESH_Q 百分位。
評估（測試段注入後的副本）：
    FPR           乾淨正常時段被判異常的比例（節日另列）
    事件 recall    注入事件中至少一個時段超過閾值的比例（依類型 × 幅度）
    時段 recall    注入時段超過閾值的比例
    AUROC         注入時段 vs 乾淨正常時段
    真實異常命中率  規則 / 統計標籤時段被判異常的比例
模型：
    B_test 用階段二（最終模型）
    A_test 同時用階段一、階段二評估，檢查 fine-tune 後迴路層級偵測能力是否退步

用法：
    uv run python scripts/evaluate.py
輸出：
    outputs/eval/scores_{A,B}.parquet、metrics.json、event_recall.csv
"""

import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import config
import features as F
from model import TwoBranchAE, masked_mse
from train import HPARAMS, MODEL_DIR
from windows import WindowSet

EVAL_DIR = config.OUT_DIR / "eval"
FEAT_DIR = config.OUT_DIR / "features"
THRESH_Q = 0.99
# max 分數設定：
#   電壓偏差仍是模型輸入，但不計入異常分數（全區共用訊號、重建誤差吵，不是偵測目標）
#   正規化分母下限 0.25（約半個 robust 尺度），避免近乎常數的迴路（攝影機 ~12W）因 1W 內的緩慢漂移觸發警報
#   注意：下限是看過 A_test / B_test 結果後選的（0.1~0.5 間差異不大）；之後調參應改在 cal 段注入異常來調
SCORE_EXCLUDE = ["err_v_dev_s"]
NORM_FLOOR = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(stage: int) -> TwoBranchAE:
    m = TwoBranchAE(len(F.POLE_FEATURES), len(F.CIRCUIT_FEATURES), len(F.COND_FEATURES),
                    len(config.POLES), **HPARAMS).to(DEVICE)
    m.load_state_dict(torch.load(MODEL_DIR / f"stage{stage}.pt", map_location=DEVICE)["state_dict"])
    return m.eval()


@torch.no_grad()
def score(model: TwoBranchAE, feats: pd.DataFrame, splits, use_circ: bool, window: int = config.WINDOW) -> pd.DataFrame:
    """回傳每個時段的 score_pole、score_circ、score，以及各特徵誤差。"""
    ds = WindowSet(feats, splits, train=False, window=window)
    n, n_p, n_c = len(ds.feats), len(F.POLE_FEATURES), len(F.CIRCUIT_FEATURES)
    sum_p, sum_c = torch.zeros(n, n_p, device=DEVICE), torch.zeros(n, n_c, device=DEVICE)
    res_p, res_c = torch.zeros(n, n_p, device=DEVICE), torch.zeros(n, n_c, device=DEVICE)  # 帶正負號的殘差
    cover = torch.zeros(n, device=DEVICE)
    for b in DataLoader(ds, batch_size=512):
        b = {k: v.to(DEVICE) for k, v in b.items()}
        out = model(b["xp"], b["mp"], b["cond"], b["pid"],
                    b["xc"] if use_circ else None, b["mc"] if use_circ else None)
        rows = b["row"].reshape(-1)
        sum_p.index_add_(0, rows, masked_mse(out["pole"], b["xp"], b["mp"], reduce=False).reshape(-1, n_p))
        res_p.index_add_(0, rows, ((b["xp"] - out["pole"]) * b["mp"]).reshape(-1, n_p))
        if use_circ:
            sum_c.index_add_(0, rows, masked_mse(out["circ"], b["xc"], b["mc"], reduce=False).reshape(-1, n_c))
            res_c.index_add_(0, rows, ((b["xc"] - out["circ"]) * b["mc"]).reshape(-1, n_c))
        cover.index_add_(0, rows, torch.ones_like(rows, dtype=torch.float))
    cover = cover.clamp_min(1).unsqueeze(1)
    err_p, err_c = (sum_p / cover).cpu().numpy(), (sum_c / cover).cpu().numpy()
    sres_p, sres_c = (res_p / cover).cpu().numpy(), (res_c / cover).cpu().numpy()
    mp = ds.feats[[f"m_{c}" for c in F.POLE_FEATURES]].to_numpy()
    mc = ds.feats[[f"m_{c}" for c in F.CIRCUIT_FEATURES]].to_numpy()
    res = ds.feats[[c for c in ["polename", "ts", "split", "label", "label_reason", "inj_event_id"] if c in ds.feats]].copy()
    res["score_pole"] = err_p.sum(1) / np.maximum(mp.sum(1), 1)
    res["score_circ"] = err_c.sum(1) / np.maximum(mc.sum(1), 1) if use_circ else 0.0
    res.loc[mp.sum(1) == 0, "score_pole"] = np.nan  # 電表完全沒資料的時段不打分數
    res["score"] = res.score_pole + res.score_circ
    for i, c in enumerate(F.POLE_FEATURES):
        res[f"err_{c}"] = np.where(mp[:, i] > 0, err_p[:, i], np.nan)
        res[f"res_{c}"] = np.where(mp[:, i] > 0, sres_p[:, i], np.nan)  # > 0：實際高於預期
    if use_circ:
        for i, c in enumerate(F.CIRCUIT_FEATURES):
            res[f"err_{c}"] = np.where(mc[:, i] > 0, err_c[:, i], np.nan)
            res[f"res_{c}"] = np.where(mc[:, i] > 0, sres_c[:, i], np.nan)
    return res.reset_index(drop=True)


def fit_norm(cal: pd.DataFrame) -> dict:
    """每 (杆, 特徵) 在 cal 段乾淨正常時段的第 99 百分位誤差，下限 NORM_FLOOR。"""
    clean = cal[cal.label == "normal"]
    err_cols = [c for c in cal.columns if c.startswith("err_") and c not in SCORE_EXCLUDE]
    norm = {}
    for c in err_cols:
        q = clean.groupby("polename")[c].quantile(0.99)
        norm[c] = {p: max(float(v), NORM_FLOOR) if pd.notna(v) else None for p, v in q.items()}
    return norm


def apply_norm(df: pd.DataFrame, norm: dict) -> pd.Series:
    parts = []
    for c, per_pole in norm.items():
        q = df.polename.map(per_pole).astype(float)
        parts.append(df[c] / q)
    s = pd.concat(parts, axis=1).max(axis=1, skipna=True)
    return s.where(df.score_pole.notna())


def metrics(test: pd.DataFrame, thr: float, events: pd.DataFrame, col: str) -> tuple[dict, pd.DataFrame]:
    test = test.dropna(subset=[col]).copy()
    test["score"] = test[col]
    test["alarm"] = test.score > thr
    test["holiday"] = test.ts.dt.date.isin(config.holiday_dates())
    clean = (test.label == "normal") & test.inj_event_id.isna()
    injected = test.inj_event_id.notna()
    real = test.label.isin(["rule", "stat"]) & test.inj_event_id.isna()

    # 事件層級：事件內任一時段超過閾值即算抓到；延遲 = 第一個警報距事件開始的小時數
    hit = test[injected].groupby("inj_event_id").agg(detected=("alarm", "any"),
                                                     first=("ts", lambda s: s[test.loc[s.index, "alarm"]].min()))
    ev = events.merge(hit, left_on="event_id", right_index=True, how="left")
    ev["detected"] = ev.detected.fillna(False).astype(bool)
    ev["delay_h"] = (pd.to_datetime(ev["first"]) - pd.to_datetime(ev["start"])).dt.total_seconds() / 3600

    # 誤報事件：乾淨正常時段中連續的警報段，換算成每杆每週幾次
    fa = test[clean].sort_values(["polename", "ts"])
    starts = fa.alarm & ~fa.groupby("polename").alarm.shift(fill_value=False)
    weeks = fa.groupby("polename").ts.agg(lambda s: s.dt.date.nunique()).sum() / 7

    y = np.r_[np.ones(injected.sum()), np.zeros(clean.sum())]
    s = np.r_[test.score[injected], test.score[clean]]
    m = {
        "threshold": thr,
        "fpr": float(test.alarm[clean].mean()),
        "fpr_holiday": float(test.alarm[clean & test.holiday].mean()) if (clean & test.holiday).any() else None,
        "fpr_non_holiday": float(test.alarm[clean & ~test.holiday].mean()),
        "false_alarm_events_per_pole_week": float(starts.sum() / weeks),
        "event_recall": float(ev.detected.mean()),
        "bin_recall": float(test.alarm[injected].mean()),
        "median_delay_h": float(ev.delay_h.median()),
        "auroc": float(roc_auc_score(y, s)),
        "real_anomaly_hit_rate": float(test.alarm[real].mean()) if real.any() else None,
        "n_clean_bins": int(clean.sum()), "n_injected_bins": int(injected.sum()), "n_real_bins": int(real.sum()),
        "fpr_by_pole": test[clean].groupby("polename").alarm.mean().round(4).to_dict(),
    }
    return m, ev


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    feats = pd.read_parquet(FEAT_DIR / "features.parquet")
    events = pd.read_csv(config.OUT_DIR / "injected_events.csv", parse_dates=["start", "end"])
    report, ev_tables = {}, []

    cases = [  # (名稱, 時段, 模型階段, 是否用迴路分支)
        ("B_test | 階段二（最終模型）", "B", 2, False),
        ("A_test | 階段一", "A", 1, True),
        ("A_test | 階段二（fine-tune 後）", "A", 2, True),
    ]
    for name, period, stage, use_circ in cases:
        model = load_model(stage)
        cal = score(model, feats, [f"{period}_cal"], use_circ)
        test_feats = pd.read_parquet(FEAT_DIR / f"features_injected_{period}.parquet")
        test = score(model, test_feats, [f"{period}_test"], use_circ)
        norm = fit_norm(cal)
        cal["score_max"], test["score_max"] = apply_norm(cal, norm), apply_norm(test, norm)
        test["score_mean"] = test["score"]
        for col, tag in [("score_mean", "mean"), ("score_max", "max")]:
            thr = float(cal.loc[cal.label == "normal", "score" if col == "score_mean" else col].quantile(THRESH_Q))
            m, ev = metrics(test, thr, events[events.split == f"{period}_test"], col)
            report[f"{name} [{tag}]"] = m
            ev_tables.append(ev.assign(case=f"{name} [{tag}]"))
            test[f"alarm_{tag}"] = test[col] > thr
            if tag == "max" and stage == 2:  # 最終模型的打分參數，推論時載入
                mode = "with_circuit" if use_circ else "pole_only"
                (MODEL_DIR / f"scoring_{mode}.json").write_text(json.dumps(
                    {"mode": mode, "threshold": thr, "norm": norm, "exclude": SCORE_EXCLUDE,
                     "calibrated_on": f"{period}_cal"}, ensure_ascii=False, indent=1), encoding="utf-8")
        test.to_parquet(EVAL_DIR / f"scores_{period}_stage{stage}.parquet", index=False)

    (EVAL_DIR / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    ev_all = pd.concat(ev_tables, ignore_index=True)
    ev_all.to_csv(EVAL_DIR / "event_recall.csv", index=False, encoding="utf-8-sig")

    # ---- 摘要 ----
    keys = ["threshold", "fpr", "fpr_non_holiday", "fpr_holiday", "false_alarm_events_per_pole_week",
            "event_recall", "bin_recall", "median_delay_h", "auroc", "real_anomaly_hit_rate"]
    print(pd.DataFrame({k: {kk: report[k][kk] for kk in keys} for k in report}).round(4).to_string())
    print("\n== 事件 recall：異常類型 × 幅度（max 分數）")
    tab = ev_all[ev_all.case.str.endswith("[max]")].pivot_table(index=["kind"], columns=["case", "magnitude"], values="detected", aggfunc="mean")
    tab = tab.reindex(columns=["small", "medium", "large"], level=1)
    print((tab * 100).round(0).to_string())
    print("\n== 各杆 FPR")
    print(pd.DataFrame({k: report[k]["fpr_by_pole"] for k in report}).round(4).to_string())


if __name__ == "__main__":
    main()
