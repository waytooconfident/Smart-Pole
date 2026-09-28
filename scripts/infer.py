"""第 7 步：推論。輸入一段時間的每杆 15 分鐘寬表，輸出異常警報清單（依優先度排序）。

流程：寬表 → 特徵轉換（scalers.json）→ 最終模型（stage2.pt）打分數 → 閾值（scoring_*.json）
      → 連續的警報時段合併成事件 → 標出最可疑的特徵與方向 → 依嚴重度 × 人流排優先度
另外列出電表斷訊（連續 >= 1 小時沒有資料）：模型無法打分數，但維運上需要處理。

有迴路資料（如 1–2 月）時使用杆體 + 迴路分數；沒有迴路資料（如 5–6 月）時只用杆體分數。

用法：
    uv run python scripts/infer.py --start 2026-06-01 --end 2026-07-01
    uv run python scripts/infer.py --start 2026-06-01 --end 2026-07-01 --input 其他寬表.parquet
輸入寬表欄位需與 outputs/pole_ts_15min.parquet 相同（由 build_pole_timeseries.py 產生）。
輸出：
    outputs/inference/alarms_{start}_{end}.csv   警報與斷訊清單
    outputs/inference/scores_{start}_{end}.parquet 每杆每 15 分鐘的分數
"""

import argparse
import json

import numpy as np
import pandas as pd

import config
import evaluate as E
import features as F

INF_DIR = config.OUT_DIR / "inference"
FEATURE_NAMES = {
    "w_total_s": "總功率", "w_signage_s": "數位看板", "w_light_s": "路燈迴路", "w_camera_s": "攝影機",
    "w_indicator_s": "智慧指標", "w_network_s": "網路設備", "w_unmetered_s": "未計量用電",
    "lamp_gap": "燈控與路燈功率落差", "v_dev_s": "電壓",
}
MERGE_GAP_BINS = 1      # 警報之間相隔 <= 1 個時段就併成同一事件
OUTAGE_MIN_BINS = 4     # 斷訊 >= 1 小時才列出
PEOPLE_WEIGHT = 1.0     # 優先度 = 嚴重度 × (1 + PEOPLE_WEIGHT × 人流係數)
FLEET_MIN_POLES = 3     # 同一時間 >= 3 根杆都有警報 → 視為全區同步事件（停電、排程調整等），不當單杆故障


def group_fleet_events(alarms: pd.DataFrame) -> pd.DataFrame:
    """時間重疊、且涉及 >= FLEET_MIN_POLES 根杆的模型警報，合併成一筆「全區同步異常」。"""
    m = alarms[alarms["類型"] == "模型警報"].sort_values("開始").reset_index(drop=True)
    if m.empty:
        return alarms
    cluster, end, cid = [], pd.Timestamp.min, -1
    for s, e in zip(m["開始"], m["結束"]):
        if s > end:
            cid += 1
            end = e
        else:
            end = max(end, e)
        cluster.append(cid)
    m["cluster"] = cluster
    keep, fleet = [], []
    for _, g in m.groupby("cluster"):
        if g["杆"].nunique() >= FLEET_MIN_POLES:
            fleet.append({
                "類型": "全區同步異常", "杆": "、".join(sorted(p.replace("SCCP-", "") for p in g["杆"].unique())),
                "開始": g["開始"].min(), "結束": g["結束"].max(),
                "持續_小時": round((g["結束"].max() - g["開始"].min()) / pd.Timedelta("1h"), 2),
                "嚴重度": g["嚴重度"].max(),
                "最可疑特徵": g["最可疑特徵"].mode().iloc[0],
                "平均人流": None, "優先度分數": None,
            })
        else:
            keep.append(g.drop(columns="cluster"))
    return pd.concat([alarms[alarms["類型"] != "模型警報"], *keep, pd.DataFrame(fleet)], ignore_index=True)


def load_table(path, start, end) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df.polename.isin(config.POLES) & (df.ts >= start) & (df.ts < end)].copy()
    if df.empty:
        raise SystemExit(f"{path} 在 {start} ~ {end} 沒有資料")
    days = (df.ts.max() - df.ts.min()) / pd.Timedelta("1D")
    if days < config.WINDOW / 96:
        raise SystemExit("至少需要 24 小時的資料（模型視窗長度）")
    return df


def people_reference() -> float:
    """人流係數的分母：訓練期間各杆 15 分鐘人流的第 90 百分位。"""
    df = pd.read_parquet(config.OUT_DIR / "pole_ts_labeled.parquet", columns=["split", "people_count"])
    return float(df[df.split.isin(config.TRAIN_SPLITS.values())].people_count.quantile(0.9))


def to_events(scores: pd.DataFrame, flag: pd.Series, gap: int) -> list[pd.DataFrame]:
    """把同一杆連續（允許 gap 個時段空隙）為 True 的時段切成事件。"""
    out = []
    for _, g in scores[flag].groupby("polename"):
        g = g.sort_values("ts")
        brk = (g.ts.diff() > pd.Timedelta(config.FREQ) * (gap + 1)).cumsum()
        out.extend(e for _, e in g.groupby(brk))
    return out


def describe_alarm(e: pd.DataFrame, norm_cols: list, thr: float, people_ref: float) -> dict:
    contrib = e[norm_cols].sum()
    top = contrib.idxmax()
    feat = top.replace("norm_err_", "")
    direction = "偏高" if e[f"res_{feat}"].mean() > 0 else "偏低"
    severity = float(e.score_max.max() / thr)
    people = float(e.people_count.mean()) if e.people_count.notna().any() else np.nan
    people_factor = float(np.clip(people / people_ref, 0, 1)) if pd.notna(people) else 0.0
    return {
        "類型": "模型警報", "杆": e.polename.iloc[0],
        "開始": e.ts.min(), "結束": e.ts.max() + pd.Timedelta(config.FREQ),
        "持續_小時": round(len(e) * 0.25, 2),
        "嚴重度": round(severity, 2),
        "最可疑特徵": f"{FEATURE_NAMES.get(feat, feat)}{direction}",
        "平均人流": round(people, 1) if pd.notna(people) else None,
        "優先度分數": round(severity * (1 + PEOPLE_WEIGHT * people_factor), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True, help="不含當天（右開區間）")
    ap.add_argument("--input", default=str(config.OUT_DIR / "pole_ts_15min.parquet"))
    args = ap.parse_args()
    INF_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_table(args.input, args.start, args.end)
    scalers = F.load(config.OUT_DIR / "features" / "scalers.json")
    feats = F.transform(raw, scalers).assign(split="infer")
    use_circ = bool(feats[[f"m_{c}" for c in F.CIRCUIT_FEATURES]].to_numpy().sum() > 0)
    mode = "with_circuit" if use_circ else "pole_only"
    scoring = json.loads((E.MODEL_DIR / f"scoring_{mode}.json").read_text(encoding="utf-8"))

    scores = E.score(E.load_model(2), feats, ["infer"], use_circ)
    norm_cols = []
    for c, per_pole in scoring["norm"].items():
        scores[f"norm_{c}"] = scores[c] / scores.polename.map(per_pole).astype(float)
        norm_cols.append(f"norm_{c}")
    scores["score_max"] = scores[norm_cols].max(axis=1).where(scores.score_pole.notna())
    thr = scoring["threshold"]
    scores["alarm"] = scores.score_max > thr
    scores = scores.merge(raw[["polename", "ts", "people_count", "w_total"]], on=["polename", "ts"], how="left")

    people_ref = people_reference()
    rows = [describe_alarm(e, norm_cols, thr, people_ref) for e in to_events(scores, scores.alarm, MERGE_GAP_BINS)]
    for e in to_events(scores, scores.w_total.isna(), 0):
        if len(e) >= OUTAGE_MIN_BINS:
            rows.append({"類型": "電表斷訊", "杆": e.polename.iloc[0], "開始": e.ts.min(),
                         "結束": e.ts.max() + pd.Timedelta(config.FREQ), "持續_小時": round(len(e) * 0.25, 2)})
    alarms = pd.DataFrame(rows)
    if not alarms.empty:
        alarms = group_fleet_events(alarms)
        order = {"模型警報": 0, "全區同步異常": 1, "電表斷訊": 2}
        alarms = alarms.sort_values(["類型", "優先度分數", "開始"], key=lambda s: s.map(order) if s.name == "類型" else s,
                                    ascending=[True, False, True], na_position="last")

    tag = f"{args.start}_{args.end}"
    alarms.to_csv(INF_DIR / f"alarms_{tag}.csv", index=False, encoding="utf-8-sig")
    scores.to_parquet(INF_DIR / f"scores_{tag}.parquet", index=False)

    n_pole_days = scores.groupby("polename").ts.agg(lambda s: s.dt.date.nunique()).sum()
    counts = alarms["類型"].value_counts() if not alarms.empty else pd.Series(dtype=int)
    n_single = int(counts.get("模型警報", 0))
    print(f"模式：{mode}（閾值 {thr:.3f}，校準於 {scoring['calibrated_on']}）")
    print(f"期間 {args.start} ~ {args.end}，{scores.polename.nunique()} 根杆，警報時段比例 {scores.alarm.mean():.2%}")
    print(f"單杆警報 {n_single} 件（平均每杆每週 {n_single / n_pole_days * 7:.1f} 件），"
          f"全區同步異常 {int(counts.get('全區同步異常', 0))} 件，電表斷訊 {int(counts.get('電表斷訊', 0))} 件")
    if not alarms.empty:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(alarms.head(25).to_string(index=False))
    print(f"\n完整清單：{INF_DIR / f'alarms_{tag}.csv'}")


if __name__ == "__main__":
    main()
