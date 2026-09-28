"""把推論與評估結果整理成展示網站用的 data.json。

前置：先跑 infer.py（2026-06 與 2026-02 兩段）與 evaluate.py。
用法：
    uv run python scripts/export_dashboard.py
輸出：
    outputs/dashboard/data.json
"""

import json

import numpy as np
import pandas as pd

import config
import features as F

DASH_DIR = config.OUT_DIR / "dashboard"
INF_DIR = config.OUT_DIR / "inference"
PERIODS = [  # (key, 標籤, 起, 迄)
    ("2026-06", "2026 年 6 月", "2026-06-01", "2026-07-01"),
    ("2026-02", "2026 年 2 月", "2026-02-09", "2026-03-01"),
]
CIRCUIT_SERIES = {"w_signage_s": ("w_signage", "數位看板"), "w_light_s": ("w_light", "路燈迴路"),
                  "w_camera_s": ("w_camera", "攝影機")}
FEATURE_NAMES = {
    "w_total_s": "總功率", "w_signage_s": "數位看板", "w_light_s": "路燈迴路", "w_camera_s": "攝影機",
    "w_indicator_s": "智慧指標", "w_network_s": "網路設備", "w_unmetered_s": "未計量用電", "lamp_gap": "燈控與路燈功率落差",
}


def r(x, nd=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def arr(s: pd.Series, nd=1):
    return [None if pd.isna(v) else round(float(v), nd) for v in s]


def pole_meta() -> list:
    dmap = pd.read_csv(config.OUT_DIR / "device_pole_map.csv", dtype=str)
    poles = pd.read_csv(config.ROOT / "杆體資訊_smartpole_list_2026.csv", dtype=str)
    ts = pd.read_parquet(config.OUT_DIR / "pole_ts_labeled.parquet")
    out = []
    for p in config.POLES:
        d = dmap[dmap.polename == p]
        info = poles[poles.polename == p].iloc[0]
        circ = d[d.id_level == "circuit"].device_type.str.replace("迴路", "").tolist()
        a = ts[(ts.polename == p) & (ts.split == "A_train")]
        out.append({
            "id": p, "short": p.replace("SCCP-", ""),
            "lat": float(info.latitude), "lon": float(info.longitude),
            "meter_model": f"-{d.meter_model.dropna().iloc[0]} 型",
            "circuits": [c for c in circ if c != "MP"],
            "n_lights": int((d.system == "路燈").sum()), "n_cameras": int((d.system == "攝影機").sum()),
            "has_env": bool((d.system == "環境感測").any()),
            "day_w": r(a[a.hour_int.between(9, 15)].w_total.median()),
            "night_w": r(a[(a.hour_int >= 19) | (a.hour_int < 5)].w_total.median()),
        })
    return out


def period_block(key, label, start, end, scalers) -> dict:
    tag = f"{start}_{end}"
    scores = pd.read_parquet(INF_DIR / f"scores_{tag}.parquet")
    alarms = pd.read_csv(INF_DIR / f"alarms_{tag}.csv", parse_dates=["開始", "結束"])
    raw = pd.read_parquet(config.OUT_DIR / "pole_ts_15min.parquet")
    raw = raw[raw.polename.isin(config.POLES) & (raw.ts >= start) & (raw.ts < end)]
    df = scores.merge(raw[["polename", "ts", "light_level"] + [v[0] for v in CIRCUIT_SERIES.values()]],
                      on=["polename", "ts"], how="left")
    mode = "with_circuit" if "err_w_signage_s" in df else "pole_only"
    scoring = json.loads((config.OUT_DIR / "models" / f"scoring_{mode}.json").read_text(encoding="utf-8"))
    thr = scoring["threshold"]

    series = {}
    for p, g in df.groupby("polename"):
        g = g.sort_values("ts")
        sc = scalers["power"][p]["w_total_s"]["scale"]
        s = {"actual": arr(g.w_total), "expected": arr(g.w_total - g.res_w_total_s * sc),
             "score": arr(g.score_max / thr, 2), "people": arr(g.people_count, 0), "light": arr(g.light_level, 0)}
        if mode == "with_circuit":
            for feat, (raw_col, _) in CIRCUIT_SERIES.items():
                sc_c = scalers["power"][p][feat]["scale"]
                s[raw_col] = arr(g[raw_col])
                s[raw_col + "_exp"] = arr(g[raw_col] - g[f"res_{feat}"] * sc_c)
        series[p] = s

    norm_cols = [c for c in df.columns if c.startswith("norm_err_")]
    items = []
    for i, a in alarms.reset_index(drop=True).iterrows():
        item = {"id": f"{key}-{i:03d}", "type": a["類型"], "pole": a["杆"], "start": a["開始"].isoformat(),
                "end": a["結束"].isoformat(), "hours": r(a["持續_小時"], 2)}
        if a["類型"] != "電表斷訊":
            item.update(severity=r(a["嚴重度"], 2), feature=a["最可疑特徵"])
        if a["類型"] == "模型警報":
            item.update(priority=r(a["優先度分數"], 2), people=r(a["平均人流"]))
            w = df[(df.polename == a["杆"]) & (df.ts >= a["開始"]) & (df.ts < a["結束"])]
            contrib = w[norm_cols].mean().sort_values(ascending=False)
            item["contrib"] = [{"name": FEATURE_NAMES.get(c.replace("norm_err_", ""), c), "value": r(v, 2)}
                               for c, v in contrib.items() if pd.notna(v)][:6]
        items.append(item)

    days = pd.date_range(start, end, freq="D", inclusive="left")
    al = pd.DataFrame(items)
    al["day"] = pd.to_datetime(al.start).dt.normalize() if not al.empty else []
    daily = []
    for d in days:
        sub = al[al.day == d] if not al.empty else al
        day_scores = df[df.ts.dt.normalize() == d]
        daily.append({"date": d.strftime("%Y-%m-%d"),
                      "single": int((sub.type == "模型警報").sum()) if len(sub) else 0,
                      "fleet": int((sub.type == "全區同步異常").sum()) if len(sub) else 0,
                      "alarm_share": r(day_scores.alarm.mean() * 100, 2)})
    n_pole_days = df.groupby("polename").ts.agg(lambda s: s.dt.date.nunique()).sum()
    counts = al.type.value_counts() if not al.empty else pd.Series(dtype=int)
    return {
        "label": label, "start": start, "end": end, "mode": mode, "threshold": thr,
        "calibrated_on": scoring["calibrated_on"], "t0": pd.Timestamp(start).isoformat(), "step_min": 15,
        "series": series, "alarms": items, "daily": daily,
        "summary": {"single": int(counts.get("模型警報", 0)), "fleet": int(counts.get("全區同步異常", 0)),
                    "outage": int(counts.get("電表斷訊", 0)),
                    "per_pole_week": r(counts.get("模型警報", 0) / n_pole_days * 7, 2),
                    "alarm_share": r(df.alarm.mean() * 100, 2)},
    }


def metrics_block() -> dict:
    m = json.loads((config.OUT_DIR / "eval" / "metrics.json").read_text(encoding="utf-8"))
    ev = pd.read_csv(config.OUT_DIR / "eval" / "event_recall.csv")
    pick = {"B": "B_test | 階段二（最終模型） [max]", "A1": "A_test | 階段一 [max]", "A2": "A_test | 階段二（fine-tune 後） [max]"}
    keys = ["fpr", "fpr_holiday", "fpr_non_holiday", "false_alarm_events_per_pole_week", "event_recall",
            "bin_recall", "auroc", "real_anomaly_hit_rate", "threshold"]
    out = {k: {kk: m[v][kk] for kk in keys} for k, v in pick.items()}
    out["mean_scoring_B"] = {kk: m["B_test | 階段二（最終模型） [mean]"][kk] for kk in ["fpr", "event_recall", "auroc"]}
    recall = {}
    for k, v in pick.items():
        t = ev[ev.case == v].pivot_table(index="kind", columns="magnitude", values="detected", aggfunc="mean")
        recall[k] = {kind: {mag: r(t.loc[kind, mag] * 100, 0) for mag in ["small", "medium", "large"] if mag in t}
                     for kind in t.index}
    out["recall"] = recall
    desc = ev.drop_duplicates("kind").set_index("kind").description.to_dict()
    out["kinds"] = desc
    logs = {s: json.loads((config.OUT_DIR / "models" / f"stage{s}_log.json").read_text()) for s in (1, 2)}
    out["training"] = {
        "stage1": {"epochs": len(logs[1]["log"]), "best_epoch": logs[1]["best_epoch"], "seconds": logs[1]["seconds"],
                   "windows": logs[1]["train_windows"], "params": logs[1]["trainable_params"]},
        "stage2": {"epochs": len(logs[2]["log"]), "best_epoch": logs[2]["best_epoch"], "seconds": logs[2]["seconds"],
                   "windows": logs[2]["train_windows"], "params": logs[2]["trainable_params"],
                   "a_cal_before": logs[2]["a_cal_before"], "a_cal_after": logs[2]["a_cal_after"]},
    }
    return out


def main():
    DASH_DIR.mkdir(parents=True, exist_ok=True)
    scalers = F.load(config.OUT_DIR / "features" / "scalers.json")
    data = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "poles": pole_meta(),
        "periods": {k: period_block(k, lab, s, e, scalers) for k, lab, s, e in PERIODS},
        "metrics": metrics_block(),
        "splits": config.SPLITS, "holidays": config.HOLIDAYS,
    }
    path = DASH_DIR / "data.json"
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{path}  {path.stat().st_size / 1e6:.2f} MB")
    for k, p in data["periods"].items():
        print(k, p["mode"], p["summary"])


if __name__ == "__main__":
    main()
