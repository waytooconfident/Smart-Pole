"""特徵轉換（fit / transform），訓練與推論共用。

設計：
- 正規化參數只用 A_train 的乾淨正常時段計算，套用到 A、B 兩時段。
  B 期的分佈位移（例如路燈改為調光）因此保留在特徵中，交由第二階段 fine-tune 去適應。
- 功率類特徵：每杆 robust scaling  x_s = (x − 中位數) / 尺度，尺度 = max(IQR, 10%·|中位數|, 2W)。
  杆間的日週期形狀差異由模型透過 pole_id embedding 與時間條件學習。
- 電壓：先扣掉同一時間的全區平均（共用電網的波動），再以全區共用尺度縮放。
- 路燈亮度：各杆同一套排程，不逐杆縮放，直接除以 100。
- 缺值：縮放後補 0，另附 m_* 遮罩（1 = 真實觀測）。模型的 loss 只算遮罩為 1 的位置。

特徵分組：
    POLE_FEATURES     杆體層級，A、B 都有
    CIRCUIT_FEATURES  迴路層級，只有 A 期
    COND_FEATURES     條件輸入（時間、節日、路燈亮度），只當輸入、不重建
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config

POLE_FEATURES = ["w_total_s", "v_dev_s"]
CIRCUIT_FEATURES = ["w_signage_s", "w_light_s", "w_camera_s", "w_indicator_s", "w_network_s",
                    "w_unmetered_s", "lamp_gap"]
COND_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "offday_f", "holiday_f", "light_level_f", "m_light_level"]

SCALED_POWER = {  # 輸出欄位 -> 原始欄位
    "w_total_s": "w_total", "w_signage_s": "w_signage", "w_light_s": "w_light", "w_camera_s": "w_camera",
    "w_indicator_s": "w_indicator", "w_network_s": "w_network", "w_unmetered_s": "w_unmetered",
}
LIGHT_FFILL_BINS = 4  # 路燈亮度缺口 <= 1 小時才往前補，其餘視為缺值


def _robust(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {"median": 0.0, "scale": 1.0}
    med = float(s.median())
    iqr = float(s.quantile(0.75) - s.quantile(0.25))
    return {"median": med, "scale": max(iqr, config.Z_FLOOR_REL * abs(med), config.Z_FLOOR_W)}


def fit(df: pd.DataFrame) -> dict:
    """df：完整寬表（含 split / train_ok）。只使用 A_train 的乾淨正常時段。"""
    tr = df[(df.split == "A_train") & df.train_ok]
    scalers = {"power": {}, "p_full_light": {}, "v_dev": None}
    for pole, g in tr.groupby("polename"):
        scalers["power"][pole] = {out: _robust(g[raw]) for out, raw in SCALED_POWER.items()}
        full = g.loc[g.light_level >= 95, "w_light"].median()
        scalers["p_full_light"][pole] = float(full) if pd.notna(full) else None
    v_dev = tr.v - tr.groupby("ts").v.transform("mean")
    scalers["v_dev"] = _robust(v_dev)
    scalers["pole_ids"] = {p: i for i, p in enumerate(config.POLES)}
    scalers["features"] = {"pole": POLE_FEATURES, "circuit": CIRCUIT_FEATURES, "cond": COND_FEATURES}
    return scalers


def transform(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """回傳特徵表：識別欄 + 特徵 + 遮罩。df 需含同一時間各杆的資料（電壓偏差要用全區平均）。"""
    df = df.sort_values(["polename", "ts"]).reset_index(drop=True)
    out = pd.DataFrame({"polename": df.polename, "ts": df.ts})
    out["pole_id"] = df.polename.map(scalers["pole_ids"]).astype("int64")

    # 功率：每杆 robust scaling
    for col, raw in SCALED_POWER.items():
        med = df.polename.map(lambda p: scalers["power"][p][col]["median"])
        sc = df.polename.map(lambda p: scalers["power"][p][col]["scale"])
        out[col] = (df[raw] - med) / sc

    # 電壓偏差：扣掉同一時間全區平均
    v_dev = df.v - df.groupby("ts").v.transform("mean")
    out["v_dev_s"] = (v_dev - scalers["v_dev"]["median"]) / scalers["v_dev"]["scale"]

    # 燈控與路燈迴路功率的落差：實際功率 / 全亮功率 − 燈控亮度比例
    p_full = df.polename.map(scalers["p_full_light"]).astype(float)
    light = df.groupby("polename").light_level.ffill(limit=LIGHT_FFILL_BINS)
    out["lamp_gap"] = df.w_light / p_full - light / 100

    # 條件輸入
    h = df.ts.dt.hour + df.ts.dt.minute / 60
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    out["dow_sin"], out["dow_cos"] = np.sin(2 * np.pi * df.dow / 7), np.cos(2 * np.pi * df.dow / 7)
    out["offday_f"] = df.offday.astype(float)
    out["holiday_f"] = df.holiday.astype(float)
    out["light_level_f"] = light / 100
    out["m_light_level"] = light.notna().astype(float)

    # 遮罩 + 補 0
    for col in POLE_FEATURES + CIRCUIT_FEATURES:
        out[f"m_{col}"] = out[col].notna().astype(float)
        out[col] = out[col].fillna(0.0)
    out["light_level_f"] = out["light_level_f"].fillna(0.0)

    keep = [c for c in ["period", "split", "label", "label_reason", "train_ok", "inj_event_id"] if c in df]
    return pd.concat([df[keep], out], axis=1)


# ---- 混合版本（H1）：統計基準殘差 -------------------------------------------------
# 功率類特徵改成「相對每杆 × 每小時 × 休息日基準的 robust z」，平常水準交給統計基準，
# 深度模型只學殘差裡的時間關係與特徵間關聯。v_dev_s、lamp_gap 本身已是偏差量，維持原樣。
RESID_FEATURES = ["w_total_s", "w_signage_s", "w_light_s", "w_camera_s", "w_indicator_s", "w_network_s", "w_unmetered_s"]
RESID_CLIP = 20.0


def fit_residual(feats: pd.DataFrame) -> dict:
    """每個時段用自己的訓練段（A_train / B_adapt）乾淨時段計算基準：中位數與 1.4826·MAD（下限 0.1）。"""
    fitted = {}
    for period, train_split in config.TRAIN_SPLITS.items():
        tr = feats[(feats.split == train_split) & (feats.label == "normal")].assign(hour=lambda x: x.ts.dt.hour)
        g = tr.groupby(["polename", "hour", "offday_f"])
        cols = [c for c in RESID_FEATURES if tr[f"m_{c}"].sum() > 0]
        med = g[cols].median()
        mad = (g[cols].agg(lambda s: (s - s.median()).abs().median()) * 1.4826).clip(lower=0.1)
        fitted[period] = (med, mad)
    return fitted


def residualize(feats: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    out = feats.copy()
    hour = out.ts.dt.hour
    for period, (med, mad) in fitted.items():
        rows = out.period == period if "period" in out else pd.Series(True, index=out.index)
        key = pd.MultiIndex.from_arrays([out.loc[rows, "polename"], hour[rows], out.loc[rows, "offday_f"]])
        for c in RESID_FEATURES:
            if c not in med:
                continue
            z = (out.loc[rows, c].values - med[c].reindex(key).values) / mad[c].reindex(key).values
            z = np.clip(np.nan_to_num(z, nan=0.0), -RESID_CLIP, RESID_CLIP)
            out.loc[rows, c] = np.where(out.loc[rows, f"m_{c}"].values > 0, z, 0.0)
    return out


def save(scalers: dict, path: Path) -> None:
    path.write_text(json.dumps(scalers, ensure_ascii=False, indent=1), encoding="utf-8")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
