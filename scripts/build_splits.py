"""第 2 步：時間切分、統計異常標籤、測試集注入模擬異常。

1. 切分：依 config.SPLITS 替每個 (杆, 15 分鐘) 標上 A_train / A_cal / A_test / B_adapt / B_cal / B_test。
2. 基準值：每杆 × 每小時 × 上班日或休息日 的中位數與 MAD，只用各時段的訓練段計算
   （A 用 A_train，B 用 B_adapt），避免測試資訊洩漏。
3. 標籤（label）：
       normal     乾淨正常
       rule       明確規則異常（燈控與路燈迴路功率矛盾）
       stat       統計異常（|扣掉全區中位數後的 robust z| > MAD_K，且連續 >= MIN_RUN 個時段）
       uncertain  斷訊或 ×100 故障切換點前後，數值可能不可靠
       missing    電表沒有資料
   訓練只用 label == normal 且屬於訓練段的時段（train_ok）。
4. 注入：複製 A_test、B_test，在 normal 時段注入已知的模擬異常，事件清單即為標準答案。

用法：
    uv run python scripts/build_splits.py
輸出：
    outputs/pole_ts_labeled.parquet     寬表 + split / label / label_reason / train_ok / z_*
    outputs/baselines.parquet           基準值（第 3 步特徵工程沿用）
    outputs/test_injected_A.parquet     A_test 注入後副本（含 inj_event_id 欄）
    outputs/test_injected_B.parquet     B_test 注入後副本
    outputs/injected_events.csv         注入事件清單
"""

import numpy as np
import pandas as pd

import config
from config import MAD_K, MIN_RUN, OUT_DIR, POLES

# 做統計標籤的特徵：A 期有迴路明細，B 期只有杆體總量
STAT_FEATURES = {
    "A": ["w_total", "w_signage", "w_light", "w_camera", "w_indicator", "w_network", "w_unmetered"],
    "B": ["w_total"],
}
SUB_CIRCUITS = ["w_signage", "w_light", "w_camera", "w_indicator", "w_network", "w_env"]
LABEL_PRIORITY = ["missing", "rule", "stat", "uncertain", "normal"]


# ---- 1. 切分 -------------------------------------------------------------------
def assign_split(df: pd.DataFrame) -> pd.Series:
    split = pd.Series(pd.NA, index=df.index, dtype="string")
    for name, (a, b) in config.SPLITS.items():
        split[(df.ts >= a) & (df.ts < b)] = name
    return split


# ---- 2. 基準值 -----------------------------------------------------------------
def mad(s: pd.Series) -> float:
    return float((s - s.median()).abs().median())


def fit_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """每 (period, 杆, 小時, 休息日, 特徵) 的中位數與 robust 尺度，只用訓練段。"""
    rows = []
    for period, train_split in config.TRAIN_SPLITS.items():
        d = df[df.split == train_split]
        for feat in STAT_FEATURES[period]:
            g = d.dropna(subset=[feat]).groupby(["polename", "hour_int", "offday"])[feat]
            b = pd.DataFrame({"median": g.median(), "mad": g.apply(mad), "n": g.size()}).reset_index()
            b["feature"], b["period"] = feat, period
            rows.append(b)
    base = pd.concat(rows, ignore_index=True)
    # robust 尺度 = 1.4826·MAD，並設下限避免近乎常數的迴路被微小波動觸發
    base["scale"] = np.maximum.reduce([
        1.4826 * base["mad"], config.Z_FLOOR_REL * base["median"].abs(),
        np.full(len(base), config.Z_FLOOR_W)])
    return base


def robust_z(df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """z_*：相對該杆基準的 robust z。zr_*：再扣掉同一時間各杆 z 的中位數。

    多根杆同時同向偏移（例如 2/23 起全區看板夜間待機功率一起上升約 28W）是營運端調整，
    不是單杆設備異常；標籤只看 zr_*，只抓「這根杆自己跟別人不一樣」的偏離。
    """
    z = pd.DataFrame(index=df.index)
    keys = ["period", "polename", "hour_int", "offday"]
    for feat in sorted({f for fs in STAT_FEATURES.values() for f in fs}):
        b = base[base.feature == feat][keys + ["median", "scale"]]
        m = df[keys].merge(b, on=keys, how="left")
        z[f"z_{feat}"] = (df[feat].values - m["median"].values) / m["scale"].values
        fleet = z[f"z_{feat}"].groupby([df.period, df.ts]).transform("median")
        z[f"zr_{feat}"] = z[f"z_{feat}"] - fleet
    return z


# ---- 3. 標籤 -------------------------------------------------------------------
def runs_at_least(flag: pd.Series, groups: pd.Series, n: int) -> pd.Series:
    """flag 為 True 且所在的連續 True 區段長度 >= n。"""
    block = (flag != flag.groupby(groups).shift()).cumsum()
    length = flag.groupby([groups, block]).transform("size")
    return flag & (length >= n)


def near(flag: pd.Series, groups: pd.Series, k: int) -> pd.Series:
    """flag 前後 k 個時段內（同一杆同一時段）。"""
    f = flag.astype(float)
    win = 2 * k + 1
    return f.groupby(groups).transform(lambda s: s.rolling(win, center=True, min_periods=1).max()).astype(bool)


def lamp_rules(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """燈控亮度 vs 路燈迴路功率（只有 A 期有路燈迴路）。

    P_full = 該杆在 A_train 中亮度 >= 95% 時的路燈迴路功率中位數。
      燈控全亮但功率 < 0.5·P_full → lamp_off_when_on
      燈控全暗但功率 > 0.5·P_full → lamp_on_when_off
    """
    ref = df[(df.split == "A_train") & (df.light_level >= 95)].groupby("polename").w_light.median()
    p_full = df.polename.map(ref)
    off_when_on = (df.light_level >= 95) & (df.w_light < 0.5 * p_full)
    on_when_off = (df.light_level <= 5) & (df.w_light > 0.5 * p_full)
    return off_when_on, on_when_off


def build_labels(df: pd.DataFrame, z: pd.DataFrame) -> pd.DataFrame:
    grp = df.polename + "|" + df.period
    reasons = pd.DataFrame(index=df.index)

    reasons["missing"] = df.w_total.isna()

    off_on, on_off = lamp_rules(df)
    reasons["rule:lamp_off_when_on"] = runs_at_least(off_on.fillna(False), grp, MIN_RUN)
    reasons["rule:lamp_on_when_off"] = runs_at_least(on_off.fillna(False), grp, MIN_RUN)

    for c in [c for c in z.columns if c.startswith("zr_")]:
        feat = c[3:]
        hit = (z[c].abs() > MAD_K).fillna(False)
        reasons[f"stat:{feat}"] = runs_at_least(hit, grp, MIN_RUN)

    fault_edge = df.scale_fault.fillna(0).groupby(grp).diff().fillna(0).ne(0)
    reasons["uncertain:outage_edge"] = near(reasons["missing"], grp, config.EDGE_BINS) & ~reasons["missing"]
    reasons["uncertain:fault_edge"] = near(fault_edge, grp, config.EDGE_BINS)

    label = pd.Series("normal", index=df.index, dtype="string")
    for lvl in reversed(LABEL_PRIORITY[:-1]):  # 優先度低的先寫，高的覆蓋
        cols = [c for c in reasons.columns if c.split(":")[0] == lvl]
        label[reasons[cols].any(axis=1)] = lvl
    reason = reasons.apply(lambda r: ";".join(r.index[r.values]), axis=1)
    return pd.DataFrame({"label": label, "label_reason": reason})


# ---- 4. 注入模擬異常 -------------------------------------------------------------
# 每種異常 × 3 種幅度 × N_PER 個事件；幅度為相對該時段基準值的比例或固定瓦數
N_PER = 4  # 使注入時段約佔測試段正常時段的 10–20%，保留足夠的乾淨正常時段算誤報率
MAGNITUDES = {"small": 0, "medium": 1, "large": 2}
INJECTIONS = {
    # name: (適用時段, 持續時段數範圍, 允許的起始小時, 說明)
    # 日 / 夜型事件的起始小時與長度搭配成整段落在白天（9–17 時）或夜間（19–5 時）內
    "spike":          ("AB", (2, 8),    None,          "總功率短暫突增"),
    "dropout":        ("AB", (4, 16),   None,          "總功率掉到接近 0（斷電/跳脫）"),
    "level_shift":    ("AB", (24, 96),  None,          "總功率持續偏移 6 小時~1 天"),
    "drift":          ("AB", (48, 144), None,          "總功率緩慢爬升 12 小時~1.5 天"),
    "lamp_on_day":    ("AB", (8, 24),   range(9, 12),  "白天燈控顯示關燈，但路燈仍耗電"),
    "lamp_off_night": ("AB", (8, 32),   range(19, 22), "夜間燈控顯示亮燈，但路燈沒耗電"),
    "camera_dead":    ("A",  (8, 48),   None,          "攝影機迴路歸零（只有迴路層級看得出來）"),
    "signage_stuck":  ("A",  (8, 32),   range(19, 22), "數位看板夜間該關未關"),
}
SPIKE_REL = [0.3, 0.6, 1.2]
DROP_KEEP = [0.5, 0.2, 0.0]          # dropout 後保留的比例
SHIFT_REL = [0.1, 0.25, 0.5]
DRIFT_REL = [0.15, 0.35, 0.7]        # 事件結束時的累積偏移
LAMP_SCALE = [0.5, 1.0, 1.0]         # 路燈功率的比例（small = 半亮）
SIGNAGE_REL = [0.3, 0.6, 1.0]        # 看板白天典型功率的比例


def pick_window(d: pd.DataFrame, length: int, hours, rng, used: np.ndarray) -> int | None:
    """在同一杆的測試段中找一段連續 normal、未被使用、起點符合 hours 的區間，回傳起始列位置。"""
    ok = (d.label.values == "normal") & ~used
    starts = np.arange(0, len(d) - length)
    if hours is not None:
        starts = starts[np.isin(d.hour_int.values[starts], list(hours))]
    rng.shuffle(starts)
    csum = np.concatenate([[0], np.cumsum(ok)])
    for s in starts[:4000]:
        if csum[s + length] - csum[s] == length:
            return int(s)
    return None


def apply_injection(d: pd.DataFrame, i0: int, i1: int, kind: str, lvl: int, p_light: float, p_sign: float):
    idx = d.index[i0:i1]
    n = i1 - i0
    base_total = d.loc[idx, "w_total"].values.copy()

    def add_total(delta):
        d.loc[idx, "w_total"] = base_total + delta

    if kind == "spike":
        add_total(base_total * SPIKE_REL[lvl])
    elif kind == "dropout":
        add_total(-base_total * (1 - DROP_KEEP[lvl]))
        for c in SUB_CIRCUITS + ["w_unmetered"]:
            if c in d and d[c].notna().any():
                d.loc[idx, c] = d.loc[idx, c] * DROP_KEEP[lvl]
    elif kind == "level_shift":
        sign = 1 if lvl != 1 else -1  # 中幅度往下移，其餘往上移，兩個方向都測
        add_total(sign * base_total * SHIFT_REL[lvl])
    elif kind == "drift":
        add_total(base_total * DRIFT_REL[lvl] * np.linspace(0, 1, n))
    elif kind == "lamp_on_day":
        delta = p_light * LAMP_SCALE[lvl]
        add_total(delta)
        if d.w_light.notna().any():
            d.loc[idx, "w_light"] = d.loc[idx, "w_light"] + delta
        d.loc[idx, "light_level"] = 0
    elif kind == "lamp_off_night":
        # 燈實際耗電 = 全亮功率 × 當下燈控亮度；B 期夜間調光到約 18%，能扣掉的功率也只有這麼多
        level = d.loc[idx, "light_level"].ffill().fillna(100).values / 100
        delta = p_light * LAMP_SCALE[lvl] * level
        add_total(-delta)
        if d.w_light.notna().any():
            d.loc[idx, "w_light"] = np.maximum(d.loc[idx, "w_light"] - delta, 0)
    elif kind == "camera_dead":
        cam = d.loc[idx, "w_camera"].values.copy()
        d.loc[idx, "w_camera"] = 0
        add_total(-cam)
    elif kind == "signage_stuck":
        delta = p_sign * SIGNAGE_REL[lvl]
        d.loc[idx, "w_signage"] = d.loc[idx, "w_signage"] + delta
        add_total(delta)
    # 總電源 = 分迴路 + 未計量：維持這個物理關係（dropout 已同比例縮放 w_unmetered）
    if d.w_unmetered.notna().any() and kind != "dropout":
        subs = d.loc[idx, [c for c in SUB_CIRCUITS if c in d]].sum(axis=1, min_count=1)
        d.loc[idx, "w_unmetered"] = d.loc[idx, "w_total"] - subs


def inject(df: pd.DataFrame, period: str, seed: int, split: str | None = None,
           n_per: int = N_PER) -> tuple[pd.DataFrame, pd.DataFrame]:
    """split 預設為測試段；傳入 A_cal / B_cal 則產生驗證用的注入副本（ablation 選模型用，不看測試集）。"""
    rng = np.random.default_rng(seed)
    split = split or f"{period}_test"
    prefix = period if split.endswith("_test") else split
    test = df[df.split == split].copy()
    test["inj_event_id"] = pd.Series(pd.NA, index=test.index, dtype="string")
    # 各杆的路燈全亮功率、看板白天典型功率：A 期用迴路資料；B 期沒有迴路 → 借用同杆 A 期的數值
    a_ref = df[df.split == "A_train"]
    p_light = a_ref[a_ref.light_level >= 95].groupby("polename").w_light.median()
    p_sign = a_ref[a_ref.hour_int.between(9, 15)].groupby("polename").w_signage.median()

    events = []
    by_pole = {p: g.sort_values("ts") for p, g in test.groupby("polename")}
    used = {p: np.zeros(len(g), bool) for p, g in by_pole.items()}
    for kind, (periods, (lo, hi), hours, desc) in INJECTIONS.items():
        if period not in periods:
            continue
        for mag, lvl in MAGNITUDES.items():
            for k in range(n_per):
                for _ in range(20):  # 隨機挑杆，找不到合適區間就換杆
                    pole = POLES[rng.integers(len(POLES))]
                    if kind in ("camera_dead",) and by_pole[pole].w_camera.isna().all():
                        continue
                    if kind == "signage_stuck" and not (p_sign.get(pole, 0) > 20):
                        continue
                    if kind.startswith("lamp") and not (p_light.get(pole, 0) > 5):
                        continue
                    length = int(rng.integers(lo, hi + 1))
                    d = by_pole[pole]
                    s = pick_window(d, length, hours, rng, used[pole])
                    if s is None:
                        continue
                    eid = f"{prefix}-{kind}-{mag}-{k}"
                    apply_injection(d, s, s + length, kind, lvl, p_light.get(pole, 0), p_sign.get(pole, 0))
                    d.iloc[s:s + length, d.columns.get_loc("inj_event_id")] = eid
                    # 事件前後各留 4 小時空白，事件之間不重疊也不緊貼
                    used[pole][max(0, s - 16):s + length + 16] = True
                    events.append(dict(event_id=eid, period=period, split=split, polename=pole, kind=kind, magnitude=mag,
                                       start=d.ts.iloc[s], end=d.ts.iloc[s + length - 1] + pd.Timedelta(config.FREQ),
                                       n_bins=length, description=desc))
                    break
    out = pd.concat(by_pole.values()).sort_values(["polename", "ts"])
    return out, pd.DataFrame(events)


# ---- main ---------------------------------------------------------------------
def main() -> None:
    df = pd.read_parquet(OUT_DIR / "pole_ts_15min.parquet")
    df = df[df.polename.isin(POLES)].sort_values(["period", "polename", "ts"]).reset_index(drop=True)
    df["hour_int"] = df.ts.dt.hour
    df["split"] = assign_split(df)

    base = fit_baselines(df)
    z = robust_z(df, base)
    labels = build_labels(df, z)
    df = pd.concat([df, labels, z], axis=1)
    df["train_ok"] = df.split.isin(config.TRAIN_SPLITS.values()) & (df.label == "normal")

    df.to_parquet(OUT_DIR / "pole_ts_labeled.parquet", index=False)
    base.to_parquet(OUT_DIR / "baselines.parquet", index=False)

    inj_a, ev_a = inject(df, "A", seed=11)
    inj_b, ev_b = inject(df, "B", seed=22)
    inj_a.to_parquet(OUT_DIR / "test_injected_A.parquet", index=False)
    inj_b.to_parquet(OUT_DIR / "test_injected_B.parquet", index=False)
    # 驗證用注入（校準段）：ablation 依此選設定，測試段只做最後確認；校準段較短，每格 2 個事件
    val_a, vev_a = inject(df, "A", seed=33, split="A_cal", n_per=2)
    val_b, vev_b = inject(df, "B", seed=44, split="B_cal", n_per=2)
    val_a.to_parquet(OUT_DIR / "val_injected_A.parquet", index=False)
    val_b.to_parquet(OUT_DIR / "val_injected_B.parquet", index=False)
    events = pd.concat([ev_a, ev_b, vev_a, vev_b], ignore_index=True)
    events.to_csv(OUT_DIR / "injected_events.csv", index=False, encoding="utf-8-sig")

    # ---- 摘要 ----
    print("== 各切分的標籤比例 (%)")
    print((pd.crosstab(df.split, df.label, normalize="index") * 100).round(1).to_string())
    print("\n== 各切分的時段數 / 可訓練時段數")
    print(df.groupby("split").agg(bins=("ts", "size"), train_ok=("train_ok", "sum"), days=("ts", lambda s: s.dt.date.nunique())).to_string())
    print("\n== 標籤原因（非 normal、非 missing），各切分時段數")
    r = df[~df.label.isin(["normal", "missing"])].assign(reason=lambda x: x.label_reason.str.split(";")).explode("reason").reset_index(drop=True)
    r = r[~r.reason.isin(["missing"])]
    print(pd.crosstab(r.reason, r.split).to_string())
    print("\n== 各杆 × 切分的 normal 比例 (%)")
    print((df.assign(n=df.label.eq("normal")).pivot_table(index="polename", columns="split", values="n") * 100).round(0).to_string())
    print(f"\n== 注入事件 {len(events)} 個")
    print(events.groupby(["split", "kind"]).size().unstack(0).to_string())


if __name__ == "__main__":
    main()
