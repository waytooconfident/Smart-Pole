"""特徵 EDA：同杆日夜變化、整段期間變化、跨杆分佈差異，以及正規化方式的比較。

輸入：outputs/pole_ts_15min.parquet（先跑 build_pole_timeseries.py）
輸出：outputs/eda/*.png 與 outputs/eda/stats.json

用法：
    uv run python scripts/eda_features.py
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "eda"

# ---- 視覺規範（dataviz 參考色盤） -------------------------------------------------
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a3a2", "#c03434"])
SEQ.set_bad("#efeeea")

plt.rcParams.update({
    "font.family": ["Microsoft JhengHei", "sans-serif"],
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "lines.linewidth": 1.6, "axes.unicode_minus": False,
})

POLES = [f"SCCP-SP-{i:02d}" for i in range(1, 11)]
SHORT = {p: p.replace("SCCP-SP-", "SP-") for p in POLES}
MODEL20 = ["SCCP-SP-02", "SCCP-SP-03", "SCCP-SP-07", "SCCP-SP-08", "SCCP-SP-09"]
# MP 是總電源（= 其餘迴路加總 + 未計量），分迴路清單不含 MP
CIRCUITS = [("w_signage", "數位看板"), ("w_light", "路燈"), ("w_camera", "攝影機"),
            ("w_indicator", "智慧指標"), ("w_network", "網路設備")]
TOTAL = ("w_total", "杆體總功率 (MP)")
UNMETERED = ("w_unmetered", "未計量 (MP − 分迴路)")
NIGHT = lambda h: (h >= 19) | (h < 5)
DAY = lambda h: (h >= 9) & (h < 16)

STATS: dict = {}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def heat(ax, mat, cmap=SEQ, norm=None, vmin=None, vmax=None, fmt=None, xt=None, yt=None, fontsize=7.5):
    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.grid(False)
    ax.set_xticks(range(mat.shape[1]), xt if xt is not None else [])
    ax.set_yticks(range(mat.shape[0]), yt if yt is not None else [])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if fmt:
        lo, hi = im.get_clim()
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                dark = (v - lo) / (hi - lo + 1e-9) > 0.55 if norm is None else abs(norm(v) - 0.5) > 0.3
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=fontsize, color="white" if dark else INK)
    return im


# ---- 0. 特徵覆蓋率 ---------------------------------------------------------------
def fig_coverage(df, feats):
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.2), sharey=True)
    for ax, p, title in zip(axes, "AB", ["時段 A：2026/1/1 – 2/28", "時段 B：2026/5/1 – 6/30"]):
        d = df[df.period == p]
        mat = d.groupby("polename")[feats].apply(lambda g: g.notna().mean()).T.values * 100
        im = heat(ax, mat, vmin=0, vmax=100, fmt=lambda v: f"{v:.0f}", xt=[SHORT[x] for x in POLES], yt=feats)
        ax.set_title(title, loc="left")
        ax.tick_params(axis="x", rotation=45)
    cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02)
    cb.set_label("非缺值比例 (%)")
    cb.outline.set_visible(False)
    save(fig, "00_coverage")


# ---- 1. 杆 × 迴路 配置矩陣 -------------------------------------------------------
def fig_config(a):
    cols = ["w_total"] + [c for c, _ in CIRCUITS] + ["w_env", "w_unmetered"]
    labels = ["總電源 MP"] + [n for _, n in CIRCUITS] + ["環境感測", "未計量"]
    night = a[NIGHT(a.hour)].groupby("polename")[cols].median().reindex(POLES)
    day = a[DAY(a.hour)].groupby("polename")[cols].median().reindex(POLES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, m, t in zip(axes, [day, night], ["白天 09–16 時 中位數功率 (W)", "夜間 19–05 時 中位數功率 (W)"]):
        heat(ax, np.log10(m.clip(lower=0).values + 1), vmin=0, vmax=3, fmt=None, xt=labels, yt=[SHORT[p] for p in POLES])
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if np.log10(max(v, 0) + 1) > 1.8 else INK)
        ax.axvline(0.5, color=SURFACE, lw=3)
        ax.set_title(t, loc="left")
        ax.tick_params(axis="x", rotation=30)
    fig.text(0.01, -0.02, "總電源 MP ≈ 右側各分迴路加總 + 未計量。灰色格 = 該杆沒有此迴路。顏色為 log 尺度。",
             color=MUTED, fontsize=8.5)
    save(fig, "01_config_matrix")
    STATS["config_day"] = day.round(1).to_dict()
    STATS["config_night"] = night.round(1).to_dict()


# ---- 2. 單杆一週的迴路組成（日夜） -----------------------------------------------
def fig_week_stack(a, pole="SCCP-SP-01", start="2026-01-12", end="2026-01-19"):
    d = a[(a.polename == pole) & (a.ts >= start) & (a.ts < end)].set_index("ts")
    cols = [c for c, _ in CIRCUITS if d[c].notna().any()] + ["w_unmetered"]
    names = dict(CIRCUITS + [UNMETERED])
    fig, ax = plt.subplots(figsize=(12, 4))
    ys = [d[c].fillna(0).clip(lower=0).values for c in cols]
    ax.stackplot(d.index, ys, colors=SERIES[:len(cols)], linewidth=0.4, edgecolor=SURFACE,
                 labels=[names[c] for c in cols])
    ax.plot(d.index, d.w_total, color=INK, lw=0.8, label="總電源 MP（實測）")
    ax.set_ylabel("功率 (W)")
    ax.set_title(f"{SHORT[pole]} 一週：各分迴路堆疊 vs 總電源 MP（{start} 起，15 分鐘平均）", loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%a"))
    ax.set_xlim(d.index.min(), d.index.max())
    ax.legend(loc="upper left", ncol=len(cols) + 1, frameon=False, fontsize=8.5, bbox_to_anchor=(0, 1.0))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    save(fig, "02_week_stack_sp01")


# ---- 3. 各迴路：杆 × 小時 熱圖 ---------------------------------------------------
def fig_hour_heatmaps(a):
    specs = [TOTAL] + CIRCUITS + [UNMETERED, ("light_level", "路燈亮度 (%)"),
                                  ("people_count", "人流 (事件數/15 分)"), ("v", "電壓 (V)")]
    fig, axes = plt.subplots(5, 2, figsize=(12, 15))
    for ax, (col, name) in zip(axes.flat, specs):
        piv = a.pivot_table(index="polename", columns=a.hour.astype(int), values=col, aggfunc="median").reindex(POLES)
        im = heat(ax, piv.values, xt=[str(h) if h % 3 == 0 else "" for h in range(24)],
                  yt=[SHORT[p] for p in POLES])
        ax.set_title(name, loc="left")
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("時 (hour of day)")
    fig.suptitle("時段 A：每根杆每小時的中位數（灰色 = 沒有此迴路）", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "03_hour_heatmaps")


# ---- 4. 整段期間：每杆日平均總功率 -----------------------------------------------
def fig_daily_trend(df):
    d = df.assign(day=df.ts.dt.floor("D")).groupby(["period", "polename", "day"]).agg(
        w=("w_total", "mean"), fault=("scale_fault", "max")).reset_index()
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.6), sharey=True)
    for ax, pole in zip(axes.flat, POLES):
        for p, color, lab in [("A", SERIES[0], "時段 A"), ("B", SERIES[1], "時段 B")]:
            s = d[(d.polename == pole) & (d.period == p)]
            x = (s.day - s.day.min()).dt.days if len(s) else []
            ax.plot(x, s.w, color=color, label=lab)
        if pole in MODEL20:
            ax.axvspan(1, 21, color=SERIES[3], alpha=0.12, lw=0)
        ax.set_title(SHORT[pole] + ("（-20 型）" if pole in MODEL20 else ""), loc="left", fontsize=9.5)
        ax.set_xlim(0, 60)
    for ax in axes[:, 0]:
        ax.set_ylabel("日平均功率 (W)")
    for ax in axes[-1]:
        ax.set_xlabel("該時段第幾天")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("每根杆的日平均總功率：A（1–2 月）vs B（5–6 月）。黃底 = 已修正的 ×100 故障區間",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "04_daily_trend")


# ---- 5. ×100 故障：修正前後 -------------------------------------------------------
def fig_scale_fault(a):
    s = a[a.polename == "SCCP-SP-02"].set_index("ts")
    raw = s.w_camera * np.where(s.scale_fault == 1, 100, 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.3))
    for ax, y, t in [(axes[0], raw, "修正前（原始 meterinfo2）"), (axes[1], s.w_camera, "修正後（區間內 ÷100）")]:
        ax.plot(s.index, y, color=SERIES[0], lw=1)
        ax.set_title(f"SP-02 攝影機迴路功率 — {t}", loc="left")
        ax.set_ylabel("W")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.axvspan(pd.Timestamp("2026-01-02 17:00"), pd.Timestamp("2026-01-22 17:30"), color=SERIES[3], alpha=0.12, lw=0)
    save(fig, "05_scale_fault")


# ---- 6. 路燈：控制器亮度 vs 路燈迴路功率 ------------------------------------------
def fig_light_consistency(a):
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.4), sharex=True, sharey=True)
    for ax, pole in zip(axes.flat, POLES):
        s = a[(a.polename == pole)].dropna(subset=["light_level", "w_light"])
        jitter = np.random.default_rng(0).normal(0, 1.5, len(s))
        ax.scatter(s.light_level + jitter, s.w_light, s=6, color=SERIES[0], alpha=0.25, lw=0)
        ax.set_title(SHORT[pole], loc="left", fontsize=9.5,
                     color=SERIES[7] if pole == "SCCP-SP-03" else INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("路燈迴路功率 (W)")
    for ax in axes[-1]:
        ax.set_xlabel("控制器亮度 (%)")
    fig.suptitle("時段 A：路燈控制器說的亮度 vs 電表量到的路燈迴路功率（每點 = 15 分鐘）",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "06_light_consistency")


# ---- 7. 時段 A vs B：亮度排程與夜間總功率 -----------------------------------------
def fig_period_shift(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    for p, color, lab in [("A", SERIES[0], "時段 A（1–2 月）"), ("B", SERIES[1], "時段 B（5–6 月）")]:
        prof = df[df.period == p].groupby(df.hour)["light_level"].mean()
        ax.plot(prof.index, prof.values, color=color, label=lab)
        ax.annotate(lab, (prof.index[-1], prof.values[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=8.5, va="center")
    ax.set_xlim(0, 27.5); ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("時"); ax.set_ylabel("平均亮度 (%)")
    ax.set_title("路燈亮度排程（10 杆平均）", loc="left")

    ax = axes[1]
    night = df[NIGHT(df.hour)].groupby(["polename", "period"]).w_total.median().unstack().reindex(POLES)
    y = np.arange(len(POLES))
    for i, pole in enumerate(POLES):
        ax.plot([night.loc[pole, "A"], night.loc[pole, "B"]], [i, i], color=AXIS, lw=1.5, zorder=1)
    ax.scatter(night.A, y, color=SERIES[0], s=46, zorder=2, label="時段 A", edgecolor=SURFACE, lw=1.5)
    ax.scatter(night.B, y, color=SERIES[1], s=46, zorder=2, label="時段 B", edgecolor=SURFACE, lw=1.5)
    ax.set_yticks(y, [SHORT[p] for p in POLES]); ax.invert_yaxis()
    ax.set_xlabel("夜間 19–05 時 總功率中位數 (W)")
    ax.set_title("同一根杆，夜間總功率在兩時段間的位移", loc="left", pad=22)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
              borderaxespad=0, handletextpad=0.3)
    fig.tight_layout()
    save(fig, "07_period_shift")
    STATS["night_median_AB"] = night.round(1).to_dict()


# ---- 8. 跨杆分佈：日 / 夜 箱型圖 --------------------------------------------------
def fig_box_daynight(a):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    pos = np.arange(len(POLES))
    for k, (mask_fn, color, lab) in enumerate([(DAY, SERIES[1], "白天 09–16"), (NIGHT, SERIES[0], "夜間 19–05")]):
        data = [a[(a.polename == p) & mask_fn(a.hour)].w_total.dropna().values for p in POLES]
        bp = ax.boxplot(data, positions=pos + (k - 0.5) * 0.36, widths=0.3, patch_artist=True,
                        showfliers=False, medianprops=dict(color=SURFACE, lw=1.6),
                        whiskerprops=dict(color=color), capprops=dict(color=color))
        for b in bp["boxes"]:
            b.set(facecolor=color, edgecolor=color)
        ax.plot([], [], color=color, lw=6, label=lab)
    ax.set_xticks(pos, [SHORT[p] for p in POLES])
    ax.set_ylabel("杆體總功率 (W)")
    ax.set_title("時段 A：各杆總功率分佈，白天 vs 夜間（箱 = IQR，鬚 = 1.5 IQR）", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    save(fig, "08_box_daynight")


# ---- 9. KS 距離矩陣：原始 / 每杆標準化 / 每杆×時段 基準殘差 ------------------------
def normalize_variants(a, col):
    d = a[["polename", "ts", "hour", col]].dropna().copy()
    d["raw"] = d[col]
    g = d.groupby("polename")[col]
    med, iqr = g.transform("median"), g.transform(lambda s: s.quantile(.75) - s.quantile(.25))
    d["robust"] = (d[col] - med) / iqr.replace(0, np.nan)
    d["hb"] = d.ts.dt.hour
    base = d.groupby(["polename", "hb"])[col].transform("median")
    mad = d.groupby(["polename", "hb"])[col].transform(lambda s: (s - s.median()).abs().median())
    d["resid"] = (d[col] - base) / (mad.replace(0, np.nan) * 1.4826)
    return d


def ks_matrix(d, col, sample=4000):
    rng = np.random.default_rng(0)
    groups = {p: d.loc[d.polename == p, col].dropna().values for p in POLES}
    groups = {p: rng.choice(v, min(sample, len(v)), replace=False) for p, v in groups.items() if len(v)}
    m = np.full((len(POLES), len(POLES)), np.nan)
    for i, p in enumerate(POLES):
        for j, q in enumerate(POLES):
            if p in groups and q in groups:
                m[i, j] = ks_2samp(groups[p], groups[q]).statistic
    return m


def fig_ks(a):
    d = normalize_variants(a, "w_total")
    variants = [("raw", "① 原始值\n（未處理）"),("robust", "② 每杆 robust 標準化\n(x − 中位數) / IQR"),
                ("resid", "③ 每杆 × 每小時基準殘差\n(x − 該杆該時中位數) / MAD")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    means = {}
    for ax, (v, t) in zip(axes, variants):
        m = ks_matrix(d, v)
        im = heat(ax, m, vmin=0, vmax=1, fmt=lambda x: f"{x:.2f}", xt=[SHORT[p] for p in POLES],
                  yt=[SHORT[p] for p in POLES], fontsize=7)
        ax.tick_params(axis="x", rotation=45)
        off = m[~np.eye(len(POLES), dtype=bool)]
        means[v] = float(np.nanmean(off))
        ax.set_title(f"{t}\n平均 KS D = {means[v]:.2f}", loc="left")
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01)
    cb.set_label("KS 統計量 D（0 = 分佈相同，1 = 完全不重疊）")
    cb.outline.set_visible(False)
    save(fig, "09_ks_matrix_w_total")
    STATS["ks_mean_w_total"] = means

    # 三種處理後的分佈形狀（箱型圖）
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, (v, t) in zip(axes, variants):
        data = [d.loc[d.polename == p, v].dropna().values for p in POLES]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.55,
                        medianprops=dict(color=SURFACE, lw=1.4),
                        whiskerprops=dict(color=SERIES[0]), capprops=dict(color=SERIES[0]))
        for b in bp["boxes"]:
            b.set(facecolor=SERIES[0], edgecolor=SERIES[0])
        ax.set_xticks(range(1, 11), [SHORT[p].replace("SP-", "") for p in POLES])
        ax.set_title(t.replace("\n", " "), loc="left", fontsize=9.5)
        ax.set_xlabel("杆號 SP-")
    save(fig, "10_normalization_boxes")


def fig_ks_circuits(a):
    """各迴路的跨杆平均 KS 距離，原始 vs 基準殘差。"""
    rows = []
    for col, name in [("w_total", "總功率 (MP)")] + CIRCUITS + [("w_unmetered", "未計量"), ("v", "電壓"),
                                                              ("people_count", "人流"), ("light_level", "路燈亮度")]:
        d = normalize_variants(a, col)
        for v in ["raw", "robust", "resid"]:
            m = ks_matrix(d, v, sample=2500)
            off = m[~np.eye(len(POLES), dtype=bool)]
            rows.append((name, v, float(np.nanmean(off))))
    r = pd.DataFrame(rows, columns=["feature", "variant", "ks"]).pivot(index="feature", columns="variant", values="ks")
    r = r[["raw", "robust", "resid"]].sort_values("raw", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(r))
    for k, (v, lab) in enumerate([("raw", "原始"), ("robust", "每杆 robust 標準化"), ("resid", "每杆×每小時基準殘差")]):
        ax.barh(y + (k - 1) * 0.27, r[v], height=0.25, color=SERIES[k], label=lab)
    ax.set_yticks(y, r.index); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("跨杆平均 KS D（越小 = 各杆分佈越接近）")
    ax.set_title("每個特徵經不同正規化後，跨杆分佈差異還剩多少", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "11_ks_by_feature")
    STATS["ks_by_feature"] = r.round(3).to_dict()


# ---- 10. 電壓：跨杆高度同步（共用電網） -------------------------------------------
def fig_voltage(a):
    piv = a.pivot_table(index="ts", columns="polename", values="v").reindex(columns=POLES)
    corr = piv.corr().values
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    wk = piv.loc["2026-01-12":"2026-01-14"]
    mean = wk.mean(axis=1)
    for p in POLES:
        ax.plot(wk.index, wk[p], color=AXIS, lw=0.8)
    ax.plot(wk.index, wk["SCCP-SP-10"], color=SERIES[1], lw=1.4, label="SP-10")
    ax.plot(wk.index, mean, color=SERIES[0], lw=1.6, label="10 杆平均")
    ax.set_ylabel("電壓 (V)"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H時"))
    ax.set_title("三天的電壓：10 杆幾乎同步起伏（灰線 = 其他各杆）", loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    heat(axes[1], corr, cmap=SEQ, vmin=0.5, vmax=1, fmt=lambda x: f"{x:.2f}",
         xt=[SHORT[p].replace("SP-", "") for p in POLES], yt=[SHORT[p] for p in POLES], fontsize=6.5)
    axes[1].set_title("杆間電壓相關係數（時段 A）", loc="left")
    fig.tight_layout()
    save(fig, "12_voltage")
    STATS["v_corr_min"] = float(np.nanmin(corr))
    STATS["v_median"] = piv.median().round(2).to_dict()


# ---- 11. 功率因數：兩種電表型號 --------------------------------------------------
def fig_pf(a):
    fig, ax = plt.subplots(figsize=(12, 3.8))
    for i, p in enumerate(POLES):
        vals = a.loc[a.polename == p, ["pf_mp", "pf_signage", "pf_light", "pf_camera"]].values.ravel()
        vals = vals[~np.isnan(vals)]
        vals = np.random.default_rng(i).choice(vals, min(3000, len(vals)), replace=False)
        color = SERIES[1] if p in MODEL20 else SERIES[0]
        ax.scatter(i + np.random.default_rng(i).normal(0, 0.09, len(vals)), vals, s=3, alpha=0.25, color=color, lw=0)
    ax.axhspan(-1, 1, color=SERIES[2], alpha=0.06, lw=0)
    ax.axhline(1, color=AXIS, lw=0.8); ax.axhline(-1, color=AXIS, lw=0.8); ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks(range(10), [SHORT[p] + ("\n-20 型" if p in MODEL20 else "\n-10 型") for p in POLES])
    ax.set_ylabel("功率因數 pf")
    ax.scatter([], [], color=SERIES[0], s=20, label="-10 型電表"); ax.scatter([], [], color=SERIES[1], s=20, label="-20 型電表")
    ax.legend(frameon=False, ncol=2, loc="lower left", fontsize=8.5)
    ax.set_title("各迴路 pf 分佈：-20 型電表出現負值與超出 ±1 的不可能值（綠底 = 合理範圍）", loc="left")
    save(fig, "13_pf_by_model")


# ---- 12. 人流：杆 × 小時、平日 vs 週末 -------------------------------------------
def fig_people(a):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    for ax, (wk, t) in zip([axes[0]], [(None, "")]):
        piv = a.pivot_table(index="polename", columns=a.hour.astype(int), values="people_count", aggfunc="mean").reindex(POLES)
        im = heat(ax, piv.values, xt=[str(h) if h % 3 == 0 else "" for h in range(24)], yt=[SHORT[p] for p in POLES])
        ax.set_title("平均人流（counting 事件數 / 15 分鐘）", loc="left"); ax.set_xlabel("時")
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01); cb.outline.set_visible(False)
    ax = axes[1]
    for is_we, color, lab in [(False, SERIES[0], "平日"), (True, SERIES[1], "週末")]:
        s = a[(a.dow >= 5) == is_we].groupby(a.hour)["people_count"].mean()
        ax.plot(s.index, s.values, color=color, label=lab)
    ax.set_xlabel("時"); ax.set_ylabel("人流（10 杆平均）")
    ax.set_title("平日 vs 週末的人流日曲線", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "14_people")


# ---- 13. 特徵相關（杆內 pooled，基準殘差後） ---------------------------------------
def fig_corr(a):
    cols = ["w_total", "w_signage", "w_light", "w_camera", "w_unmetered", "v", "light_level", "people_count"]
    names = ["總功率 MP", "數位看板", "路燈", "攝影機", "未計量", "電壓", "路燈亮度", "人流"]
    raw = a[cols].corr(method="spearman").values
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    heat(ax, raw, cmap=DIV, norm=TwoSlopeNorm(0, -1, 1), fmt=lambda x: f"{x:.2f}", xt=names, yt=names, fontsize=7.5)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("時段 A 特徵間 Spearman 相關（所有杆合併）", loc="left")
    save(fig, "15_feature_corr")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ROOT / "outputs" / "pole_ts_15min.parquet")
    a = df[df.period == "A"].copy()
    feats = [c for c in df.columns if c not in ("polename", "ts", "period", "hour", "dow", "n_circuits_b", "scale_fault")]

    steps = [
        lambda: fig_coverage(df, feats), lambda: fig_config(a), lambda: fig_week_stack(a),
        lambda: fig_hour_heatmaps(a), lambda: fig_daily_trend(df), lambda: fig_scale_fault(a),
        lambda: fig_light_consistency(a), lambda: fig_period_shift(df), lambda: fig_box_daynight(a),
        lambda: fig_ks(a), lambda: fig_ks_circuits(a), lambda: fig_voltage(a), lambda: fig_pf(a),
        lambda: fig_people(a), lambda: fig_corr(a),
    ]
    for i, s in enumerate(steps):
        s()
        print(f"fig {i} done")

    STATS["near_constant"] = {c: float(df[c].std()) for c in ["hz", "light_status_ok", "camera_status_ok", "person_detected"]}
    STATS["missing_A"] = a.groupby("polename")[feats].apply(lambda g: g.notna().mean()).round(3).to_dict()
    (OUT / "stats.json").write_text(json.dumps(STATS, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
