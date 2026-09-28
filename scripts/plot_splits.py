"""把切分、異常標籤、注入事件畫成時間軸，檢查第 2 步的產出。

用法：
    uv run python scripts/plot_splits.py
輸出：
    outputs/eda/16_split_label_timeline.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

import config
from config import OUT_DIR, POLES

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
LABEL_COLORS = {  # 狀態色：只有異常類使用，正常留白
    "missing": "#c3c2b7", "uncertain": "#fab219", "stat": "#ec835a", "rule": "#d03b3b",
}
LABEL_NAMES = {"missing": "電表無資料", "uncertain": "不確定", "stat": "統計異常", "rule": "規則異常"}
INJ_COLOR = "#2a78d6"
HOLIDAY_BG = "#e3ecf8"

plt.rcParams.update({
    "font.family": ["Microsoft JhengHei", "sans-serif"], "font.size": 9.5,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "text.color": INK, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.titleweight": "bold", "axes.titlesize": 10.5, "axes.unicode_minus": False,
})
BIN = pd.Timedelta(config.FREQ)


def runs(ts: pd.Series, mask: pd.Series):
    """把連續為 True 的時段合併成 (start, width) 區間。"""
    t = ts[mask.values]
    if t.empty:
        return []
    brk = t.diff().ne(BIN).cumsum()
    return [(g.iloc[0], g.iloc[-1] + BIN - g.iloc[0]) for _, g in t.groupby(brk.values)]


def main():
    df = pd.read_parquet(OUT_DIR / "pole_ts_labeled.parquet")
    events = pd.read_csv(OUT_DIR / "injected_events.csv", parse_dates=["start", "end"])
    fig, axes = plt.subplots(2, 1, figsize=(15, 7.6))
    for ax, period in zip(axes, ["A", "B"]):
        d = df[df.period == period]
        a, b = config.PERIODS[period]
        for i, pole in enumerate(POLES):
            g = d[d.polename == pole].sort_values("ts")
            y = len(POLES) - 1 - i
            ax.broken_barh([(g.ts.min(), g.ts.max() - g.ts.min())], (y - 0.32, 0.46), color="#efeeea", lw=0)
            for lab, color in LABEL_COLORS.items():
                spans = [(mdates.date2num(s), w / pd.Timedelta("1D")) for s, w in runs(g.ts, g.label == lab)]
                ax.broken_barh(spans, (y - 0.32, 0.46), color=color, lw=0)
            ev = events[(events.period == period) & (events.polename == pole)]
            spans = [(mdates.date2num(s), (e - s) / pd.Timedelta("1D")) for s, e in zip(ev.start, ev.end)]
            ax.broken_barh(spans, (y + 0.18, 0.16), color=INJ_COLOR, lw=0)
        for name, (s, e) in config.HOLIDAYS.items():
            s, e = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta("1D")
            if s >= pd.Timestamp(a) and s < pd.Timestamp(b):
                ax.axvspan(s, e, color=HOLIDAY_BG, zorder=0, lw=0)
                ax.text(s, len(POLES) - 0.35, name, fontsize=8, color=MUTED, va="bottom")
        for name, (s, e) in config.SPLITS.items():
            if not name.startswith(period):
                continue
            ax.axvline(pd.Timestamp(s), color=INK2, lw=0.9)
            mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
            ax.text(mid, -1.05, name, ha="center", va="top", fontsize=9, color=INK, fontweight="bold")
        ax.set_xlim(pd.Timestamp(a), pd.Timestamp(b))
        ax.set_ylim(-1.4, len(POLES) + 0.2)
        ax.set_yticks(range(len(POLES)), [p.replace("SCCP-", "") for p in reversed(POLES)])
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.tick_params(length=0)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.set_title(f"時段 {period}：每根杆的標籤（下方粗條）與注入的模擬異常（上方藍色細條，僅測試段）", loc="left")
    handles = [Patch(color="#efeeea", label="乾淨正常")] + \
              [Patch(color=c, label=LABEL_NAMES[k]) for k, c in LABEL_COLORS.items()] + \
              [Patch(color=INJ_COLOR, label="注入的模擬異常"), Patch(color=HOLIDAY_BG, label="國定假日（背景）")]
    fig.legend(handles=handles, loc="lower left", ncol=7, frameon=False, bbox_to_anchor=(0.01, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_DIR / "eda" / "16_split_label_timeline.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
