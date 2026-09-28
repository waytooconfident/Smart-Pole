"""專案共用設定：杆號、時段、切分日期、節日、標籤規則參數。各 script 以 `import config` 使用。"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"

# SP-03 的路燈 / 看板 / 智慧指標迴路整期 0W，整根杆排除
EXCLUDED_POLES = ["SCCP-SP-03"]
POLES = [f"SCCP-SP-{i:02d}" for i in range(1, 11) if f"SCCP-SP-{i:02d}" not in EXCLUDED_POLES]

FREQ = "15min"
PERIODS = {  # 右開區間
    "A": ("2026-01-01", "2026-03-01"),  # 電表有迴路明細
    "B": ("2026-05-01", "2026-07-01"),  # 電表只有杆體總量
}

# 依時間順序切分（右開區間）；參考 Xu et al. 2024：訓練段與定閾值的校準段分開
SPLITS = {
    "A_train": ("2026-01-01", "2026-01-26"),
    "A_cal":   ("2026-01-26", "2026-02-09"),
    "A_test":  ("2026-02-09", "2026-03-01"),
    "B_adapt": ("2026-05-01", "2026-05-25"),
    "B_cal":   ("2026-05-25", "2026-06-01"),
    "B_test":  ("2026-06-01", "2026-07-01"),
}
TRAIN_SPLITS = {"A": "A_train", "B": "B_adapt"}  # 基準值只用這兩段計算

# 115 年行政院人事行政總處辦公日曆表（含補假）
HOLIDAYS = {
    "元旦": ("2026-01-01", "2026-01-01"),
    "春節": ("2026-02-14", "2026-02-22"),
    "和平紀念日": ("2026-02-27", "2026-03-01"),
    "勞動節": ("2026-05-01", "2026-05-03"),
    "端午節": ("2026-06-19", "2026-06-21"),
}

# 異常標籤規則
MAD_K = 4.0          # |robust z| 超過幾倍 MAD 視為偏離
MIN_RUN = 2          # 連續幾個 15 分鐘時段才算事件（2 = 30 分鐘）
Z_FLOOR_W = 2.0      # robust 尺度下限（W），避免近乎常數的迴路被微小波動觸發
Z_FLOOR_REL = 0.10   # robust 尺度下限（相對基準值比例）；看板夜間待機 MAD 常 < 1W，5% 會過度敏感
EDGE_BINS = 4        # 斷訊 / 故障切換點前後幾個時段標為不確定（4 = 1 小時）

# 模型輸入視窗（第 4 步起使用）
WINDOW = 96          # 24 小時
STRIDE = 4           # 每 1 小時滑一次


def holiday_dates() -> set:
    days = set()
    for a, b in HOLIDAYS.values():
        days.update(pd.date_range(a, b, freq="D").date)
    return days
