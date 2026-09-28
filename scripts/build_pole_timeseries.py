"""把各子系統資料併成「每根杆 × 每 15 分鐘」一列的寬表，供 EDA 與後續建模使用。

兩個時段（電表資料只有這兩段）：
    A: 2026-01-01 ~ 2026-02-28  電表有迴路明細（meterinfo2）
    B: 2026-05-01 ~ 2026-06-30  電表只有杆體層級（xlsx，各迴路混在一起、無 circuitid）

資料修正：
    -20 型電表（SP-02/03/07/08/09）在 2026-01-02 ~ 01-22 期間 w、a 皆被放大 100 倍
    （攝影機迴路平常 ~13W，此期間讀到 ~1280W；a 亦飽和在 99.99）。
    依各杆實際切換時間偵測後除以 100，並以 scale_fault 欄位標記。

MP 迴路是杆體總電源（MP ≈ 其他各迴路加總，9 杆相關係數 0.995~1.000），因此：
    w_total     = MP 功率（不是所有迴路相加，否則重複計算）
    w_unmetered = MP − 各分迴路加總（未被分迴路量到的用電）
    時段 B 每個時間點的多筆中，最大者即 MP。

範圍：台北 SCCP-SP-01~10，排除 config.EXCLUDED_POLES（SP-03）；
依賴 outputs/device_pole_map.csv（先跑 build_device_pole_map.py）。

用法：
    uv run python scripts/build_pole_timeseries.py
輸出：
    outputs/pole_ts_15min.parquet
"""

import duckdb
import pandas as pd

import config
from config import FREQ, OUT_DIR, PERIODS, POLES, ROOT

METER_DIR = ROOT / "智慧電表" / "115年1月-2月_各設備耗能_台北"
XLSX_B = ROOT / "智慧電表" / "115年5月-6月_各杆體總耗能_台北_v2.xlsx"
CIRCUIT_FEATURE = {  # category_code -> 欄位後綴
    "01": "mp", "06": "network", "10": "light", "11": "indicator",
    "12": "signage", "13": "camera", "14": "env",
}
CSV_OPTS = "header=true, all_varchar=true, encoding='latin-1'"
TS = "COALESCE(TRY_STRPTIME(reporttime, '%Y/%m/%d %H:%M'), TRY_CAST(reporttime AS TIMESTAMP))"


def load_map() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "device_pole_map.csv", dtype=str)


def period_filter(col: str = "ts") -> str:
    return " OR ".join(f"({col} >= '{a}' AND {col} < '{b}')" for a, b in PERIODS.values())


def register_map(con, dmap: pd.DataFrame) -> None:
    con.register("dmap", dmap[["join_key", "polename", "system", "id_level", "category_code"]])


def meter_period_a(con) -> pd.DataFrame:
    """meterinfo2 → 每杆每迴路 15 分鐘平均，再 pivot 成寬表。"""
    src = (METER_DIR / "meterinfo2_202604011708.csv").as_posix()
    df = con.execute(f"""
        WITH r AS (
            SELECT m.polename, m.category_code, CAST(r.reporttime AS TIMESTAMP) AS t,
                   TRY_CAST(r.w AS DOUBLE) AS w, TRY_CAST(r.a AS DOUBLE) AS a,
                   TRY_CAST(r.pf AS DOUBLE) AS pf, TRY_CAST(r.v AS DOUBLE) AS v, TRY_CAST(r.hz AS DOUBLE) AS hz
            FROM read_csv('{src}', {CSV_OPTS}) r
            JOIN dmap m ON m.join_key = r.meterid AND m.id_level = 'circuit'
        ),
        -- ×100 故障區間：攝影機 / MP 迴路正常不會超過 200W 且電流 > 5A
        fault AS (
            SELECT polename, min(t) AS t0, max(t) AS t1 FROM r
            WHERE category_code IN ('01', '13') AND w > 200 AND a > 5
            GROUP BY 1
        )
        SELECT r.polename, r.category_code,
               time_bucket(INTERVAL '{FREQ}', r.t) AS ts,
               avg(r.w / CASE WHEN f.polename IS NOT NULL THEN 100 ELSE 1 END) AS w,
               avg(r.a / CASE WHEN f.polename IS NOT NULL THEN 100 ELSE 1 END) AS a,
               avg(r.pf) AS pf, avg(r.v) AS v, avg(r.hz) AS hz,
               max(CASE WHEN f.polename IS NOT NULL THEN 1 ELSE 0 END) AS scale_fault
        FROM r LEFT JOIN fault f ON f.polename = r.polename AND r.t BETWEEN f.t0 AND f.t1
        GROUP BY ALL
    """).df()
    df["feat"] = df["category_code"].map(CIRCUIT_FEATURE)

    wide = df.pivot_table(index=["polename", "ts"], columns="feat", values=["w", "pf", "a"])
    wide.columns = [f"{m}_{f}" for m, f in wide.columns]
    # 同一杆各迴路的 v、hz 幾乎相同（同一電源），取平均當杆體層級特徵
    pole_level = df.groupby(["polename", "ts"]).agg(
        v=("v", "mean"), hz=("hz", "mean"), scale_fault=("scale_fault", "max"))
    wide = wide.join(pole_level)
    sub_cols = [c for c in wide.columns if c.startswith("w_") and c != "w_mp"]
    wide["w_total"] = wide["w_mp"]
    wide["w_unmetered"] = wide["w_mp"] - wide[sub_cols].sum(axis=1, min_count=1)
    return wide.reset_index().assign(period="A")


def meter_period_b() -> pd.DataFrame:
    """5-6月 xlsx：每個時間點有 n 筆（各迴路），無 circuitid → 只能還原杆體總量。

    同一次回報的各迴路時間戳可能差 1 秒，先取到分鐘再分組；組內最大的一筆即 MP（總電源）。
    """
    frames = []
    sheets = pd.read_excel(XLSX_B, sheet_name=None, engine="calamine")
    for pole, df in sheets.items():
        df = df.dropna(subset=["reporttime"])
        df["t"] = pd.to_datetime(df["reporttime"]).dt.floor("min")
        snap = df.groupby("t").agg(w_total=("w", "max"), v=("v", "mean"), hz=("hz", "mean"), n=("w", "size"))
        snap["ts"] = snap.index.floor(FREQ)
        g = snap.groupby("ts").agg(w_total=("w_total", "mean"), v=("v", "mean"), hz=("hz", "mean"))
        g["n_circuits_b"] = int(snap["n"].mode().iloc[0])
        frames.append(g.reset_index().assign(polename=pole))
    return pd.concat(frames, ignore_index=True).assign(period="B")


def light_features(con) -> pd.DataFrame:
    pat = (ROOT / "路燈照明" / "lightnumberinfo_*.csv").as_posix()
    return con.execute(f"""
        WITH r AS (
            SELECT deviceid, attrid, {TS} AS ts0, TRY_CAST(value AS DOUBLE) AS val
            FROM read_csv('{pat}', {CSV_OPTS})
        )
        SELECT m.polename, time_bucket(INTERVAL '{FREQ}', ts0) AS ts,
               avg(val) FILTER (WHERE attrid = '100800') AS light_level,
               avg(CASE WHEN val = 1 THEN 1.0 ELSE 0.0 END) FILTER (WHERE attrid = '103800') AS light_status_ok
        FROM r JOIN dmap m ON m.join_key = r.deviceid
        WHERE {period_filter('ts0')}
        GROUP BY ALL
    """).df()


def camera_features(con) -> pd.DataFrame:
    """人流 = counting 事件數；每支攝影機各自計數後，同杆加總。"""
    str_pat = (ROOT / "人流辨識" / "camerastringinfo_*.csv").as_posix()
    num_pat = (ROOT / "人流辨識" / "cameranumberinfo_*.csv").as_posix()
    people = con.execute(f"""
        SELECT m.polename, time_bucket(INTERVAL '{FREQ}', CAST(r.reporttime AS TIMESTAMP)) AS ts,
               count(*) FILTER (WHERE r.value = 'counting')        AS people_count,
               count(*) FILTER (WHERE r.value = 'person_detected') AS person_detected
        FROM read_csv('{str_pat}', {CSV_OPTS}) r
        JOIN dmap m ON m.join_key = r.deviceid
        WHERE r.attrid = '103903' AND {period_filter('CAST(r.reporttime AS TIMESTAMP)')}
        GROUP BY ALL
    """).df()
    status = con.execute(f"""
        SELECT m.polename, time_bucket(INTERVAL '{FREQ}', CAST(r.reporttime AS TIMESTAMP)) AS ts,
               avg(CASE WHEN r.value = '1' THEN 1.0 ELSE 0.0 END) AS camera_status_ok
        FROM read_csv('{num_pat}', {CSV_OPTS}) r
        JOIN dmap m ON m.join_key = r.deviceid
        WHERE r.attrid = '103800' AND {period_filter('CAST(r.reporttime AS TIMESTAMP)')}
        GROUP BY ALL
    """).df()
    return people.merge(status, on=["polename", "ts"], how="outer")


def build_grid() -> pd.DataFrame:
    """完整時間格線：每杆每 15 分鐘都有一列，缺資料就留 NaN（缺值本身是 EDA 對象）。"""
    frames = []
    for p, (a, b) in PERIODS.items():
        ts = pd.date_range(a, b, freq=FREQ, inclusive="left")
        idx = pd.MultiIndex.from_product([POLES, ts], names=["polename", "ts"])
        frames.append(idx.to_frame(index=False).assign(period=p))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    con = duckdb.connect()
    register_map(con, load_map())

    print("meter A ..."); meter_a = meter_period_a(con)
    print("meter B ..."); meter_b = meter_period_b()
    print("light ...");   light = light_features(con)
    print("camera ...");  camera = camera_features(con)

    meter = pd.concat([meter_a, meter_b], ignore_index=True)
    df = build_grid()
    df = df.merge(meter.drop(columns="period"), on=["polename", "ts"], how="left")
    df = df.merge(light, on=["polename", "ts"], how="left")
    df = df.merge(camera, on=["polename", "ts"], how="left")
    # 有攝影機資料回報的時段，沒有事件 = 0 人；完全沒回報則保留 NaN
    has_cam = df["camera_status_ok"].notna()
    for c in ["people_count", "person_detected"]:
        df.loc[has_cam & df[c].isna(), c] = 0

    df["hour"] = df["ts"].dt.hour + df["ts"].dt.minute / 60
    df["dow"] = df["ts"].dt.dayofweek
    df["holiday"] = df["ts"].dt.date.isin(config.holiday_dates())
    df["offday"] = df["holiday"] | (df["dow"] >= 5)  # 週末或國定假日
    df = df.sort_values(["period", "polename", "ts"]).reset_index(drop=True)

    OUT_DIR.mkdir(exist_ok=True)
    df.to_parquet(OUT_DIR / "pole_ts_15min.parquet", index=False)
    print(df.shape)
    print(df.groupby("period").apply(lambda g: g.notna().mean().round(2), include_groups=False).T.to_string())


if __name__ == "__main__":
    main()
