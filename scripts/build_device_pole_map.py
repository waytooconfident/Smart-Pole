"""建立 device_pole_map：把各子系統的設備 / 迴路 ID 對應到台北智慧杆 (SCCP-SP-01~10)。

每一列 = 一個會出現在數據表裡的 ID（join_key），告訴你它屬於哪根杆、是什麼設備、
要跟哪張表的哪個欄位 join。

範圍：只含台北場域；metercircuit 中 loadid=0 的迴路視為異常資料，已排除。

用法：
    uv run python scripts/build_device_pole_map.py
輸出：
    outputs/device_pole_map.csv   (UTF-8 BOM，Excel 可直接開)
    outputs/device_pole_map.xlsx  (含「欄位說明」工作表)
"""

from pathlib import Path
import re

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
METER_DIR = ROOT / "智慧電表" / "115年1月-2月_各設備耗能_台北"

TAIPEI_PREFIX = "SCCP-SP-"

CATEGORY_NAMES = {
    "01": "MP",
    "06": "網路設備",
    "10": "路燈",
    "11": "智慧指標",
    "12": "數位看板",
    "13": "攝影機",
    "14": "環境感測",
}

COLUMNS = [
    "polename", "system", "id_level", "device_type", "join_key", "join_to",
    "dcid", "loadid", "category_code", "meter_model", "latitude", "longitude",
]


def pole_from_number(n: int) -> str:
    return f"{TAIPEI_PREFIX}{n:02d}"


def normalize_id(s: str) -> str:
    """代號對照表寫 'INL1 --- 123'，數據檔寫 'INL1---123'，統一去掉空白。"""
    return re.sub(r"\s+", "", str(s))


def read_code_table_mapping(path: Path) -> pd.DataFrame:
    """代號對照表的下半段是 group / deviceid / polename 對照，找到表頭後往下讀。"""
    if path.suffix == ".xlsx":
        raw = pd.read_excel(path, header=None, dtype=str, engine="calamine")
    else:
        raw = pd.read_csv(path, header=None, dtype=str, encoding="cp950")
    header_row = raw.index[raw.iloc[:, 1].eq("deviceid")][0]
    df = raw.iloc[header_row + 1:, :3].copy()
    df.columns = ["group", "deviceid", "polename"]
    df = df.dropna(subset=["deviceid", "polename"])
    df["deviceid"] = df["deviceid"].map(normalize_id)
    df["polename"] = df["polename"].str.strip()
    return df[df["polename"].str.startswith(TAIPEI_PREFIX)]


def build_sensor_rows() -> pd.DataFrame:
    sources = [
        ("路燈", "路燈控制器", ROOT / "路燈照明" / "lightinfo_代號對照.xlsx",
         "lightnumberinfo_*.deviceid"),
        ("攝影機", "攝影機", ROOT / "人流辨識" / "camerainfo_代號對照.xlsx",
         "cameranumberinfo_*.deviceid / camerastringinfo_*.deviceid"),
        ("環境感測", "環境感測器", ROOT / "環境感測" / "sensorinfo_代號對照.csv",
         "sensornumberinfo_*.deviceid"),
    ]
    frames = []
    for system, device_type, path, join_to in sources:
        m = read_code_table_mapping(path)
        frames.append(pd.DataFrame({
            "polename": m["polename"],
            "system": system,
            "id_level": "device",
            "device_type": device_type,
            "join_key": m["deviceid"],
            "join_to": join_to,
        }))
    return pd.concat(frames, ignore_index=True)


def read_deviceconfig() -> pd.DataFrame:
    # devicename 的中文已是無法還原的亂碼，但 '#01' 還在，用它取得杆號
    dc = pd.read_csv(METER_DIR / "deviceconfig_202604011636.csv", dtype=str,
                     encoding="utf-8", encoding_errors="replace")
    dc["polename"] = dc["devicename"].str.extract(r"#(\d+)")[0].astype(int).map(pole_from_number)
    # deviceid 後綴 -10 / -20 對應兩種電表型號（-20 另有 va/var 欄位）
    dc["meter_model"] = dc["deviceid"].str.extract(r"-(\d+)$")[0]
    return dc[["polename", "dcid", "deviceid", "meter_model"]]


def build_meter_rows(dc: pd.DataFrame) -> pd.DataFrame:
    pole_meter = pd.DataFrame({
        "polename": dc["polename"],
        "system": "電表",
        "id_level": "device",
        "device_type": "杆體電表(收集器)",
        "join_key": dc["deviceid"],
        "join_to": "115年5月-6月_各杆體總耗能_台北_v2.xlsx.deviceid（sheet 名稱 = polename）",
        "dcid": dc["dcid"],
        "meter_model": dc["meter_model"],
    })

    mc = pd.read_csv(METER_DIR / "metercircuit_202604011646.csv", dtype=str, encoding="utf-8-sig")
    mc = mc[mc["loadid"] != "0"]  # loadid=0：無名稱、數值異常，排除
    mc = mc.merge(dc[["dcid", "polename", "meter_model"]], on="dcid", how="left")

    # 交叉檢查：loadname 裡的杆號應與 dcid 推得的杆號一致
    name_pole = mc["loadname"].str.extract(r"#(\d+)")[0].astype(int).map(pole_from_number)
    mismatch = mc[name_pole != mc["polename"]]
    if not mismatch.empty:
        raise ValueError(f"loadname 與 dcid 的杆號不一致：\n{mismatch}")

    circuits = pd.DataFrame({
        "polename": mc["polename"],
        "system": "電表",
        "id_level": "circuit",
        "device_type": mc["category"].map(CATEGORY_NAMES) + "迴路",
        "join_key": mc["circuitid"],
        "join_to": "meterinfo2.meterid / meterprofile.meterid",
        "dcid": mc["dcid"],
        "loadid": mc["loadid"],
        "category_code": mc["category"],
        "meter_model": mc["meter_model"],
    })
    return pd.concat([pole_meter, circuits], ignore_index=True)


def attach_pole_info(df: pd.DataFrame, dc: pd.DataFrame) -> pd.DataFrame:
    poles = pd.read_csv(ROOT / "杆體資訊_smartpole_list_2026.csv", dtype=str, encoding="utf-8")
    df = df.drop(columns=["dcid", "meter_model"], errors="ignore").merge(
        dc[["polename", "dcid", "meter_model"]], on="polename", how="left")
    df = df.merge(poles[["polename", "latitude", "longitude"]], on="polename", how="left")
    return df


def validate_against_data(df: pd.DataFrame) -> pd.DataFrame:
    """確認每個 join_key 都真的出現在對應的數據檔裡，回報命中筆數。"""
    con = duckdb.connect()
    checks = {
        "路燈": ("路燈照明/lightnumberinfo_*.csv", "deviceid"),
        "攝影機": ("人流辨識/cameranumberinfo_*.csv", "deviceid"),
        "環境感測": ("環境感測/sensornumberinfo_*.csv", "deviceid"),
    }
    # 各檔編碼不一（部分為 cp950），ID 本身是純 ASCII，用 latin-1 讀即可避免解碼錯誤
    opts = "header=true, all_varchar=true, encoding='latin-1'"
    counts = {}
    for system, (pattern, col) in checks.items():
        q = f"SELECT {col} AS k, count(*) AS n FROM read_csv('{(ROOT / pattern).as_posix()}', {opts}) GROUP BY 1"
        counts[system] = dict(con.execute(q).fetchall())
    q = f"SELECT meterid, count(*) FROM read_csv('{(METER_DIR / 'meterinfo2_202604011708.csv').as_posix()}', {opts}) GROUP BY 1"
    circuit_counts = dict(con.execute(q).fetchall())

    def rows_in_data(r):
        if r["id_level"] == "circuit":
            return circuit_counts.get(r["join_key"], 0)
        if r["system"] == "電表":
            return pd.NA  # 5-6月 xlsx 已人工驗證 10/10 吻合，此處不重讀 70MB 檔
        return counts[r["system"]].get(r["join_key"], 0)

    df["rows_in_data"] = df.apply(rows_in_data, axis=1).astype("Int64")
    return df


FIELD_DOC = pd.DataFrame([
    ("polename", "杆號（SCCP-SP-01~10），跨系統串接的共同鍵"),
    ("system", "子系統：電表 / 路燈 / 攝影機 / 環境感測"),
    ("id_level", "device = 一台實體設備；circuit = 電表下的一個迴路"),
    ("device_type", "設備或迴路類型"),
    ("join_key", "要拿去 join 的 ID 值（deviceid 或 circuitid），已去除空白"),
    ("join_to", "join_key 對應到哪個檔案的哪個欄位"),
    ("dcid", "該杆的資料收集器編號（deviceconfig.dcid）"),
    ("loadid", "迴路序號（僅 circuit）；loadid=0 為異常迴路，已排除"),
    ("category_code", "迴路分類代碼（僅 circuit）：01=MP、06=網路設備、10=路燈、11=智慧指標、12=數位看板、13=攝影機、14=環境感測"),
    ("meter_model", "電表型號（deviceid 後綴）：10 或 20；20 型另有 va/var 欄位"),
    ("latitude / longitude", "杆體座標（杆體資訊_smartpole_list_2026.csv）"),
    ("rows_in_data", "該 ID 在數據檔中的筆數（驗證用）；杆體電表未重新統計，留空"),
], columns=["欄位", "說明"])

NOTES = pd.DataFrame({"注意事項": [
    "範圍：僅台北場域 SCCP-SP-01~10，不含高雄。",
    "一根杆可能有多台同類設備：SP-09 有 2 台路燈控制器，SP-07、SP-08 各有 2 台攝影機。聚合到杆層級時需決定合併方式。",
    "環境感測只有 SP-05 一台。",
    "迴路（circuit）ID 只適用 115年1-2月的 meterinfo2 / meterprofile；5-6月的電表 xlsx 沒有 circuitid 欄位。",
    "5-6月「各杆體總耗能」xlsx 實際上每個時間點有 6 筆（各迴路），並非杆體總量，使用時需加總或另行還原迴路。",
]})


def main() -> None:
    dc = read_deviceconfig()
    df = pd.concat([build_sensor_rows(), build_meter_rows(dc)], ignore_index=True)
    df = attach_pole_info(df, dc)
    df = df[COLUMNS]
    df = df.sort_values(["polename", "system", "id_level", "category_code", "join_key"],
                        na_position="first").reset_index(drop=True)
    df = validate_against_data(df)

    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "device_pole_map.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT_DIR / "device_pole_map.xlsx", engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="device_pole_map", index=False)
        FIELD_DOC.to_excel(xw, sheet_name="欄位說明", index=False)
        NOTES.to_excel(xw, sheet_name="注意事項", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width * 1.3), 80)
            ws.freeze_panes = "A2"

    print(f"共 {len(df)} 列")
    print(df.groupby(["system", "id_level"]).size().to_string())
    missing = df[df["rows_in_data"].eq(0)]
    print(f"\n在數據檔中找不到的 ID：{len(missing)} 個")
    if not missing.empty:
        print(missing[["polename", "system", "device_type", "join_key"]].to_string(index=False))


if __name__ == "__main__":
    main()
