# Dataset 關聯圖（Top-down）

> 每條線都標明「用哪個欄位當連接鍵（PK/FK）」。
> 最上層的共用主鍵是 `杆體資訊_smartpole_list_2026.csv` 的 `polename`（15 根杆：台北 SCCP-SP-01~10、高雄 A1~A5）。

```
                          ┌───────────────────────────────────────────────┐
                          │  杆體資訊_smartpole_list_2026.csv               │
                          │  ── 全資料集的最上層主鍵 ──                     │
                          │  PK: polename  (SCCP-SP-01~10, A1~A5)          │
                          │  其他: latitude, longitude, groupid            │
                          └───────────────────────────────────────────────┘
                             │              │              │            │
        ┌────────────────────┘              │              │            └────────────────────┐
        │ 連接鍵：                          │ 連接鍵：      │ 連接鍵：                        │ 連接鍵：
        │ latitude + longitude              │ polename      │ polename                        │ polename
        │ （電表側無 polename，             │ （對照表      │ （對照表                        │ （對照表
        │   靠經緯度對回，10/10 對上）      │   deviceid→   │   deviceid→                     │   deviceid→
        ▼                                   ▼   polename)   ▼   polename)                     ▼   polename)
┌───────────────────────┐   ┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│ 電表（1月-2月_台北）  │   │ 路燈照明                │ │ 人流辨識                │ │ 環境感測                │
└───────────────────────┘   └─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
        │                             │                           │                           │
        │  ↓ 見下方電表區塊展開       │                           │                           │
        │                             ▼                           ▼                           ▼
        │                   ┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
        │                   │ lightinfo_        │       │ camerainfo_       │       │ sensorinfo_       │
        │                   │ 代號對照.xlsx      │       │ 代號對照.xlsx      │       │ 代號對照.csv(Big5)│
        │                   │ PK: deviceid      │       │ PK: deviceid      │       │ PK: deviceid      │
        │                   │ (INL1---…)        │       │ (IN30---…)        │       │ (IN21---…)        │
        │                   │ → polename        │       │ → polename        │       │ → polename        │
        │                   │ + attrid 代號說明 │       │ + attrid 代號說明 │       │ + attrid 代號說明 │
        │                   └───────────────────┘       └───────────────────┘       └───────────────────┘
        │                             │ 連接鍵：deviceid          │ 連接鍵：deviceid          │ 連接鍵：deviceid
        │                             ▼                           ▼                           ▼
        │                   ┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
        │                   │ lightnumberinfo   │       │ cameranumberinfo  │       │ sensornumberinfo  │
        │                   │ _114.12-115.04    │       │ _114.12-115.04 ⚠ │       │ _114.12-115.04    │
        │                   │ _115.05-115.06    │       │ _115.05-115.06    │       │ _115.05-115.06    │
        │                   │ FK: deviceid      │       │ camerastringinfo  │       │ FK: deviceid      │
        │                   │ 欄:attrid,        │       │ _115.05-115.06    │       │ 欄:attrid,        │
        │                   │   reporttime,value│       │ FK: deviceid      │       │   reporttime,value│
        │                   └───────────────────┘       └───────────────────┘       └───────────────────┘
        │
        │  ── 電表區塊展開（資料夾：115年1月-2月_各設備耗能_台北）──
        ▼
┌─────────────────────────────┐
│ deviceconfig                │
│ _202604011636.csv           │
│ PK: dcid                    │
│ 對回杆體: latitude+longitude│
│ 也有 deviceid, devicename   │
└─────────────────────────────┘
        │ 連接鍵：dcid
        ▼
┌─────────────────────────────┐        ┌──────────────────────────────────────────────┐
│ metercircuit                │        │ 說明：一根杆(dcid) 拆成 6~7 個迴路(circuitid)  │
│ _202604011646.csv           │───────▶│ loadname 標明是哪個設備                        │
│ PK: circuitid               │        │ category=10 → 路燈                             │
│ FK: dcid                    │        │ category=1→MP, 12→數位看板, 11→智慧指標,      │
│ 欄: loadname, category      │        │          13→攝影機, 6→網路設備, 14→環境感測   │
└─────────────────────────────┘        └──────────────────────────────────────────────┘
        │ 連接鍵：circuitid  ==  meterid   （同一個值，換了欄位名）
        ▼
┌─────────────────────────────┐        ┌──────────────────────────────────────────────┐
│ meterinfo2                  │        │ 主檔，152 萬列，2026-01-01 ~ 02-28            │
│ _202604011708.csv (158MB)   │◀───────│ 每 3 分鐘一筆                                  │
│ FK: meterid (=circuitid)    │        │ 用電量看 w 這一欄                              │
│ 欄: reporttime, v, a, w,    │        └──────────────────────────────────────────────┘
│     pf, whplus …            │
└─────────────────────────────┘

  （meterprofile_202604011700.csv 是輔助對照，欄位 meterid/dcid/circuitid，
    用來確認 meterid == circuitid，不參與主資料流。）
```

---

## 連接鍵一覽（每條線用哪個欄位當 PK/FK）

| 從 | 到 | 連接鍵 | 說明 |
|---|---|---|---|
| `杆體資訊`.`polename` | `deviceconfig`（電表側） | `latitude` + `longitude` | 電表側沒有 `polename`，靠經緯度對回，10/10 對得上 |
| `杆體資訊`.`polename` | `lightinfo_代號對照`.`polename` | `polename` | 對照表把 `deviceid` 對到 `polename` |
| `杆體資訊`.`polename` | `camerainfo_代號對照`.`polename` | `polename` | 同上 |
| `杆體資訊`.`polename` | `sensorinfo_代號對照`.`polename` | `polename` | 同上 |
| `deviceconfig`.`dcid` | `metercircuit`.`dcid` | `dcid` | 一根杆對到它底下的迴路 |
| `metercircuit`.`circuitid` | `meterinfo2`.`meterid` | `circuitid` == `meterid` | 同一個值，兩檔欄位名不同 |
| `lightinfo_代號對照`.`deviceid` | `lightnumberinfo`.`deviceid` | `deviceid` | `INL1---` 開頭 |
| `camerainfo_代號對照`.`deviceid` | `cameranumberinfo` / `camerastringinfo`.`deviceid` | `deviceid` | `IN30---` 開頭 |
| `sensorinfo_代號對照`.`deviceid` | `sensornumberinfo`.`deviceid` | `deviceid` | `IN21---` 開頭 |

---

## 備註

- ⚠ `cameranumberinfo_114.12-115.04.csv`（173MB）目前被系統占用讀不到。
- 電表側要連到 `polename`，路徑是：`meterinfo2`.`meterid` → `metercircuit`.`circuitid`/`dcid` → `deviceconfig`.`dcid`，再用 `deviceconfig` 的 `latitude`+`longitude` 對回 `杆體資訊`.`polename`。
- 三份代號對照表（`lightinfo` / `camerainfo` / `sensorinfo`）除了 `deviceid`→`polename`，還帶各自 `attrid` 的代號說明（例如 `100800`=亮度、`103800`=設備狀態、`400100`=溫度…）。
- 環境感測的對照表只列 2 個 `deviceid`（對到 `SCCP-SP-05`、`A2`），所以只有這 2 根杆有環境資料。
