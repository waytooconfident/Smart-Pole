# Smart-Pole：智慧杆異常用電預警

用台北松菸 10 根智慧杆的電表、路燈、人流資料，自動找出「用電不正常」的杆子，並依人流多寡排出維修優先順序。

- **輸入**：每根杆每 15 分鐘的用電、路燈亮度、人流
- **輸出**：異常警報清單（哪根杆、什麼時候、哪個數值怪、該多快處理）
- **成果展示**：一個網頁儀表板（dashboard），打開就能看

---

## 快速開始：打開 Dashboard

Dashboard 只需要兩個檔案，都在 `outputs/dashboard/`：

| 檔案 | 內容 |
|---|---|
| `index.html` | 頁面本身 |
| `data.json` | 目前所有的分數與警報資料（已經算好，不用再跑模型） |

> ⚠️ **不能直接雙擊 `index.html`**。瀏覽器用 `file://` 開檔時會擋掉讀取 `data.json`，畫面會停在「資料載入失敗」。
> 要先開一個本機小伺服器，步驟如下。

### 方法一：用 Python（最簡單）

1. 確認電腦有 Python 3（在終端機輸入 `python --version` 看得到版本就可以）
2. 在終端機切換到 dashboard 資料夾，啟動伺服器：

   ```bash
   cd outputs/dashboard
   python -m http.server 8765
   ```

3. 用瀏覽器打開 <http://localhost:8765>
4. 看完後回到終端機按 `Ctrl + C` 關掉伺服器

### 方法二：沒有 Python

- **VS Code**：安裝「Live Server」擴充套件，在 `index.html` 上按右鍵 →「Open with Live Server」
- **Node.js**：在 `outputs/dashboard` 裡執行 `npx serve`，再打開它顯示的網址

### 要傳給別人看

把整個 `outputs/dashboard` 資料夾壓成 zip 傳過去即可，對方照上面的方法打開。
不需要模型檔、原始資料或安裝本專案的 Python 環境。

### 常見問題

| 狀況 | 原因與解法 |
|---|---|
| 顯示「資料載入失敗」 | 直接雙擊開檔了，請改用上面的伺服器方式；或 `data.json` 沒跟 `index.html` 放在同一個資料夾 |
| 字型跟截圖不一樣 | 字型從 Google Fonts 載入，沒網路時會改用系統字型，內容不受影響 |
| 顯示 `Address already in use` | 8765 埠被佔用了，換一個數字，例如 `python -m http.server 8800`，網址也跟著改 |

---

## 其他可以直接看的成果

| 檔案 | 內容 | 怎麼看 |
|---|---|---|
| `outputs/eda/report.html` | 資料探索報告（建模前的統計圖表與結論） | 跟 dashboard 一樣要開伺服器：`cd outputs/eda` 再 `python -m http.server 8766` |
| `outputs/eda/*.png` | 報告裡的各張圖 | 直接點開 |
| `outputs/ablation/results.csv` | 各種模型設定的比較實驗結果 | Excel 開啟 |
| `outputs/inference/alarms_*.csv` | 2026 年 2 月、6 月的警報清單 | Excel 開啟 |
| `outputs/device_pole_map.xlsx` | 各設備 ID 對應到哪根杆 | Excel 開啟 |

---

## 專案在做什麼

### 資料

資料來自資策會數據包，只用台北場域 `SCCP-SP-01 ~ 10`（SP-03 因路燈、看板迴路整期為 0W 而排除）。

| 時段 | 期間 | 有什麼 |
|---|---|---|
| A 期 | 2026/1/1 – 2/28 | 電表有各迴路明細（看板、路燈、攝影機…） |
| B 期 | 2026/5/1 – 6/30 | 電表只有整根杆的總用電 |

路燈亮度與人流兩期都有。

### 方法

1. **整理資料**：把電表、路燈、人流併成「每根杆 × 每 15 分鐘」一列的表格，並修正電表讀值放大 100 倍的故障。
2. **標記正常 / 異常**：用每杆每小時的中位數當基準，偏離太多的時段標為異常，只拿正常時段訓練模型。
3. **模型**：雙分支 Autoencoder（LSTM）。模型學「正常時該長什麼樣子」，還原不好的時段就是可疑的。
   - 先用 A 期（有迴路明細）訓練，再用 B 期（只有總用電）微調。
4. **評估**：因為沒有真實的故障紀錄，我們在測試資料裡人工注入 8 種模擬異常（突增、斷電、漂移、該亮不亮等），看模型抓不抓得到。
5. **優先度**：警報依嚴重程度 × 當地人流排序，人多的地方先修。

更詳細的說明見：

- [提案.md](提案.md)：應用情境與提案
- [資料盤點與欄位分析.md](資料盤點與欄位分析.md)、[dataset_schema.md](dataset_schema.md)：原始資料有哪些欄位、怎麼串接
- [EDA_findings.md](EDA_findings.md)：原始資料探索結果
- [異常偵測模型規劃.md](異常偵測模型規劃.md)：模型設計規劃

---

## 從頭重跑（開發者用）

只看 dashboard 的話不需要這一段。

### 1. 安裝環境

需要 Python 3.12 與 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync
```

`pyproject.toml` 預設安裝 CUDA 12.8 版 PyTorch，沒有 NVIDIA 顯卡也能跑，只是比較慢。

### 2. 放原始資料

原始資料是競賽提供的，檔案很大（約 1.4 GB），**不放在 GitHub 上**。請另外取得後，照下面的位置放在專案根目錄：

```
Smart-Pole/
├── 杆體資訊_smartpole_list_2026.csv
├── 智慧電表/
│   ├── 115年1月-2月_各設備耗能_台北/      (deviceconfig、metercircuit、meterinfo2、meterprofile 四個 csv)
│   └── 115年5月-6月_各杆體總耗能_台北_v2.xlsx
├── 路燈照明/        (lightinfo_代號對照.xlsx、lightnumberinfo_*.csv)
├── 人流辨識/        (camerainfo_代號對照.xlsx、cameranumberinfo_*.csv、camerastringinfo_*.csv)
└── 環境感測/        (sensorinfo_代號對照.csv、sensornumberinfo_*.csv)
```

### 3. 依序執行

| 步驟 | 指令 | 產出 |
|---|---|---|
| 設備對應表 | `uv run python scripts/build_device_pole_map.py` | `outputs/device_pole_map.*` |
| 每杆 15 分鐘寬表 | `uv run python scripts/build_pole_timeseries.py` | `outputs/pole_ts_15min.parquet` |
| 資料探索圖表 | `uv run python scripts/eda_features.py` | `outputs/eda/` |
| 切分、標籤、注入模擬異常 | `uv run python scripts/build_splits.py` | `outputs/pole_ts_labeled.parquet` 等 |
| 特徵工程 | `uv run python scripts/build_features.py` | `outputs/features/` |
| 訓練（兩階段） | `uv run python scripts/train.py --stage 1`<br>`uv run python scripts/train.py --stage 2` | `outputs/models/` |
| 評估 | `uv run python scripts/evaluate.py` | `outputs/eval/` |
| 推論 | `uv run python scripts/infer.py --start 2026-06-01 --end 2026-07-01`<br>`uv run python scripts/infer.py --start 2026-02-09 --end 2026-03-01` | `outputs/inference/` |
| 更新 dashboard 資料 | `uv run python scripts/export_dashboard.py` | `outputs/dashboard/data.json` |
| 比較實驗（選用，較久） | `uv run python scripts/ablation.py` | `outputs/ablation/` |

各腳本最上方的說明文字有更詳細的輸入、輸出與參數。共用設定（杆號、切分日期、節日、閾值）都在 `scripts/config.py`。

---

## 資料夾結構

```
Smart-Pole/
├── scripts/            所有程式
├── outputs/
│   ├── dashboard/      網頁儀表板（index.html + data.json）
│   ├── eda/            資料探索報告與圖表
│   ├── features/       特徵表
│   ├── models/         訓練好的模型
│   ├── eval/           評估結果
│   ├── inference/      警報清單
│   └── ablation/       比較實驗結果
├── *.md                提案、規劃、資料說明文件
└── pyproject.toml      Python 套件清單
```
