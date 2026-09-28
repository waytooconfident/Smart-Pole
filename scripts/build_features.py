"""第 3 步：特徵工程。轉換邏輯在 features.py（推論時共用）。

用法：
    uv run python scripts/build_features.py
輸入：
    outputs/pole_ts_labeled.parquet、outputs/test_injected_{A,B}.parquet（build_splits.py 產出）
輸出：
    outputs/features/features.parquet            全部時段（未注入）
    outputs/features/features_injected_A.parquet A_test 注入後
    outputs/features/features_injected_B.parquet B_test 注入後
    outputs/features/scalers.json                正規化參數與特徵清單
"""

import pandas as pd

import features as F
from config import OUT_DIR

FEAT_DIR = OUT_DIR / "features"


def main() -> None:
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(OUT_DIR / "pole_ts_labeled.parquet")
    scalers = F.fit(df)
    F.save(scalers, FEAT_DIR / "scalers.json")

    feats = F.transform(df, scalers)
    feats.to_parquet(FEAT_DIR / "features.parquet", index=False)
    # 混合版本（H1）：統計基準殘差特徵，另存 *_resid 檔；原本的特徵檔不變
    resid_fit = F.fit_residual(feats)
    F.residualize(feats, resid_fit).to_parquet(FEAT_DIR / "features_resid.parquet", index=False)
    for p in "AB":
        inj = F.transform(pd.read_parquet(OUT_DIR / f"test_injected_{p}.parquet"), scalers)
        inj.to_parquet(FEAT_DIR / f"features_injected_{p}.parquet", index=False)
        F.residualize(inj, resid_fit).to_parquet(FEAT_DIR / f"features_resid_injected_{p}.parquet", index=False)
        val = F.transform(pd.read_parquet(OUT_DIR / f"val_injected_{p}.parquet"), scalers)  # 驗證用（校準段注入）
        val.to_parquet(FEAT_DIR / f"features_val_injected_{p}.parquet", index=False)
        F.residualize(val, resid_fit).to_parquet(FEAT_DIR / f"features_resid_val_injected_{p}.parquet", index=False)

    # ---- 摘要：乾淨正常時段各特徵的平均 / 標準差，看 A、B 之間的位移 ----
    cols = F.POLE_FEATURES + F.CIRCUIT_FEATURES
    n = feats[feats.label == "normal"]
    obs = {c: n[c].where(n[f"m_{c}"] == 1) for c in cols}
    obs = pd.DataFrame(obs).assign(split=n.split.values)
    print("== 各切分乾淨正常時段：平均")
    print(obs.groupby("split")[cols].mean().round(2).T.to_string())
    print("\n== 標準差")
    print(obs.groupby("split")[cols].std().round(2).T.to_string())
    print("\n== 遮罩覆蓋率（觀測比例）")
    print(n.groupby("split")[[f"m_{c}" for c in cols] + ["m_light_level"]].mean().round(2).T.to_string())
    night = n[(n.ts.dt.hour >= 20) | (n.ts.dt.hour < 5)]
    print("\n== 夜間 w_total_s 中位數：A_train vs B_adapt（B 期路燈調光造成的位移）")
    print(night[night.split.isin(["A_train", "B_adapt"])].pivot_table(
        index="polename", columns="split", values="w_total_s", aggfunc="median").round(2).to_string())
    print(f"\n特徵表 {feats.shape}，特徵：pole={F.POLE_FEATURES} circuit={F.CIRCUIT_FEATURES} cond={F.COND_FEATURES}")


if __name__ == "__main__":
    main()
