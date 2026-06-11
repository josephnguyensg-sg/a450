# tool3.py
# tool này được sử dụng cho bước 4
# TRAIN MODE ONLY: Chỉ train Isolation Forest + lưu model.pkl
# Input : a450labeled.parquet (từ tool2)
# Output: behavior_model.pkl trong thư mục output/models
#
# Chạy độc lập khi cần retrain model (không export report)

import os
import gc
import time
import pickle
import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
import json as _json

_HERE     = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.environ.get("A450_BASE_DIR", _HERE)

# ── Load cấu hình từ tool3.json ───────────────────────────────────────────
with open(os.path.join(_HERE, "tool3.json"), encoding="utf-8") as _f:
    _CFG = _json.load(_f)

LABELED_FILE = os.environ.get("A450_LABELED_FILE", os.path.join(_BASE_DIR, _CFG["labeled_file"]))
OUTPUT_DIR   = os.environ.get("A450_OUTPUT_DIR",   os.path.join(_BASE_DIR, _CFG["output_dir"]))
MODEL_DIR    = os.path.join(OUTPUT_DIR, "models")  # ← Lưu model vào output/models

ML_FEATURES = _CFG["ml_features"]

CONFIG_ML = {
    "iso_contamination": _CFG["iso_contamination"],
    "iso_n_estimators" : _CFG["iso_n_estimators"],
    "random_state"     : _CFG["random_state"],
}



# =============================================================================
# TRAIN & SAVE MODEL
# =============================================================================
def _train_isolation_forest(
    labeled_file: str = None,
    output_dir: str = None,
) -> str:
    """
    Train Isolation Forest trên non-rule transactions và lưu model vào output/models.
    """
    t_total = time.time()

    if labeled_file is None:
        labeled_file = LABELED_FILE
    if output_dir is None:
        output_dir = OUTPUT_DIR

    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("III. ISOLATION FOREST TRAINING")
    print("=" * 60)

    # Load dữ liệu
    print(f"Đọc file tại: {labeled_file}")
    df = pl.read_parquet(labeled_file)
    print(f"Đã load {df.shape[0]:,} giao dịch")

    # Lấy bookie_set
    bookie_set = set(
        df.filter(pl.col("is_bookie_tx") == 1)["appuser"].unique().to_list()
    )
    print(f"Tìm thấy {len(bookie_set):,} bookies")

    # Train trên non-rule only
    df_norule = df.filter(pl.col("hit_any_rule") == 0)
    n_norule = df_norule.shape[0]
    print(f"Training dựa trên {n_norule:,} giao dịch non-rule...")

    if n_norule < 1000:
        print("⚠️ Cảnh báo: Số lượng non-rule transactions khá ít.")

    t0 = time.time()
    X = (
        df_norule.select(ML_FEATURES)
        .fill_null(0)
        .cast(pl.Float32)
        .to_numpy()
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=CONFIG_ML["iso_n_estimators"],
        contamination=CONFIG_ML["iso_contamination"],
        random_state=CONFIG_ML["random_state"],
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    # Lưu model
    model_path = os.path.join(model_dir, "behavior_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "config": CONFIG_ML,
            "scaler": scaler,
            "iso_model": iso,
            "ml_features": ML_FEATURES,
            "bookie_set": bookie_set,
            "train_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_train_samples": n_norule,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Machine learning model lưu tại: {model_path}")

    # Thống kê nhanh
    raw = iso.decision_function(X_scaled)
    ml_scores = np.clip(
        (raw.max() - raw) / (raw.max() - raw.min() + 1e-9) * 100,
        0, 100
    ).round(2)

    print(f"\nKết quả training")
    print(f"  Samples trained : {n_norule:,}")
    print(f"  Mean score      : {ml_scores.mean():.2f}")
    print(f"  P95 score       : {np.percentile(ml_scores, 95):.2f}")
    print(f"  Max score       : {ml_scores.max():.2f}")

    del X, X_scaled, raw, ml_scores, df, df_norule
    gc.collect()

    print(f"\n✅ TRAIN HOÀN TẤT ({time.time()-t_total:.1f}s)")
    return model_path


# --- Chạy trực tiếp ---
if __name__ == "__main__":
    _train_isolation_forest()