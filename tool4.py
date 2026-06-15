# tool4.py
# INFERENCE + FULL REPORT + DASHBOARD (Hoàn chỉnh)
# Load model → Score → Export reports + Visualizations

import os
import gc
import time
import pickle
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from tool6 import export_html_and_send

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import json as _json

_HERE     = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.environ.get("A450_BASE_DIR", _HERE)

# ── Load cấu hình từ tool4.json ───────────────────────────────────────────
with open(os.path.join(_HERE, "tool4.json"), encoding="utf-8") as _f:
    _CFG = _json.load(_f)

LABELED_FILE       = os.environ.get("A450_LABELED_FILE",       os.path.join(_BASE_DIR, _CFG["labeled_file"]))
OUTPUT_DIR         = os.environ.get("A450_OUTPUT_DIR",         os.path.join(_BASE_DIR, _CFG["output_dir"]))
MODEL_PATH         = os.environ.get("A450_MODEL_PATH",         os.path.join(_BASE_DIR, _CFG["model_path"]))
ML_SCORE_THRESHOLD = int(os.environ.get("A450_ML_SCORE_THRESHOLD", _CFG["ml_score_threshold"]))
ML_FEATURES        = _CFG["ml_features"]

# ============================== 1. INFERENCE SCORING ======================================

def _inference_score(df: pl.DataFrame, scaler, iso, ml_features) -> pl.DataFrame:
    t0 = time.time()
    df_norule = df.filter(pl.col("hit_any_rule") == 0)

    if df_norule.height == 0:
        print("[Inference] Không có giao dịch non-rule.")
        return df.with_columns(pl.lit(None).cast(pl.Float32).alias("ml_score"))

    X = df_norule.select(ml_features).fill_null(0).cast(pl.Float32).to_numpy()
    X_scaled = scaler.transform(X)

    raw = iso.decision_function(X_scaled)
    ml_scores = np.clip(
        (raw.max() - raw) / (raw.max() - raw.min() + 1e-9) * 100,
        0, 100
    ).round(2)

    df_norule = df_norule.with_columns(pl.Series("ml_score", ml_scores, dtype=pl.Float32))
    df_rule = df.filter(pl.col("hit_any_rule") == 1).with_columns(
        pl.lit(None).cast(pl.Float32).alias("ml_score")
    )

    df = pl.concat([df_rule, df_norule]).sort("reqdate")
    print(f"☕ Scoring hoàn tất trong {time.time()-t0:.1f}s")
    return df


# ============================== 2. USER REPORT =====================================

def _step9_user_report(df: pl.DataFrame, bookie_set: set) -> pl.DataFrame:
    t0 = time.time()

    all_users = pl.concat([
        df.select(pl.col("userid").alias("user_id")),
        df.select(pl.col("appuser").alias("user_id")),
    ]).unique()

    def count_hits(flag_col: str, id_col: str, out_col: str) -> pl.DataFrame:
        return (
            df.lazy()
            .filter(pl.col(flag_col) == 1)
            .group_by(id_col)
            .agg(pl.len().alias(out_col))
            .collect()
            .rename({id_col: "user_id"})
        )

    rule_tbls = [
        count_hits("is_bookie_tx",     "appuser", "bookie_tx_count"),
        count_hits("is_gambler_tx",    "userid",  "gambler_tx_count"),
        count_hits("is_recipient1_tx", "appuser", "recipient1_tx_count"),
        count_hits("is_depositor1_tx", "userid",  "depositor1_tx_count"),
        count_hits("is_recipient2_tx", "appuser", "recipient2_tx_count"),
        count_hits("is_depositor2_tx", "userid",  "depositor2_tx_count"),
    ]

    ml_df = df.filter(pl.col("ml_score").is_not_null())
    ml_sender = ml_df.lazy().group_by("userid").agg(pl.col("ml_score").mean().alias("ml_s")).collect().rename({"userid": "user_id"})
    ml_recv = ml_df.lazy().group_by("appuser").agg(pl.col("ml_score").mean().alias("ml_r")).collect().rename({"appuser": "user_id"})
    del ml_df; gc.collect()

    sent_sum = (
        df.lazy().group_by("userid").agg([
            pl.count("amount").alias("total_sent_tx"),
            pl.col("amount").sum().alias("total_sent_amount"),
            pl.col("appuser").n_unique().alias("unique_receivers"),
        ]).collect().rename({"userid": "user_id"})
    )
    recv_sum = (
        df.lazy().group_by("appuser").agg([
            pl.count("amount").alias("total_recv_tx"),
            pl.col("amount").sum().alias("total_recv_amount"),
            pl.col("userid").n_unique().alias("unique_senders"),
        ]).collect().rename({"appuser": "user_id"})
    )

    rpt = all_users
    for tbl in rule_tbls + [ml_sender, ml_recv, sent_sum, recv_sum]:
        rpt = rpt.join(tbl, on="user_id", how="left")

    cnt_cols = [
        "bookie_tx_count", "gambler_tx_count", "recipient1_tx_count", "depositor1_tx_count",
        "recipient2_tx_count", "depositor2_tx_count", "total_sent_tx", "total_sent_amount",
        "unique_receivers", "total_recv_tx", "total_recv_amount", "unique_senders"
    ]
    rpt = rpt.with_columns([pl.col(c).fill_null(0) for c in cnt_cols])

    rpt = rpt.with_columns(
        ((pl.col("ml_s").fill_null(0) + pl.col("ml_r").fill_null(0)) /
         (pl.col("ml_s").is_not_null().cast(pl.Float32) + pl.col("ml_r").is_not_null().cast(pl.Float32) + 1e-9)
        ).round(2).alias("ml_score")
    )

    GRP_MAP = {
        "bookie": "bookie_tx_count", "gambler": "gambler_tx_count",
        "recipient1": "recipient1_tx_count", "depositor1": "depositor1_tx_count",
        "recipient2": "recipient2_tx_count", "depositor2": "depositor2_tx_count",
    }
    rpt = rpt.with_columns(
        pl.sum_horizontal([pl.col(c) for c in GRP_MAP.values()]).alias("total_flagged_tx")
    )

    label_exprs = [
        pl.when(pl.col(col) > 0).then(pl.lit(grp)).otherwise(pl.lit(""))
        for grp, col in GRP_MAP.items()
    ]
    rpt = rpt.with_columns(
        pl.concat_str(label_exprs, separator="|")
        .str.replace_all(r"\|{2,}", "|")
        .str.strip_chars("|")
        .str.replace_all(r"^$", "normal")
        .alias("user_group")
    )
    rpt = rpt.sort(["total_flagged_tx", "ml_score"], descending=True)

    print(f"📤 Tổng hợp reports hoàn thành trong {time.time()-t0:.1f}s | Flagged users: {rpt.filter(pl.col('user_group') != 'normal').shape[0]:,}")
    return rpt

# ================================= 3. EXPORT ====================================

def _export_reports_and_figures(df: pl.DataFrame, rpt: pl.DataFrame, bookie_set: set, output_dir: str):
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    def out(name): return os.path.join(output_dir, name)

    # I. Nhóm Transactions
    TX_COLS = [c for c in ["reqdate", "userid", "appuser", "amount", "desc_clean", "platform", "userip", "iploc",
                           "is_bookie_tx", "is_gambler_tx", "is_recipient1_tx", "is_depositor1_tx",
                           "is_recipient2_tx", "is_depositor2_tx", "hit_any_rule", "ml_score",
                           "is_bet_tail", "amount_tail3", "desc_has_win", "desc_match_b1", "desc_match_b2"]
               if c in df.columns]

    # ===================== 1. transactions_labeled.parquet =============================
    #df.select(TX_COLS).write_parquet(out("transactions_labeled.parquet"), compression="snappy") #-> file nặng, chưa xuất
    
    # ===================== 2. transactions_flagged.parquet =============================
    df.filter(pl.col("hit_any_rule") == 1).select(TX_COLS).write_parquet(out("transactions_flagged.parquet"), compression="snappy")

    # II. Nhóm Reports
    REPORT_COLS = ["user_id", "user_group", "total_flagged_tx", "ml_score", "bookie_tx_count", "gambler_tx_count",
                   "recipient1_tx_count", "depositor1_tx_count", "recipient2_tx_count", "depositor2_tx_count",
                   "total_sent_tx", "total_sent_amount", "unique_receivers", "total_recv_tx", "total_recv_amount",
                   "unique_senders", "ml_s", "ml_r"]

    # ===================== 3. report_users.parquet =============================
    rpt.select(REPORT_COLS).write_parquet(out("report_users.parquet"), compression="snappy")
    
    # ===================== 4. report_users.csv =================================
    #rpt.select(REPORT_COLS).write_csv(out("report_users.csv")) -> file nặng, chưa xuất
    
    # ===================== 5. report_flagged_users.csv =================================
    #nếu chỉ loại normal thì còn nặng nên đoạn dưới tạm chưa dùng
    #rpt.filter(pl.col("user_group") != "normal").select(REPORT_COLS).write_csv(out("report_flagged_users.csv"))
    rpt.filter((pl.col("user_group") != "normal") & (pl.col("user_group") != "depositor1") & (pl.col("user_group") != "depositor2") & (pl.col("user_group") != "gambler")).select(REPORT_COLS).write_csv(out("report_flagged_users.csv"))

    # ===================== 6. report_flagged_users.csv_70 =================================
    # lọc các userid nhóm normal với điểm ML vượt ngưỡng config
    high_score = rpt.filter((pl.col("user_group") == "normal") & (pl.col("ml_score") > ML_SCORE_THRESHOLD)).sort("ml_score", descending=True)
    # xuất file
    high_score.write_parquet(out(f"report_high_score_users_{ML_SCORE_THRESHOLD}.parquet"), compression="snappy")
    high_score.write_csv(out(f"report_high_score_users_{ML_SCORE_THRESHOLD}.csv"))

    # ===================== 7. report_bookie =================================
    current_bookie_set = set(
        rpt.filter(pl.col("bookie_tx_count") > 0)["user_id"].to_list()
    )
    bookie_recv = df.filter(pl.col("appuser").is_in(current_bookie_set)).group_by("appuser").agg([
        pl.len().alias("recv_tx_count"), pl.col("userid").n_unique().alias("recv_unique_senders"),
        pl.col("amount").sum().alias("total_recv_amount")
    ]).rename({"appuser": "user_id"})

    bookie_send = df.filter(pl.col("userid").is_in(current_bookie_set)).group_by("userid").agg([
        pl.len().alias("send_tx_count"), pl.col("appuser").n_unique().alias("send_unique_receivers"),
        pl.col("amount").sum().alias("total_send_amount")
    ]).rename({"userid": "user_id"})

    bookie_report = bookie_recv.join(bookie_send, on="user_id", how="left").with_columns([
        pl.col(c).fill_null(0) for c in ["send_tx_count", "send_unique_receivers", "total_send_amount"]
    ]).with_columns(
        (pl.col("total_recv_amount") / (pl.col("total_send_amount") + 1)).round(4).alias("recv_send_ratio")
    ).sort("total_recv_amount", descending=True)

    # xuất Bookie Report
    bookie_report.write_parquet(out("report_bookie.parquet"), compression="snappy")
    bookie_report.write_csv(out("report_bookie.csv")) # tra cứu nhanh, trong report_user.parquet có rồi

    print(f"Report được lưu trong {time.time()-t0:.1f}s")


# ============================= 4. VISUALIZATIONS =================================

def _create_visualizations(df: pl.DataFrame, rpt: pl.DataFrame, bookie_set: set, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print("🐌 Tải dữ liệu ...")

    # ========================== 1. ML Score Histogram ==========================
    ml_vals = df.filter(pl.col('ml_score').is_not_null())['ml_score'].to_list()
    plt.figure(figsize=(14, 6))
    sns.histplot(ml_vals, bins=60, color="#1E88E5")
    plt.title("ML Score Distribution - Non-rule transactions", fontsize=16)
    plt.xlabel("ML Score (0=normal, 100=abnormal)")
    plt.ylabel("Transactions")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "F1_histogram.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("✅ Chart 1 đã được tạo thành công!")

    # ============================= 2. Scatter Plot ============================
    try:
        rpt_flag = rpt.filter(pl.col('user_group') != 'normal').to_pandas()
        plt.figure(figsize=(14, 8))
        sns.scatterplot(data=rpt_flag, x='total_flagged_tx', y='ml_score', hue='user_group',
                        size='total_sent_amount', sizes=(20, 400), alpha=0.8, palette="tab10", edgecolor='black')
        plt.title('Rule-based Flags vs ML Anomaly Score', fontsize=16)
        plt.xlabel('Total Flagged Transactions')
        plt.ylabel('Average ML Score')
        plt.savefig(os.path.join(output_dir, "F2_scatter.png"), dpi=200, bbox_inches='tight') # file .png
        plt.close()
        print("✅ Chart 2 đã được tạo thành công!")
    except Exception as e:
        print(f"  ⚠️ Scatter plot skipped: {e}")

    # ===================== 3. Top Bookies (Fixed colorscale) ===================
    if bookie_set:
        try:
            top_bk = (
                rpt.filter(pl.col('bookie_tx_count') > 0)
                .select([
                    pl.col('user_id').alias('appuser'),
                    pl.col('unique_senders'),
                ])
                .sort('unique_senders', descending=True)
                .head(20)
                .to_pandas()
            )

            top_bk = top_bk.sort_values("unique_senders", ascending=True)
            plt.figure(figsize=(10, 6))
            colors3 = sns.color_palette("Reds", n_colors=max(len(top_bk), 1))
            plt.barh(top_bk["appuser"], top_bk["unique_senders"], color=colors3)
            for idx, value in enumerate(top_bk["unique_senders"]):
                plt.text(value, idx, f" {value}", va="center", fontsize=9)
            plt.title("Top 20 Bookies by Unique Senders", fontsize=14)
            plt.xlabel("Unique Senders")
            plt.ylabel("Bookie")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "F3_top_bookies.png"), dpi=200, bbox_inches="tight")
            plt.close()
            print("✅ Chart 3 đã được tạo thành công!")
        except Exception as e:
            print(f"  ⚠️ Top Bookies chart skipped: {e}")
    
    # ========================= 4 Tổng tiền theo nhóm =======================
    # bản thêm vào thủ công, nếu có lỗi thì xoá
    
    #groups = ['bookie', 'gambler', 'recipient1', 'depositor1', 'recipient2', 'depositor2']
    group_config = {
        'bookie':     {'flag_col': 'bookie_tx_count',     'amt_col': 'total_recv_amount'}, # Bookie chủ yếu nhận tiền
        'gambler':    {'flag_col': 'gambler_tx_count',    'amt_col': 'total_sent_amount'}, # Gambler chủ yếu nạp tiền đi
        'recipient1': {'flag_col': 'recipient1_tx_count', 'amt_col': 'total_recv_amount'}, # Nhận tiền từ con bạc
        'depositor1': {'flag_col': 'depositor1_tx_count', 'amt_col': 'total_sent_amount'}, # Đẩy tiền đi tiếp
        'recipient2': {'flag_col': 'recipient2_tx_count', 'amt_col': 'total_recv_amount'},
        'depositor2': {'flag_col': 'depositor2_tx_count', 'amt_col': 'total_sent_amount'},
    }    
    colors = ['#E53935', '#1E88E5', '#43A047', '#8E24AA', '#FB8C00', '#00ACC1']
    
    amt = {}
    for g, config in group_config.items():
        # Lọc những user thuộc nhóm này (hoặc có phát sinh giao dịch của nhóm này)
        # và sum đúng cột số tiền (sent hoặc recv) của user đó
        if config['flag_col'] in rpt.columns:
            total_amount = int(
                rpt.filter(pl.col(config['flag_col']) > 0)[config['amt_col']].sum()
            )
            amt[g] = total_amount
            print(f"  {g:12s}: {total_amount:>15,} VND")
        else:
            amt[g] = 0
            print(f"  {g:12s}: Không có cột cấu hình")
    
    # Tạo biểu đồ
    groups = list(amt.keys())
    values = [v / 1e9 for v in amt.values()]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(groups, values, color=colors)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}B", ha="center", va="bottom")
    plt.title("Fig.4 Tong tien giao dich theo nhom (Ty VND)", fontsize=14)
    plt.xlabel("Nhom")
    plt.ylabel("Ty VND")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "F4_vol_by_gr.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("\n✅ Chart 4 đã được tạo thành công!")

    # ========================= 5. Bookie PnL ===========================
    # bản thêm vào thủ công, nếu có lỗi thì xoá
    BOOKIE_REPORT = os.path.join(output_dir, "report_bookie.parquet")
    bookie_report = pl.read_parquet(BOOKIE_REPORT)
    total_in = bookie_report['total_recv_amount'].sum()
    total_out = bookie_report['total_send_amount'].sum()
    
    summary = {
        'total_in_bil': total_in / 1e9,
        'total_out_bil': total_out / 1e9,
        'num_bookies': bookie_report.height,
        'total_gamblers': bookie_report['recv_unique_senders'].sum(),
        'avg_in_out_ratio': total_in / (total_out + 1),
    }
    
    print(f"Tổng In     : {summary['total_in_bil']:.2f}B VND")
    print(f"Tổng Out    : {summary['total_out_bil']:.2f}B VND")
    print(f"Overall Ratio: {summary['avg_in_out_ratio']:.2f}x")
    
        # Phân loại 5 nhóm PnL
    df_cat = (
        bookie_report.with_columns(
            pl.when(pl.col('recv_send_ratio') < 0.80).then(pl.lit('1. Significantly lose (<0.8x)'))
            .when(pl.col('recv_send_ratio').is_between(0.81, 0.99)).then(pl.lit('2. Slightly lose (0.81x-0.99x)'))
            .when(pl.col('recv_send_ratio').is_between(1.00, 1.10)).then(pl.lit('3. Slightly win (1.0x-1.10x)'))
            .when(pl.col('recv_send_ratio').is_between(1.10, 1.20)).then(pl.lit('4. Significantly win (1.10x-1.20x)'))
            .otherwise(pl.lit('5. Substantially win  (>1.20x)'))
            .alias('behavior_group')
        )
        .group_by('behavior_group')
        .agg(pl.len().alias('count'))
        .sort('behavior_group')
        .to_pandas()
    )
    
    # Tạo biểu đồ
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 2]})
    vol_values = [summary["total_in_bil"], summary["total_out_bil"]]
    axes[0].bar(["In", "Out"], vol_values, color=["#00CC33", "#FF6600"])
    for idx, value in enumerate(vol_values):
        axes[0].text(idx, value, f"{value:.1f}B", ha="center", va="bottom")
    axes[0].set_title("Bookie Volume (Bil VND)")
    axes[0].set_ylabel("Bil VND")

    pnl_colors = ["#CC3300", "#FF0000", "#33CC00", "#339900", "#006600"]
    axes[1].bar(df_cat["behavior_group"], df_cat["count"], color=pnl_colors[:len(df_cat)])
    for idx, value in enumerate(df_cat["count"]):
        axes[1].text(idx, value, f"{value}", ha="center", va="bottom")
    axes[1].set_title("PnL Distribution by In/Out Ratio")
    axes[1].tick_params(axis="x", rotation=25)

    fig.suptitle(f"Fig.5 Bookie Money Flow & PnL | Overall Ratio: {summary['avg_in_out_ratio']:.2f}x")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "F5_bookie_pnl.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\n✅ Chart 5 đã được tạo thành công!")
    
    # ======================6. Thống kê theo kèo đặt cược===============================
    # bản thêm vào thủ công, nếu có lỗi thì xoá
        # Tính thống kê Gambler Tail
    gambler_tail = (
        df.filter(pl.col('is_gambler_tx') == 1)
        .with_columns(
            pl.when(pl.col('amount_tail3') == 11)
              .then(pl.lit('011'))
              .otherwise(pl.lit('012'))
              .alias('bet_tail_type')
        )
        .group_by('bet_tail_type')
        .agg([
            pl.count('amount').alias('tx_count'),
            pl.col('amount').sum().alias('total_amount'),
            pl.col('userid').n_unique().alias('unique_gamblers'),
            pl.col('amount').mean().alias('avg_amount'),
        ])
        .sort('bet_tail_type')
    )
    
    print("Thống kê Gambler theo đuôi tiền:")
    print(gambler_tail)
    
    # Tạo biểu đồ
    tail_df = gambler_tail.to_pandas()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (col, title) in zip(axes, [
        ("tx_count", "Txns"),
        ("total_amount", "Vol. (Bil VND)"),
        ("unique_gamblers", "Gamblers"),
    ]):
        y = tail_df[col] / (1e9 if col == 'total_amount' else 1)
        ax.bar(tail_df["bet_tail_type"], y, color=["#E53935", "#1E88E5"][:len(tail_df)])
        for idx, value in enumerate(y):
            ax.text(idx, value, f"{value:,.0f}", ha="center", va="bottom")
        ax.set_title(title)

    fig.suptitle("Fig. 6 Thong ke Gambler theo keo dat (011 vs 012)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "F6_gambler_picks.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    print("\n✅ Chart 6 đã được tạo thành công!")

# ================================= MAIN =======================================

def _inference_and_report():
    t_total = time.time()
    print("=" * 70)
    print("✨ INFERENCE + REPORT + DASHBOARD")
    print("=" * 70)

    # Load model
    with open(MODEL_PATH, "rb") as f:
        model_dict = pickle.load(f)

    scaler = model_dict["scaler"]
    iso = model_dict["iso_model"]
    bookie_set = model_dict.get("bookie_set", set())
    ml_features = model_dict.get("ml_features", ML_FEATURES)

    print(f"📥 ML model loaded from: {MODEL_PATH}")

    df = pl.read_parquet(LABELED_FILE)
    print(f"📥 Data loaded: {df.shape[0]:,} rows")

    df = _inference_score(df, scaler, iso, ml_features)
    rpt = _step9_user_report(df, bookie_set)

    _export_reports_and_figures(df, rpt, bookie_set, OUTPUT_DIR)
    _create_visualizations(df, rpt, bookie_set, OUTPUT_DIR)

    print(f"\n📊 Đã hoàn tất 6 charts ({time.time()-t_total:.1f}s)")
    print(f"📁 Files được lưu tại: {OUTPUT_DIR}")

    # ── Xuất HTML & gửi email ──────────────────────────────────────────────
    export_html_and_send(OUTPUT_DIR)
    print(f"Báo cáo đã được gửi email tới aml-zion@vng.com.vn")


if __name__ == "__main__":
    _inference_and_report()
