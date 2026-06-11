# tool2.py

import os
import gc
import time
import polars as pl

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
import json as _json

_HERE     = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.environ.get("A450_BASE_DIR", _HERE)

# ── Load cấu hình từ tool2.json ───────────────────────────────────────────
with open(os.path.join(_HERE, "tool2.json"), encoding="utf-8") as _f:
    _CFG = _json.load(_f)

ETL_FILE     = os.environ.get("A450_ETL_FILE",     os.path.join(_BASE_DIR, _CFG["etl_file"]))
LABELED_FILE = os.environ.get("A450_LABELED_FILE", os.path.join(_BASE_DIR, _CFG["labeled_file"]))
TEMP_DIR     = os.environ.get("A450_TEMP_DIR",     os.path.join(_BASE_DIR, _CFG["temp_dir"]))

CONFIG = {k: _CFG[k] for k in [
    "bet_tail_digits", "win_pattern_prefix", "zalopay_excl",
    "pattern_b1", "pattern_b2",
    "bookie_min_senders", "bookie_recv_send_ratio",
    "bookie_recv_bet_tail", "bookie_send_win_desc",
    "bookie_send_non_bet_tail",
    "bookie_cond_min",
    "weight_min_senders", "weight_recv_send_ratio",
    "weight_recv_bet_tail",
    "weight_send_win_non_bet",
    "str_ip_change_min",
    "str_smurfing_recv_users",
    "str_smurfing_recv_amount",
    "str_smurfing_send_amount",
    "str_rapid_tx_seconds",
    "str_rapid_tx_min_pairs",
]}

def _tmp(name: str) -> str:
    return os.path.join(TEMP_DIR, name)


# =============================================================================
# Derived columns & Pattern flags
# =============================================================================
def _step4_derived(df: pl.DataFrame) -> pl.DataFrame:
    t0 = time.time()

    ZALOPAY   = CONFIG["zalopay_excl"]
    PAT_B1    = CONFIG["pattern_b1"]
    PAT_B2    = CONFIG["pattern_b2"]
    BET_TAILS = CONFIG["bet_tail_digits"]

    df = (
        df.lazy()
        .with_columns([
            # Ép amount về số trước khi tính toán
            pl.col("amount").cast(pl.Int64, strict=False).alias("amount"),
            
            (pl.col("amount").cast(pl.Int64, strict=False) % 1000)
              .cast(pl.Int16).alias("amount_tail3"),
            
            (pl.col("amount").cast(pl.Int64, strict=False) % 1000)
              .is_in(BET_TAILS).cast(pl.Int8).alias("is_bet_tail"),
            
            pl.col("desc_clean").str.contains(r"\bwin\d+").cast(pl.Int8).alias("desc_has_win"),
            pl.col("desc_clean").str.contains(ZALOPAY, literal=True).cast(pl.Int8).alias("is_zalopay"),
        ])
        .with_columns([
            (
                (pl.col("is_zalopay") == 0) &
                pl.col("desc_clean").str.contains(PAT_B1)
            ).cast(pl.Int8).alias("desc_match_b1"),
            (
                (pl.col("is_zalopay") == 0) &
                pl.col("desc_clean").str.contains(PAT_B2)
            ).cast(pl.Int8).alias("desc_match_b2"),
        ])
        .collect()
    )

    # drop cot trung gian
    df = df.drop(["desc_clean", "is_zalopay"])

    print(f"Derived columns ({time.time()-t0:.1f}s). Kết quả:")
    for col in ["is_bet_tail", "desc_has_win", "desc_match_b1", "desc_match_b2"]:
        print(f"  {col:15s}: {df[col].sum():>10,}")

    return df

# =============================================================================
# Feature Engineering: sender + receiver aggregation
# =============================================================================
def _step5_features(df: pl.DataFrame) -> pl.DataFrame:
    t0 = time.time()

    #1 Sender features
    sender_agg = (
        df.lazy()
        .group_by("userid")
        .agg([
            pl.count("amount")            .alias("sender_tx_count"),
            pl.col("appuser").n_unique()  .alias("sender_unique_receivers"),
            pl.col("amount").sum()        .alias("sender_total_sent"),
            pl.col("amount").mean()       .alias("sender_avg_amount"),
            pl.col("amount").max()        .alias("sender_max_amount"),
            pl.col("amount").std()        .alias("sender_std_amount"),
            pl.col("userip").n_unique()   .alias("sender_unique_ips"),
            pl.col("is_bet_tail").mean()  .alias("sender_bet_tail_ratio"),
            pl.col("desc_has_win").mean() .alias("sender_win_desc_ratio"),
            pl.col("desc_match_b1").mean().alias("sender_b1_ratio"),
            pl.col("desc_match_b2").mean().alias("sender_b2_ratio"),
        ])
        .with_columns(pl.col("sender_std_amount").fill_null(0))
        .collect()
    )
    sender_agg.write_parquet(_tmp("s05a_sender_agg.parquet"), compression="lz4")
    del sender_agg; gc.collect()

    df = (
        df.lazy()
        .join(pl.scan_parquet(_tmp("s05a_sender_agg.parquet")), on="userid", how="left")
        .collect()
    )
    print(f"Hoàn tất Sender features ({time.time()-t0:.1f}s)")

    #2 Receiver features
    t0 = time.time()
    recv_agg = (
        df.lazy()
        .group_by("appuser")
        .agg([
            pl.count("amount")             .alias("recv_tx_count"),
            pl.col("userid").n_unique()    .alias("recv_unique_senders"),
            pl.col("amount").sum()         .alias("recv_total_received"),
            pl.col("amount").mean()        .alias("recv_avg_amount"),
            pl.col("amount").max()         .alias("recv_max_amount"),
            pl.col("amount").std()         .alias("recv_std_amount"),
            pl.col("is_bet_tail").mean()   .alias("recv_bet_tail_ratio"),
            pl.col("desc_has_win").mean()  .alias("recv_win_desc_ratio"),
            pl.col("desc_match_b1").mean() .alias("recv_b1_ratio"),
            pl.col("desc_match_b2").mean() .alias("recv_b2_ratio"),
        ])
        .with_columns(pl.col("recv_std_amount").fill_null(0))
        .collect()
    )
    recv_agg.write_parquet(_tmp("s05b_recv_agg.parquet"), compression="lz4")
    del recv_agg; gc.collect()

    df = (
        df.lazy()
        .join(pl.scan_parquet(_tmp("s05b_recv_agg.parquet")), on="appuser", how="left")
        .with_columns(
            (
                pl.col("recv_tx_count") /
                (pl.col("sender_tx_count") + pl.col("recv_tx_count") + 1)
            ).alias("recv_send_tx_ratio")
        )
        .collect()
    )

    print(f"Hoàn tất Receiver features ({time.time()-t0:.1f}s) | receivers: {df['appuser'].n_unique():,}")
    return df


# =============================================================================
# STR Flags (Suspicious Transaction Report criteria)
# =============================================================================
def _step5b_str_flags(df: pl.DataFrame) -> pl.DataFrame:
    """
    Tính 4 cờ cảnh báo STR theo user, join lại vào df giao dịch.

    Cờ mới (cấp user, Int8):
      str_ip_change    — user dùng ≥ str_ip_change_min IP khác nhau
      str_device_share — deviceid hoặc IP bị dùng bởi >1 user
      str_smurfing     — trong 1 ngày: nhận từ >N user với amount<500k VÀ có ≥1 GD gửi >5tr
      str_rapid_tx     — trong 1 ngày: có ≥ str_rapid_tx_min_pairs cặp GD liên tiếp cách <10s
    """
    t0 = time.time()

    IP_MIN       = CONFIG["str_ip_change_min"]
    RECV_USERS   = CONFIG["str_smurfing_recv_users"]
    RECV_AMT     = CONFIG["str_smurfing_recv_amount"]
    SEND_AMT     = CONFIG["str_smurfing_send_amount"]
    RAPID_SEC    = CONFIG["str_rapid_tx_seconds"]
    RAPID_PAIRS  = CONFIG["str_rapid_tx_min_pairs"]

    # ── Tiêu chí 1: user thay đổi IP hoặc thiết bị ──────────────────────
    ip_change = (
        df.lazy()
        .group_by("userid")
        .agg([
            pl.col("userip").n_unique().alias("_n_ips"),
            pl.col("deviceid").n_unique().alias("_n_devices"),
        ])
        .with_columns(
            (
                (pl.col("_n_ips") >= IP_MIN) |
                (pl.col("_n_devices") >= IP_MIN)
            ).cast(pl.Int8).alias("str_ip_change")
        )
        .drop(["_n_ips", "_n_devices"])
        .collect()
    )

    # ── Tiêu chí 2: thiết bị / IP dùng chung bởi >1 user ─────────────────
    device_multi = (
        df.lazy()
        .group_by("deviceid")
        .agg(pl.col("userid").n_unique().alias("_n_users_dev"))
        .filter(pl.col("_n_users_dev") > 1)
        .select("deviceid")
        .collect()
    )
    ip_multi = (
        df.lazy()
        .group_by("userip")
        .agg(pl.col("userid").n_unique().alias("_n_users_ip"))
        .filter(pl.col("_n_users_ip") > 1)
        .select("userip")
        .collect()
    )
    shared_devices = set(device_multi["deviceid"].to_list())
    shared_ips     = set(ip_multi["userip"].to_list())

    device_share = (
        df.lazy()
        .group_by("userid")
        .agg([
            pl.col("deviceid").is_in(shared_devices).any().alias("_dev_shared"),
            pl.col("userip").is_in(shared_ips).any().alias("_ip_shared"),
        ])
        .with_columns(
            (pl.col("_dev_shared") | pl.col("_ip_shared")).cast(pl.Int8).alias("str_device_share")
        )
        .drop(["_dev_shared", "_ip_shared"])
        .collect()
    )

    # ── Tiêu chí 3: smurfing — tính trong 1 ngày ─────────────────────────
    # 3a. Ngày nào appuser nhận từ >N user khác nhau với amount < RECV_AMT
    recv_day = (
        df.lazy()
        .with_columns(pl.col("reqdate").str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", strict=False).dt.date().alias("_date"))
        .filter(pl.col("amount") < RECV_AMT)
        .group_by(["appuser", "_date"])
        .agg(pl.col("userid").n_unique().alias("_recv_users"))
        .filter(pl.col("_recv_users") > RECV_USERS)
        .select(["appuser", "_date"])
        .collect()
    )

    # 3b. Ngày nào userid gửi ít nhất 1 GD > SEND_AMT
    send_day = (
        df.lazy()
        .with_columns(pl.col("reqdate").str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", strict=False).dt.date().alias("_date"))
        .filter(pl.col("amount") > SEND_AMT)
        .group_by(["userid", "_date"])
        .agg(pl.len().alias("_cnt"))
        .filter(pl.col("_cnt") >= 1)
        .select(["userid", "_date"])
        .collect()
    )

    # 3c. User thoả cả 2 điều kiện TRÊN CÙNG 1 NGÀY
    #     (appuser trong recv_day = userid trong send_day, cùng _date)
    smurf_users = set(
        recv_day
        .rename({"appuser": "userid"})
        .join(send_day, on=["userid", "_date"], how="inner")
        ["userid"]
        .to_list()
    )

    smurfing = (
        df.lazy()
        .select("userid")
        .unique()
        .with_columns(
            pl.col("userid").is_in(smurf_users).cast(pl.Int8).alias("str_smurfing")
        )
        .collect()
    )

    # ── Tiêu chí 4: giao dịch liên tiếp < RAPID_SEC giây ─────────────────
    rapid = (
        df.lazy()
        .with_columns([
            pl.col("reqdate").str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", strict=False).alias("_ts"),
            pl.col("reqdate").str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", strict=False).dt.date().alias("_date"),
        ])
        .sort(["userid", "_ts"])
        .with_columns(
            pl.col("_ts")
              .shift(1)
              .over(["userid", "_date"])
              .alias("_prev_ts")
        )
        .with_columns(
            ((pl.col("_ts") - pl.col("_prev_ts"))
              .dt.total_seconds()
              .lt(RAPID_SEC)
              .cast(pl.Int8))
            .alias("_rapid_pair")
        )
        .group_by(["userid", "_date"])
        .agg(pl.col("_rapid_pair").sum().alias("_n_rapid"))
        .filter(pl.col("_n_rapid") >= RAPID_PAIRS)
        .select("userid")
        .unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("str_rapid_tx"))
        .collect()
    )

    # ── Join tất cả flag về df ────────────────────────────────────────────
    str_flags = (
        df.lazy()
        .select("userid")
        .unique()
        .join(ip_change.lazy(),    on="userid", how="left")
        .join(device_share.lazy(), on="userid", how="left")
        .join(smurfing.lazy(),     on="userid", how="left")
        .join(rapid.lazy(),        on="userid", how="left")
        .with_columns([
            pl.col("str_ip_change")   .fill_null(0).cast(pl.Int8),
            pl.col("str_device_share").fill_null(0).cast(pl.Int8),
            pl.col("str_smurfing")    .fill_null(0).cast(pl.Int8),
            pl.col("str_rapid_tx")    .fill_null(0).cast(pl.Int8),
        ])
        .collect()
    )

    df = df.join(str_flags, on="userid", how="left")

    # drop cot chi dung de tinh feature (userip/deviceid can cho STR, drop o day)
    df = df.drop(["userip", "deviceid", "platform"])

    STR_COLS = ["str_ip_change", "str_device_share", "str_smurfing", "str_rapid_tx"]
    print(f"STR flags ({time.time()-t0:.1f}s). Kết quả:")
    for col in STR_COLS:
        n = df[col].sum()
        print(f"  {col:20s}: {df.filter(pl.col(col)==1)['userid'].n_unique():>8,} users  ({n:,} GD)")

    return df


# =============================================================================
# Rule-based Labeling
# =============================================================================
def _step6_label(df: pl.DataFrame) -> pl.DataFrame:
    t0 = time.time()

    # 1 Xac dinh bookie_set
    bk_recv = (
        df.lazy()
        .group_by("appuser")
        .agg([
            pl.count("amount").alias("recv_tx"),
            pl.col("userid").n_unique().alias("recv_src"),
            pl.col("is_bet_tail").mean().alias("recv_bet_tail"),
            # pl.col("desc_has_win").mean().alias("recv_win_desc"), tạm bỏ ra vì gambler cũng nhập win
        ])
        .collect()
    )

    bk_send = (
        df.lazy()
        .group_by("userid")
        .agg([
            pl.count("amount").alias("send_tx"),
            pl.col("desc_has_win").mean().alias("send_win_desc"),
            # thêm điều kiện kèm theo cho desc có win nhưng GD không phải đuôi 011 012
            (1-pl.col("is_bet_tail")).mean().alias("non_bet_tail_ratio"),
        ])
        .collect()
        .rename({"userid": "appuser"})
    )

    bk = (
        bk_recv.join(bk_send, on="appuser", how="left")
        .with_columns([
            pl.col("send_tx").fill_null(0),
            pl.col("send_win_desc").fill_null(0),
            pl.col("non_bet_tail_ratio").fill_null(0),
        ])
        .with_columns(
            (pl.col("recv_tx") / (pl.col("send_tx") + 1)).alias("recv_send_ratio")
        )
        .with_columns((
            (pl.col("recv_src")        >= CONFIG["bookie_min_senders"]    ).cast(pl.Float64) * CONFIG["weight_min_senders"]    +
            (pl.col("recv_send_ratio") >  CONFIG["bookie_recv_send_ratio"]).cast(pl.Float64) * CONFIG["weight_recv_send_ratio"] +
            (pl.col("recv_bet_tail")   >  CONFIG["bookie_recv_bet_tail"]  ).cast(pl.Float64) * CONFIG["weight_recv_bet_tail"]   +
            (
                (pl.col("send_win_desc")      > CONFIG["bookie_send_win_desc"]) &
                (pl.col("non_bet_tail_ratio") > CONFIG["bookie_send_non_bet_tail"])
        ).cast(pl.Float64) * CONFIG["weight_send_win_non_bet"]
    ).alias("bookie_score"))
    )

    bookie_set = set(bk.filter(pl.col("bookie_score") >= CONFIG["bookie_cond_min"])["appuser"].to_list())
    del bk_recv, bk_send; gc.collect()

    print(f"Số lượng bookies: {len(bookie_set):,}")
    print(f"Số GD đến bookie: {df.filter(pl.col('appuser').is_in(bookie_set)).shape[0]:,}")
    print(f"Số GD đến bookie có bet_tail: {df.filter(pl.col('appuser').is_in(bookie_set) & (pl.col('is_bet_tail')==1)).shape[0]:,}")

    # 2 Gan nhan tung giao dich
    RULE_COLS = [
        "is_bookie_tx", "is_gambler_tx",
        "is_recipient1_tx", "is_depositor1_tx",
        "is_recipient2_tx", "is_depositor2_tx",
    ]

    df = (
        df.lazy()
        .with_columns([
            pl.col("appuser").is_in(bookie_set).cast(pl.Int8).alias("is_bookie_tx"),
            (
                pl.col("appuser").is_in(bookie_set) &
                (pl.col("is_bet_tail") == 1)
            ).cast(pl.Int8).alias("is_gambler_tx"),
            pl.col("desc_match_b1").alias("is_recipient1_tx"),
            pl.col("desc_match_b1").alias("is_depositor1_tx"),
            pl.col("desc_match_b2").alias("is_recipient2_tx"),
            pl.col("desc_match_b2").alias("is_depositor2_tx"),
        ])
        .collect()
    )

    print(f"Số GD is_gambler_tx=1: {df['is_gambler_tx'].sum():,}")
    print(f"Số userid unique là gambler: {df.filter(pl.col('is_gambler_tx')==1)['userid'].n_unique():,}")

    df = df.with_columns(
        pl.max_horizontal([pl.col(c) for c in RULE_COLS])
        .cast(pl.Int8).alias("hit_any_rule")
    )

    n_hit = df["hit_any_rule"].sum()
    print(f"Label xong ({time.time()-t0:.1f}s). Kết quả:")
    for col in RULE_COLS:
        print(f"  {col:20s}: {df[col].sum():>10,}")
    print(f"  {'hit_any_rule':20s}: {n_hit:>10,}  ({n_hit/df.shape[0]*100:.1f}%)")
    print(f"  {'ML (non-rule)':20s}: {df.shape[0]-n_hit:>10,}  ({(df.shape[0]-n_hit)/df.shape[0]*100:.1f}%)")

    return df


# =============================================================================
# HAM CHINH
# =============================================================================
def _feature_engineering(
    etl_file:     str = ETL_FILE,
    labeled_file: str = LABELED_FILE,
    temp_dir:     str = TEMP_DIR,
) -> str:
    """
    Doc a450etl.parquet (output tool1), chay buoc 4+5+6, ghi a450labeled.parquet.
    Tra ve duong dan file output.
    """
    t_total = time.time()

    print("=" * 60)
    print("II. FEATURE ENGINEERING")
    print("=" * 60)

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(labeled_file)), exist_ok=True)

    # Doc input
    print(f"☕ Đang load {etl_file} ...")
    df = pl.read_parquet(etl_file)
    print(f"Loaded {df.shape[0]:,} dòng | {df.shape[1]} cột")

    # Buoc 4
    df = _step4_derived(df)

    # Buoc 5
    df = _step5_features(df)

    # Buoc 5b — STR flags (truoc khi drop userip/deviceid)
    df = _step5b_str_flags(df)

    # Buoc 6
    df = _step6_label(df)

    # Ghi output
    df.write_parquet(labeled_file, compression="snappy")

    # Don dep file tam
    for f in ["s05a_sender_agg.parquet", "s05b_recv_agg.parquet"]:
        fpath = _tmp(f)
        if os.path.exists(fpath):
            os.remove(fpath)

    print("=" * 60)
    print(f"Hoàn thành feature engineering ({time.time()-t_total:.1f}s) - File lưu tại: {labeled_file}")
    print(f"  Shape: {df.shape[0]:,} dòng | {df.shape[1]} cột")
    print("=" * 60)
    return labeled_file


# --- chay truc tiep de test ---
if __name__ == "__main__":
    _feature_engineering()
