# tool1.py

import os
import re
import unicodedata
import bisect
import ipaddress
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Resolve BASE_DIR
# ---------------------------------------------------------------------------
import json as _json

_HERE     = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.environ.get("A450_BASE_DIR", _HERE)

# ── Load cấu hình từ tool1.json ───────────────────────────────────────────
with open(os.path.join(_HERE, "tool1.json"), encoding="utf-8") as _f:
    _CFG = _json.load(_f)

RAW_DIR    = os.environ.get("A450_RAW_DIR",    os.path.join(_BASE_DIR, _CFG["raw_dir"]))
IPREF_FILE = os.environ.get("A450_IPREF_FILE", os.path.join(_BASE_DIR, _CFG["ipref_file"]))
OUT_FILE   = os.environ.get("A450_OUT_FILE",   os.path.join(_BASE_DIR, _CFG["out_file"]))
CHUNKSIZE  = int(os.environ.get("A450_CHUNKSIZE", _CFG["chunksize"]))


# =============================================================================
# BUOC 1 -- Stream tung chunk CSV trong RAW_DIR (khong ghi file trung gian)
# =============================================================================
def _iter_raw_chunks(raw_dir: str):
    """Yield tung DataFrame chunk tu tat ca *.csv trong raw_dir."""
    csv_files = sorted(
        f for f in os.listdir(raw_dir) if f.lower().endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError(f"Khong tim thay file CSV nao trong: {raw_dir}")

    print(f"Tìm thấy {len(csv_files)} file *.csv:")
    for fname in csv_files:
        print(f"  - {fname}")
        fpath = os.path.join(raw_dir, fname)
        yield from pd.read_csv(fpath, chunksize=CHUNKSIZE, dtype=str)


# =============================================================================
# BUOC 2 -- Xay bang tra cuu IP -> country code
# =============================================================================
def _build_ip_lookup(ipref_file: str):
    """Doc allip.csv, tra ve closure lookup_country(ip) -> str."""
    print(f"Tra cứu IP location từ {ipref_file}")
    ref_df = pd.read_csv(ipref_file)

    ipv4_starts, ipv4_ranges = [], []
    ipv6_starts, ipv6_ranges = [], []

    for _, row in ref_df.iterrows():
        cc       = row["country"]
        iptype   = row["type"]
        start_ip = str(row["start"]).strip()
        value    = int(row["value"])

        if iptype == "ipv4":
            start_int = int(ipaddress.IPv4Address(start_ip))
            end_int   = start_int + value - 1
            ipv4_starts.append(start_int)
            ipv4_ranges.append((start_int, end_int, cc))
        elif iptype == "ipv6":
            start_int = int(ipaddress.IPv6Address(start_ip))
            end_int   = start_int + value - 1
            ipv6_starts.append(start_int)
            ipv6_ranges.append((start_int, end_int, cc))

    def _sort(starts, ranges):
        combined = sorted(zip(starts, ranges), key=lambda x: x[0])
        return [x[0] for x in combined], [x[1] for x in combined]

    ipv4_starts, ipv4_ranges = _sort(ipv4_starts, ipv4_ranges)
    ipv6_starts, ipv6_ranges = _sort(ipv6_starts, ipv6_ranges)

    print(f"  Kết quả IPv4: {len(ipv4_ranges):,} địa chỉ  |  IPv6: {len(ipv6_ranges):,} địa chỉ")

    def lookup_country(ip: str) -> str:
        try:
            ip = str(ip).strip()
            if not ip or ip == "nan":
                return ""
            if ":" in ip:
                ip_int = int(ipaddress.IPv6Address(ip))
                starts, ranges = ipv6_starts, ipv6_ranges
            else:
                ip_int = int(ipaddress.IPv4Address(ip))
                starts, ranges = ipv4_starts, ipv4_ranges

            idx = bisect.bisect_right(starts, ip_int) - 1
            if idx >= 0:
                start, end, cc = ranges[idx]
                if start <= ip_int <= end:
                    return cc
            return ""
        except Exception:
            return ""

    return lookup_country


# =============================================================================
# BUOC 3 -- Transform moi chunk
# =============================================================================
def _transform(chunk: pd.DataFrame, lookup_country) -> pd.DataFrame:
    # 3a. Sua deviceid null -> ghep userid-userip
    mask = chunk["deviceid"].isna()
    chunk.loc[mask, "deviceid"] = (
        chunk.loc[mask, "userid"].fillna("") + "-" +
        chunk.loc[mask, "userip"].fillna("")
    )

    # 3c. Chen iploc truoc cot deviceid
    deviceid_pos = chunk.columns.get_loc("deviceid")
    chunk.insert(deviceid_pos, "iploc", chunk["userip"].apply(lookup_country))

    # 3d. Lam sach description -> desc_clean, bo cot goc
    # Bo dau truoc (NFD + xoa combining marks) de chuan hoa ve ASCII,
    # sau do xoa ky tu dac biet va chuan hoa khoang trang.
    # Vi du: "chuyen khoan $" -> "chuyen khoan"
    # Ket qua khop voi pattern_b1/b2 va zalopay_excl viet khong dau.
    def _normalize(text):
        text = unicodedata.normalize("NFD", str(text).lower())
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+",     " ", text).strip()
        return text

    chunk["desc_clean"] = chunk["description"].fillna("").apply(_normalize)
    chunk = chunk.drop(columns=["description"])

    return chunk


# =============================================================================
# HAM CHINH
# =============================================================================
def _etl(
    raw_dir:    str = RAW_DIR,
    ipref_file: str = IPREF_FILE,
    out_file:   str = OUT_FILE,
) -> str:
    """
    Pipeline ETL -> Parquet (snappy), khong co file CSV trung gian:
      1. Stream tung chunk CSV trong raw_dir
      2. Nap bang IP reference, xay lookup closure
      3. Transform: drop null, sua deviceid, gan iploc, lam sach description
      4. Ghi tung chunk vao Parquet qua ParquetWriter (append)
    Tra ve duong dan file Parquet.
    """
    import time
    t0 = time.time()

    print("=" * 60)
    print("I. ETL")
    print("=" * 60)

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)

    # Buoc 2: build IP lookup
    lookup_country = _build_ip_lookup(ipref_file)

    # Buoc 1 + 3: stream -> transform -> ghi Parquet
    print("Stream CSV -> transform -> Parquet ...")
    writer    = None
    total_in  = 0
    total_out = 0

    try:
        for chunk in _iter_raw_chunks(raw_dir):
            total_in += len(chunk)
            chunk = _transform(chunk, lookup_country)
            total_out += len(chunk)

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_file, table.schema, compression="snappy")
            writer.write_table(table)

    finally:
        if writer:
            writer.close()

    elapsed = time.time() - t0
    print(f"Hoàn thành viết parquet file — Input: {total_in:,} dòng  |  Out put: {total_out:,} dòng")
    print("=" * 60)
    print(f"ETL hoàn thành ({elapsed:.1f}s) File được ghi vào {out_file}")
    print("=" * 60)
    return out_file


# --- chay truc tiep de test ---
if __name__ == "__main__":
    _etl()
