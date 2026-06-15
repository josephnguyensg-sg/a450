# agent.py
# =============================================================================
# CẤU TRÚC FILE:
#
#   1.  IMPORTS & ĐƯỜNG DẪN ENV
#   2.  LLM KHỞI TẠO
#   3.  SCHEMA & PROMPT
#       3a. Schema ngắn (nhánh C fallback)
#       3b. Schema tool5 nội tuyến (nhánh B — LLM sinh SQL trên rpt_users/highmls/txn_flg/maindb)
#       3c. Schema a450labeled (nhánh C tra cứu sâu — DuckDB)
#       3d. System prompt agent nhánh C
#   4.  INTENT RECOGNITION
#       - Phân loại rõ: pipeline / tra cứu report / tra cứu giao dịch / meta / sửa sai
#   5.  LOGIC TỪNG NHÁNH
#       5a. Nhánh B  — query tự nhiên "? câu hỏi" → LLM sinh SQL → tool5
#       5b. Nhánh C1 — tra cứu report (rpt_users, highmls qua tool5)
#       5c. Nhánh C2 — tra cứu giao dịch sâu (a450labeled qua DuckDB)
#   6.  WRAPPER FUNCTIONS (tool1–4)
#   7.  ĐĂNG KÝ TOOLS
#   8.  PIPELINE GUARD
#   9.  HELPERS
#  10.  HÀM ĐIỀU PHỐI CHÍNH — chay_agent_aml()
#  11.  ENTRY POINT
# =============================================================================


# =============================================================================
# 1. IMPORTS & ĐƯỜNG DẪN ENV
# =============================================================================

import os
import re
import unicodedata

from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

import tool5
from tool1 import _etl
from tool2 import _feature_engineering
from tool3 import _train_isolation_forest
from tool4 import _inference_and_report

import json as _json

_HERE     = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.environ.get("A450_BASE_DIR", _HERE)

# ── Load cấu hình từ agent.json ───────────────────────────────────────────
with open(os.path.join(_HERE, "agent.json"), encoding="utf-8") as _f:
    _CFG = _json.load(_f)

ETL_FILE     = os.environ.get("A450_ETL_FILE",     os.path.join(_BASE_DIR, _CFG["etl_file"]))
LABELED_FILE = os.environ.get("A450_LABELED_FILE", os.path.join(_BASE_DIR, _CFG["labeled_file"]))
MODEL_PATH   = os.environ.get("A450_MODEL_PATH",   os.path.join(_BASE_DIR, _CFG["model_path"]))
OUTPUT_DIR   = os.environ.get("A450_OUTPUT_DIR",   os.path.join(_BASE_DIR, _CFG["output_dir"]))


# =============================================================================
# 2. LLM KHỞI TẠO
# =============================================================================

_AI_PLATFORM_BASE_URL = os.environ.get(
    "AI_PLATFORM_BASE_URL",
    "https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1",
)
_llm = None
_llm_with_tools = None


def _get_api_key_and_source():
    for env_name in ("AI_PLATFORM_API_KEY", "OPENAI_API_KEY"):
        api_key = os.environ.get(env_name)
        if api_key:
            return api_key, env_name
    return None, None


def _format_llm_auth_error() -> str:
    _, api_key_source = _get_api_key_and_source()
    api_key_source = api_key_source or "AI_PLATFORM_API_KEY/OPENAI_API_KEY"
    model = _CFG.get("llm_model", "(unknown)")
    return (
        "Lỗi xác thực LLM (401 Unauthorized). "
        f"Runtime đang dùng `{api_key_source}` với "
        f"`AI_PLATFORM_BASE_URL={_AI_PLATFORM_BASE_URL}` và `llm_model={model}`. "
        "Hãy kiểm tra key còn hiệu lực và thuộc đúng endpoint. "
        "Nếu dùng OpenAI API key, đặt `AI_PLATFORM_BASE_URL=https://api.openai.com/v1` "
        "và đổi `llm_model` sang model OpenAI hợp lệ. "
        "Nếu dùng endpoint VNG AI Platform, đặt `AI_PLATFORM_API_KEY` của VNG; "
        "`AI_PLATFORM_API_KEY` được ưu tiên hơn `OPENAI_API_KEY` khi cả hai cùng tồn tại."
    )


def _is_llm_auth_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 401:
        return True
    exc_text = str(exc)
    exc_name = exc.__class__.__name__
    return (
        "AuthenticationError" in exc_name
        or "Error code: 401" in exc_text
        or "Unauthorized" in exc_text
    )


def _raise_friendly_llm_error(exc: Exception):
    if _is_llm_auth_error(exc):
        raise RuntimeError(_format_llm_auth_error()) from exc


def _get_llm():
    """Khởi tạo LLM khi thật sự cần, tránh làm container chết lúc startup."""
    global _llm
    if _llm is None:
        api_key, _ = _get_api_key_and_source()
        if not api_key:
            raise RuntimeError(
                "Thiếu AI_PLATFORM_API_KEY hoặc OPENAI_API_KEY. Hãy khai báo secret/env này trên runtime VCR "
                "trước khi dùng các chức năng cần LLM."
            )
        _llm = ChatOpenAI(
            model=_CFG["llm_model"],
            temperature=_CFG["llm_temperature"],
            max_tokens=_CFG["llm_num_ctx"],
            timeout=_CFG.get("llm_timeout"),
            openai_api_key=api_key,
            openai_api_base=_AI_PLATFORM_BASE_URL,
        )
    return _llm


def _invoke_llm(payload):
    try:
        return _get_llm().invoke(payload)
    except Exception as exc:
        _raise_friendly_llm_error(exc)
        raise


# =============================================================================
# 3. SCHEMA & PROMPT
# =============================================================================

# ---------------------------------------------------------------------------
# 3a. Schema ngắn — dùng trong SYSTEM_PROMPT_FULL (nhánh C fallback LLM)
# ---------------------------------------------------------------------------
_SCHEMA_SHORT = _CFG["schema_short"]

# ---------------------------------------------------------------------------
# 3b. Schema tool5 — bảng trong Polars SQLContext (nhánh B và C1)
#
#   rpt_users : báo cáo tổng hợp theo user (sau khi chạy bước 3)
#   highmls   : user có ml_score > 70
#   txn_flg   : giao dịch bị flag (hit_any_rule = 1)
#   maindb    : toàn bộ giao dịch đã labeled (a450labeled.parquet)
#
# Dùng bảng này khi user hỏi về report, user rủi ro, thống kê nhóm.
# ---------------------------------------------------------------------------
_SCHEMA_TOOL5 = _CFG["schema_tool5"]

# Prompt sinh SQL cho nhánh B và C1 (Polars SQLContext, tool5)
_SQL_PROMPT_TOOL5 = _CFG["sql_prompt_tool5"]

# ---------------------------------------------------------------------------
# 3c. Schema a450labeled — dùng cho nhánh C2 (DuckDB, phân tích giao dịch thô)
#
# Dùng khi user hỏi về đặc điểm giao dịch ở mức thấp hơn: feature engineering,
# tỉ lệ bet_tail, phân phối amount, sender/receiver behavior...
# ---------------------------------------------------------------------------
_SCHEMA_LABELED = _CFG["schema_labeled"]

# Prompt sinh SQL cho nhánh C2 (DuckDB, bảng "transactions")
_SQL_PROMPT_DUCKDB = _CFG["sql_prompt_duckdb"]

# ---------------------------------------------------------------------------
# 3d. System prompt nhánh C (agent điều phối pipeline)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_PIPELINE = _CFG["system_prompt_pipeline"]

_SYSTEM_PROMPT_FULL = (
    _SYSTEM_PROMPT_PIPELINE
    + "\n\nQUY TẮC TOOL-CALL BẮT BUỘC:\n"
    + "- Khi cần chạy tool, phải dùng cơ chế tool/function calling thật của runtime.\n"
    + "- Không bao giờ in ra các thẻ dạng <tool_call>, <function=...>, </function> trong nội dung trả lời.\n"
    + "\n\nMÔ TẢ CỘT THAM KHẢO:\n"
    + _SCHEMA_SHORT
)

_MO_TA_CHUC_NANG = _CFG["mo_ta_chuc_nang"]


# =============================================================================
# 4. INTENT RECOGNITION
#
# Mục tiêu: phân loại câu hỏi user vào đúng 1 trong 5 nhóm:
#   - PIPELINE   : yêu cầu chạy bước 1/2/3/4
#   - REPORT     : tra cứu kết quả từ report (rpt_users, highmls) — dùng tool5
#   - TRANSACTION: phân tích giao dịch thô (a450labeled) — dùng DuckDB
#   - META       : hỏi về chức năng, hướng dẫn
#   - CORRECTION : sửa sai câu trả lời trước
# =============================================================================

# --- Từ khoá pipeline (ưu tiên cao nhất, check trước) ---
_KEYWORD_PIPELINE = _CFG["keywords_pipeline"]

# --- Từ khoá tra cứu REPORT (tool5, rpt_users/highmls) ---
# Dùng khi user hỏi về kết quả đã có sau bước 3
_KEYWORD_REPORT = _CFG["keywords_report"]

# --- Từ khoá tra cứu GIAO DỊCH (DuckDB, a450labeled) ---
# Dùng khi user hỏi về đặc điểm, phân phối giao dịch
_KEYWORD_TRANSACTION = _CFG["keywords_transaction"]

# --- Từ khoá META (hỏi về chức năng) ---
_KEYWORD_META = _CFG["keywords_meta"]

# --- Từ khoá CORRECTION (sửa sai) ---
_KEYWORD_CORRECTION = _CFG["keywords_correction"]


def _nhan_dien_pipeline(q: str) -> str:
    """Trả về tên tool nếu câu hỏi là yêu cầu chạy pipeline, else ''."""
    for tool_name, keywords in _KEYWORD_PIPELINE.items():
        if any(k in q for k in keywords):
            return tool_name
    return ""

def _bo_dau(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "").replace("đ", "d").replace("Đ", "D")
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

def _la_lenh_tiep_tuc(q: str) -> bool:
    plain = re.sub(r"\s+", " ", _bo_dau(q).lower()).strip(" .!?,")
    return plain in {
        "next",
        "continue",
        "go on",
        "tiep",
        "tiep tuc",
        "tiep di",
        "lam tiep",
        "chay tiep",
        "co",
        "yes",
        "y",
        "ok",
        "okay",
        "dong y",
    }

def _la_phan_anh_pipeline(q: str) -> bool:
    plain = re.sub(r"\s+", " ", _bo_dau(q).lower()).strip()
    if not any(step in plain for step in ("buoc 1", "buoc 2", "buoc 3", "buoc 4")):
        return False
    return any(
        marker in plain
        for marker in (
            "bug",
            "loi",
            "sai",
            "khong dung",
            "tai sao",
            "vi sao",
            "sao lai",
            "moi chay",
            "ma da",
            "nhay",
            "xo",
            "skip",
            "bo qua",
            "logic",
        )
    )

def _last_completed_pipeline_step(chat_history) -> int:
    """Tìm bước pipeline gần nhất đã hoàn tất trong lịch sử chat."""
    for msg in reversed(chat_history or []):
        if msg.get("role") != "assistant":
            continue
        content = _bo_dau(msg.get("content") or "").lower()
        has_done = "hoan tat" in content or "hoan thanh" in content
        is_negative_notice = any(
            term in content
            for term in ("chua co", "cu hon", "vui long chay", "hay bat dau")
        )

        if ("buoc 3" in content and has_done) or "f1-f6 charts" in content:
            return 3
        if ("buoc 2" in content and has_done) or (
            "a450labeled.parquet" in content and not is_negative_notice
        ):
            return 2
        if ("buoc 1" in content and has_done) or (
            "a450etl.parquet" in content and not is_negative_notice
        ):
            return 1
    return 0

def _file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0

def _labeled_outdated() -> bool:
    return (
        os.path.exists(ETL_FILE)
        and os.path.exists(LABELED_FILE)
        and _file_mtime(LABELED_FILE) < _file_mtime(ETL_FILE)
    )

def _infer_next_pipeline_tool(chat_history) -> str:
    last_step = _last_completed_pipeline_step(chat_history)
    if last_step == 1:
        return "tool_buoc2_feature_engineering"
    if last_step == 2:
        return "tool_buoc3_score_report"
    if last_step >= 3:
        return ""

    if not os.path.exists(ETL_FILE):
        return "tool_buoc1_etl"
    if not os.path.exists(LABELED_FILE) or _labeled_outdated():
        return "tool_buoc2_feature_engineering"
    return "tool_buoc3_score_report"

def _la_tra_cuu_report(q: str) -> bool:
    """True nếu câu hỏi liên quan đến báo cáo user (rpt_users, highmls)."""
    return any(k in q for k in _KEYWORD_REPORT)

def _la_tra_cuu_giao_dich(q: str) -> bool:
    """True nếu câu hỏi liên quan đến phân tích giao dịch thô (a450labeled)."""
    return any(k in q for k in _KEYWORD_TRANSACTION)

def _la_cau_hoi_meta(q: str) -> bool:
    return any(k in q for k in _KEYWORD_META)

def _la_phan_hoi_sua_sai(q: str) -> bool:
    return any(k in q for k in _KEYWORD_CORRECTION)


# =============================================================================
# 5. LOGIC TỪNG NHÁNH
# =============================================================================

# ---------------------------------------------------------------------------
# Helper dùng chung
# ---------------------------------------------------------------------------

def _lam_sach_sql(sql: str) -> str:
    """Bỏ markdown fence, chỉ giữ phần SELECT."""
    sql = sql.strip().replace("```sql", "").replace("```SQL", "").replace("```", "")
    match = re.search(r"(SELECT[\s\S]*)", sql, re.IGNORECASE)
    return match.group(1).strip() if match else sql.strip()

def _format_polars_df(df, rows: int) -> str:
    """Format Polars DataFrame thành string trả về chat."""
    if df.shape == (1, 1):
        val = df.row(0)[0]
        col = df.columns[0]
        val_str = f"{val:,}" if isinstance(val, (int, float)) else str(val)
        return f"📊 **{col}**: {val_str}"
    tbl = df.to_pandas().to_string(index=False)
    return f"📊 Kết quả ({rows} dòng):\n\n```\n{tbl}\n```"


# ---------------------------------------------------------------------------
# 5a. NHÁNH B — "? câu hỏi" → LLM sinh SQL → tool5 (Polars SQLContext)
#
#   Bảng dùng được: rpt_users, highmls, txn_flg, maindb
#   Phù hợp: hỏi về user, nhóm, ml_score, giao dịch flagged
# ---------------------------------------------------------------------------

def _query_tu_nhien(cau_hoi: str) -> str:
    prompt = _SQL_PROMPT_TOOL5.format(schema=_SCHEMA_TOOL5, cau_hoi=cau_hoi)
    sql_raw = _invoke_llm(prompt).content.strip()
    sql = _lam_sach_sql(sql_raw)

    print(f"\n[NHÁNH B] SQL sinh ra:\n{sql}\n")

    tool   = tool5.get_tool()
    result = tool.execute_sql_safe(sql)

    if not result["success"]:
        return (
            f"❌ SQL lỗi: {result['error']}\n\n"
            f"SQL đã sinh:\n```sql\n{sql}\n```\n\n"
            f"Gợi ý: Thử diễn đạt lại câu hỏi, hoặc dùng tab **Truy vấn dữ liệu** để viết SQL thủ công."
        )

    df   = result["data"]
    rows = result["rows"]

    if rows == 0:
        return f"ℹ️ Không có dữ liệu thoả điều kiện.\n\nSQL đã dùng:\n```sql\n{sql}\n```"

    return _format_polars_df(df, rows)


# ---------------------------------------------------------------------------
# 5b. NHÁNH C1 — tra cứu report (tool5, rpt_users/highmls)
#
#   Gọi khi câu hỏi liên quan đến user, nhóm, ml_score, báo cáo
#   Giống nhánh B nhưng được kích hoạt tự động từ câu chat thường
# ---------------------------------------------------------------------------

def _tra_cuu_report(cau_hoi: str) -> str:
    """Sinh SQL → chạy trên tool5 (Polars SQLContext). Dùng cho câu hỏi về báo cáo user."""
    prompt = _SQL_PROMPT_TOOL5.format(schema=_SCHEMA_TOOL5, cau_hoi=cau_hoi)
    sql_raw = _invoke_llm(prompt).content.strip()
    sql = _lam_sach_sql(sql_raw)

    print(f"\n[NHÁNH C1] SQL sinh ra:\n{sql}\n")

    tool   = tool5.get_tool()
    result = tool.execute_sql_safe(sql)

    if not result["success"]:
        return (
            f"❌ Không thể truy vấn report: {result['error']}\n\n"
            f"SQL đã sinh:\n```sql\n{sql}\n```\n\n"
            f"💡 Bạn có thể dùng tab **Truy vấn dữ liệu** để viết SQL thủ công, "
            f"hoặc thêm dấu `?` ở đầu câu hỏi để tôi thử lại với thêm ngữ cảnh."
        )

    df   = result["data"]
    rows = result["rows"]

    if rows == 0:
        return "ℹ️ Không tìm thấy dữ liệu thoả điều kiện trong báo cáo."

    return _format_polars_df(df, rows)


# ---------------------------------------------------------------------------
# 5c. NHÁNH C2 — tra cứu giao dịch thô (DuckDB, a450labeled)
#
#   Gọi khi câu hỏi liên quan đến feature, phân phối, hành vi giao dịch
#   Có retry 3 lần nếu SQL lỗi. Hiển thị SQL để user kiểm chứng.
# ---------------------------------------------------------------------------

_DUCKDB_CFG = _CFG["duckdb"]

_PATH_MAP = {
    "{labeled_file}": LABELED_FILE,
    "{etl_file}":     ETL_FILE,
}

def _tao_views_duckdb(con):
    """Tạo tất cả VIEW được khai báo trong agent.json['duckdb']['views']."""
    for view_name, tmpl in _DUCKDB_CFG["views"].items():
        path = tmpl
        for k, v in _PATH_MAP.items():
            path = path.replace(k, v)
        con.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{path}')")

# Từ khoá câu hỏi mơ hồ — hỏi lại thay vì đoán
_TU_KHOA_MO_HO   = _CFG["keywords_ambiguous"]
_TU_KHOA_RO_RANG = _CFG["keywords_clear"]

def _tra_cuu_giao_dich(cau_hoi: str) -> str:
    """Sinh SQL → chạy trên DuckDB (a450labeled). Dùng cho phân tích giao dịch thô."""
    import duckdb

    if not os.path.exists(LABELED_FILE):
        return (
            f"⚠️ Chưa tìm thấy `{os.path.basename(LABELED_FILE)}`. "
            f"Vui lòng chạy **Bước 2 (Feature Engineering)** trước."
        )

    q_lower = cau_hoi.lower()
    if any(k in q_lower for k in _TU_KHOA_MO_HO) and not any(k in q_lower for k in _TU_KHOA_RO_RANG):
        return (
            "🤔 Câu hỏi còn chung chung. Bạn muốn phân tích theo tiêu chí nào?\n\n"
            "Ví dụ:\n"
            "  • Thống kê số giao dịch theo từng loại flag\n"
            "  • Phân phối ml_score của giao dịch non-rule\n"
            "  • Top 10 sender gửi nhiều tiền nhất\n"
            "  • Tỉ lệ giao dịch có đuôi cá độ (is_bet_tail)\n"
            "  • Giao dịch từ IP ngoài VN"
        )

    last_error = ""
    generated_sql = ""
    max_retry = _DUCKDB_CFG["max_retry"]

    try:
        con = duckdb.connect(config={"threads": 4})
        _tao_views_duckdb(con)
    except Exception as e:
        return f"❌ Không thể khởi tạo dữ liệu: {e}"

    for attempt in range(1, max_retry + 1):
        error_hint = f"\n\nLẦN TRƯỚC BỊ LỖI: {last_error}\nHãy sửa lại." if last_error else ""
        prompt = _SQL_PROMPT_DUCKDB.format(schema=_SCHEMA_LABELED, cau_hoi=cau_hoi) + error_hint

        try:
            raw = _invoke_llm(prompt).content.strip()
        except Exception as e:
            con.close()
            return f"❌ Lỗi gọi LLM: {e}"

        generated_sql = _lam_sach_sql(raw)
        # Nối thành 1 dòng, bỏ comment
        lines = [l.strip() for l in generated_sql.splitlines()
                 if l.strip() and not l.strip().startswith("--")]
        generated_sql = " ".join(lines)

        sql_display = f"📋 SQL (lần {attempt}):\n```sql\n{generated_sql}\n```\n"
        print(f"\n[NHÁNH C2] {sql_display}")

        try:
            import threading
            timeout_sec = _DUCKDB_CFG["timeout"]
            result_holder = {}
            error_holder  = {}

            def _run():
                try:
                    result_holder["df"] = con.execute(generated_sql).df()
                except Exception as ex:
                    error_holder["err"] = str(ex)

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=timeout_sec)

            if t.is_alive():
                con.interrupt()
                last_error = f"Query vượt quá {timeout_sec}s timeout"
                if attempt == max_retry:
                    con.close()
                    return f"❌ Sau {max_retry} lần thử, SQL vẫn lỗi.\nLỗi cuối: {last_error}\n\n{sql_display}"
                continue

            if "err" in error_holder:
                raise Exception(error_holder["err"])

            result_df = result_holder["df"]
            con.close()
        except Exception as e:
            last_error = str(e)
            if attempt == max_retry:
                con.close()
                return (
                    f"❌ Sau {max_retry} lần thử, SQL vẫn lỗi.\n"
                    f"Lỗi cuối: {last_error}\n\n{sql_display}"
                )
            continue

        if result_df.empty:
            return f"{sql_display}ℹ️ Không có dòng nào thoả điều kiện."

        if result_df.shape == (1, 1):
            val = result_df.iloc[0, 0]
            col = result_df.columns[0]
            val_str = f"{val:,}" if isinstance(val, (int, float)) else str(val)
            return f"{sql_display}📊 **{col}**: {val_str}"

        tbl = result_df.to_string(index=False)
        return f"{sql_display}📊 Kết quả ({len(result_df)} dòng):\n\n```\n{tbl}\n```"

    return f"❌ Không thể hoàn thành sau {max_retry} lần thử."


# =============================================================================
# 6. WRAPPER FUNCTIONS (tool1–4 với path rõ ràng)
#
# Lưu ý ánh xạ bước ↔ tool file:
#   Bước 1 → tool1._etl
#   Bước 2 → tool2._feature_engineering
#   Bước 3 → tool4._inference_and_report   (tool4, KHÔNG phải tool3)
#   Bước 4 → tool3._train_isolation_forest (tool3, KHÔNG phải tool4)
# =============================================================================

def _run_etl() -> str:
    return _etl(
        raw_dir=os.environ.get("A450_RAW_DIR",     os.path.join(_BASE_DIR, "raw")),
        ipref_file=os.environ.get("A450_IPREF_FILE", os.path.join(_BASE_DIR, "ref", "allip.csv")),
        out_file=ETL_FILE,
    )

def _run_feature_engineering() -> str:
    return _feature_engineering(
        etl_file=ETL_FILE,
        labeled_file=LABELED_FILE,
        temp_dir=os.environ.get("A450_TEMP_DIR", os.path.join(_BASE_DIR, ".tmp")),
    )

def _run_score_report() -> str:
    _inference_and_report()  # tool4: inference + export report + charts
    return (
        f"✅ Bước 3 hoàn tất. File lưu tại: {OUTPUT_DIR}\n"
        f"  - transactions_flagged.parquet\n"
        f"  - report_users.parquet / report_flagged_users.csv\n"
        f"  - report_high_score_users_{os.environ.get('A450_ML_SCORE_THRESHOLD', 70)}.parquet\n"
        f"  - report_bookie.parquet / .csv\n"
        f"  - F1–F6 charts (.png)"
    )

def _run_retrain() -> str:
    return _train_isolation_forest(  # tool3: chỉ train, không export report
        labeled_file=LABELED_FILE,
        output_dir=OUTPUT_DIR,
    )


# =============================================================================
# 7. ĐĂNG KÝ TOOLS
# =============================================================================

class NoInput(BaseModel):
    pass

class QueryInput(BaseModel):
    cau_hoi: str


buoc1_tool = StructuredTool.from_function(
    func=_run_etl,
    name="tool_buoc1_etl",
    args_schema=NoInput,
    description=(
        "Bước 1: Gộp CSV thô, tra IP location (iploc), chuẩn hoá nội dung giao dịch. "
        "Đầu ra: a450etl.parquet. "
        "Kích hoạt: ETL, gộp file, tra IP, chuẩn hoá, bước 1. "
    ),
)

buoc2_tool = StructuredTool.from_function(
    func=_run_feature_engineering,
    name="tool_buoc2_feature_engineering",
    args_schema=NoInput,
    description=(
        "Bước 2: Tạo cột phái sinh, flag pattern, feature engineering, gán nhãn rule-based. "
        "Đầu ra: a450labeled.parquet. "
        "Kích hoạt: feature engineering, tạo biến, gán nhãn, labeling, bước 2. "
    ),
)

buoc3_tool = StructuredTool.from_function(
    func=_run_score_report,
    name="tool_buoc3_score_report",
    args_schema=NoInput,
    description=(
        "Bước 3: Chạy model ML (inference/scoring), xuất report, tạo charts. "
        "Đầu ra: report_users.parquet, transactions_flagged.parquet, report_bookie.parquet, report_high_score_users_70.parquet. "
        "Kích hoạt: chạy model, score, xuất báo cáo, tạo chart/biểu đồ, bước 3. "
        "KHÔNG kích hoạt khi user chỉ muốn XEM kết quả report đã có. "
    ),
)

buoc4_tool = StructuredTool.from_function(
    func=_run_retrain,
    name="tool_buoc4_retrain_model",
    args_schema=NoInput,
    description=(
        "Bước 4: Retrain Isolation Forest, lưu model mới. "
        "CHỈ kích hoạt khi user nói rõ: retrain, đào tạo lại, huấn luyện lại, bước 4. "
        "KHÔNG kích hoạt khi user chỉ muốn chạy model → dùng Bước 3."
    ),
)

tra_cuu_report_tool = StructuredTool.from_function(
    func=_tra_cuu_report,
    name="tool_tra_cuu_report",
    args_schema=QueryInput,
    description=(
        "Tra cứu báo cáo user: bookie, gambler, recipient1, recipient2, depositor1, depositor2, ml_score, nhóm rủi ro. "
        "Truy vấn rpt_users, highmls, txn_flg, maindb qua Polars SQLContext. "
        "Kích hoạt: hỏi về số lượng user, nhóm, điểm ML, danh sách bookie/gambler."
    ),
)

tra_cuu_giao_dich_tool = StructuredTool.from_function(
    func=_tra_cuu_giao_dich,
    name="tool_tra_cuu_giao_dich",
    args_schema=QueryInput,
    description=(
        "Phân tích giao dịch thô trên a450labeled.parquet qua DuckDB. "
        "Kích hoạt: hỏi về số tiền, đuôi cá độ, IP, phân phối, feature sender/receiver, "
        "tỉ lệ bet_tail, thống kê giao dịch theo ngày."
    ),
)

DANH_SACH_TOOLS = [buoc1_tool, buoc2_tool, buoc3_tool, buoc4_tool,
                   tra_cuu_report_tool, tra_cuu_giao_dich_tool]


def _get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = _get_llm().bind_tools(DANH_SACH_TOOLS)
    return _llm_with_tools


def _invoke_llm_with_tools(payload):
    try:
        return _get_llm_with_tools().invoke(payload)
    except Exception as exc:
        _raise_friendly_llm_error(exc)
        raise


# =============================================================================
# 8. PIPELINE GUARD
# =============================================================================

def _kiem_tra_dieu_kien(tool_name: str) -> str | None:
    """Trả về thông báo lỗi nếu chưa đủ điều kiện chạy tool, None nếu OK."""
    if tool_name == "tool_buoc2_feature_engineering":
        if not os.path.exists(ETL_FILE):
            return (
                f"⚠️ Chưa có `{os.path.basename(ETL_FILE)}`. "
                f"Vui lòng chạy **Bước 1 (ETL)** trước."
            )

    elif tool_name in ("tool_buoc3_score_report", "tool_buoc4_retrain_model"):
        if not os.path.exists(ETL_FILE):
            return f"⚠️ Chưa có `{os.path.basename(ETL_FILE)}`. Hãy bắt đầu từ **Bước 1**."
        if not os.path.exists(LABELED_FILE):
            return (
                f"⚠️ Chưa có `{os.path.basename(LABELED_FILE)}`. "
                f"Vui lòng chạy **Bước 2 (Feature Engineering)** trước."
            )
        if _labeled_outdated():
            return (
                f"⚠️ `{os.path.basename(LABELED_FILE)}` đang cũ hơn "
                f"`{os.path.basename(ETL_FILE)}`. "
                f"Vui lòng chạy **Bước 2 (Feature Engineering)** trước khi chạy bước này."
            )
        if tool_name == "tool_buoc3_score_report" and not os.path.exists(MODEL_PATH):
            return (
                f"⚠️ Chưa có model `{os.path.basename(MODEL_PATH)}`.\n"
                f"Nếu đã retrain: copy từ `output/models/` vào `models/`.\n"
                f"Nếu chưa có: chạy **Bước 4 (Retrain)** trước."
            )

    return None


# =============================================================================
# 9. HELPERS
# =============================================================================

def _rut_gon_lich_su(chat_history, limit=8) -> list:
    """Lấy tối đa `limit` tin nhắn cuối, bỏ tin rỗng và cắt bớt nếu quá dài."""
    if not chat_history:
        return []
    cleaned = []
    for msg in chat_history[-limit:]:
        role    = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content[:4000]})
    return cleaned

def _invoke_tool(ten_tool: str, arg_tool: dict) -> str:
    dispatch = {
        "tool_buoc1_etl":                 buoc1_tool,
        "tool_buoc2_feature_engineering": buoc2_tool,
        "tool_buoc3_score_report":        buoc3_tool,
        "tool_buoc4_retrain_model":       buoc4_tool,
        "tool_tra_cuu_report":            tra_cuu_report_tool,
        "tool_tra_cuu_giao_dich":         tra_cuu_giao_dich_tool,
    }
    tool = dispatch.get(ten_tool)
    if tool is None:
        return f"Lỗi: Không tìm thấy tool '{ten_tool}'."
    return tool.invoke(arg_tool)


_PSEUDO_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([A-Za-z0-9_]+)>\s*([\s\S]*?)\s*</function>\s*</tool_call>",
    re.IGNORECASE,
)


def _parse_pseudo_tool_args(raw_args: str) -> dict:
    raw_args = (raw_args or "").strip()
    if not raw_args:
        return {}
    try:
        parsed = _json.loads(raw_args)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _chay_pseudo_tool_calls_if_any(text: str) -> str:
    """Fallback cho model in pseudo tool-call XML thay vì gọi tool thật."""
    matches = list(_PSEUDO_TOOL_CALL_RE.finditer(text or ""))
    if not matches:
        return text

    cleaned_text = _PSEUDO_TOOL_CALL_RE.sub("", text or "").strip()
    outputs = []
    for match in matches:
        tool_name = match.group(1)
        tool_args = _parse_pseudo_tool_args(match.group(2))
        err = _kiem_tra_dieu_kien(tool_name)
        if err:
            outputs.append(err)
            continue
        print(f"[LOG] Pseudo tool-call → kích hoạt: {tool_name}")
        outputs.append(str(_invoke_tool(tool_name, tool_args)))

    return "\n\n".join(part for part in [cleaned_text, *outputs] if part).strip()

def _log(nhanh: str, input_str: str, output_str: str):
    print(f"\n{'='*60}")
    print(f"[NHÁNH {nhanh}] IN : {input_str[:120]}")
    print(f"[NHÁNH {nhanh}] OUT: {output_str[:300]}")
    print(f"{'='*60}\n")


# =============================================================================
# 10. HÀM ĐIỀU PHỐI CHÍNH
# =============================================================================

def chay_agent_aml(user_question: str, chat_history=None) -> str:
    """
    Entry point — app.py gọi hàm này.

    Thứ tự ưu tiên xử lý:
      B  → "? câu hỏi"      : query tự nhiên → LLM sinh SQL → tool5
      A  → "[query] SQL"    : SQL trực tiếp → tool5
      C0 → câu hỏi META    : trả lời mô tả chức năng
      C1 → PIPELINE keyword : chạy bước 1/2/3/4
      C2 → tra cứu REPORT  : LLM sinh SQL → tool5 (rpt_users/highmls)
      C3 → tra cứu GIAO DỊCH: LLM sinh SQL → DuckDB (a450labeled)
      C4 → sửa sai         : LLM với tools + lịch sử
      C5 → fallback LLM    : LLM với tools
    """
    user_question = (user_question or "").strip()
    q = user_question.lower()

    # =========================================================
    # NHÁNH B — "? bao nhiêu bookie có ml_score > 70?"
    # =========================================================
    if user_question.startswith("?"):
        cau_hoi = user_question[1:].strip()
        if not cau_hoi:
            return "Vui lòng nhập câu hỏi sau dấu `?`."
        result = _query_tu_nhien(cau_hoi)
        _log("B", user_question, result)
        return result

    # =========================================================
    # NHÁNH A — "[query] SELECT ..."  (thêm [e] ở cuối để xuất CSV)
    # =========================================================
    if q.startswith("[query]"):
        body = user_question[7:].strip()
        if not body:
            return "⚠️ Vui lòng nhập câu SQL sau `[query]`."

        # Kiểm tra cú pháp [e] ở cuối (không phân biệt hoa thường, cho phép khoảng trắng)
        export_csv = bool(re.search(r"\[e\]\s*$", body, re.IGNORECASE))
        sql = re.sub(r"\[e\]\s*$", "", body, flags=re.IGNORECASE).strip()

        tool        = tool5.get_tool()
        result_data = tool.execute_sql_safe(sql)

        if result_data["success"]:
            df   = result_data["data"]
            rows = result_data["rows"]
            if rows == 0:
                return "ℹ️ Query thành công nhưng không có dòng nào thoả điều kiện."

            if export_csv:
                import datetime
                _query_dir = os.path.join(_BASE_DIR, "output", "query")
                os.makedirs(_query_dir, exist_ok=True)
                ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = os.path.join(_query_dir, f"export_{ts}.csv")
                df.write_csv(csv_path)
                out = (
                    f"✅ Xuất CSV thành công — **{rows} dòng**\n\n"
                    f"📁 `{csv_path}`\n\n"
                    + _format_polars_df(df, rows)
                )
            else:
                out = _format_polars_df(df, rows)
        else:
            out = f"❌ Lỗi SQL: {result_data['error']}"

        _log("A", user_question, out)
        return out

    # =========================================================
    # NHÁNH C0 — câu hỏi META
    # =========================================================
    if _la_cau_hoi_meta(q):
        _log("C0", user_question, "→ mô tả chức năng")
        return _MO_TA_CHUC_NANG

    # =========================================================
    # NHÁNH C0.25 — user phản ánh lỗi logic pipeline, không chạy tool
    # =========================================================
    if _la_phan_anh_pipeline(q):
        out = (
            "Bạn nói đúng: pipeline phải chạy tuần tự **Bước 1 → Bước 2 → Bước 3**. "
            "Câu này đang phản ánh lỗi logic nên tôi sẽ không kích hoạt tool nào. "
            "Nếu muốn chạy tiếp từ trạng thái hiện tại, hãy nhập `next` hoặc `tiếp tục`."
        )
        _log("C0.25", user_question, out)
        return out

    # =========================================================
    # NHÁNH C0.5 — user xác nhận / yêu cầu chạy bước kế tiếp
    # =========================================================
    if _la_lenh_tiep_tuc(q):
        next_tool = _infer_next_pipeline_tool(chat_history)
        if not next_tool:
            out = "✅ Bước 3 đã hoàn tất. Bạn có thể tra cứu báo cáo hoặc chạy query SQL."
            _log("C0.5", user_question, out)
            return out

        err = _kiem_tra_dieu_kien(next_tool)
        if err:
            _log("C0.5", user_question, f"→ guard chặn: {next_tool}")
            return err
        _log("C0.5", user_question, f"→ bước kế tiếp: {next_tool}")
        return str(_invoke_tool(next_tool, {}))

    # =========================================================
    # NHÁNH C1 — PIPELINE keyword (ưu tiên trước tra cứu)
    # =========================================================
    tool_name = _nhan_dien_pipeline(q)
    if tool_name:
        err = _kiem_tra_dieu_kien(tool_name)
        if err:
            _log("C1", user_question, f"→ guard chặn: {tool_name}")
            return err
        _log("C1", user_question, f"→ kích hoạt: {tool_name}")
        return str(_invoke_tool(tool_name, {}))

    # =========================================================
    # NHÁNH C2 — tra cứu REPORT (rpt_users, highmls qua tool5)
    # =========================================================
    if _la_tra_cuu_report(q):
        out = _tra_cuu_report(user_question)
        _log("C2", user_question, out)
        return out

    # =========================================================
    # NHÁNH C3 — tra cứu GIAO DỊCH thô (a450labeled qua DuckDB)
    # =========================================================
    if _la_tra_cuu_giao_dich(q):
        out = _tra_cuu_giao_dich(user_question)
        _log("C3", user_question, out)
        return out

    # =========================================================
    # NHÁNH C4 — SỬASAI (LLM với tools + lịch sử)
    # =========================================================
    if _la_phan_hoi_sua_sai(q):
        messages = [{"role": "system", "content": _SYSTEM_PROMPT_FULL}]
        messages.extend(_rut_gon_lich_su(chat_history))
        if not messages or messages[-1].get("content") != user_question:
            messages.append({"role": "user", "content": user_question})
        ai_msg = _invoke_llm_with_tools(messages)
        if ai_msg.tool_calls:
            messages.append(ai_msg)
            for tc in ai_msg.tool_calls:
                err = _kiem_tra_dieu_kien(tc["name"])
                if err:
                    return err
                out = _invoke_tool(tc["name"], tc["args"])
                messages.append({"role": "tool", "content": str(out), "tool_call_id": tc["id"]})
            result = _invoke_llm(messages).content
        else:
            result = ai_msg.content
        result = _chay_pseudo_tool_calls_if_any(str(result))
        _log("C4", user_question, result)
        return result

    # =========================================================
    # NHÁNH C5 — FALLBACK LLM với tools
    # =========================================================
    messages = [{"role": "system", "content": _SYSTEM_PROMPT_FULL}]
    messages.extend(_rut_gon_lich_su(chat_history))
    messages.append({"role": "user", "content": user_question})
    ai_msg = _invoke_llm_with_tools(messages)

    if ai_msg.tool_calls:
        messages.append(ai_msg)
        for tc in ai_msg.tool_calls:
            err = _kiem_tra_dieu_kien(tc["name"])
            if err:
                return err
            print(f"[LOG] Fallback LLM → kích hoạt: {tc['name']}")
            out = _invoke_tool(tc["name"], tc["args"])
            messages.append({"role": "tool", "content": str(out), "tool_call_id": tc["id"]})
        result = _invoke_llm(messages).content
    else:
        result = ai_msg.content

    result = _chay_pseudo_tool_calls_if_any(str(result))
    _log("C5", user_question, result)
    return result


# =============================================================================
# 11. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=== TEST AGENT LOCAL ===\n")
    test_cases = [
        ("META",       "bạn có thể làm gì?"),
        ("PIPELINE",   "chạy feature engineering đi"),
        ("PIPELINE",   "bước 3 đi"),
        ("PIPELINE",   "retrain model"),
        ("REPORT B",   "? bao nhiêu user là bookie?"),
        ("REPORT B",   "? top 5 user có ml_score cao nhất"),
        ("REPORT C2",  "bao nhiêu bookie?"),
        ("TXNS C3",    "tỉ lệ giao dịch có đuôi cá độ là bao nhiêu?"),
        ("TXNS C3",    "giao dịch từ IP ngoài VN có bao nhiêu?"),
        ("QUERY A",    "[query] SELECT COUNT(*) as cnt FROM rpt_users WHERE group = 'bookie'"),
    ]
    for label, cau_hoi in test_cases:
        print(f"[{label}] {cau_hoi}")
        print(f"→ {chay_agent_aml(cau_hoi)[:200]}\n")
