import time
import json
import subprocess
import streamlit as st
from streamlit_ace import st_ace  # Thư viện Editor dành cho chế độ Query SQL
import agent
try:
    import iagent
except ImportError:
    iagent = None
# tool5 được gọi qua agent

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load cấu hình từ app.json ─────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "app.json"), encoding="utf-8") as _f:
    _CFG = json.load(_f)

_WEBCHAT_DIR = os.path.join(BASE_DIR, _CFG["webchat_dir"])
BOT_AVATAR   = os.path.join(_WEBCHAT_DIR, "bot_avatar.png")
USER_AVATAR  = os.path.join(_WEBCHAT_DIR, "user_avatar.png")
FAVICON      = os.path.join(_WEBCHAT_DIR, "favicon.png")

st.set_page_config(
    page_title=_CFG["page_title"],
    page_icon=FAVICON,
    layout=_CFG["layout"],
)

MAX_HISTORY          = _CFG["max_history"]
DEFAULT_MESSAGE_A450 = _CFG["default_message_a450"]
DEFAULT_MESSAGE_IBFT = _CFG["default_message_ibft"]
_AGENT_MODES         = _CFG["agent_modes"]
_SELECTBOX_LABEL     = _CFG["selectbox_label"]
_BASH_BLACKLIST      = _CFG["bash_blacklist"]
_BASH_TIMEOUT        = _CFG["bash_timeout"]
_TYPING_SPEED        = _CFG["typing_speed"]
_TYPING_MAX_CHARS    = _CFG["typing_max_chars"]

# ── Khởi tạo session state ────────────────────────────────────────────────
if "agent_mode" not in st.session_state:
    st.session_state.agent_mode = "A450"
if "messages" not in st.session_state:
    st.session_state.messages = [DEFAULT_MESSAGE_A450]

st.title("Zalopay - AML AI Agent")
st.caption("_**Lưu ý:**_")
st.caption("  _AI Agent này chỉ phân tích giao dịch chuyển tiền tại Zalopay trên môi trường local_")
st.caption("  _Dữ liệu giao dịch là giả để thử nghiệm mô hình machine learning_")
st.caption("_____________________________________________________________________________________")

with st.sidebar:
    # ── Chọn chế độ Agent (góc trên sidebar) ─────────────────────────────
    selected_mode = st.selectbox(_SELECTBOX_LABEL, _AGENT_MODES, index=0 if st.session_state.agent_mode == _AGENT_MODES[0] else 1)
    if selected_mode != st.session_state.agent_mode:
        st.session_state.agent_mode = selected_mode
        st.session_state.messages = [DEFAULT_MESSAGE_A450 if selected_mode == "A450" else DEFAULT_MESSAGE_IBFT]
        st.rerun()

    st.header("Tiện ích")
    if st.button("🗑️ Xóa hội thoại"):
        st.session_state.messages = [DEFAULT_MESSAGE_A450 if st.session_state.agent_mode == "A450" else DEFAULT_MESSAGE_IBFT]
        st.rerun()

    with st.expander("🗂️ Cấu trúc các bảng dữ liệu"):
        with st.expander("`rpt_users`"):
            st.code("""user_id             str
user_group          str
total_flagged_tx    u32
ml_score            f32
bookie_tx_count     u32
gambler_tx_count    u32
recipient1_tx_count u32
depositor1_tx_count u32
recipient2_tx_count u32
depositor2_tx_count u32
total_sent_tx       u32
total_sent_amount   i64
unique_receivers    u32
total_recv_tx       u32
total_recv_amount   i64
unique_senders      u32
ml_s                f32
ml_r                f32""", language=None)

        with st.expander("`highmls`"):
            st.code("""user_id             str
bookie_tx_count     u32
gambler_tx_count    u32
recipient1_tx_count u32
depositor1_tx_count u32
recipient2_tx_count u32
depositor2_tx_count u32
ml_s                f32
ml_r                f32
total_sent_tx       u32
total_sent_amount   i64
unique_receivers    u32
total_recv_tx       u32
total_recv_amount   i64
unique_senders      u32
ml_score            f32
total_flagged_tx    u32
user_group               str""", language=None)

        with st.expander("`txn_flg`"):
            st.code("""reqdate             str
userid              str
appuser             str
amount              i64
iploc               str
is_bookie_tx        i8
is_gambler_tx       i8
is_recipient1_tx    i8
is_depositor1_tx    i8
is_recipient2_tx    i8
is_depositor2_tx    i8
hit_any_rule        i8
ml_score            f32
is_bet_tail         i8
amount_tail3        i16
desc_has_win        i8
desc_match_b1       i8
desc_match_b2       i8""", language=None)

        with st.expander("`maindb`"):
            st.code("""reqdate                 str
userid                  str
amount                  i64
appuser                 str
iploc                   str
amount_tail3            i16
is_bet_tail             i8
desc_has_win            i8
desc_match_b1           i8
desc_match_b2           i8
sender_tx_count         u32
sender_unique_receivers u32
sender_total_sent       i64
sender_avg_amount       f64
sender_max_amount       i64
sender_std_amount       f64
sender_unique_ips       u32
sender_bet_tail_ratio   f64
sender_win_desc_ratio   f64
sender_b1_ratio         f64
sender_b2_ratio         f64
recv_tx_count           u32
recv_unique_senders     u32
recv_total_received     i64
recv_avg_amount         f64
recv_max_amount         i64
recv_std_amount         f64
recv_bet_tail_ratio     f64
recv_win_desc_ratio     f64
recv_b1_ratio           f64
recv_b2_ratio           f64
recv_send_tx_ratio      f64
str_ip_change           i8
str_device_share        i8
str_smurfing            i8
str_rapid_tx            i8
is_bookie_tx            i8
is_gambler_tx           i8
is_recipient1_tx        i8
is_depositor1_tx        i8
is_recipient2_tx        i8
is_depositor2_tx        i8
hit_any_rule            i8""", language=None)

# 1. HIỂN THỊ LỊCH SỬ CHAT
for message in st.session_state.messages:
    avatar = BOT_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

st.write("---")

# 2. THIẾT KẾ GIAO DIỆN TÁCH BIỆT: CHAT THƯỜNG VS QUERY SQL
# Sử dụng st.tabs để người dùng vừa có thể gõ [query] ở chat thường, vừa có không gian gõ SQL chuyên nghiệp
tab_chat, tab_sql = st.tabs(["💬 Chat Thông Thường", "🗄️ Truy vấn dữ liệu"])

user_input = ""
is_query_mode = False

# --- LÝ TRÍ CHAT THƯỜNG ---
with tab_chat:
    chat_input = st.chat_input("Nhập câu hỏi hoặc câu truy vấn như hướng dẫn ở trên nhé")
    if chat_input:
        user_input = chat_input
        # Kiểm tra xem có ký tự [query] ở đầu không
        if chat_input.strip().lower().startswith("[query]"):
            is_query_mode = True

# --- LÝ TRÍ TRUY VẤN SQL (EDITOR THÔNG MINH) ---
with tab_sql:
    st.caption("Chỉ nhập câu SQL vào bên dưới")
    sql_input = st_ace(
        value="SELECT COUNT(*) as cnt FROM highmls",
        language="sql",
        theme="clouds",
        height=120,
        auto_update=False,
        key="sql_editor_panel"
    )
    if st.button("Thực thi SQL 🚀"):
        if sql_input:
            # Tự động thêm tag [query] nếu viết ở tab SQL để Agent nhận diện
            user_input = f"[query]\n{sql_input}"
            is_query_mode = True

# 3. XỬ LÝ LOGIC KHI CÓ DỮ LIỆU ĐẦU VÀO
if user_input:
    # Tránh gửi trùng tin nhắn liên tiếp
    if len(st.session_state.messages) == 1 or user_input != st.session_state.messages[-2].get("content"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.rerun()

# ── Helper: chạy lệnh bash an toàn ──────────────────────────────────────
def _run_bash(cmd: str) -> str:
    cmd = cmd.strip()
    for banned in _BASH_BLACKLIST:
        if banned in cmd:
            return f"❌ Lệnh bị chặn: `{banned}`"
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=_BASH_TIMEOUT,
            cwd=BASE_DIR
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if out and err:
            return f"{out}\n\n[stderr]\n{err}"
        return out or err or "(không có output)"
    except subprocess.TimeoutExpired:
        return "❌ Timeout sau 30 giây"
    except Exception as e:
        return f"❌ Lỗi: {e}"

# LUỒNG XỬ LÝ AGENT PHẢN HỒI
if st.session_state.messages[-1]["role"] == "user":
    last_query = st.session_state.messages[-1]["content"]
    is_bash  = last_query.strip().lower().startswith("[bash]")
    is_query = last_query.strip().lower().startswith("[query]")

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        try:
            import io, contextlib
            _stdout_buf = io.StringIO()
            _stderr_buf = io.StringIO()
            with st.spinner("Tôi đang suy nghĩ và xử lý..."):
                with contextlib.redirect_stdout(_stdout_buf), contextlib.redirect_stderr(_stderr_buf):
                    if is_bash:
                        cmd = last_query.strip()[6:].strip()
                        bash_out = _run_bash(cmd)
                        response = f"```\n{bash_out}\n```"
                    elif st.session_state.agent_mode == "IBFT":
                        if iagent is None:
                            response = "❌ Mode này chưa phát triển :)) Hãy chọn Agent Mode `A450` nhé."
                        else:
                            response = iagent.chay_agent_ibft(
                                last_query,
                                chat_history=st.session_state.messages[:-1][-MAX_HISTORY:]
                            )
                    else:
                        response = agent.chay_agent_aml(
                            last_query,
                            chat_history=st.session_state.messages[:-1][-MAX_HISTORY:]
                        )
        except Exception as e:
            response = f"❌ Lỗi hệ thống: {e}"

        # ── Hiển thị stdout/stderr logs nếu có ───────────────────────────
        _logs = _stdout_buf.getvalue().strip()
        _errs = _stderr_buf.getvalue().strip()
        if _logs or _errs:
            _all_logs = "\n".join(filter(None, [_logs, _errs]))
            with st.expander("📋 Logs", expanded=False):
                st.code(_all_logs, language=None)

        response = str(response)
        full_response = response

        # Tách phần text và code block để render đúng
        import re as _re
        _code_pattern = _re.compile(r"```[\s\S]*?```")
        _parts = _code_pattern.split(response)

        if len(_parts) == 1:
            # Không có code block — hiệu ứng gõ chữ bình thường
            message_placeholder = st.empty()
            if len(response) < _TYPING_MAX_CHARS:
                _typed = ""
                for word in response.split():
                    _typed += word + " "
                    message_placeholder.markdown(_typed + "▌")
                    time.sleep(_TYPING_SPEED)
                message_placeholder.markdown(_typed)
                full_response = _typed
            else:
                message_placeholder.markdown(response)
        else:
            # Có code block — split rồi render từng phần
            _segments = _code_pattern.split(response)
            # _segments = [text, code, text, code, ...]
            for i, seg in enumerate(_segments):
                if i % 2 == 0:
                    if seg.strip():
                        st.markdown(seg.strip())
                else:
                    st.code(seg, language=None)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()
