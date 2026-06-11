# telegram_bot.py
import asyncio
import logging
import os
import re
from collections import deque

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

import json
import agent

# ── Load cấu hình từ telegram_bot.json ───────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "telegram_bot.json"), encoding="utf-8") as _f:
    _CFG = json.load(_f)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", _CFG.get("telegram_token"))
TG_MAX_CHARS    = _CFG["tg_max_chars"]
TG_WEBCHAT_HINT = _CFG["tg_webchat_hint"]
MAX_HISTORY     = _CFG["max_history"]
_START_MESSAGE  = _CFG["start_message"]
MAX_USERS = _CFG.get("max_users", 1)
ALLOWED_USERS = set(_CFG.get("allowed_users", []))

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# =============================================================================
# QUEUE & WORKER
# =============================================================================

_request_queue: asyncio.Queue = asyncio.Queue()
_agent_busy = False


async def _agent_worker():
    """Worker tuần tự — đảm bảo agent chỉ xử lý 1 request tại một thời điểm."""
    global _agent_busy
    logger.info("Agent worker khởi động.")

    while True:
        update, context, user_question, chat_history = await _request_queue.get()
        _agent_busy = True

        user = update.effective_user
        logger.info("Xử lý: %s (%s): %s", user.full_name, user.id, user_question[:80])

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: agent.chay_agent_aml(user_question, chat_history),
            )
        except Exception as e:
            response = "❌ Lỗi hệ thống: " + str(e)
            logger.exception("Agent lỗi: %s", e)

        await _send_response(update, context, str(response))

        _agent_busy = False
        _active_users.discard(user.id)
        _request_queue.task_done()


# =============================================================================
# GỬI RESPONSE
# =============================================================================

_CODE_PATTERN = re.compile(r"```[\s\S]*?```")
_LANG_STRIP   = re.compile(r"^```\w*\n?")


async def _send_response(update: Update, context: ContextTypes.DEFAULT_TYPE, response: str):
    """
    Gửi response về Telegram.
    - Tách text và code block để render đúng.
    - Cắt nếu vượt TG_MAX_CHARS và nhắc dùng webchat.
    """
    is_truncated = False
    if len(response) > TG_MAX_CHARS:
        response = response[:TG_MAX_CHARS]
        is_truncated = True

    parts = _CODE_PATTERN.split(response)
    codes = _CODE_PATTERN.findall(response)

    segments = []
    for i, text in enumerate(parts):
        if text.strip():
            segments.append(("text", text.strip()))
        if i < len(codes):
            code_content = _LANG_STRIP.sub("", codes[i]).rstrip("`").strip()
            segments.append(("code", code_content))

    if not segments:
        await update.message.reply_text("_(Không có nội dung)_", parse_mode="Markdown")
        return

    for kind, seg in segments:
        if kind == "text":
            try:
                await update.message.reply_text(seg, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(seg)
        else:
            # Chia nhỏ nếu code quá dài
            chunks = [seg[i:i+3400] for i in range(0, len(seg), 3400)]
            for chunk in chunks:
                msg = "```\n" + chunk + "\n```"
                try:
                    await update.message.reply_text(msg, parse_mode="Markdown")
                except Exception:
                    await update.message.reply_text(chunk)

    if is_truncated:
        try:
            await update.message.reply_text(TG_WEBCHAT_HINT, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text("💡 Nội dung bị cắt ngắn. Vui lòng dùng webchat để xem đầy đủ.")


# =============================================================================
# CHAT HISTORY PER USER
# =============================================================================

_user_histories: dict[int, deque] = {}
_active_users = set()
# MAX_HISTORY đã load từ telegram_bot.json


def _get_history(user_id: int) -> list:
    if user_id not in _user_histories:
        _user_histories[user_id] = deque(maxlen=MAX_HISTORY)
    return list(_user_histories[user_id])


def _add_to_history(user_id: int, role: str, content: str):
    if user_id not in _user_histories:
        _user_histories[user_id] = deque(maxlen=MAX_HISTORY)
    _user_histories[user_id].append({"role": role, "content": content})


# =============================================================================
# HANDLERS
# =============================================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_text(
        "Chào " + user.first_name + "! 👋\n\n"
        "Tôi là AML AI Agent của Zalopay.\n"
        + _START_MESSAGE,
        parse_mode="Markdown",
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    _user_histories.pop(user_id, None)
    await update.message.reply_text("✅ Đã xóa lịch sử hội thoại.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    queue_size = _request_queue.qsize()
    status = "🔴 Đang bận" if _agent_busy else "🟢 Rảnh"
    await update.message.reply_text(
        "Trạng thái agent: " + status + "\n"
        "Số yêu cầu đang chờ: " + str(queue_size)
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user    = update.effective_user
    user_id = user.id
    if ALLOWED_USERS and user_id not in ALLOWED_USERS:
        await update.message.reply_text("⛔ Bạn không là top nên không được xài bot.")
        return
    if user_id not in _active_users:
        if len(_active_users) >= MAX_USERS:
            await update.message.reply_text(
                "⚠️ Bot đang đủ người sử dụng, thử lại sau."
            )
            return
        _active_users.add(user_id)

    user_question = (update.message.text or "").strip()

    if not user_question:
        return

    # Nhận diện chế độ [query] — giống tab SQL bên streamlit
    is_query = user_question.lower().startswith("[query]")
    if is_query:
        clean_sql = user_question[7:].strip()
        if not clean_sql:
            await update.message.reply_text(
                "⚠️ Vui lòng nhập câu SQL sau cú pháp `[query]`.",
                parse_mode="Markdown",
            )
            return
        # Preview SQL cho user xác nhận
        preview = "```sql\n" + clean_sql + "\n```"
        await update.message.reply_text(
            "🗄️ Nhận lệnh SQL:\n" + preview,
            parse_mode="Markdown",
        )

    # Lưu vào history
    _add_to_history(user_id, "user", user_question)
    chat_history = _get_history(user_id)

    # Thông báo vị trí queue
    queue_size = _request_queue.qsize()
    if _agent_busy or queue_size > 0:
        pos = queue_size + 1
        await update.message.reply_text(
            "⏳ Agent đang bận. Yêu cầu của bạn xếp hàng (vị trí #" + str(pos) + ").\n"
            "Vui lòng chờ..."
        )
    else:
        await update.message.reply_text("⚙️ Đang xử lý...")

    await _request_queue.put((update, context, user_question, chat_history))


# =============================================================================
# MAIN
# =============================================================================

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("clear",  cmd_clear))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Khởi động worker trong post_init để đảm bảo event loop đã sẵn sàng
    async def _post_init(application):
        asyncio.create_task(_agent_worker())
        logger.info("Agent worker khởi động.")

    app.post_init = _post_init

    logger.info("Bot đang chạy...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
