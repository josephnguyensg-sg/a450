# iagent.py
# =============================================================================
# Module tra cứu hình phạt rửa tiền (Điều 324 BLHS)
# Được gọi từ app.py khi agent_mode != "A450"
# Config (system_prompt, opening_message, llm_*) đọc từ agent.json key "iagent"
# =============================================================================

import os
import json
from langchain_openai import ChatOpenAI

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Load config từ agent.json ─────────────────────────────────────────────────
with open(os.path.join(_HERE, "agent.json"), encoding="utf-8") as _f:
    _CFG = json.load(_f)["iagent"]

_AI_PLATFORM_BASE_URL = os.environ.get(
    "AI_PLATFORM_BASE_URL",
    "https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1",
)

_llm = None

_NO_EMPTY_FINAL_INSTRUCTION = (
    "\n\nYÊU CẦU KỸ THUẬT BẮT BUỘC:\n"
    "- Trả lời trực tiếp bằng tiếng Việt trong final answer/content.\n"
    "- Không để phần trả lời rỗng.\n"
    "- Không chỉ suy nghĩ nội bộ; phải sinh nội dung người dùng đọc được."
)


def _get_api_key_and_source():
    for env_name in ("AI_PLATFORM_API_KEY", "OPENAI_API_KEY"):
        api_key = os.environ.get(env_name)
        if api_key:
            return api_key, env_name
    return None, None


def _format_llm_auth_error() -> str:
    _, api_key_source = _get_api_key_and_source()
    api_key_source = api_key_source or "AI_PLATFORM_API_KEY/OPENAI_API_KEY"
    return (
        "❌ Lỗi xác thực LLM (401 Unauthorized). "
        f"Runtime đang dùng `{api_key_source}` với "
        f"`AI_PLATFORM_BASE_URL={_AI_PLATFORM_BASE_URL}` và "
        f"`llm_model={_CFG.get('llm_model', '(unknown)')}`. "
        "Hãy kiểm tra key còn hiệu lực và thuộc đúng endpoint."
    )


def _is_llm_auth_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    exc_text = str(exc)
    exc_name = exc.__class__.__name__
    return (
        status_code == 401
        or "AuthenticationError" in exc_name
        or "Error code: 401" in exc_text
        or "Unauthorized" in exc_text
    )


def _message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _llm_extra_body() -> dict:
    model_name = str(_CFG.get("llm_model", "")).lower()
    llm_think = bool(_CFG.get("llm_think", False))
    if "qwen" not in model_name or llm_think:
        return {}
    return {
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _get_llm():
    global _llm
    if _llm is None:
        api_key, _ = _get_api_key_and_source()
        if not api_key:
            raise RuntimeError("Thiếu AI_PLATFORM_API_KEY hoặc OPENAI_API_KEY.")
        llm_kwargs = {
            "model": _CFG["llm_model"],
            "temperature": _CFG["llm_temperature"],
            "max_tokens": _CFG["llm_max_tokens"],
            "timeout": _CFG["llm_timeout"],
            "openai_api_key": api_key,
            "openai_api_base": _AI_PLATFORM_BASE_URL,
        }
        extra_body = _llm_extra_body()
        if extra_body:
            llm_kwargs["extra_body"] = extra_body
        _llm = ChatOpenAI(**llm_kwargs)
    return _llm


def get_opening_message() -> str:
    return _CFG["opening_message"]


def chay_agent_ibft(user_question: str, chat_history=None) -> str:
    """
    Entry point — app.py gọi hàm này khi agent_mode != 'A450'.

    Args:
        user_question: câu hỏi/mô tả hành vi của user
        chat_history:  list[dict] với keys 'role' và 'content'
    """
    user_question = (user_question or "").strip()
    if not user_question:
        return get_opening_message()

    messages = [{"role": "system", "content": _CFG["system_prompt"] + _NO_EMPTY_FINAL_INSTRUCTION}]

    # Thêm lịch sử hội thoại, bỏ opening message
    opening = get_opening_message()
    if chat_history:
        for msg in chat_history[-_CFG["max_history"]:]:
            role = msg.get("role", "")
            content = (msg.get("content") or "").strip()
            if role in {"user", "assistant"} and content and content != opening:
                messages.append({"role": role, "content": content[:3000]})

    messages.append({"role": "user", "content": user_question})

    try:
        response = _get_llm().invoke(messages)
        answer = _message_content_to_text(response.content)
        if answer:
            return answer
        retry_messages = messages[:-1] + [{
            "role": "user",
            "content": (
                f"{user_question}\n\n"
                "Yêu cầu kỹ thuật: hãy trả lời ngay trong `content` bằng tiếng Việt, "
                "không để rỗng và không chỉ tạo reasoning/thinking."
            ),
        }]
        response = _get_llm().invoke(retry_messages)
        answer = _message_content_to_text(response.content)
        if answer:
            return answer
        return (
            "❌ LLM đã trả về phản hồi rỗng. "
            "Với Qwen, hãy kiểm tra endpoint có hỗ trợ `enable_thinking=False` không, "
            "hoặc thử đổi model sang bản instruct/non-thinking."
        )
    except Exception as exc:
        if _is_llm_auth_error(exc):
            return _format_llm_auth_error()
        return f"❌ Lỗi hệ thống: {exc}"


# ── Entry point test local ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== TEST IAGENT LOCAL ===\n")
    test_cases = [
        "Tôi nhận tiền từ bạn bè rồi chuyển qua nhiều tài khoản khác nhau để che giấu nguồn gốc",
        "Tôi dùng tiền cờ bạc thắng được để mở quán cà phê",
        "Công ty tôi có tổ chức chuyên nhận tiền bẩn rồi đầu tư vào bất động sản, làm nhiều lần rồi",
    ]
    history = []
    for q in test_cases:
        print(f"USER: {q}")
        ans = chay_agent_ibft(q, chat_history=history)
        print(f"BOT : {ans[:400]}\n{'─'*60}\n")
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": ans})
