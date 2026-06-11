# tool6.py
# Xuất file rpt.html (nhúng tất cả PNG trong output_dir) và gửi email

import os
import json
import base64
import smtplib
from email.message import EmailMessage

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Load cấu hình từ tool6.json ───────────────────────────────────────────
with open(os.path.join(_HERE, "tool6.json"), encoding="utf-8") as _f:
    _CFG = json.load(_f)

GMAIL_ACCOUNT      = os.environ.get("A450_GMAIL_ACCOUNT",  _CFG["gmail_account"])
GMAIL_APP_PASSWORD = os.environ.get("A450_GMAIL_PASSWORD", _CFG["gmail_app_password"])
TO_EMAIL           = os.environ.get("A450_TO_EMAIL",        _CFG["to_email"])
_EMAIL_SUBJECT     = _CFG["email_subject"]
_EMAIL_BODY        = _CFG["email_body"]
_HTML_TITLE        = _CFG["html_title"]
_HTML_H1           = _CFG["html_h1"]


def export_html_and_send(output_dir: str,
                          descriptions: dict | str | None = None,
                          gmail_account: str    = GMAIL_ACCOUNT,
                          gmail_password: str   = GMAIL_APP_PASSWORD,
                          to_email: str         = TO_EMAIL) -> str:
    """
    Nhúng toàn bộ file *.png trong output_dir vào output_dir/rpt.html,
    rồi gửi file đó qua Gmail.

    Parameters
    ----------
    output_dir     : thư mục chứa PNG (và nơi sẽ lưu rpt.html)
    descriptions   : ánh xạ tên file PNG → mô tả hiển thị trong HTML.
                     Có thể truyền vào theo 3 cách:
                       - dict  : {"chart1.png": "Biểu đồ doanh thu", ...}
                       - str   : đường dẫn tới file JSON chứa dict trên
                       - None  : không có mô tả (hành vi cũ)
    gmail_account  : địa chỉ Gmail người gửi
    gmail_password : App Password của Gmail
    to_email       : địa chỉ người nhận

    Returns
    -------
    Đường dẫn tuyệt đối tới file rpt.html vừa tạo.
    """
    # ── 0. Xử lý descriptions ─────────────────────────────────────────────
    default_json = os.path.join(_HERE, "desc.json")
    if isinstance(descriptions, str):
        json_path = descriptions
    elif descriptions is None and os.path.exists(default_json):
        json_path = default_json
    else:
        json_path = None

    if json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            desc_map: dict = json.load(f)
        print(f"[tool6] 📋 Đọc mô tả từ {json_path}")
    elif isinstance(descriptions, dict):
        desc_map = descriptions
    else:
        desc_map = {}
    output_html = os.path.join(output_dir, "rpt.html")

    # ── 1. Thu thập PNG (bỏ qua rpt.html nếu có) ──────────────────────────
    png_files = sorted(
        f for f in os.listdir(output_dir)
        if f.lower().endswith(".png")
    )

    if not png_files:
        print(f"[tool6] ⚠️  Không tìm thấy file PNG nào trong '{output_dir}'")

    # ── 2. Dựng HTML ──────────────────────────────────────────────────────
    parts = [f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{_HTML_TITLE}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
h3   {{ margin-top: 30px; color: #333; margin-bottom: 8px; }}
img  {{ max-width: 1100px; display: block;
       margin-bottom: 10px; border: 1px solid #ccc; }}
</style>
</head>
<body>
<h1>{_HTML_H1}</h1>
"""]

    for filename in png_files:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        desc_text = desc_map.get(filename, "")
        heading_html = f'<h3>{desc_text}</h3>\n' if desc_text else ""
        parts.append(heading_html
                     + f'<img src="data:image/png;base64,{encoded}" alt="{filename}">\n')

    parts.append("</body>\n</html>\n")

    html_content = "".join(parts)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[tool6] ✅ Đã tạo {output_html}  ({len(png_files)} ảnh nhúng)")

    # ── 3. Gửi email ──────────────────────────────────────────────────────
    # msg = EmailMessage()
    # msg["Subject"] = _EMAIL_SUBJECT
    # msg["From"]    = gmail_account
    # msg["To"]      = to_email
    # msg.set_content(_EMAIL_BODY)

    # with open(output_html, "rb") as f:
    #     msg.add_attachment(
    #         f.read(),
    #         maintype="text",
    #         subtype="html",
    #         filename="rpt.html",
    #     )

    # print("[tool6] 📤 Đang kết nối Gmail...")
    # with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
    #     smtp.login(gmail_account, gmail_password)
    #     smtp.send_message(msg)

    # print("[tool6] ✅ Đã gửi email thành công!")
    # return output_html
