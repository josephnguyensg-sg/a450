A. Bối cảnh & Vấn đề

Team CPL - ZaloPay chịu trách nhiệm giám sát giao dịch phục vụ tuân thủ quy định phòng, chống rửa tiền (AML). Quy trình hiện tại yêu cầu nhân sự viết query thủ công, xuất CSV, và sàng lọc bằng Excel — dẫn đến nhiều hạn chế: tốn thời gian, dễ sai sót (typo, sai query, đánh giá cảm tính), tiêu chí rule-based phải thường xuyên cập nhật, và Excel không xử lý được file dữ liệu lớn.

B. Giải pháp - AI Agent

Hiệu năng xử lý dữ liệu
- Toàn bộ pipeline viết bằng Python, giao diện webchat chạy bằng Streamlit (tiết kiệm RAM).
- Dữ liệu lưu dạng Parquet thay vì CSV; xử lý nặng với Polars (lazy evaluation), DuckDB, Scikit-learn.
- Các tham số quan trọng (đường dẫn, tiêu chí lọc, model features...) không hard-code — lưu trong file cấu hình để dễ vận hành và bảo trì.
- Ngoài webchat, user có thể tương tác qua Telegram (hiện mở cho 3 tài khoản nội bộ team CPL).

Giảm thao tác thủ công & tăng độ chính xác
- User chỉ cần 1 câu query tải dữ liệu thô, sau đó ra lệnh bằng ngôn ngữ tự nhiên để Agent tự động thực hiện toàn bộ: ETL & gán nhãn giao dịch, chạy model ML để scoring, xuất báo cáo dạng HTML kèm biểu đồ, và retrain model khi cần.

Tra cứu thông tin nhanh
- User có thể viết SQL trực tiếp hoặc đặt câu hỏi bằng ngôn ngữ tự nhiên — Agent tự chuyển thành SQL, thực thi và trả kết quả ngay trên màn hình chat.

Tính năng mở rộng
- Agent có thêm một mode "for fun": user mô tả tình huống, Agent đánh giá có phải hành vi rửa tiền không và thông báo hình phạt tương ứng — kèm nội dung quảng bá sản phẩm ZaloPay.