FROM python:3.11.15-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV A450_BASE_DIR=/app
ENV ENABLE_TELEGRAM_BOT=false

WORKDIR /app

# Cài OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY *.py .
COPY *.json .

# Copy các thư mục tĩnh vào image
COPY raw/     ./raw/
COPY ref/     ./ref/
COPY webchat/ ./webchat/
COPY models/  ./models/

# Tạo thư mục output và tmp (output sẽ được mount từ ngoài)
RUN mkdir -p output .tmp

EXPOSE 8501

# start.sh chạy Streamlit; Telegram chỉ bật khi ENABLE_TELEGRAM_BOT=true
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
