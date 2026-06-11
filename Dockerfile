FROM python:3.11.15-slim

# Tránh interactive prompt khi cài package
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Cài OS dependencies tối thiểu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY *.py .
COPY *.json .

# Copy raw data vào image (output/ sẽ được mount từ ngoài)
COPY raw/ ./raw/

# Tạo thư mục output và tmp
RUN mkdir -p output .tmp

# Mặc định chạy pipeline rồi start agent
EXPOSE 8501

# Biến môi trường — override khi chạy container
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV A450_BASE_DIR=/app

# Chạy app.py bằng streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
