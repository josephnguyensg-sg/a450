#!/bin/bash
set -euo pipefail

APP_PORT="${PORT:-8501}"

echo "=== Khởi động Streamlit ==="
streamlit run app.py \
  --server.port="$APP_PORT" \
  --server.address=0.0.0.0 \
  --server.headless=true &
streamlit_pid=$!

if [[ "${ENABLE_TELEGRAM_BOT:-false}" == "true" ]]; then
  if [[ -z "${TELEGRAM_BOT_TOKEN:-}" ]]; then
    echo "ENABLE_TELEGRAM_BOT=true nhưng TELEGRAM_BOT_TOKEN chưa được set."
    exit 1
  fi

  echo "=== Khởi động Telegram Bot ==="
  python telegram_bot.py &
  telegram_pid=$!

  wait -n "$streamlit_pid" "$telegram_pid"
  echo "Một service đã dừng, tắt container."
  exit 1
fi

echo "=== Telegram Bot đang tắt (ENABLE_TELEGRAM_BOT=false) ==="
wait "$streamlit_pid"
