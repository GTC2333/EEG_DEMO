#!/usr/bin/env bash

# Default ports in this OpenHands demo runtime
HTTP_PORT=${EEG_DEMO_HTTP_PORT:-51730}
WS_PORT=${EEG_DEMO_WS_PORT:-55141}

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

# stop previous
if [ -f "$LOG_DIR/frontend_http.pid" ]; then
  kill "$(cat $LOG_DIR/frontend_http.pid)" 2>/dev/null || true
  rm -f "$LOG_DIR/frontend_http.pid"
fi
if [ -f "$LOG_DIR/backend_mock.pid" ]; then
  kill "$(cat $LOG_DIR/backend_mock.pid)" 2>/dev/null || true
  rm -f "$LOG_DIR/backend_mock.pid"
fi

EEG_DEMO_HTTP_PORT="$HTTP_PORT" python frontend/serve_frontend.py > "$LOG_DIR/frontend_http.log" 2>&1 &
echo $! > "$LOG_DIR/frontend_http.pid"

EEG_DEMO_WS_PORT="$WS_PORT" python backend/server.py > "$LOG_DIR/backend_mock.log" 2>&1 &
echo $! > "$LOG_DIR/backend_mock.pid"

sleep 1

echo "Frontend: http://0.0.0.0:${HTTP_PORT}/frontend/index.html?ws_port=${WS_PORT}"
echo "WebSocket: ws://0.0.0.0:${WS_PORT}"
