#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${BACKEND_DIR}/.run"
PID_FILE="${RUN_DIR}/backend.pid"
LOG_FILE="${RUN_DIR}/backend.log"

HOST="${RAG_SERVER_HOST:-127.0.0.1}"
PORT="${RAG_SERVER_PORT:-8000}"
APP_MODULE="${RAG_SERVER_APP:-app.main:app}"
HEALTH_URL="http://${HOST}:${PORT}/health"

PYTHON_BIN="${BACKEND_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

pid_from_file() {
  if [[ ! -f "${PID_FILE}" ]]; then
    return 1
  fi
  cat "${PID_FILE}"
}

listener_pids() {
  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi
  lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true
}

show_port_owner() {
  if ! command -v lsof >/dev/null 2>&1; then
    echo "Unable to inspect port owners (lsof not available)."
    return 0
  fi
  lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true
}

cleanup_stale_pid_file() {
  local pid
  pid="$(pid_from_file || true)"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f "${PID_FILE}"
  fi
}

is_running() {
  cleanup_stale_pid_file
  if [[ ! -f "${PID_FILE}" ]]; then
    return 1
  fi
  local pid
  pid="$(pid_from_file || true)"
  if [[ -z "${pid}" ]]; then
    return 1
  fi
  kill -0 "${pid}" 2>/dev/null
}

start_server() {
  mkdir -p "${RUN_DIR}"
  cleanup_stale_pid_file
  if is_running; then
    echo "Backend server already running (pid=$(pid_from_file))."
    exit 0
  fi

  local in_use_pids
  in_use_pids="$(listener_pids)"
  if [[ -n "${in_use_pids}" ]]; then
    echo "Port ${PORT} is already in use. Refusing to start a second server."
    show_port_owner
    echo "Stop the existing process first or run '$0 stop' to clean it up."
    exit 1
  fi

  echo "Starting backend server on ${HOST}:${PORT}..."
  (
    cd "${BACKEND_DIR}"
    nohup "${PYTHON_BIN}" -m uvicorn "${APP_MODULE}" --host "${HOST}" --port "${PORT}" \
      </dev/null \
      >"${LOG_FILE}" 2>&1 &
    echo $! >"${PID_FILE}"
  )

  local pid
  pid="$(cat "${PID_FILE}")"

  for _ in $(seq 1 40); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "Backend process exited during startup. Recent logs:"
      tail -n 80 "${LOG_FILE}" || true
      rm -f "${PID_FILE}"
      exit 1
    fi
    if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
      echo "Backend started (pid=${pid}). Health check passed."
      return
    fi
    sleep 0.5
  done

  echo "Backend process is running (pid=${pid}) but health check did not pass yet."
  echo "Use '$0 logs' to inspect startup logs."
}

stop_server() {
  local stopped_any=0
  cleanup_stale_pid_file

  if is_running; then
    local pid
    pid="$(pid_from_file)"
    echo "Stopping backend server (pid=${pid})..."
    kill "${pid}" 2>/dev/null || true
    stopped_any=1

    for _ in $(seq 1 20); do
      if ! kill -0 "${pid}" 2>/dev/null; then
        rm -f "${PID_FILE}"
        break
      fi
      sleep 0.25
    done

    if kill -0 "${pid}" 2>/dev/null; then
      echo "Process did not stop gracefully, sending SIGKILL..."
      kill -9 "${pid}" 2>/dev/null || true
      rm -f "${PID_FILE}"
    fi
  fi

  local remaining_pids
  remaining_pids="$(listener_pids)"
  if [[ -n "${remaining_pids}" ]]; then
    echo "Cleaning up remaining listener(s) on port ${PORT}: ${remaining_pids}"
    kill ${remaining_pids} 2>/dev/null || true
    sleep 0.4
    remaining_pids="$(listener_pids)"
    if [[ -n "${remaining_pids}" ]]; then
      echo "Force-killing remaining listener(s): ${remaining_pids}"
      kill -9 ${remaining_pids} 2>/dev/null || true
    fi
    stopped_any=1
  fi

  if [[ "${stopped_any}" -eq 1 ]]; then
    echo "Backend stopped."
  else
    echo "Backend server is not running."
  fi
}

status_server() {
  cleanup_stale_pid_file
  if is_running; then
    local pid
    pid="$(pid_from_file)"
    echo "Backend is running (pid=${pid}) at ${HOST}:${PORT}."
    if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
      echo "Health: OK"
    else
      echo "Health: unavailable"
    fi
  else
    local in_use_pids
    in_use_pids="$(listener_pids)"
    if [[ -n "${in_use_pids}" ]]; then
      echo "Backend PID file is not active, but port ${PORT} is in use by:"
      show_port_owner
    else
      echo "Backend is not running."
    fi
  fi
}

show_logs() {
  if [[ ! -f "${LOG_FILE}" ]]; then
    echo "No log file at ${LOG_FILE}"
    return
  fi
  tail -n 120 "${LOG_FILE}"
}

usage() {
  cat <<EOF
Usage: $0 {start|stop|restart|status|logs}

Environment overrides:
  RAG_SERVER_HOST   (default: 127.0.0.1)
  RAG_SERVER_PORT   (default: 8000)
  RAG_SERVER_APP    (default: app.main:app)
EOF
}

case "${1:-}" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  restart)
    stop_server
    start_server
    ;;
  status)
    status_server
    ;;
  logs)
    show_logs
    ;;
  *)
    usage
    exit 1
    ;;
esac
