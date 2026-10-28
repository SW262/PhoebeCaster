#!/usr/bin/env bash
# PhoebeCaster 演示服务。默认只绑 127.0.0.1。
# 要从别的机器看, ssh -L 8900:127.0.0.1:8900 过来。
# 停服务: ./start_server.sh --stop  (读 webui.pid)
set -e
cd "$(dirname "$0")"
PIDFILE=webui.pid
if [[ "${1:-}" == "--stop" ]]; then
  if [[ -f $PIDFILE ]] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
    kill "$(cat $PIDFILE)"; echo "已停 (pid $(cat $PIDFILE))"; rm -f $PIDFILE
  else echo "没在跑"; fi
  exit 0
fi
if [[ -f $PIDFILE ]] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
  echo "已在跑 (pid $(cat $PIDFILE)); 要重启先 ./start_server.sh --stop"; exit 1
fi
PORT=${1:-8900}
BIND=${BIND:-127.0.0.1}
WORKERS=${2:-2}
echo "PhoebeCaster -> http://$BIND:$PORT  ($WORKERS workers)"
exec gunicorn phoebeweb.wsgi:application \
    --chdir "$(pwd)" \
    --pid "$(pwd)/$PIDFILE" \
    --bind "$BIND:$PORT" \
    --workers "$WORKERS" \
    --timeout 180 \
    --access-logfile - \
    --error-logfile -
