#!/usr/bin/env bash
# GPUを空けないためのジョブキュー実行役。
# 使い方: nohup scripts/queue_worker.sh outputs/queue_<名前>.txt &
# キューファイルの先頭行を1つずつ取り出して実行する（空なら30秒待機）。
set -u
Q="$1"
LOCK="$Q.lock"
touch "$Q"
while true; do
  CMD=$(flock "$LOCK" bash -c "head -n 1 '$Q'; sed -i '1d' '$Q'")
  if [ -z "$CMD" ]; then sleep 30; continue; fi
  echo "[$(date +%H:%M:%S)] START $CMD" >> "$Q.history"
  bash -c "$CMD"
  echo "[$(date +%H:%M:%S)] END rc=$? $CMD" >> "$Q.history"
done
