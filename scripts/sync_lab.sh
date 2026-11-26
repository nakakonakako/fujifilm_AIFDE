#!/usr/bin/env bash
# 研究室サーバの学習結果（予測値・設定・ログ）を WSL の outputs/ に回収する。モデル重みは回収しない。
set -u
cd "$(dirname "$0")/.."
timeout 120 ssh.exe -o BatchMode=yes lab-server \
  'cd ~/fujifilm_AIFDE/outputs && tar cf - --exclude="*.pt" --exclude="*.npy" */oof.csv */test_logit.csv */args.json */summary.json *.log queue_lab.txt* 2>/dev/null' \
  | tar xf - -C outputs/ 2>/dev/null
ls -1d outputs/*/ | wc -l
