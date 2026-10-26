#!/usr/bin/env bash
# SPU 학습 K겹 교차검증. fold마다 run_spu_parties.sh를 한 번씩 돌린다 (WSL, 프로젝트 루트에서 실행).
# 사용법: bash mpc/run_spu_cv.sh [K=5] [train_spu_party 추가 인자...]
# 로그: mpc/logs/cv/fold{k}/hospital{0,1,2}.log, 체크포인트: mpc/checkpoints_cv/fold{k}/
set -u
cd "$(dirname "$0")/.."
K=${1:-5}
shift || true
for ((k = 0; k < K; k++)); do
    echo "=== fold $((k + 1))/$K ==="
    LOG_DIR=mpc/logs/cv/fold$k bash mpc/run_spu_parties.sh \
        --n-folds "$K" --fold "$k" --ckpt-dir "mpc/checkpoints_cv/fold$k" "$@" || exit 1
done
echo "=== fold별 마지막 test_acc ==="
for ((k = 0; k < K; k++)); do
    grep test_acc "mpc/logs/cv/fold$k/hospital0.log" | tail -1 | sed "s/.*test_acc=/fold$k: /"
done | tee mpc/logs/cv/summary.txt
awk '{s += $2; ss += $2 * $2; n++} END {m = s / n; printf "mean=%.4f  std=%.4f\n", m, sqrt(ss / n - m * m)}' \
    mpc/logs/cv/summary.txt | tee -a mpc/logs/cv/summary.txt
