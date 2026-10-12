#!/usr/bin/env bash
# 병원 3개를 같은 머신에서 각각 별도 프로세스로 띄운다 (WSL, 프로젝트 루트에서 실행).
# 포트: mpc/cluster_local.json (병원당 sf_port + spu_port).
# 사용법: bash mpc/run_spu_parties.sh --epochs 30 --max-batches 1
# 로그: mpc/logs/hospital{0,1,2}.log
set -u
cd "$(dirname "$0")/.."
PY=${PY:-venv_wsl/bin/python}
mkdir -p mpc/logs
pids=()
for party in hospital0 hospital1 hospital2; do
    PYTHONPATH=. "$PY" -m mpc.train_spu_party --party "$party" "$@" > "mpc/logs/$party.log" 2>&1 &
    pids+=($!)
    echo "[launch] $party pid=$!"
done
trap 'kill "${pids[@]}" 2>/dev/null' INT TERM
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "[done] exit status=$status  (로그: mpc/logs/)"
exit $status
