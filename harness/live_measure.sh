#!/usr/bin/env bash
# 网页"现场实测"按钮: 单点采集, 负载定义与 testv2_run.sh 的 one 逐字相同(b0, 245 遍, 同模板)。
#   ./live_measure.sh <bal_test|test2> <srv> <th 1|2> <agents>
# 结果落 testv2_live/<test|test2>/s{srv}_l4_t{th}_a{ag}_b0/point.json —— 单独目录,
# 不混进 testv2/(论文数字 paper_numbers_v2 / demo_failmap 只读 testv2/, 不受影响)。
# 状态写 testv2_live/status.json 给网页轮询。一次只跑一个点(live_measure.pid)。
# 闸同 testv2: 磁盘 < 400G 不跑; PG 不就绪不跑; 有采集在跑不跑; steps 必须 = passes x questions, 否则作废。
# 不用 pgrep -f / pkill -f。
set -uo pipefail
cd "$(dirname "$0")"
P="$(pwd)"; OUT=../data/testv2_live; mkdir -p "$OUT"; LOG="$OUT/run.log"; ST="$OUT/status.json"
wlw=$1; s=$2; th=$3; ag=$4
case "$wlw" in bal_test) wl=test ;; test2) wl=test2 ;; *) echo "bad workload"; exit 2 ;; esac
status(){ python3 - "$ST" "$@" <<'PY'
import json,sys,time
p,state,msg=sys.argv[1:4]; extra=dict(a.split("=",1) for a in sys.argv[4:])
try: d=json.load(open(p))
except Exception: d={}
d.update(state=state,msg=msg,updated=time.strftime("%Y-%m-%d %H:%M:%S"),**extra)
json.dump(d,open(p,"w"),ensure_ascii=False)
PY
}
if [[ -f live_measure.pid ]] && kill -0 "$(cat live_measure.pid)" 2>/dev/null; then echo "busy"; exit 3; fi
for pf in testv2.pid rc3.pid rc4.pid rc5.pid rc5_chain.pid retest.pid ndbv2.pid ndbv3.pid ndbv3_pilot.pid build_ndb3_v2.pid daily_collect.pid; do
  [[ -f "$pf" ]] || continue; pid=$(cat "$pf" 2>/dev/null)
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && { echo "collection $pf running"; exit 4; }
done
echo $$ > live_measure.pid
trap 'rm -f live_measure.pid' EXIT
tag="s${s}_l4_t${th}_a${ag}_b0"; d="$OUT/$wl"; mkdir -p "$d"
status running "measuring ${s} cores / HT $([[ $th == 2 ]] && echo on || echo off) / ${ag} agents" \
  workload="$wlw" srv="$s" th="$th" agents="$ag" started="$(date '+%F %T')" started_ts="$(date +%s)"
g=$(df -BG --output=avail "${DATA_DISK:-$PWD}"|tail -1|tr -dc '0-9')
(( g < 400 )) && { status failed "disk ${g}G < 400G"; exit 5; }
pg_isready -p 5433 -U postgres >/dev/null 2>&1 || { status failed "PostgreSQL not ready"; exit 6; }
cpus_for(){ local n=$1 t=${2:-2} b=${3:-0} a=()
            for ((i=0;i<n;i++)); do a+=("$((b+i))"); ((t==2)) && a+=("$((b+i+24))"); done
            local IFS=,; echo "${a[*]}"; }
case "$wl" in
  test)  TPL=bird_test_v2;  RUN=bird_test_run;  Q=testset_100.json;  V=test_question_vecs.npz;  TR=test_trace.json ;;
  test2) TPL=bird_test2_v2; RUN=bird_test2_run; Q=testset2_100.json; V=test2_question_vecs.npz; TR=test2_trace.json ;;
esac
rm -rf "$d/$tag"
echo "════ live $(date '+%F %T') $wl/$tag ════" >>"$LOG"
export BIRD_BIZ_SCHEMA="" BIRD_TEMPLATE_DB="$TPL" BIRD_RUN_DB="$RUN" BIRD_QUESTIONS="$P/$Q" BIRD_VECS="$P/$V"
timeout -s TERM 2400 ./collect_bird.sh --server-cpus "$(cpus_for "$s" "$th")" \
  --load-cpus "$(cpus_for 4 2 16)" --agents "$ag" --mem-gb 16 --passes 245 \
  --questions 100 --ef-search 64 --writes-per-step 1 --tp-updates 1 \
  --business-write-pct 0 --delete-rate 0.01 --embedding-update-rate 0 \
  --value-sync atomic --trace-file "$P/$TR" --run-tag "$tag" --results-dir "$d" >>"$LOG" 2>&1
rc=$?
read st ex < <(python3 -c "import json;d=json.load(open('$d/$tag/point.json'));print(d.get('steps',0), d.get('passes',0)*d.get('questions',0))" 2>/dev/null || echo "0 -1")
if [[ "$st" != "$ex" ]]; then
  rm -rf "$d/$tag"; status failed "run incomplete (steps $st != $ex, rc $rc); discarded"; exit 7
fi
w=$(python3 -c "import json;print(json.load(open('$d/$tag/point.json'))['completion_seconds'])")
status done "measured ${w} s" completion_seconds="$w" finished="$(date '+%F %T')"
