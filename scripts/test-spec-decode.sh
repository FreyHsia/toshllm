#!/bin/zsh
# Measures speculative decoding (MTP or DFlash) against the plain autoregressive baseline,
# on BOTH predictable and unpredictable content, and sweeps the acceptance-backoff threshold.
#
#   ./scripts/test-spec-decode.sh <model-with-mtp.gguf> [threshold ...]
#
# Measuring only predictable content overstates the win: the backoff exists for the other
# side, and removing it drops unpredictable content below the baseline. Defaults sweep the
# thresholds that bracket real acceptance.
#
# Uses the engine in vendor/ by default; override with TOSH_BIN=/path/to/bin.
set -u
cd "$(dirname "$0")/.."

MODEL="${1:?usage: $0 <model.gguf> [threshold ...]}"
shift
THRESHOLDS=("$@")
[ ${#THRESHOLDS[@]} -gt 0 ] || THRESHOLDS=(0.65 0.60)

BIN="${TOSH_BIN:-vendor/llama.cpp/build-static/bin}"
[ -x "$BIN/llama-server" ] || { echo "no engine at $BIN (build it, or set TOSH_BIN)" >&2; exit 1; }

PORT="${PORT:-8099}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"; kill $SRV 2>/dev/null' EXIT

export GGML_METAL_CONCURRENCY_DISABLE=1
export TOSH_FA_AMD=1

# predictable: the case speculation is good at. unpredictable: the case the backoff guards.
GOOD=(
 'Write a Python function that validates an email with regex, with a docstring and examples.'
 'Explain what recursion is and give two commented Python examples.'
 'Write a Python class for a stack with push, pop, peek and len.'
)
BAD=(
 'Escribe un cuento original de 200 palabras sobre un farero que una mañana descubre que el mar ha desaparecido.'
 'Escribe un poema de 20 versos libres, sin rima, sobre la nostalgia que produce un objeto cotidiano.'
 'Write a list of 20 random 8-character combinations of letters and digits, one per line.'
)

serve() {
    "$BIN/llama-server" --host 127.0.0.1 --port "$PORT" -m "$MODEL" --load-mode none \
        -ngl 99 -c 8192 -np 1 --metrics "$@" > "$TMP/server.log" 2>&1 &
    SRV=$!
    local i
    for i in $(seq 1 400); do
        curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && return 0
        kill -0 $SRV 2>/dev/null || { echo "  server died, see $TMP/server.log"; tail -5 "$TMP/server.log"; return 1; }
        sleep 1
    done
    echo "  timeout waiting for the server"; return 1
}

stop() {
    kill $SRV 2>/dev/null
    local i
    for i in $(seq 1 30); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
    kill -9 $SRV 2>/dev/null
    sleep 1
    # a kill that silently failed would poison the next measurement
    pgrep -f "llama-server .*--port $PORT" >/dev/null && echo "  WARNING: server still alive"
    sleep 15
}

measure() {
    local label="$1" kind="$2"; shift 2
    serve "$@" || return
    local prompts n=0
    if [ "$kind" = bad ]; then prompts=("${BAD[@]}"); else prompts=("${GOOD[@]}"); fi
    # a metrics snapshot per prompt: the cell mean hides which prompt the
    # backoff actually engaged on
    curl -s -m 30 "http://127.0.0.1:$PORT/metrics" > "$TMP/m-1.txt"
    for p in "${prompts[@]}"; do
        curl -s -m 900 -o "$TMP/r$n.json" "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":400,\"temperature\":0,\"cache_prompt\":false}"
        curl -s -m 30 "http://127.0.0.1:$PORT/metrics" > "$TMP/m$n.txt"
        n=$((n+1))
    done
    python3 - "$label" "$TMP" "$n" <<'EOF'
import json,re,sys
label,tmp,n=sys.argv[1],sys.argv[2],int(sys.argv[3])
def metrics(i):
    t=open(f"{tmp}/m{i}.txt").read()
    def g(name):
        m=re.search(r'^llamacpp:spec_decode_%s (\d+)$'%name,t,re.M); return int(m.group(1)) if m else 0
    return g('num_draft_tokens_total'),g('num_accepted_tokens_total'),g('num_drafts_total')
sp=[json.load(open(f"{tmp}/r{i}.json")).get('timings',{}).get('predicted_per_second',0) for i in range(n)]
prev=metrics(-1)
rows=[]
for i in range(n):
    cur=metrics(i)
    d,a,dr=(c-p for c,p in zip(cur,prev))
    prev=cur
    rows.append("      prompt %d: tg %6.2f  drafts=%-4d acceptance=%s" %
                (i+1, sp[i], dr, "%.3f"%(a/d) if d else "  -  "))
d,a,dr=metrics(n-1)
t=open(f"{tmp}/m{n-1}.txt").read()
pos={int(m.group(1)):int(m.group(2)) for m in re.finditer(r'per_pos_total\{position="(\d+)"\} (\d+)',t)}
by=' '.join('p%d=%.0f%%'%(i,100*pos[i]/dr) for i in sorted(pos)) if dr else ''
print("  %-28s tg %6.2f t/s   drafts=%-5d acceptance=%.3f  %s" % (label, sum(sp)/len(sp), dr, a/d if d else 0, by))
print("\n".join(rows))
EOF
    stop
}

echo "=== GPU"
"$BIN/llama-bench" -m "$MODEL" -p 0 -n 1 -r 1 2>&1 | grep -iE "probed SIMD-group width" || true

for kind in good bad; do
    echo "\n=== ${kind:u} case"
    measure "baseline (no speculation)" $kind
    for thr in "${THRESHOLDS[@]}"; do
        TOSH_MTP_ACC_MIN=$thr measure "MTP, backoff $thr" $kind --spec-type draft-mtp
    done
done

echo "\nA threshold only wins if it beats the baseline in BOTH cases."
