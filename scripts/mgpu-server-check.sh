#!/bin/zsh
# Server-side half of issue #51: the layer-split GPU timeout, and a profile of where
# tensor-split decode spends its time. Send back the whole output plus sample.txt.
#
#   ./scripts/mgpu-server-check.sh <model.gguf>
#
# Uses the installed app by default; override with TOSH_BIN=/path/to/bin.
# TOSH_DEV=MTL0,MTL0 splits over a single card, which is how we reproduce this here.
set -uo pipefail

MODEL="${1:?usage: $0 <model.gguf>}"
PORT=18082

BIN="${TOSH_BIN:-/Applications/ToshLLM.app/Contents/Resources/bin}"
[ -x "$BIN/llama-server" ] || BIN="$(dirname "$0")/../vendor/llama.cpp/build-static/bin"
[ -x "$BIN/llama-server" ] || { echo "llama-server not found; set TOSH_BIN to the app's Resources/bin" >&2; exit 1; }

export GGML_METAL_CONCURRENCY_DISABLE=1
export TOSH_FA_AMD=1

OUT="$(mktemp -d)"
echo "logs in $OUT"

serve() { # serve <log> <extra args...>
    local log="$1"; shift
    local dev=()
    [ -n "${TOSH_DEV:-}" ] && dev=(-dev "$TOSH_DEV")
    "$BIN/llama-server" -m "$MODEL" -ngl 99 --no-mmap -fa on --jinja "${dev[@]}" \
        --host 127.0.0.1 --port $PORT "$@" > "$log" 2>&1 &
    SRV=$!
    for i in $(seq 1 300); do
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && return 0
        kill -0 $SRV 2>/dev/null || { echo "server died during load"; return 1; }
        sleep 1
    done
    echo "server never became ready"; return 1
}

show() { # show <json file>: models that think answer in reasoning_content, not content
    python3 -c "import json,sys
m=json.load(open(sys.argv[1]))['choices'][0]['message']
print((m.get('content') or m.get('reasoning_content') or '')[:200])" "$1" 2>/dev/null || head -c 200 "$1"
}

ask() { # ask <n_predict>
    curl -sf -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
        -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed technical explanation of how virtual memory paging works.\"}],\"max_tokens\":$1}"
}

stop() { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 3; }

report() { # report <log> <label>
    grep -iE "status 5|Timeout|not supported|error|abort" "$1" | head -5
    grep -E "model loaded" "$1" | head -1
    grep -E "eval time" "$1" | tail -2
    echo "[$2] done"
}

echo "\n=== 1. layer split, --fit off (the failing config)"
if serve "$OUT/fitoff.log" --split-mode layer --parallel 2 --fit off; then
    ask 64 > "$OUT/fitoff.json"; show "$OUT/fitoff.json"
    stop
fi
report "$OUT/fitoff.log" "fit off"

echo "\n=== 2. layer split, --fit on (same run, fitting enabled)"
if serve "$OUT/fiton.log" --split-mode layer --parallel 2 --fit on; then
    ask 64 > "$OUT/fiton.json"; show "$OUT/fiton.json"
    stop
fi
report "$OUT/fiton.log" "fit on"

echo "\n=== 3. tensor split, profile of decode"
if serve "$OUT/tensor.log" --split-mode tensor; then
    ask 400 > "$OUT/gen.json" &
    GEN=$!
    sleep 3
    sample $SRV 10 -f "$OUT/sample.txt" > /dev/null 2>&1 || echo "sample failed"
    wait $GEN
    echo "output (must be coherent, not 0000 or garbage):"
    show "$OUT/gen.json"
    stop
fi
report "$OUT/tensor.log" "tensor"
echo "\nattach $OUT/sample.txt"
