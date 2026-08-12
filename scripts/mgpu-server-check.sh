#!/bin/zsh
# Server-side half of issue #51: the layer-split GPU timeout, and a profile of where
# tensor-split decode spends its time. Send back the whole output plus sample.txt.
#
#   ./scripts/mgpu-server-check.sh <model.gguf> [3+ GPU list, e.g. 1,2,3]
#
# The second argument is optional and only for machines with three or more cards: it runs
# section 6, the reduction across all of them.
#
# Uses the installed app by default; override with TOSH_BIN=/path/to/bin.
# Pick the GPUs with GGML_METAL_DEVICE_LIST (0,0 splits over a single card, which is how we
# reproduce this here). Never with -dev: on tensor split it lands on a path that is neither
# split nor single and hides the effect of the sync flags.
set -uo pipefail

MODEL="${1:?usage: $0 <model.gguf>}"
PORT=18082

BIN="${TOSH_BIN:-/Applications/ToshLLM.app/Contents/Resources/bin}"
[ -x "$BIN/llama-server" ] || BIN="$(dirname "$0")/../vendor/llama.cpp/build-static/bin"
[ -x "$BIN/llama-server" ] || { echo "llama-server not found; set TOSH_BIN to the app's Resources/bin" >&2; exit 1; }

echo "=== engine"
echo "  $BIN"
"$BIN/llama-server" --version 2>&1 | head -2

export GGML_METAL_CONCURRENCY_DISABLE=1
export TOSH_FA_AMD=1

OUT="$(mktemp -d)"
echo "logs in $OUT"

MGPU_ENV=(env -u TOSH_MGPU_PEER -u TOSH_MGPU_EVENTS)

serve() { # serve <log> <extra args...>
    local log="$1"; shift
    "${MGPU_ENV[@]}" "$BIN/llama-server" -m "$MODEL" -ngl 99 --no-mmap -fa on --jinja \
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

ask() { # ask <n_predict> [prompt]
    local prompt="${2:-Write a detailed technical explanation of how virtual memory paging works.}"
    curl -sf -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
        -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$1}"
}

# the batch threshold only bites above it, so section 5 needs a prompt that clears every value
LONG_PROMPT="$(printf 'Explain in precise technical detail how virtual memory paging works, covering page tables, the TLB, page faults, replacement policies and how the kernel backs pages with swap. %.0s' $(seq 1 24))"

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

# "none" runs here too: section 3 generates 400 tokens, so its t/s is not comparable with these
for cfg in "none:" "peer:TOSH_MGPU_PEER=1" "events:TOSH_MGPU_EVENTS=1" \
           "peer+events:TOSH_MGPU_PEER=1 TOSH_MGPU_EVENTS=1"; do
    label="${cfg%%:*}"
    echo "\n=== 4. tensor split, coherence with $label"
    MGPU_ENV=(env -u TOSH_MGPU_PEER -u TOSH_MGPU_EVENTS ${=cfg#*:})
    if serve "$OUT/$label.log" --split-mode tensor; then
        ask 120 > "$OUT/$label.json"
        echo "output (must be coherent, no mixed scripts or garbage):"
        show "$OUT/$label.json"
        stop
    fi
    report "$OUT/$label.log" "$label"
done

# Where the all-reduce should stop preferring the peer copy: with both transfers enabled,
# batches under the threshold go by events. 0 = always peer, 999999 = always events.
# Same 120 tokens as section 4, so those rows are comparable with these.
for th in 0 8 32 128 999999; do
    echo "\n=== 5. tensor split, peer+events, TOSH_MGPU_PEER_MIN_BATCH=$th"
    MGPU_ENV=(env -u TOSH_MGPU_PEER -u TOSH_MGPU_EVENTS \
              TOSH_MGPU_PEER=1 TOSH_MGPU_EVENTS=1 TOSH_MGPU_PEER_MIN_BATCH=$th)
    if serve "$OUT/th$th.log" --split-mode tensor; then
        ask 120 "$LONG_PROMPT" > "$OUT/th$th.json"
        show "$OUT/th$th.json"
        stop
    fi
    report "$OUT/th$th.log" "min-batch $th"
done

# Only with three or more cards: the butterfly reduction of the Metal backend against the
# generic one. The butterfly is the default since 0.83.17, so the generic row is the one that
# needs a variable; "reducing across N devices" in the log tells the two rows apart.
DEVS="${2:-}"
if [ -n "$DEVS" ]; then
    export GGML_METAL_DEVICE_LIST="$DEVS"
    for cfg in "generic:TOSH_MGPU_COMM_DISABLE=1" "butterfly:"; do
        label="${cfg%%:*}"
        echo "\n=== 6. tensor split on $DEVS, $label reduction"
        MGPU_ENV=(env -u TOSH_MGPU_PEER -u TOSH_MGPU_COMM_DISABLE TOSH_MGPU_EVENTS=1 ${=cfg#*:})
        if serve "$OUT/$label.log" --split-mode tensor; then
            ask 120 "$LONG_PROMPT" > "$OUT/$label.json"
            echo "output (must be coherent):"
            show "$OUT/$label.json"
            stop
        fi
        grep "reducing across" "$OUT/$label.log" | head -1
        report "$OUT/$label.log" "$label"
    done
fi
