#!/bin/zsh
# Validates the AMD Flash-Attention kernels on a wave64 GPU (GCN/Vega), which we
# have no hardware for. Run it on the card and send back the whole output.
#
#   ./scripts/test-wave64-fa.sh /path/to/model.gguf
#
# A Turbo-compatible model also exercises the padded DK and cooperative-store paths.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?usage: $0 <model.gguf>}"
BIN="vendor/llama.cpp/build-static/bin"
[ -x "$BIN/test-backend-ops" ] || { echo "build the engine first: ./scripts/build-engines.sh" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

export TOSH_FA_AMD=1

echo "=== GPU"
"$BIN/llama-bench" -m "$MODEL" -p 0 -n 1 -r 1 2>&1 | grep -iE "probed SIMD-group width|found device" || true

WIDTH="$("$BIN/llama-bench" -m "$MODEL" -p 0 -n 1 -r 1 2>&1 | grep -oE "probed SIMD-group width = [0-9]+" | grep -oE "[0-9]+$" | head -1)"
if [ "$WIDTH" != "64" ]; then
    echo "NOTE: this GPU reports simd width $WIDTH, not 64. The run is still valid but"
    echo "      it exercises the wave32 paths, which we already test here."
fi

# The shapes the model actually uses, at a depth that reaches the split path.
echo "\n=== correctness (must be all OK)"
# Older llama.cpp checkouts name the target export-graph-ops (no test- prefix).
if [ -x "$BIN/test-export-graph-ops" ]; then EXPORT="$BIN/test-export-graph-ops"; else EXPORT="$BIN/export-graph-ops"; fi
TOSH_METAL_SIMD_WIDTH="$WIDTH" "$EXPORT" -m "$MODEL" -c 8192 -ub 512 -b 512 -fa on -o "$TMP/ops.txt" > /dev/null 2>&1
awk '$1==74' "$TMP/ops.txt" > "$TMP/fa.txt"
echo "shapes: $(wc -l < "$TMP/fa.txt")"
"$BIN/test-backend-ops" test -b MTL0 --test-file "$TMP/fa.txt" 2>&1 | grep -E "tests passed|FAIL"

echo "\n=== directed wave64 matrix"
"$BIN/test-backend-ops" test -b MTL0 -o FLASH_ATTN_EXT -p 'kv=257' 2>&1 | grep -E "tests passed|FAIL"
"$BIN/test-backend-ops" test -b MTL0 -o FLASH_ATTN_EXT -p 'kv=1025' 2>&1 | grep -E "tests passed|FAIL"
"$BIN/test-backend-ops" test -b MTL0 -o FLASH_ATTN_EXT -p 'hsk=(128|384|640).*kv=113.*type_K=turbo' 2>&1 | grep -E "tests passed|FAIL"
"$BIN/test-backend-ops" test -b MTL0 -o SET_ROWS -p 'ne=\[(128|256|384|512|640),7,1,2\]' 2>&1 | grep -E "tests passed|FAIL"

echo "\n=== TurboQuant model smoke"
for kv in turbo4 turbo3; do
    "$BIN/llama-bench" -m "$MODEL" -mmp 0 -ngl 99 -fa on -ctk "$kv" -ctv "$kv" -p 64 -n 32 -r 1 2>&1 |
        grep -E "probed SIMD-group width|model|pp|tg|error|failed"
done

echo "\n=== speed, Flash Attention on vs off (higher is better)"
for d in 0 8192; do
    a="$("$BIN/llama-bench" -m "$MODEL" -mmp 0 -fa auto -p 0 -n 64 -d $d -r 3 2>&1 | grep -oE "[0-9]+\.[0-9]+ ± [0-9.]+" | tail -1)"
    b="$("$BIN/llama-bench" -m "$MODEL" -mmp 0 -fa 0    -p 0 -n 64 -d $d -r 3 2>&1 | grep -oE "[0-9]+\.[0-9]+ ± [0-9.]+" | tail -1)"
    echo "  depth $d  tg64  FA: $a   no-FA: $b"
done

echo "\n=== perplexity, must match between the two (a real difference means a broken kernel)"
CORPUS="$TMP/corpus.txt"
cat README.md CHANGELOG.md > "$CORPUS"
for fa in auto 0; do
    p="$("$BIN/llama-perplexity" -m "$MODEL" --load-mode none -fa $fa -c 4096 -f "$CORPUS" --chunks 6 2>&1 | grep -oE "PPL = [0-9.]+" | tail -1)"
    echo "  -fa $fa: $p"
done

echo "\nDone. Send this whole output back."
