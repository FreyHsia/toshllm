Frozen copies of the shared Metal hunks for the image engine.

`build-engines.sh` feeds `patches/llama/metal/0001-*` to stable-diffusion.cpp too, but its
vendored ggml is pinned older than llama.cpp's. When a llama bump moves those hunks past what
the older tree still matches, drop the pre-bump version of that patch here and the image build
picks it up instead.

Each copy carries a `.sha256` of the Metal changes its llama counterpart had at freeze time
(`scripts/shared-hunks-sha.py`, context ignored so a rebase is not a change). If that patch
later gains anything, the image build stops and asks for the change to be ported here, so the
image engine cannot fall behind on Metal work without someone noticing.

Delete a file and its `.sha256` once this engine's ggml catches up.
