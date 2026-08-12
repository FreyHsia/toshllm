# Patches

Patches against the pinned upstream commits, grouped by engine and by area:

```
llama/metal/    Metal backend: ToshGEMM, FA-AMD, tiles, wave64, multi-GPU, turbo KV
llama/core/     ggml core: scheduler expert prefetch, mmap residency and loader
llama/model/    graph and model code: DFlash planner, M-RoPE, MTP
llama/server/   llama-server: OpenAI-compatible endpoints
image/          stable-diffusion.cpp
```

One patch should be one thing. `0001` and `0005` are still the exception: they predate this
layout and each carries several unrelated features. `0001` has been reduced from 28 files to 15
by moving out the DFlash planner (`0002`), the scheduler expert prefetch (`0015`) and the mmap
residency work (`0016`); what is left is the Metal backend, and splitting that means splitting
`ggml-metal.metal` by kernel family, which is still to do. `0005` has not been touched yet.

`scripts/build-engines.sh` discovers them per engine and applies them in **numeric order of the
file name, across areas**, so a new patch only has to be dropped in the right folder. The number
is the apply order, not a per-folder counter: never reuse one, and keep it when moving a file.

`llama/metal/0001` is applied to **both** engines. The image engine takes only its
`ggml/src/ggml-metal/*` hunks, in two passes: `ggml-metal-impl.h` needs `-C2` because that
engine's ggml is on a different commit and its decode block does not match at `-U5`. The build
script asserts afterwards that the hunks landed where they should; do not remove that check.

## Regenerating one

The vendor tree carries every patch at once, so `git diff` there is **all of them**. To
regenerate a single patch, diff its files against a copy that has the rest applied, and write
only that hunk back. Two rules that have cost real time:

- **`-U5` or wider.** With three lines of context a hunk lands in a different kernel and the
  build still exits 0.
- **Verify the round trip**: apply the regenerated patch to the pre-change copy and confirm it
  reproduces the file byte for byte.

`scripts/build-engines.sh` resets and re-clones the vendor tree, so regenerate before building,
never after.
