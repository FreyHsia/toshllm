#!/usr/bin/env python3
"""Hash of a patch's Metal-backend changes, ignoring context so a rebase is not a change."""
import hashlib, re, sys

lines, keep, current = open(sys.argv[1], encoding="utf-8", errors="replace").read().split("\n"), [], False
for line in lines:
    m = re.match(r"^(?:\+\+\+ b/|--- a/)(.*)$", line)
    if m:
        current = m.group(1).startswith("ggml/src/ggml-metal/")
        continue
    if current and line[:1] in "+-" and not line.startswith(("+++", "---")):
        keep.append(line.rstrip())
print(hashlib.sha256("\n".join(keep).encode()).hexdigest())
