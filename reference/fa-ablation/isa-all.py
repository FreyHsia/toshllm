import os, re, subprocess, sys

# Dumps every kernel that carries no function constants in its name: attention and the rest of
# the catalogue. Writes each row as it lands so an interrupted run leaves a usable file, and
# skips names already present so it can be resumed.
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
LIB, NAMES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
OBJ = "/usr/local/opt/llvm/bin/llvm-objdump"
MCPU = os.environ.get("ISA_MCPU", "gfx1032")
MACH = os.environ.get("ISA_ELF_MACH", "0x03a")

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        done.add(line.split()[0])

f = open(OUT, "a", buffering=1)
if not done:
    f.write("%-60s %7s %7s %7s %6s %7s %6s %7s %8s\n" %
            ("kernel", "instr", "narrow", "loads", "ds", "mac", "vgpr", "th_max", "smem"))

ok = fail = 0
for name in sorted(set(l.strip() for l in open(NAMES) if l.strip())):
    if name in done or name[:60] in done:
        continue
    arch = "/tmp/isa_all.archive"
    for p in (arch, "/tmp/isa_all.compute", "/tmp/isa_all.elf"):
        if os.path.exists(p):
            os.remove(p)
    r = subprocess.run([os.path.join(WORK, "dump-isa"), LIB, name, arch],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(arch):
        f.write("%-60s %7s\n" % (name[:60], "FALLA")); fail += 1; continue
    thmax = re.search(r"maxTotalThreadsPerThreadgroup: (\d+)", r.stdout)
    smem = re.search(r"staticThreadgroupMemoryLength: (\d+)", r.stdout)
    e = subprocess.run(["python3", os.path.join(HERE, "extract-isa2.py"), arch, "/tmp/isa_all.compute"],
                       capture_output=True, text=True)
    if e.returncode != 0:
        f.write("%-60s %7s\n" % (name[:60], "FALLA")); fail += 1; continue
    subprocess.run(["python3", os.path.join(HERE, "mkelf.py"), "/tmp/isa_all.compute",
                    "/tmp/isa_all.elf", "468", MACH], check=True)
    d = subprocess.run([OBJ, "-d", "--mcpu=" + MCPU, "/tmp/isa_all.elf"],
                       capture_output=True, text=True)
    body = [l for l in d.stdout.splitlines()[6:] if re.match(r"^\t[a-z]", l)]
    def count(pat):
        r = re.compile(r"^\t" + pat)
        return sum(1 for l in body if r.match(l))
    vg = [int(m) for l in body for m in re.findall(r"\bv(\d+)\b", l)]
    f.write("%-60s %7d %7d %7d %6d %7d %6d %7d %8d\n" % (
        name[:60], len(body),
        count(r"(global|buffer|flat)_load_(u?byte|u?short|sbyte|sshort)"),
        count(r"(global|buffer|flat)_load"), count(r"ds_(read|write)"),
        count(r"v_(pk_)?(fmac|fma|dot|mad)"),
        max(vg) if vg else 0,
        int(thmax.group(1)) if thmax else 0,
        int(smem.group(1)) if smem else 0))
    ok += 1
print("ok: %d, fallos: %d" % (ok, fail))
