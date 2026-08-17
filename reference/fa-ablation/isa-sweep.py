import os, re, subprocess, sys, collections

# Drives dump-isa over every pipeline the runtime actually compiled. The pipeline name carries
# its function constants, so the dumped kernel is the one that really ran, not a guess.
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
LIB = sys.argv[1]
NAMES = sys.argv[2]
OUT = sys.argv[3]
OBJ = "/usr/local/opt/llvm/bin/llvm-objdump"
# the card decides how the compiler allocates, so a dump is only valid for the arch it names
MCPU = os.environ.get("ISA_MCPU", "gfx1032")
MACH = os.environ.get("ISA_ELF_MACH", "0x03a")
NW = int(os.environ.get("ISA_NW", "32"))

MV, MM = 600, 700
MV_MAP = {"nsg": MV+0, "nxpsg": MV+1, "ne12": MV+2, "r2": MV+3, "r3": MV+4, "nw": MV+5}
MV_BOOL = {"a4": MV+6}
MM_MAP = {"ne12": MM+2, "ne13": MM+3, "r2": MM+4, "r3": MM+5}
MM_BOOL = {"bci": MM+0, "bco": MM+1, "db": MM+8, "af": MM+9}


def parse(name):
    fields = dict(re.findall(r"([a-z0-9]+)=(-?\d+)", name))
    base = re.sub(r"_[a-z0-9]+=-?\d+.*$", "", name)
    cv = []
    if base.startswith("kernel_mul_mm"):
        for k, idx in MM_MAP.items():
            if k in fields:
                cv.append("%d:s:%s" % (idx, fields[k]))
        for k, idx in MM_BOOL.items():
            if k in fields:
                cv.append("%d:b:%d" % (idx, int(fields[k]) != 0))
        mm = int(fields.get("mm", 1))
        cv.append("%d:b:%d" % (MM+6, mm & 1))
        cv.append("%d:b:%d" % (MM+7, (mm >> 1) & 1))
    elif base.startswith("kernel_mul_mv"):
        for k, idx in MV_MAP.items():
            if k in fields:
                cv.append("%d:s:%s" % (idx, fields[k]))
        for k, idx in MV_BOOL.items():
            if k in fields:
                cv.append("%d:b:%d" % (idx, int(fields[k]) != 0))
        # the _id name omits them, but the host pins all three to 1
        if base.startswith("kernel_mul_mv_id_"):
            for idx in (MV+2, MV+3, MV+4):
                cv.append("%d:s:1" % idx)
    else:
        return None, None
    return base, cv


def rank(name):
    f = dict(re.findall(r"([a-z0-9]+)=(-?\d+)", name))
    # prefer the plain case: no broadcast, single batch
    return (int(f.get("a4", 1)) != 1, int(f.get("nw", NW)) != NW,
            int(f.get("ne12", 1)) != 1, int(f.get("ne13", 1)) != 1,
            int(f.get("r2", 1)) != 1, int(f.get("r3", 1)) != 1, name)


best = {}
for line in open(NAMES):
    line = line.strip()
    base, cv = parse(line)
    if not base:
        continue
    if base not in best or rank(line) < rank(best[base][0]):
        best[base] = (line, cv)

rows = []
for base, (name, cv) in sorted(best.items()):
    arch = "/tmp/isa_sweep.archive"
    for f in (arch, "/tmp/isa_sweep.compute", "/tmp/isa_sweep.elf"):
        if os.path.exists(f):
            os.remove(f)
    p = subprocess.run([os.path.join(WORK, "dump-isa"), LIB, base, arch] + cv,
                       capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(arch):
        rows.append((base, None))
        continue
    thmax = re.search(r"maxTotalThreadsPerThreadgroup: (\d+)", p.stdout)
    smem = re.search(r"staticThreadgroupMemoryLength: (\d+)", p.stdout)
    e = subprocess.run(["python3", os.path.join(HERE, "extract-isa2.py"), arch, "/tmp/isa_sweep.compute"],
                       capture_output=True, text=True)
    if e.returncode != 0:
        rows.append((base, None))
        continue
    subprocess.run(["python3", os.path.join(HERE, "mkelf.py"), "/tmp/isa_sweep.compute",
                    "/tmp/isa_sweep.elf", "468", MACH], check=True)
    d = subprocess.run([OBJ, "-d", "--mcpu=" + MCPU, "/tmp/isa_sweep.elf"],
                       capture_output=True, text=True)
    lines = d.stdout.splitlines()[6:]
    # objdump sprinkles "\t\t..." elision markers through the listing; a real instruction
    # is a tab followed by a mnemonic, and counting anything looser inflates every row
    body = [l for l in lines if re.match(r"^\t[a-z]", l)]
    def count(pat):
        r = re.compile(r"^\t" + pat)
        return sum(1 for l in body if r.match(l))
    vg = [int(m) for l in body for m in re.findall(r"\bv(\d+)\b", l)]
    rows.append((base, {
        "instr": len(body),
        "narrow": count(r"(global|buffer|flat)_load_(u?byte|u?short|sbyte|sshort)"),
        "loads": count(r"(global|buffer|flat)_load"),
        "ds": count(r"ds_(read|write)"),
        "mac": count(r"v_(pk_)?(fmac|fma|dot|mad)"),
        "vgpr": max(vg) if vg else 0,
        "thmax": int(thmax.group(1)) if thmax else 0,
        "smem": int(smem.group(1)) if smem else 0,
        "name": name,
    }))

with open(OUT, "w") as f:
    f.write("%-52s %7s %7s %7s %6s %7s %6s %7s %8s\n" %
            ("kernel", "instr", "narrow", "loads", "ds", "mac", "vgpr", "th_max", "smem"))
    for base, m in rows:
        if m is None:
            f.write("%-52s %7s\n" % (base[:52], "FALLA"))
            continue
        f.write("%-52s %7d %7d %7d %6d %7d %6d %7d %8d\n" %
                (base[:52], m["instr"], m["narrow"], m["loads"], m["ds"], m["mac"],
                 m["vgpr"], m["thmax"], m["smem"]))
print("kernels: %d, fallos: %d" % (len(rows), sum(1 for _, m in rows if m is None)))
