import re, sys, collections

# Turns test-backend-ops perf output into bandwidth per type for the memory-bound shapes
# (n=1 mat-vec): the weight is read once per run, so bytes/time is what the kernel achieves.
BPW = {"f32":32,"f16":16,"bf16":16,"q4_0":4.5,"q4_1":5,"q5_0":5.5,"q5_1":6,"q8_0":8.5,
       "q2_K":2.5625,"q3_K":3.4375,"q4_K":4.5,"q5_K":5.5,"q6_K":6.5625,"q8_K":8.0,
       "iq1_s":1.5625,"iq1_m":1.75,"iq2_xxs":2.0625,"iq2_xs":2.3125,"iq2_s":2.5,
       "iq3_xxs":3.0625,"iq3_s":3.4375,"iq4_nl":4.5,"iq4_xs":4.25,"mxfp4":4.25,
       "q1_0":1.5,"q2_0":2.5}

rows = collections.defaultdict(list)
pend = None
for line in open(sys.argv[1]):
    m = re.search(r"MUL_MAT(?:_ID)?\(type_a=(\w+),type_b=\w+,m=(\d+),n=(\d+),k=(\d+)", line)
    if m:
        pend = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)))
    t = re.search(r"([\d.]+) us/run", line)
    if t and pend:
        ta, mm, n, k = pend
        if n == 1 and ta in BPW and mm*k >= 4_000_000:
            gb = mm*k*BPW[ta]/8 / 1e9
            rows[ta].append(gb / (float(t.group(1))/1e6))
        pend = None

print("%-9s %6s %9s %9s" % ("tipo", "casos", "GB/s med", "GB/s max"))
for ta in sorted(rows, key=lambda x: -max(rows[x])):
    v = rows[ta]
    print("%-9s %6d %9.1f %9.1f" % (ta, len(v), sum(v)/len(v), max(v)))
