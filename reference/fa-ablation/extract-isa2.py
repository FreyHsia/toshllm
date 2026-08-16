import struct, sys

# the archive can hold more than one __compute section; the original script took the first,
# which is why repeated dumps of the same kernel disagreed. collect them all, keep the largest
d = open(sys.argv[1], "rb").read()
n = struct.unpack(">I", d[4:8])[0]
off = 8
found = []
for _ in range(n):
    ct, cs, o, s, a = struct.unpack(">5I", d[off:off+20]); off += 20
    if ct != 0x1000014:
        continue
    sl = d[o:o+s]
    ncmds = struct.unpack("<I", sl[16:20])[0]; co = 32
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack("<2I", sl[co:co+8])
        if cmd == 0x19:
            nsects = struct.unpack("<I", sl[co+64:co+68])[0]; so = co+72
            for _ in range(nsects):
                sect = sl[so:so+16].rstrip(b"\0").decode()
                size = struct.unpack("<Q", sl[so+40:so+48])[0]
                foff = struct.unpack("<I", sl[so+48:so+52])[0]
                if sect == "__compute":
                    found.append(sl[foff:foff+size])
                so += 80
        co += cmdsize

if not found:
    sys.exit("no __compute")
found.sort(key=len)
open(sys.argv[2], "wb").write(found[-1])
sys.stderr.write("secciones=%d tamanos=%s\n" % (len(found), [len(f) for f in found]))
