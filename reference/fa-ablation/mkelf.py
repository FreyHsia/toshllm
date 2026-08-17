import struct, sys
code=open(sys.argv[1],"rb").read()[int(sys.argv[3]) if len(sys.argv)>3 else 0:]
shstr=b"\0.text\0.shstrtab\0"
ehsize=64; code_off=ehsize
shstr_off=code_off+len(code); sh_off=(shstr_off+len(shstr)+15)//16*16
eh=b"\x7fELF"+bytes([2,1,1,64,0])+b"\0"*7+struct.pack("<HHI",1,224,1)
mach=int(sys.argv[4],0) if len(sys.argv)>4 else 0x03a
eh+=struct.pack("<QQQ",0,0,sh_off)+struct.pack("<IHHHHHH",mach,ehsize,0,0,64,3,2)
def sh(n,t,f,a,o,s,al=1): return struct.pack("<IIQQQQIIQQ",n,t,f,a,o,s,0,0,al,0)
out=bytearray(eh)+code+shstr
out+=b"\0"*(sh_off-len(out))
out+=sh(0,0,0,0,0,0)+sh(1,1,0x6,0,code_off,len(code),4)+sh(7,3,0,0,shstr_off,len(shstr),1)
open(sys.argv[2],"wb").write(out)
