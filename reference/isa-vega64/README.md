# ISA del metallib, RX Vega 64

Instruction counts of the compiled Metal kernels on the wave64 card, taken off the metallib with
the driver's own binary archive. Same purpose as `../isa-rx6700xt/`, different architecture: the
numbers there do **not** apply here.

## Última actualización

| | |
|---|---|
| Fecha | 2026-08-16 |
| Tarjeta | AMD Radeon RX Vega 64, GCN5/Vega10, `gfx900`, wave64 |
| Metallib | build de vendor con la serie completa de `patches/llama/`, incluidos 0035 arreglado y 0037 |
| Motor | llama.cpp `84e908c62` más la serie |

## Regenerar

Las tres variables son obligatorias aquí; sin ellas el barrido usa los valores de RDNA2 y
**escribe una tabla falsa sin fallar**:

```bash
cd vendor/llama.cpp
for OP in MUL_MAT MUL_MAT_ID FLASH_ATTN_EXT; do
  ./build-static/bin/test-backend-ops test -o $OP 2>&1 >/dev/null |
    grep -oE "name = '[^']+'" | sed "s/name = '//; s/'//"
done | sort -u > /tmp/pipelines.txt

cd ../../reference/fa-ablation
ISA_MCPU=gfx900 ISA_ELF_MACH=0x02c ISA_NW=64 \
  python3 isa-sweep.py <metallib> /tmp/pipelines.txt ../isa-vega64/mul.txt
```

`isa-all.py` hace el resto del catálogo (atención y todo lo demás), que no lleva constantes de
función en el nombre:

```bash
ISA_MCPU=gfx900 ISA_ELF_MACH=0x02c \
  python3 isa-all.py <metallib> ../isa-vega64/otros.txt
```

## La comprobación que demuestra que un volcado no vale para la otra tarjeta

Desensamblar el binario de Vega con el `mcpu` de RDNA2 no da error: da **basura silenciosa**.

```
--mcpu=gfx900    s_load_dwordx2 s[6:7], s[0:1], 0x10
--mcpu=gfx1032   .long 0xc0060180 ... v_illegal
```

Si una tabla sale con conteos absurdamente bajos o con `vgpr` ridículos, es esto.

## Ficheros

| fichero | contenido |
|---|---|
| `mul.txt` | 177 kernels de `mul_mm`, `mul_mv` y sus `_id`, con las constantes que el runtime compiló |
| `fa.txt` | flash attention |
| `otros.txt` | el resto del catálogo |

Columnas: `kernel instr narrow loads ds mac vgpr th_max smem`.

## Qué no dicen estas columnas

Lo mismo que en la otra tarjeta, y por los mismos motivos: `th_max` no es ocupación (cada familia
despacha un número fijo de hilos muy por debajo), `vgpr` es el índice más alto referenciado y no lo
que asigna el compilador, y `narrow` **no predice tiempo** — seis cambios guiados por esa columna
dieron cero. Sirve para localizar un kernel, no para ordenarlos.
