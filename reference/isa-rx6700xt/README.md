# ISA del metallib, RX 6700 XT

Instruction counts of the compiled Metal kernels, taken off the packaged metallib with the
driver's own binary archive. Used to locate where a kernel spends its work before touching it.

## Última actualización

| | |
|---|---|
| Fecha | 2026-08-16 |
| Tarjeta | AMD Radeon RX 6700 XT, RDNA2, `gfx1032`, wave32 |
| Metallib | build de vendor con los patches 0033 a 0036 aplicados |
| Motor | llama.cpp `84e908c62` más la serie de `patches/llama/` |
| Cobertura | 1393 de 1602 kernels |

Los conteos cambian con cualquier edición de un `.metal`, así que hay que regenerarlos y difearlos
tras tocar un kernel. En una tarjeta wave64 el compilador reparte registros de otra forma y estos
números no valen; para esa hace falta su propio volcado.

Los 209 que faltan no son un fallo del barrido: 201 kernels genéricos de flash attention y 8 de
`lightning_indexer` no compilan en esta GPU, con `SC compilation failure: There is a call to an
undefined label`. Es la miscompilación del driver que motivó escribir los kernels FA-AMD, y por eso
no se ejecutan aquí.

## Ficheros

| fichero | contenido |
|---|---|
| `completo.txt` | los 1393, una línea por kernel |
| `mul.txt` | 221 de `mul_mm`, `mul_mv`, `_id` y `_ext`, con sus constantes de función |
| `fa.txt` | 503 de flash attention AMD, todas las combinaciones de K y V |
| `all.txt` | barrido sin constantes: 1036 volcados, 566 fallos |
| `otros.txt` | lo que no es matmul ni atención |
| `missing.txt` | segunda pasada, con constantes por familia |

Columnas: `kernel  instr  estrechas  ds  vgpr  th_max`. `estrechas` cuenta cargas de byte y de
short, `ds` suma lecturas y escrituras de memoria compartida.

## Regenerar

El nombre del pipeline lleva codificadas sus constantes de función, así que se cosechan del runtime
en vez de adivinarlas:

```bash
cd vendor/llama.cpp
for OP in MUL_MAT MUL_MAT_ID; do
  ./build-static/bin/test-backend-ops test -o $OP 2>&1 >/dev/null |
    grep -oE "name = '[^']+'" | sed "s/name = '//; s/'//"
done | sort -u > /tmp/pipelines.txt

cd ../../reference/fa-ablation
python3 isa-sweep.py <metallib> /tmp/pipelines.txt ../isa-rx6700xt/mul-nuevo.txt
diff ../isa-rx6700xt/mul.txt ../isa-rx6700xt/mul-nuevo.txt
```

El tile ancho (`_w`) y algunos `_id` no los dispara la suite de tests y hay que sembrarlos con el
nombre completo. Un tipo con `-` es que falta sembrarlo, no que no exista.

## Cuatro formas de obtener números falsos

1. `extract-isa.py` devuelve la primera sección `__compute` del archive. Si un `dump-isa` falla,
   parsea el archive anterior sin avisar. El síntoma es q4_K y q5_K con el mismo conteo, que es
   imposible. Usar `extract-isa2.py` y borrar el archive antes de cada volcado.
2. `objdump` intercala marcadores `\t\t...` por todo el listado, no solo al final. Contar `^\t` mete
   32 líneas de más y cortar en el primer `...` se come unas 50 reales. Contar `^\t[a-z]`.
3. Los `_id` omiten `ne12`, `r2` y `r3` del nombre, pero el host los fija a 1. Sin pasarlos fallan.
4. zsh no divide `$VAR` en palabras. Pasar las constantes en una variable las manda como un solo
   argumento, `dump-isa` las descarta y compila sin especializar: salen `th_max` de 1024 en todo y
   parece que no hay anomalía. Pasarlas literales o en un array.

## Qué no dicen estas columnas

`th_max` no es ocupación. Cada familia despacha un número fijo de hilos muy por debajo de él: el
`mul_mm` ancho lanza 128, el flash attention de prefill 256 y el de decode 512. Solo sirve como
suelo, porque el kernel falla si baja de lo que se despacha.

`vgpr` es el índice más alto referenciado, no lo que asigna el compilador.

`estrechas` no predice tiempo. Seis cambios guiados por esa columna dieron cero, e `iq2_s` subió de
instrucciones al mejorar un 65 por ciento. Sirve para localizar un kernel, no para ordenarlos. Lo
que sí ha funcionado es comparar tiempos entre dos kernels equivalentes en la misma forma.
