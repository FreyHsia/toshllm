# Dynamic MoE: medidas de viabilidad

Respuesta medida a las preguntas del milestone 1 de `experiments/dinamic-moe.md`.
Todo en la caja de desarrollo: RX 6700 XT 12 GB (NootRX), i5-10400 6c/12t,
32 GB DDR4, macOS 26.5.2, motor en el commit `3dc7285b4`.

Herramientas en este directorio (`bw.swift`, `trace.cpp`, `analyze.py`,
`gguf_info.py`) y las trazas crudas en `*.txt.gz`.

## 1. Anchos de banda

`bw.swift`. Mediana de 11 corridas por punto, tiempos de GPU
(`gpuEndTime - gpuStartTime`), 2 de calentamiento.

| medida | GB/s |
|---|---:|
| PCIe H2D lineal, blit Shared a Private | **13.09** |
| PCIe D2H lineal | 14.15 |
| **kernel Metal leyendo Shared (host) con gather disperso, trozos >= 256 KiB** | **13.25** |
| kernel Metal leyendo Shared lineal | 13.27 |
| gather por blit, trozos de 4303 KiB | 12.99 |
| gather por blit, trozos de 864 KiB | 12.61 |
| gather por blit, trozos de 256 KiB | 12.24 |
| gather por blit, trozos de 64 KiB | 8.71 |
| VRAM a VRAM, blit (control) | 157.1 |
| kernel leyendo VRAM lineal (control) | 375.9 |
| CPU, lectura de RAM tipo STREAM, 12 hilos | 34.04 |
| CPU, lectura de RAM, 6 hilos | 33.10 |

Dos resultados que cambian el plan.

**Un kernel de Metal SI puede leer memoria del host en esta GPU, y a velocidad
de enlace completa.** Un `MTLBuffer` con `StorageModeShared` leído desde un
kernel de cómputo da 13.25 GB/s haciendo gather disperso, por encima del blit
lineal (13.09) y del gather por blit (12.6). Es la primitiva que hace
viable toda la idea, y existe aquí.

**La granularidad deja de importar a partir de 256 KiB**, y solo para el blit.
El kernel aguanta 12.94 GB/s incluso con trozos de 64 KiB, donde el blit se cae
a 8.71. Los bancos reales del zoo van de 296 a 4303 KiB, o sea que ninguno de
los dos caminos está en la zona mala, pero el kernel es el robusto.

## 2. El GEMV de expertos en CPU, medido de verdad

El STREAM sintético (34 GB/s) sobreestima. El número que importa es el que
consigue el kernel MoE de llama.cpp leyendo los expertos de RAM.

Barrido de `ncmoe` con `llama-bench` sobre OLMoE-1B-7B Q5_K_M, `-ngl 99 -n 128
-r 3 -lm mlock`, con `GGML_SCHED_PREFETCH_EXPERTS=1` y `GGML_CPU_NO_REPACK=1`
como los pone la app, 45 s de pausa entre corridas:

| ncmoe | tg (t/s) | ms/token | MiB de expertos en CPU |
|---:|---:|---:|---:|
| 0 | 209.06 ± 0.56 | 4.783 | 0 |
| 4 | 96.15 ± 0.60 | 10.401 | 140.5 |
| 8 | 62.36 ± 0.22 | 16.035 | 281.0 |
| 12 | 46.62 ± 0.10 | 21.451 | 421.5 |
| 16 | 36.94 ± 0.01 | 27.073 | 562.0 |

Regresión de ms/token contra MiB: **r² = 0.999958**, residuo máximo 0.086 ms.

- **GEMV de expertos en CPU = 26.48 GB/s** (78% del STREAM).
- Intercepto = 4.823 ms, que es el resto del token con todo en GPU. El
  `ncmoe=0` medido da 4.783 ms, o sea que el modelo cierra solo.

**Ratio `cpu_bw / pcie_bw` = 26.48 / 13.25 = 2.00.** Es el número que decide si
conviene repartir los fallos entre CPU y bus o mandarlos todos por el bus. Esta
caja cae justo en la frontera.

## 3. Sesgo del ruteo y aciertos de la caché

`trace.cpp` engancha `cb_eval` y vuelca `ffn_moe_topk` por capa y token. **No
lleva ni un parche al motor**, solo la API pública. `analyze.py` simula un LRU
global sobre los pares `(capa, experto)`.

El estático de la tabla es lo que hace `ncmoe` hoy: con S slots caben `S/E`
capas enteras, y su acierto es `capas_residentes / capas`.

| modelo | capas | expertos | top-k | activo/capa | experto | expertos totales |
|---|---:|---:|---:|---:|---:|---:|
| OLMoE-1B-7B Q5_K_M | 16 | 64 | 8 | 12.5% | 4.39 MiB | 4.4 GiB |
| Qwen3.6-35B-A3B Q2_K_XL | 40 | 256 | 8 | 3.1% | 0.96 MiB | 9.6 GiB |
| gpt-oss-20B MXFP4 | 24 | 32 | 4 | 12.5% | 12.64 MiB | 9.5 GiB |

Concentración del ruteo (800-900 pasos de decode):

| modelo | working set | expertos para el 90% | entropía norm. | el token anterior predice |
|---|---:|---:|---:|---:|
| OLMoE | 62.0 / 64 | 38.1 (59%) | 0.886 | 39.0% |
| Qwen3.6-35B | 201.2 / 256 | 95.5 (37%) | 0.805 | 37.8% |
| gpt-oss-20B | 29.2 / 32 | 16.9 (53%) | 0.839 | 37.7% |

**El ruteo NO está muy sesgado.** Se usan casi todos los expertos y la entropía
es alta. El LRU no gana por sesgo, gana porque el estático desperdicia: guarda
los 64 (o 256) expertos de una capa cuando solo se usan 8 por token.

Aciertos a igual VRAM (LRU global contra el estático de `ncmoe`):

**OLMoE-1B-7B Q5_K_M** (562 MiB de expertos por token si nada reside)

| slots | VRAM | LRU | estático | usos por carga | MiB/token |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.5G | 26.5% | 12.5% | 1.36 | 413.2 |
| 256 | 1.1G | 50.2% | 25.0% | 2.01 | 279.8 |
| 384 | 1.6G | 65.6% | 37.5% | 2.91 | 193.1 |
| 512 | 2.2G | **79.0%** | 50.0% | 4.76 | 118.2 |
| 768 | 3.3G | 94.5% | 75.0% | 18.33 | 30.7 |

**Qwen3.6-35B-A3B Q2_K_XL** (308 MiB por token)

| slots | VRAM | LRU | estático | usos por carga | MiB/token |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.5G | 41.2% | 5.0% | 1.70 | 180.8 |
| 1024 | 1.0G | 60.0% | 10.0% | 2.50 | 123.1 |
| 2048 | 1.9G | **75.4%** | 20.0% | 4.06 | 75.7 |
| 4096 | 3.8G | 90.6% | 40.0% | 10.58 | 29.1 |
| 6144 | 5.8G | 95.8% | 60.0% | 23.71 | 13.0 |

**gpt-oss-20B MXFP4** (1213 MiB por token)

| slots | VRAM | LRU | estático | usos por carga | MiB/token |
|---:|---:|---:|---:|---:|---:|
| 96 | 1.2G | 28.3% | 12.5% | 1.40 | 869.8 |
| 192 | 2.4G | 51.4% | 25.0% | 2.06 | 590.2 |
| 256 | 3.2G | 66.3% | 33.3% | 2.97 | 409.0 |
| 384 | 4.7G | **92.1%** | 50.0% | 12.73 | 95.3 |

Control con un segundo prompt en OLMoE (español, tema distinto): 30.3 / 55.7 /
72.5 / 83.9% a 128 / 256 / 384 / 512 slots, contra 26.5 / 50.2 / 65.6 / 79.0%
del primero. Los aciertos no son un artefacto del prompt.

**Los usos por carga a tamaños realistas son 2 a 6, no 1.** La amortización que
hacía falta para justificar la transferencia existe.

## 4. Qué gana cada modo

Tiempo del tráfico de expertos por token. Los aciertos se sirven de VRAM a 376
GB/s y cuestan lo mismo en los dos esquemas, así que se cancelan.

- `estático`: los fallos los calcula la CPU a 26.48 GB/s. Es lo de hoy.
- `offload`: caché LRU, los fallos suben por PCIe a 13.25 GB/s y todo se calcula
  en GPU. Es el modo simple.
- `híbrido`: caché LRU, los fallos se reparten entre PCIe y CPU en paralelo,
  26.48 + 13.25 = 39.73 GB/s.

**OLMoE**

| slots | VRAM | estático | offload | híbrido | offl | hibr |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.5G | 19.0 ms | 31.9 ms | 10.7 ms | 0.60x | 1.79x |
| 256 | 1.1G | 16.3 ms | 21.6 ms | 7.2 ms | 0.75x | 2.26x |
| 384 | 1.6G | 13.6 ms | 14.9 ms | 5.0 ms | 0.91x | 2.73x |
| 512 | 2.2G | 10.9 ms | 9.1 ms | 3.0 ms | **1.19x** | **3.57x** |
| 768 | 3.3G | 5.4 ms | 2.4 ms | 0.8 ms | 2.29x | 6.88x |

**Qwen3.6-35B-A3B**

| slots | VRAM | estático | offload | híbrido | offl | hibr |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.5G | 11.3 ms | 14.0 ms | 4.7 ms | 0.81x | 2.42x |
| 1024 | 1.0G | 10.7 ms | 9.5 ms | 3.2 ms | 1.13x | 3.37x |
| 2048 | 1.9G | 9.5 ms | 5.9 ms | 2.0 ms | 1.63x | **4.88x** |
| 4096 | 3.8G | 7.1 ms | 2.2 ms | 0.7 ms | 3.18x | 9.53x |

**gpt-oss-20B**

| slots | VRAM | estático | offload | híbrido | offl | hibr |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.6G | 39.1 ms | 58.4 ms | 19.5 ms | 0.67x | 2.01x |
| 256 | 3.2G | 31.3 ms | 31.6 ms | 10.5 ms | 0.99x | 2.97x |
| 384 | 4.7G | 23.5 ms | 7.4 ms | 2.5 ms | 3.18x | 9.55x |

**El modo simple PIERDE contra lo de hoy hasta que el acierto llega al 55-65%.**
Con la caché pequeña, transferir por un tubo de 13.25 GB/s es peor que dejar que
la CPU lea de RAM a 26.48 y de paso calcule. De ahí que el goteo correcto sea de
un experto por capa y paso, lo justo para calentar la caché sin pagar el enlace.

El híbrido gana siempre, entre 1.8x y 4.9x en los tamaños de caché que caben en
12 GB junto al resto del modelo.

Traducido a token completo en OLMoE con 2.2 GB de caché: 10.9 + 4.82 = 15.7 ms
(63.7 t/s, y el `ncmoe=8` medido da 62.36) contra 3.0 + 4.82 = 7.8 ms, o sea
alrededor de 128 t/s. Es una estimación del modelo, no una medida.

## 5. El bloqueador real, y no es el que estaba en el documento

El documento pone la sincronización por capa para leer los ids del router como
Riesgo 1. Aquí el gather desde host funciona, así que ese camino está abierto.

Lo que no existe es el solape. En `ggml/src/ggml-backend.cpp:1919`,
`ggml_backend_sched_compute_splits` recorre los splits en orden y llama a
`ggml_backend_graph_compute_async` uno detrás de otro. El backend CPU calcula
en línea, o sea que **hoy la rama CPU y la rama GPU no corren a la vez**. Todo
el beneficio del modo híbrido, que es el único que gana en toda la curva,
depende de montar ese solape.

Son dos piezas separables:

1. La caché de slots por experto (gather desde host, LRU, remapeo de ids). Da el
   salto de acierto de la sección 3 y por sí sola solo gana por encima del 55-65%
   de acierto.
2. El solape CPU/GPU de los fallos dentro de una misma capa MoE. Es lo que
   convierte la caché en ganancia en todo el rango.

## 6. Lo que ya está construido

`patches/llama/core/0015-sched-prefetch-experts.patch` ya trae la segunda
instancia de backend sobre el mismo device, N slots de staging dimensionados al
mayor tensor de experto, `MTLSharedEvent` de ida y vuelta, doble buffer, y
degradación limpia si falla la reserva. Las fases 5 y 7 del documento son
extender eso, no construirlo.

El reparto de VRAM entre slots y páginas de KV con prioridad MoE es aritmética de
bytes pura, sin GPU, y cubre la fase 8 con unas 130 líneas.

## 7. Respuestas a las preguntas del milestone 1

1. **¿Hay localidad temporal suficiente?** Sí, pero no por sesgo del ruteo. Los
   usos por carga van de 2 a 6 en los tamaños de caché que caben.
2. **¿Qué porcentaje del pool hay que tener residente?** Para superar el
   estático a igual VRAM, cualquiera. Para que el modo simple gane, del 25% al
   40% del pool según el modelo.
3. **¿Cuántos bytes por token?** Sin caché: 308 MiB (Qwen3.6-35B), 562 MiB
   (OLMoE), 1213 MiB (gpt-oss). Con caché al 75-80%: 76, 118 y 409 MiB.
4. **¿Qué modelos se benefician más?** Los de experto grande y pocos expertos
   activos. gpt-oss (12.64 MiB por experto) salta a 92% de acierto con 4.7 GB.
5. **¿Qué cuantizaciones primero?** Da igual para la caché: los bancos son
   opacos, se copian por bytes. Lo que decide es el tamaño del banco, y todos
   los del zoo pasan de 256 KiB. Q4_K y MXFP4 cubren el zoo real.
6. **¿Caché recomendada?** A 12 GB de VRAM, contando el resto del modelo:
   OLMoE 512 slots (2.2G), Qwen3.6-35B 2048-4096 slots (1.9-3.8G), gpt-oss
   384 slots (4.7G).
7. **¿Justifica tocar el backend Metal?** Sí, pero solo si se hace el solape
   CPU/GPU. La caché sola no llega.

## 8. Salvedades

- El modelo de tiempo cuenta bytes. No incluye la gestión de slots, el remapeo
  de ids, ni el coste de encolar las copias. El solape se asume perfecto.
- Los aciertos salen de un LRU global ideal sobre trazas de 800-900 pasos con
  un prompt cada una (dos en OLMoE). No hay multi-turno ni contexto largo.
- Las trazas de Qwen3.6-35B y gpt-oss se tomaron con `-ngl 0`. El ruteo puede
  variar en el último bit contra la GPU, pero la estadística no.
- 13.25 GB/s es lo que da el gather en esta caja. Vale para esta GPU, este
  chipset y este enlace, y para nada más.
- El GEMV de expertos en CPU se midió con `GGML_CPU_NO_REPACK=1` puesto, como
  lo pone la app. Sin él el número puede ser otro.

---

# Segunda tanda: validación en el grafo real

Añadido `patches/llama/metal/0055-metal-host-resident-weight-buft.patch`, 233
líneas. Expone el buft `MTL0` (memoria del host envuelta en un `MTLBuffer`
Shared) como **buffer type extra** en una GPU discreta, para que el kernel de
Metal lea pesos de RAM por el bus. Apagado salvo que se ponga
`TOSH_METAL_HOST_BUFFERS=1`, nunca se elige como buft por defecto, y se
selecciona con `-ot`. También enumera los bufts extra en `-ot` (`common/arg.cpp`
y `llama-bench`), que antes solo veía el buft por defecto de cada device.

## El modelo predice el tg real dentro del 4%

OLMoE-1B-7B Q5_K_M, `-ngl 99 -n 64 -r 3 -lm mlock`, 45 s entre corridas.
Predicción = 4.823 ms de base + bytes de expertos / ancho de banda del camino.

| config | expertos | medido | predicho | error |
|---|---|---:|---:|---:|
| A | 16 capas en host (`-ot exps=MTL0`) | 21.04 t/s | 20.3 | +3.6% |
| B | 16 capas en CPU (`-ncmoe 16`) | 36.94 t/s | 36.9 | ajuste |
| C | 8 en host + 8 en CPU | **25.96 t/s** | 26.73 serie | -2.9% |
| D | 8 en host + 8 en VRAM | 38.10 t/s | 36.62 | +4.0% |
| E | 8 en CPU + 8 en VRAM | 61.98 t/s | 63.74 | -2.8% |

Cinco configuraciones de 21 a 62 t/s, todas dentro del 4%. Los 13.25 GB/s del
gather sintético describen el grafo real.

## MEDIDO: la CPU y la GPU no solapan

La configuración C es la que discrimina. Mitad de las capas leen los expertos
por el bus (13.25 GB/s) y la otra mitad los calcula la CPU (26.48 GB/s). Los dos
caminos usan recursos distintos:

- si corrieran en paralelo: 4.82 + max(22.24, 11.13) = 26.5 ms → **37.68 t/s**
- si corren en serie: 4.82 + 22.24 + 11.13 = 37.4 ms → **26.73 t/s**

**Medido: 25.96 t/s.** Es el caso serie, incluso un pelo por debajo. Confirma
por medida lo que se leía en `ggml/src/ggml-backend.cpp:1919`: los splits se
ejecutan uno detrás de otro.

## Qué cierra esto

1. El modo simple está confirmado muerto: 21.04 t/s contra los 36.94 de hoy con
   los mismos expertos fuera de VRAM. Coincide con la columna `offl` de la
   primera tanda (0.50x con la caché fría).
2. **Todo el premio del híbrido (1.8x a 4.9x) está detrás del solape**, y el
   solape no existe hoy. No es un detalle de implementación, es la obra.
3. El modelo de bytes es fiable, así que las tablas de la primera tanda se
   pueden usar para decidir sin construir la caché.

## Lo que NO se puede probar con `-ot`

En GGUF los expertos de una capa son UN tensor (`ffn_gate_exps.weight` =
`[n_embd, n_ff, n_expert]`), así que `-ot` solo reparte por capa, nunca por
experto. Y las capas son una cadena serie, o sea que repartir por capa no crea
hermanos que puedan solapar. El hermano solo aparece partiendo los expertos de
UNA capa entre dos backends, y eso pide tocar el grafo del modelo.

## Siguiente paso

Partir el `mul_mat_id` de una capa en dos ramas sobre backends distintos y
sumarlas, primero con un reparto estático de expertos (sin LRU). Es lo mínimo
que crea el hermano y permite medir si el solape se materializa. Si con reparto
estático el solape aparece, la caché LRU encima es la que aporta los saltos de
acierto de la primera tanda. Si no aparece, no hay proyecto.

---

# Tercera tanda: el MoE grande (Qwen3.6-35B-A3B Q4_K_S, 19.45 GiB)

40 capas, 256 expertos, top-8, experto de 1.688 MiB sobre tres bancos de 576 KiB.
16.9 GiB solo de expertos, o sea que no cabe en 12 GB. Es el caso real de la app.
`-ngl 99 -n 64 -r 3 -lm mlock`, 45 s entre corridas.

| config | expertos | medido | predicho | error |
|---|---|---:|---:|---:|
| A | `-ncmoe 40` (todos en CPU) | 22.61 ± 0.24 | 22.94 | -1.4% |
| B | `-ncmoe 30` | 26.84 ± 0.82 | 27.25 | -1.5% |
| C | `-ncmoe 22` | **31.75 ± 0.17** | 32.08 | -1.0% |
| F | `-ncmoe 24` | 29.86 ± 0.70 | 30.72 | -2.8% |
| D | 22 capas en host (`MTL0`) | 25.28 ± 0.03 | 25.68 | -1.6% |
| G | 24 capas en host | 24.02 ± 0.02 | 24.38 | -1.5% |
| E | 11 en host + 11 en CPU | 27.61 ± 0.01 | 28.53 serie | -3.2% |

Ajuste sobre A/B/C: **r² = 0.99991**, intercepto **15.98 ms**.

`ncmoe 22` bate a `ncmoe 24` (31.75 contra 29.86): mientras quepa, menos es mejor.

## El GEMV de expertos en CPU depende del MODELO, no solo de la máquina

**20.02 GB/s en este modelo, contra los 26.48 de OLMoE Q5_K.** Q4_K con 256
expertos estrechos (filas de 512) rinde peor en CPU que Q5_K con 64 expertos
anchos. El bus no cambia (13.25), así que el ratio cae de 2.00 a **1.51**.

Eso cruza el umbral en la otra dirección: para este modelo el modo correcto es
`offload`, no `hybrid`. **El modo se decide por modelo, no por caja**, así que el
bench de calibración tiene que ser por formato de cuantización.

## Se confirma otra vez que no hay solape

La configuración E, ahora sobre el modelo grande: 11 capas leyendo expertos por
el bus y 11 calculándolos en CPU dan 27.61 t/s. La predicción en serie es 28.53
y la de solape 36.42. Vuelve a salir el caso serie.

## CAMBIA LA CONCLUSIÓN: aquí el modo simple SÍ gana

Con `ncmoe 22` quedan 18 capas de expertos en VRAM, o sea 7.6 GiB, que son 4608
slots de 10240. El LRU a ese tamaño da ~92% de acierto contra el 45% del
estático. Con el ancho de banda de CPU real de este modelo (20.02):

| esquema | tiempo de expertos | token completo | tg |
|---|---:|---:|---:|
| hoy, `ncmoe 22` | 15.2 ms | 31.2 ms | 31.75 (medido) |
| LRU + todo por el bus (`offload`) | 3.5 ms | 19.5 ms | **~51** |
| LRU + híbrido con solape | 1.3 ms | 17.3 ms | ~58 |
| techo (coste de expertos cero) | 0 | 15.98 ms | 62.6 |

**La caché LRU sola, sin nada de solape, vale 1.6x en este modelo.** El solape
añade otro 12%. Es lo contrario de lo que salía en OLMoE, y la razón es doble:
el GEMV de CPU es peor aquí (20.02 contra 26.48) y el acierto es mucho más alto
al mismo porcentaje de residencia (top-8 de 256 = 3.1% activo por capa).

El cruce en este modelo está en el 64% de acierto, y a 4608 slots estamos en 92%.
En la tabla proyectada el `offload` ya gana desde 512 slots (0.8 GiB).

**El techo end-to-end es 62.6 t/s** por el intercepto de 15.98 ms, así que el
proyecto entero no puede dar más de 1.97x de tg en este modelo por bien que salga.

## Proyección para este modelo, con 20.02 GB/s medidos

| slots | VRAM | LRU | estático | t_estático | t_offload | t_híbrido | offl | hibr |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.8G | 41.2% | 5.0% | 26.2 ms | 24.5 ms | 9.8 ms | 1.07x | 2.68x |
| 1024 | 1.7G | 60.0% | 10.0% | 24.9 ms | 16.7 ms | 6.7 ms | 1.49x | 3.74x |
| 2048 | 3.4G | 75.4% | 20.0% | 22.1 ms | 10.3 ms | 4.1 ms | 2.15x | 5.40x |
| 3072 | 5.1G | 84.4% | 30.0% | 19.3 ms | 6.5 ms | 2.6 ms | 2.97x | 7.47x |
| 4096 | 6.8G | 90.6% | 40.0% | 16.6 ms | 3.9 ms | 1.6 ms | 4.20x | 10.55x |
| 6144 | 10.1G | 95.8% | 60.0% | 11.0 ms | 1.8 ms | 0.7 ms | 6.28x | 15.76x |

Salvedad: los aciertos vienen de la traza del Q2_K_XL del mismo modelo. La
arquitectura y el número de expertos son idénticos, pero el router está
cuantizado distinto, así que el ruteo exacto puede variar. Falta retrazar sobre
el Q4_K_S para cerrarlo.

## Qué cambia en el plan

El primer paso deja de ser el solape y pasa a ser **la caché de slots por
experto sola**, en modo `offload` puro: los fallos suben por el bus con el
gather ya validado, todo se calcula en GPU, y no hace falta tocar el scheduler.
Vale 1.6x en el modelo grande. El solape queda como segunda fase, para el 12%
que falta y para los modelos con GEMV de CPU eficiente (tipo OLMoE), donde sí
es la diferencia entre ganar y empatar.

---

# Cuarta tanda: la salvedad cerrada, y un atajo descartado

## La traza del Q4_K_S confirma la proyección

Retrazado el `Qwen3.6-35B-A3B-UD-Q4_K_S` con `-ngl 0`, 800 pasos. El router está
cuantizado distinto que el Q2_K_XL, pero las estadísticas son las mismas:
working set 197.9/256 contra 201.2, entropía 0.816 contra 0.805.

| slots | LRU en Q2_K_XL | LRU en Q4_K_S |
|---:|---:|---:|
| 1024 | 60.0% | 56.7% |
| 2048 | 75.4% | 75.4% |
| 3072 | 84.4% | 85.3% |
| 4096 | 90.6% | **91.1%** |
| 6144 | 95.8% | **96.2%** |

Dentro de 3 puntos en todo el rango y algo mejor donde importa. La proyección de
la tercera tanda se mantiene.

## DESCARTADO: el hot-set estático elegido offline

Si un conjunto fijo de expertos calientes empatara con el LRU, la fase 1 no
necesitaría ni política ni evicción: bastaría con una tabla de residencia fija
elegida en un perfilado. Empata, pero solo si se evalúa en el mismo prompt.

OLMoE, hot-set elegido en la primera mitad del prompt 1:

| slots | LRU | hot-set, 2ª mitad del MISMO prompt | hot-set en OTRO prompt | `ncmoe` |
|---:|---:|---:|---:|---:|
| 128 | 25.8% | 40.3% | 15.2% | 12.5% |
| 256 | 49.0% | 58.6% | 24.4% | 25.0% |
| 384 | 64.5% | 72.5% | 34.3% | 37.5% |
| 512 | 78.4% | 83.5% | **45.7%** | 50.0% |
| 768 | 93.8% | 95.2% | 72.9% | 75.0% |

El prompt 2 es de otro idioma y otro tema. **Un perfil offline cae por debajo del
`ncmoe` de hoy en cuanto cambia la tarea.** La adaptatividad no es un refinamiento,
es el proyecto.

Detalle secundario: a 256 slots el LRU da 0.0% porque el working set de un solo
token (40 capas x 8 = 320 expertos) no cabe y la caché se destroza sola. El
hot-set no se destroza. Cualquier política tiene que respetar ese suelo.

## El diseño que sale de aquí: sin parada por fallo

El planteamiento habitual copia el experto a VRAM antes de usarlo, así que un fallo
es una parada y hace falta esconderla. Aquí no: con el buft `MTL0` el kernel de
Metal lee el experto directamente de RAM. **Un fallo no es una parada, es una
lectura lenta.** Es lo que permite sacar la política del camino crítico.

    tabla de residencia en VRAM: (capa, experto) -> slot | -1
    mul_mat_id elige el puntero base por experto:
        slot en VRAM   -> 376 GB/s
        banco en host  -> 13.25 GB/s
    promotor asincrono que rellena slots segun el uso reciente

**La política de caché queda fuera del camino crítico del token.** El kernel
siempre tiene un puntero válido; el LRU solo decide qué mejora, y puede ir tarde,
ser aproximado o fallar sin romper nada. Eso quita el riesgo de no tener un
equivalente a los CUDA graphs para esconder el ida y vuelta por capa.

## Lo siguiente que hay que medir, antes de construir la caché

El coste de la indirección en sí. Si `mul_mat_id` pasa de calcular
`src0 + experto * stride` a leer un puntero de una tabla, esa suma se paga en
TODOS los tokens, también en los aciertos. Hay que medir el A/B con todo residente
antes de escribir ninguna política: si la indirección cuesta un 5% de decode, se
come un tercio de la ganancia.

---

# Quinta tanda: la tabla de residencia, medida

`patches/llama/metal/0056-metal-moe-expert-residency-table.patch`, 148 líneas.
Añade a `kernel_mul_mv_id` una tabla que remapea el experto antes de calcular el
puntero base:

    int32_t i02s = i02;
    if (args.use_slots) {
        const int32_t s = slots[i02];
        if (s >= 0) i02s = s;
    }
    device const char * src0_cur = src0s + i02s*args.nb02;

Una lectura y una rama **por threadgroup**, no por elemento; el bucle interno no
cambia. Inerte salvo `TOSH_MOE_SLOTS`: 1 llena la tabla de -1 (nada residente),
2 la llena con la identidad (toda búsqueda tomada).

## Coste en la ruta de acierto: por debajo del ruido

OLMoE `-ncmoe 0` (todos los expertos en VRAM, 209 t/s, el caso más sensible),
`-n 128 -r 5`, 45 s entre corridas:

| modo | tg128 |
|---|---:|
| 0 sin tabla | 209.12 ± 1.84 |
| 1 tabla de -1 | 209.64 ± 0.78 |
| 2 tabla identidad | 210.21 ± 0.33 |
| 0 sin tabla (repetición) | 210.60 ± 0.51 |

**Las dos corridas de la base se separan más entre sí (1.48) que cualquier modo
de la base.** El coste de la indirección no es medible aquí.

## Corrección

`llama-completion --temp 0 --seed 1 -n 96`, salida **idéntica byte a byte** en
los tres modos (591 bytes). La tabla identidad recorre el camino de remapeo
completo y produce el mismo texto.

## Lo que queda para la fase 1

Con esto, las tres piezas del camino de datos están medidas y funcionan:

1. Los expertos pueden vivir en RAM y el kernel los lee a 13.25 GB/s (0055).
2. El kernel puede elegir de dónde lee cada experto, gratis (0056).
3. Falta el **segundo puntero**: hoy la tabla remapea dentro del mismo tensor.
   Hace falta un buffer de slots en VRAM aparte, y que el kernel elija entre él y
   el banco en host.

Y encima, la política: un promotor asíncrono que rellene slots por uso reciente.
No va en el camino crítico, porque el kernel siempre tiene un puntero válido.

**Trampa anotada**: la tabla viaja hoy por `set_bytes`, que la deja en memoria
constante. La versión real necesita un buffer en VRAM, y hay que volver a medir
el coste: la memoria constante puede ser más rápida que un buffer de device.

---

# Sexta tanda: la caché de slots, construida y medida

`patches/llama/metal/0057-metal-moe-expert-slot-cache.patch`, 266 líneas. Cierra
la pieza 3: el kernel elige entre DOS punteros base.

    device const char * src0_base = src0s;   // banco en RAM
    int32_t i02s = i02;
    if (args.use_slots) {
        const int32_t s = slots[i02];
        if (s >= 0) { src0_base = src0c; i02s = s; }   // espejo en VRAM
    }
    device const char * src0_cur = src0_base + i02s*args.nb02;

El espejo lo mantiene el backend (`ggml_metal_device_moe_slots`), indexado por
banco. ggml no se entera: para el grafo sigue siendo un `mul_mat_id` normal.
`TOSH_MOE_SLOTS_N=K` fija cuántos expertos de cada banco viven en VRAM.

## El barrido sigue el modelo en un orden de magnitud

OLMoE con los expertos en RAM (`-ot exps=MTL0`), `-n 64 -r 3`, 45 s de pausa. El
acierto NO es una estimación: es la fracción de activaciones que la traza real
manda a expertos `< K`.

| slots | acierto | medido | predicho | error | espejo |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0% | 21.08 ± 0.01 | 21.44 | -1.7% | 0 |
| 16 | 25.3% | 27.09 ± 0.11 | 27.74 | -2.3% | 1.10 GiB |
| 32 | 51.8% | 39.19 ± 0.32 | 40.06 | -2.2% | 2.20 GiB |
| 48 | 78.5% | 66.57 ± 0.43 | 72.55 | -8.2% | 3.29 GiB |
| 64 | 100.0% | 202.21 ± 1.43 | 209.07 | -3.3% | 4.39 GiB |

De 21 a 202 t/s siguiendo la curva. Corrección: `t(h) = 4.783 + 41.86*(1-h)` ms.
La versión anterior del modelo contaba dos veces la lectura de los residentes,
porque el intercepto de 4.783 ms YA incluye leer los expertos desde VRAM.

Corrección de salida: `llama-completion --temp 0 --seed 1 -n 96` da los mismos
591 bytes con K = 0, 32 y 64.

## Dos cosas abiertas, anotadas y sin explicar

- **A K=64 faltan 3.3%.** Con todo residente el espejo debería igualar a los
  expertos nativos en VRAM (209.06 y 210.60 en dos corridas) y da 202.21 ± 1.43,
  fuera del ruido. Sospecha razonable, sin medir: presión de working set, porque
  el banco en host (4.39 GiB) y el espejo (4.39 GiB) están vivos a la vez sobre
  12.87 GiB. Es el mismo mecanismo que explicó el acantilado de `ncmoe`.
- **A K=48 el error sube a -8.2%.** El modelo es lineal en el acierto y a
  aciertos altos mandan los costes fijos. Sirve para decidir, no para prometer.

## Trampas de esta tanda

1. El relleno del espejo colgó la primera corrida 600 s: hacía `commit` +
   `waitUntilCompleted` en la MISMA cola que tenía el command buffer abierto
   encodeando. Arreglado con una cola dedicada al relleno.
2. Tras arreglarlo solo se recompiló `llama-bench`, así que el chequeo de
   corrección corrió con un `llama-completion` que seguía colgándose y salió un
   falso "DIFIERE" con un fichero de 0 bytes. Recompilar TODOS los binarios que
   entran en una medida.
3. `llama-cli` se queda esperando entrada y lo mata el timeout. Para una pasada
   única: `llama-completion -no-cnv -st < /dev/null`.

## Lo que falta para cerrar la fase 1

La política. Un promotor que decida qué entra en el espejo, alimentado por los
ids del router. El diseño no lo pone en el camino crítico: el kernel siempre
tiene un puntero válido, así que el promotor puede ir un token por detrás.
La vía barata es leer los ids de todas las capas en el punto de sincronización
que llama.cpp ya hace al final de cada token (40 capas x 32 bytes = 1.3 KiB),
actualizar el LRU en host y encolar los blits para el token siguiente.

---

# Septima tanda: el promotor. FASE 1 CERRADA

`patches/llama/metal/0058-metal-moe-expert-promoter.patch`, 542 líneas.

## El diseño cambió por una medida, no por una idea

El plan era leer los ids del router (`src[2]` del `mul_mat_id`) después de que el
grafo completase, en el `waitUntilCompleted` que ya existe. **No funciona**: la
instrumentación devolvió estos ids con 64 expertos:

    banco 0 n_ids=8 n_experts=64 ids: -1104412344 -1144510976 1058699388 ...

Son patrones de bits de float. El buffer de cómputo se reutiliza, así que cuando
el grafo termina la región de los ids ya la han pisado activaciones. Leer los ids
a posteriori no es válido, y sin la instrumentación habría salido como "el
promotor no hace nada" sin causa.

**Lo que se hace en su lugar es mejor**: el kernel estampa la recencia.

    if (args.use_slots && tiitg == 0) {
        used[i02] = args.moe_tick;
    }

Un store plano, sin atómicos, en un buffer persistente de `n_experts` int32. Todos
los que escriben ponen el MISMO valor, así que la carrera es benigna. El promotor
lee ese buffer, que sí sobrevive al grafo, y sabe qué expertos se tocaron y
cuándo. Da recencia real, no pertenencia a un token.

## El promotor

Al terminar el grafo: publica lo que aterrizó, lee la recencia por banco, refresca
el LRU, elige un fallo, escoge la víctima menos usada saltando lo tocado en este
eval, **despublica el slot**, lanza el blit asíncrono y lo republica cuando
completa. Entre despublicar y republicar ningún kernel puede alcanzar ese slot.
Un banco cuyo barrido toca más de la mitad de sus expertos es prefill y se salta.

## Medido sobre generación de TEXTO REAL

OLMoE, expertos en RAM, 32 de 64 espejados (2.2 GiB), `--temp 0.8 --seed 1`, 792
tokens generados:

| presupuesto | acierto | eval |
|---|---:|---:|
| sin promoción (estático) | 51.3% | 33.74 t/s |
| 16 MiB/token | 62.1% | 39.49 t/s |
| **64 MiB/token** | **73.4%** | **48.57 t/s** |
| 256 MiB/token | 73.4% | 48.30 t/s |

**+44% con la misma VRAM**, solo por hacer la caché adaptativa. Satura a 64 MiB:
por encima manda el límite de una promoción en vuelo por banco, no el presupuesto.

El 73.4% queda a 5 puntos del **78.6%** que el simulador offline predijo para un
LRU por capa con 32 slots. El hueco es el estrangulamiento de una en vuelo más el
periodo de calentamiento, que entra en la media.

**Cuidado con el número de `llama-bench`**: ahí el acierto converge al 92% y el
tg1024 sale 66.34. Su flujo de tokens es repetitivo y concentra el ruteo
artificialmente. El número honesto es el de texto real.

## Corrección

`--temp 0` con 400 tokens: salida **idéntica byte a byte** con el promotor apagado
y encendido. Mover expertos entre slots en caliente no cambia el resultado.

## Estado de las fases

| fase | estado |
|---|---|
| 0 aislar, 1 trazado, 2 simulador, 3 calibración | hechas |
| 4 caché lógica | superada |
| **5 caché Metal en decode** | **CERRADA** (0055-0058) |
| 6 híbrido CPU/GPU | pendiente, bloqueada por el solape del scheduler |
| 7 prefill doble buffer | ya venía shippeada en el parche 0015 |
| 8 memoria elástica | pendiente |
| 9 multi-GPU, 10 caché semántica | otro proyecto |

## Abierto

- El 2-3% del espejo a residencia total. Residency sets **refutado** por medida.
  Quedan el binding extra de un buffer distinto y el puntero base variable.
- El límite de una promoción en vuelo por banco cuesta 5 puntos de acierto.
- El presupuesto es fijo; debería bajar solo al converger, porque mientras
  promueve compite por el bus con las lecturas de fallo.
- Falta medirlo en el 35B, que es el caso que motiva todo esto.

---

# Octava tanda: el MoE grande, con el promotor

## El 35B Q4_K_S NO cabe en modo nuevo en esta caja

Intento fallido, y la causa es de diseño, no un bug. La caché **duplica**:

| | RAM del host | VRAM |
|---|---:|---:|
| `-ncmoe 24` | 24 de 40 capas = 10.1 GiB | 16 capas = 6.75 GiB |
| modo nuevo | banco COMPLETO = 16.9 GiB | espejo = 6.75 GiB |

23.65 GiB de almacenamiento de expertos contra los 16.9 repartidos del `ncmoe`.
En 32 GB con un modelo de 19.45 GiB, esos 6.8 GiB de más son la diferencia entre
caber y hacer swap: el proceso quedó en estado `U` durante la carga con 0.5 GiB
libres y hubo que matarlo. Respondió a SIGTERM, o sea que era thrashing y no el
deadlock del driver; la máquina volvió limpia.

**Regla práctica: el modo nuevo pide RAM >= todos los expertos, con holgura.**
Tiene arreglo (dejar en RAM solo lo que no cabe en VRAM y marcar el resto como
residente permanente, sin duplicar), pero rompe la invariante "la RAM es la
fuente de verdad" del plan original y no es de esta fase.

## Qwen3.6-35B-A3B Q2_K_XL: +25% a igual VRAM

Mismo modelo y arquitectura (40 capas, 256 expertos, top-8), 9.6 GiB de expertos.
`-ncmoe 24` deja 3.84 GiB de expertos en VRAM, que son K=102 de 256 por banco.
599 tokens de generación real, `--temp 0.8 --seed 1`:

| | acierto | eval |
|---|---:|---:|
| A `-ncmoe 24` (modo de hoy) | | 26.94 t/s |
| B nuevo, espejo estático K=102 | 39.7% | 26.23 t/s |
| **C nuevo, con promotor** | **90.2%** | **33.67 t/s** |

**+25.0% sobre el modo actual, a igual VRAM.**

El acierto del promotor, 90.2%, cae encima del **90.6%** que el simulador offline
predijo para 4096 slots en este modelo, horas antes de que el código existiera.

**B es lo que hace honesta la lectura**: con la caché pero sin política queda por
DEBAJO del `ncmoe` (26.23 contra 26.94), porque con 102 de 256 residentes el
acierto estático es 39.7%, casi exactamente la fracción residente, y los fallos
van por el bus a 13.25 GB/s en vez de calcularse en CPU a ~20. Toda la ganancia
viene de la adaptatividad; ninguna del mecanismo por sí solo.

Referencia del Q4 que sí se pudo medir: `-ncmoe 24` da 30.22 t/s en generación
real, consistente con los 29.86 de `llama-bench`.

---

# Novena tanda: una regresión propia, y el arreglo

## La referencia buena era la app instalada, no mi binario

Comparar contra mi propio binario con la función apagada daba una línea base
**5.1% por debajo** del build real. El control que lo destapó fue medir contra el
`llama-bench` de `/Applications/ToshLLM.app` 0.85.7.

Lotes alternados, Qwen3.6-35B Q2_K_XL `-ncmoe 24`:

| | rondas | media |
|---|---|---:|
| app 0.85.7 | 28.55 · 28.82 · 28.88 | 28.75 |
| parcheado, apagado | 26.93 · 27.06 · 27.32 | 27.10 |

## Dos hipótesis mías, las dos refutadas

1. **Los bindings extra del encoder.** A/B en el mismo binario: sin bindings
   26.27/26.96, con bindings 26.66. No es.
2. **La compilación del metallib.** La app carga un metallib precompilado con
   `metal -O3 -std=metal3.1`; mi binario lo compilaba en runtime porque yo había
   borrado `build-static/bin/default.metallib` (para que mis ediciones del .metal
   no quedaran tapadas) y CMake no lo regenera: lo hace un paso aparte de
   `build-engines.sh`, que necesita Xcode completo (`xcrun metal` no está en las
   CLT). Con el metallib precompilado: 26.90. Tampoco es.

**Trampa que queda anotada**: borrar el metallib es correcto para que las
ediciones del `.metal` surtan efecto, pero cambia el camino de compilación
respecto al build oficial. Para comparar contra la app hay que regenerarlo:

    export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
    "$(xcrun -f metal)" -O3 -mmacosx-version-min=14.0 -std=metal3.1 \
        -DGGML_METAL_EMBED_LIBRARY -DGGML_METAL_HAS_BF16 \
        -I ggml/src -I ggml/src/ggml-metal \
        -c build-static/ggml/src/ggml-metal/autogenerated/ggml-metal-embed.metal \
        -o build-static/bin/ggml-metal.air
    "$(xcrun -f metallib)" build-static/bin/ggml-metal.air -o build-static/bin/default.metallib

Con el árbol revertido a base+54, ese metallib sale **byte a byte idéntico** al
de la app (20830646), o sea que el build local reproduce el oficial.

## La bisección: es el 0057, y es codegen

Todo con metallib precompilado, `-ncmoe 24`:

| configuración | tg128 | vs limpio |
|---|---:|---:|
| limpio (= app) | 28.35 ± 0.05 | |
| +0055 buft en host | 27.93 ± 0.73 | ruido, no toca el kernel |
| +0056 tabla de residencia | 28.29 ± 0.14 | -0.2% |
| **+0057 caché de slots** | **26.91 ± 0.06** | **-5.1%** |
| +0058 promotor (apagado) | 26.90 ± 0.06 | -5.1% |

Aislado forzando la base a `src0s` y dejando TODO lo demás del 0057+0058 en su
sitio: **28.31 ± 0.13**. Se recupera el 5% entero.

**El coste es elegir entre dos punteros base dentro del kernel**, no ejecutar la
rama. El compilador pierde la procedencia de `src0_cur`, que se recorre millones
de veces en el bucle de dequant, y el código generado empeora aunque la rama
nunca se tome. El 0056, que remapea el índice sobre UNA sola base, mide -0.2%.

## El arreglo: un nivel por despacho (0059)

Descartada la constante de función: arregla el camino apagado pero deja el
encendido pagando el 5%, y duplica las pipelines de `mul_mv_id`.

Lo que se hace es despachar dos veces, cada una con **una sola base**: el espejo
en una, el banco en la otra, y el kernel descarta los expertos que no son de su
nivel. `src0s` vuelve a ser un argumento plano.

| | tg128 |
|---|---:|
| camino apagado, tras el arreglo | **28.35 ± 0.04** (= limpio) |

## La ganancia real

Qwen3.6-35B Q2_K_XL, 599 tokens de generación real, misma VRAM de expertos:

| | eval |
|---|---:|
| A `-ncmoe 24` (modo de hoy) | 28.44 t/s |
| **C nuevo, K=102 + promotor** | **37.30 t/s** |

**+31.2%.** Salida idéntica byte a byte con la caché encendida y apagada.

Al arreglar, A subió 26.94 → 28.44 (+5.6%) pero C subió 33.67 → 37.30 (+10.8%):
**el camino encendido pagaba el coste dos veces.** Por eso la constante de
función habría dejado la mitad de la ganancia sin recoger. Y el segundo despacho
no se nota: si costara algo, C no habría subido más que A.

---

# Decima tanda: el readback del promotor, y el numero real

## Revisitar lo dado por cerrado destapo dos cosas

1. **El prefill del modo nuevo esta roto**: Q2_K_XL pp512 606.20 contra 267.03,
   **-56%**. La cache vive en `mul_mv_id` (decode); el prefill va por
   `mul_mm_id`, que no la tiene, y lee los expertos del host sin espejo.
   **La fase 5 NO esta cerrada.**
2. **El promotor costaba un 40%** y no se habia medido con residencia alta.

## Seis hipotesis refutadas antes de acertar

OLMoE K=64 (todo residente) daba 120.42 contra 210.65 nativo. Refutadas por
medida, en este orden: los bindings extra del encoder, la compilacion del
metallib, el doble despacho del 0059, el store de recencia del kernel, el
residency set, y "el solape es donde esta el premio".

**La causa**: el promotor leia el buffer de recencia **de cada banco y en cada
eval**, un blit y una espera por banco (48 en OLMoE, 120 en el 35B), sobre un
grafo de decode que es 83% serie. Con `TOSH_MOE_PROMOTE_MB=0` se pagaba igual,
porque el presupuesto se comprobaba despues de leer.

Lo que ordeno la busqueda fue una contradiccion: la misma configuracion media
~202 en la epoca del 0057, o sea antes del promotor.

| OLMoE K=64 | tg128 |
|---|---:|
| promotor, readback por banco | 121.36 |
| **promotor, readback agrupado (0061)** | **192.59** |
| sin promotor (0060) | 207.15 |
| nativo | 210.65 |

## El numero real del Q2_K_XL

599 tokens de generacion real, misma VRAM de expertos:

| | eval | antes de agrupar |
|---|---:|---:|
| A `-ncmoe 24` | 28.12 t/s | 28.44 |
| **C nuevo, K=102 + promotor** | **51.48 t/s** | 37.30 |

**+83.1% sobre el modo actual.** A no se mueve (no usa la cache); C sube un 38%
solo por agrupar el readback.

**Y el techo que di antes, 40.7 t/s, era FALSO**: se derivo de una medida que
llevaba dentro el impuesto del readback. Rehecho sobre C: base sin coste de
expertos 17.18 ms, techo **58.2 t/s**, o sea que quedan ~13% en la ruta de
expertos.

## Pendiente

- La cache en `mul_mm_id`: sin eso el prefill hace el modo inusable.
- Leer la recencia cada N tokens, para el 7% que aun cuesta el promotor.
- Barrer K, hoy fijado a mano para igualar la VRAM del `ncmoe 24`.
- El 35B Q4 sigue sin medir: el banco completo en RAM no cabe en 32 GB.

---

# Undecima tanda: prefill con niveles, y donde esta la frontera

## La cache llevada a `mul_mm_id` (0062)

Mismo patron que en decode: un nivel por despacho, `im` remapeado a `ims` por la
tabla de slots, y el kernel descarta los expertos que no son de su nivel.

| Q2_K_XL pp512 | |
|---|---:|
| A `-ncmoe 24` | 609.93 ± 3.86 |
| C nuevo, antes de 0062 | 267.03 ± 7.08 |
| **C nuevo, con 0062** | **344.06 ± 7.77** |

+29%, pero sigue -44% contra el modo actual, **y ahora se sabe por que**:
`-ncmoe` no lee los expertos por el bus en prefill, los **sube con solape**
gracias al doble buffer del parche 0015. Las lecturas del nivel 2 ocurren dentro
del kernel, asi que la GPU se para en el bus.

La cuenta cuadra: el nivel 2 son 154 de 256 expertos, 5.8 GiB por pasada de 512
tokens, que a 13.25 GB/s son 437 ms. La diferencia medida entre A (0.84 s) y C
(1.49 s) es 650 ms. **Solapar esa subida deberia recuperar casi todo.**

Eso reabre la fase 7, que se habia dado por shippeada: lo esta para el reparto
por capas, pero con un espejo por experto la politica correcta es subir los
fallos con solape, no leerlos en linea.

## El estado del modo, y su frontera

| Q2_K_XL | actual | nuevo | |
|---|---:|---:|---|
| decode | 28.12 t/s | **51.48 t/s** | **+83%** |
| prefill | 609.93 pp512 | 344.06 | **-44%** |

Cruce end-to-end: **el modo nuevo gana mientras el prompt sea menor que 12.7
tokens por cada token generado.**

| prompt | generacion | actual | nuevo | |
|---:|---:|---:|---:|---|
| 512 | 500 | 18.62 s | 11.20 s | +39.8% |
| 2048 | 500 | 21.14 s | 15.66 s | +25.9% |
| 4096 | 500 | 24.50 s | 21.62 s | +11.8% |
| 8192 | 200 | 20.54 s | 27.69 s | **-34.8%** |
| 512 | 2000 | 71.96 s | 40.34 s | +43.9% |

Gana en chat, pierde en prompt largo con respuesta corta (RAG, resumen de
documento, analisis de codigo pegado). No es un empate ambiguo: es una frontera
medible, y da el criterio para cuando el modo deba activarse solo.

---

# CORRECCION IMPORTANTE: el prefill de `llama-bench` no vale para decidir

Todo lo de prefill de las tandas anteriores estaba medido con `llama-bench pp512`
y **es una referencia inflada**. Con prompt real, mismo modelo y `-ncmoe 24`:

| arnes | prefill |
|---|---:|
| `llama-server` de la app 0.85.7 | **349.85 t/s** |
| `llama-completion`, prompt de 1541 tokens | **360.68 t/s** |
| `llama-bench pp512` | **609.93 t/s** |

Los dos arneses realistas coinciden dentro del 3%. `llama-bench` da **1.7x de
mas**: mide un batch de 512 en regimen, con calentamiento y repeticion, a
profundidad 0. Un prompt real se procesa una vez, en frio, y en trozos de
`n_ubatch` a profundidad creciente.

## El cuadro real: el modo nuevo gana en los DOS ejes

Qwen3.6-35B-A3B Q2_K_XL, misma VRAM de expertos:

| | A `-ncmoe 24` | C nuevo | |
|---|---:|---:|---|
| prefill (prompt real) | 360.68 t/s | **412.77** | **+14.5%** |
| decode (599 tokens) | 28.12 t/s | **51.53** | **+83%** |

**No hay compromiso.** Queda ANULADO lo escrito antes:

- "el prefill del modo nuevo esta roto, -44%": FALSO, era la referencia.
- "el modo gana mientras el prompt sea menor que 12.7 tokens por token
  generado": ANULADO, el cruce no existe porque gana en los dos ejes.
- "hay que subir los fallos con solape porque son 437 ms de bus": la premisa
  venia de la misma referencia mala.

## Lo que aporta cada parche en prefill, con prompt real

- **0062** (lectura por niveles en `mul_mm_id`): es el que da el +14.5%.
- **staging de banco completo** (una capa por delante, 2 x 246 MiB): 417.40
  contra 412.77, **~1%, dentro del ruido**. Descartado, no llega a parche.
- **0064** (promocionar durante prefill): media +11.4% con `llama-bench`, o sea
  con la referencia mala. Pendiente de remedir con prompt real.

## Leccion

Es el segundo arnes sintetico que engana hoy en la misma sesion. En decode,
`llama-bench` inflaba el acierto del promotor (92% contra 62% real) porque su
flujo de tokens es repetitivo. En prefill infla la referencia 1.7x porque mide a
profundidad 0 y en caliente. **Ningun numero de `llama-bench` sirve para decidir
si algo compensa; solo para comparar dos builds en el mismo punto.**

## El 1.7x es arranque en frio, NO profundidad de contexto

`llama-bench pp512` con `-d` (ncmoe 24), que son los cuatro trozos en que se parte
un prompt de 1541 tokens:

| profundidad | pp512 |
|---|---:|
| 0 | 605.58 ± 6.63 |
| 512 | 595.50 ± 3.90 |
| 1024 | 588.80 ± 0.50 |
| 1536 | 581.66 ± 1.70 |

Media ponderada: **596.50**, contra los **360.68** que da el prompt real. La
profundidad solo cuesta 4%. El resto es coste de una vez: compilacion de
pipelines, primer toque de los pesos, construccion del grafo (`graphs reused = 0`).

**Consecuencia**: los 350-360 son el PRIMER prompt de la sesion. A partir del
segundo el prefill sube hacia los ~600. Cualquier comparacion tiene que decir si
es en frio o en caliente.

## 0064 RETIRADO: era una regresion del 30%

Promocionar expertos durante el prefill medido con prompt real:

| | prefill |
|---|---:|
| sin promocion en prefill | **432.46 t/s** |
| con promocion en prefill | **304.14 t/s** |

**-30%.** Con `llama-bench` habia medido +11.4%. El parche se elimina de la serie.
Tercer numero de `llama-bench` que se cae hoy al remedirlo con un arnes realista.

## Prefill en frio y en caliente: el numero bueno, por fin

Tres prompts seguidos de 1541 tokens en la misma sesion de `llama-server`,
`cache_prompt` en false, asi que reprocesa entero cada vez:

| | prompt 1 (frio) | prompts 2-3 (caliente) |
|---|---:|---:|
| A app 0.85.7 `-ncmoe 24` | 357.70 | **569.02 / 569.47** |
| C nuevo K=102 | 423.83 | 426.00 / 425.64 |

**A calienta un 59%, C no calienta nada.** En frio gana C (+18.5%); en regimen
gana A y C queda **-25.2%**. C esta plano porque su prefill choca contra el bus,
que es un limite duro y no mejora con el uso.

Trampa de medida, tres veces en esta tanda: ficheros temporales rancios
(`resp.json`, `h.json`) hacian que un fallo se disfrazara del numero anterior.
**Sintoma siempre igual: valores identicos donde deberian variar.**

## El cuadro final del dia

| Q2_K_XL, misma VRAM | actual | nuevo | |
|---|---:|---:|---|
| decode | 28.12 t/s | **51.53** | **+83%** |
| prefill en frio | 357.70 | 423.83 | +18.5% |
| prefill en caliente | 569.25 | 425.82 | **-25.2%** |

Cruce con los numeros en caliente: **el modo nuevo gana con prompts de menos de
27.3 tokens por cada token generado.**

| prompt | generacion | actual | nuevo | |
|---:|---:|---:|---:|---|
| 512 | 500 | 18.68 s | 10.91 s | +41.6% |
| 1541 | 500 | 20.49 s | 13.32 s | +35.0% |
| 4096 | 500 | 24.98 s | 19.32 s | +22.6% |
| 8192 | 500 | 32.17 s | 28.94 s | +10.0% |
| 16384 | 500 | 46.56 s | 48.18 s | -3.5% |
| 8192 | 200 | 21.50 s | 23.12 s | -7.5% |

Gana en todo el chat normal y en RAG con respuestas de tamano razonable. Pierde
solo con prompts enormes y respuestas cortas.

**Sustituye al cruce de 12.7 que se publico antes**, que salia de comparar contra
`llama-bench` (referencia 1.7x inflada) y contra un prefill de C mal medido.

---

# DISENO CORRECTO: dos depositos, no cache con copia

## El defecto de fondo del 0055-0064

La cache **duplica**: el banco se reserva y se rellena entero al cargar, y el
espejo es una copia encima. Medido con RSS: `-ot exps=MTL0` usa 10.91 GiB de RAM
mas 3.83 de espejo = 14.74 para un modelo de 11.44 GiB. `ncmoe 24` usa 6.72 +
3.84 = 10.56, o sea MENOS que el modelo, porque no duplica.

A igual VRAM el modo gana (decode 49.87 contra 28.71, +74%), pero gasta 3.83 GiB
de RAM de mas, y contra la mejor opcion del baseline (`ncmoe 0`, 65.41) pierde
usando el doble de memoria total.

Dos ideas descartadas por medida o por analisis, no por pereza:

- **`madvise` de las paginas ya copiadas**: Metal las tiene wired para que la GPU
  las vea; el sistema no las suelta.
- **Promocion por intercambio sobre las estructuras actuales**: evita duplicados
  obsoletos pero NO baja la reserva, que ya esta hecha al cargar. No sirve.

## El diseno que si lo resuelve

Dos **depositos de tamano fijo**, no dos conjuntos fijos:

    tensor caliente   K expertos     buft privado de Metal (VRAM)
    tensor frio       n-K expertos   buft host-shared (RAM)
    tabla             experto -> (deposito, indice), cambia en runtime
    promocion         INTERCAMBIO entre depositos
    grafo             dos mul_mat_id, uno por deposito, y suma

Memoria total = exactamente el modelo, igual que `ncmoe`. La adaptatividad se
conserva porque migra el contenido, no la capacidad, lo cual importa: un conjunto
caliente ESTATICO ya se refuto (45.7% de acierto al cambiar de prompt contra
78.4% del LRU).

El kernel de dos niveles del 0059 **ya es esa suma**; lo que cambia es que
operaria sobre dos tensores reales en vez de vistas del mismo tensor.

## Las tres piezas

1. **Cargador** (`llama-model.cpp:2930`, `llama-model-loader`): crear dos tensores
   desde un tensor del GGUF. Es viable porque los expertos son contiguos por
   experto, asi que un subrango es contiguo: bastan dos entradas sinteticas en
   `weights_map` con los offsets calculados.
2. **Grafo** (`build_moe_ffn`): dos `mul_mat_id` sobre el mismo ruteo y suma. El
   riesgo esta aqui, porque el ruteo no se puede duplicar.
3. **Backend**: la tabla pasa a mapear a deposito e indice, y el promotor
   intercambia en vez de copiar.

## Lo que YA funciona y se queda

- `0059` kernel de dos niveles, un deposito por despacho: sin el, elegir base
  dentro del kernel cuesta 5%.
- `0061` readback de recencia agrupado: 121 -> 193 t/s.
- `0064` gather del banco a VRAM una capa por delante, con los residentes
  copiados VRAM a VRAM: prefill 361 -> 498 (+37.9%). La sincronizacion correcta
  en Metal NO es el evento dentro del command buffer (deadlockea, medido) sino
  **esperar en el host durante el encode**, que solapa con la GPU ejecutando las
  capas anteriores.

## 23-08 tarde: la trampa del prompt de ruido

Todas las medidas de decode de la tanda anterior estaban infladas. El prompt era
una lista de palabras aleatorias; el modelo degeneraba a repetir un solo token, y
repetir un token rutea siempre a los mismos expertos. El acierto de la cache subia
a 98.8% y el tg con depositos parecia ganar +17%.

Rehecho con prosa de `wikitext-2-raw` (mismo modelo, mismo prompt de 1459 tokens,
600 tokens generados, Qwen3.6-35B-A3B Q4_K_S):

| | modo actual (ncmoe 24) | depositos K=102 |
|---|---:|---:|
| prompt | 226.78 t/s | **1310.70** (5.8x) |
| generacion | **29.30 t/s** | 28.42 (-3%) |

Acierto real 86-87% (no 98.8%) y ~2200 intercambios por cada 64 evals.

**El prefill gana de verdad; el decode NO gana.** Todo lo que se dijo del +15/17%
de generacion queda retirado.

Regla para este arnes: el prompt tiene que producir texto coherente, y hay que
mirar la salida, no solo el numero. Un tg alto con acierto >95% en un MoE es
sospechoso de generacion degenerada.

### Intercambio asincrono (dos etapas por filas de repuesto)

Correcto (salida identica a la referencia en OLMoE con K=32). Sobre el prompt de
ruido daba 34.80 contra 34.21 del sincrono. Pendiente de remedir sobre texto real:
la sonda que quitaba la espera sin pagar nada prometia 6.2% y el arreglo bien hecho
dio 1.7%, porque el intercambio tarda ahora dos evals en completarse.

### Barrido de residencia sobre texto real (400 tokens generados)

| K | acierto | prompt t/s | generacion t/s |
|---|---:|---:|---:|
| 102 | 86% | 1310.70 | 28.42 |
| **112** | 96% | 1291.17 | **30.58** |
| 120 | 96% | 1265.24 | 29.61 |

Referencia `ncmoe 24`: 226.78 / 29.30.

La residencia SI es la palanca del decode, al contrario de lo que dijo el barrido
anterior: aquel corria sobre el prompt de ruido, donde el acierto ya estaba en 98%
y subir K no podia mejorarlo. K=120 empeora pese a igualar el acierto de 112, y
hace mas intercambios (1089 contra 921).

### A/B final (misma sesion, VRAM por el registro IOAccelerator)

| | modo actual (ncmoe 24) | depositos K=112 |
|---|---:|---:|
| prompt | 223.00 t/s | **1291.76** (5.8x) |
| generacion | 28.94 t/s | **30.55** (+5.6%) |
| VRAM pico | 10.11 GiB | 10.34 GiB |

Reproducible: dos corridas de K=112 dieron 30.58/30.55 de tg y 1291.17/1291.76 de pp.

K=102 es el otro punto util de la curva: 9.62 GiB de VRAM (medio giga por debajo
del modo actual) con el mismo prefill y el decode a la par.

### Correccion: la "salida degradada" era divergencia de muestreo

A temperatura 0.8 los depositos y `ncmoe 24` divergen en el PRIMER token por
redondeo (los expertos se calculan en GPU contra AVX2), y a partir de ahi son
trayectorias distintas. Una salio como un ensayo y la otra degenero copiando el
prompt. Eso NO es corrupcion.

Verificado a temperatura 0 en el 35B, mismo prompt, tres caminos:

- `ncmoe 24`                         -> "Paris is the capital of France. It became the capital due to its strategic location on the Seine River..."
- depositos K=112 SIN intercambio    -> identico
- depositos K=112 CON intercambio    -> identico

Y en OLMoE los tres son identicos caracter a caracter. **El intercambio asincrono
no corrompe nada.**

Regla: para juzgar correctitud, temperatura 0. A temperatura >0 dos caminos
numericamente distintos SIEMPRE divergen y la comparacion no dice nada.

**El PPL de este arnes esta roto**, no el motor: `llama-perplexity` da 23704 tambien
en la referencia `ncmoe 24`. No usarlo hasta averiguar por que.

### El despacho del deposito frio cuesta el 36% del decode

Sonda (salida incorrecta a proposito, solo acota): quitando el despacho frio el tg
pasa de 30.55 a **41.58**. A 96% de acierto los fallos son ~11 MiB por token, o sea
0.85 ms de los 8.68 ms de diferencia; **los otros ~7.8 ms son coste de lanzar 120
despachos casi vacios por token**, no trabajo util.

Siguiente paso: un solo despacho que elija deposito por experto. Elegir entre dos
punteros base dentro del kernel se midio en 5% (parche 0059), muy por debajo del
24% que cuesta el despacho aparte.

### Un solo despacho para los dos depositos

El kernel `mul_mv_id` gana un modo 4 que elige el deposito por experto (base
variable) en vez de un despacho por deposito. A/B en los dos ordenes:

| | prompt | generacion |
|---|---:|---:|
| dos despachos | 1293.08 / 1292.31 | 30.48 / 30.42 |
| **un despacho** | 1292.50 / 809.24* | **31.14 / 31.17** |

(*) la primera corrida tras recompilar da el prefill hundido porque el deposito
frio vive en RAM del host y sus paginas estan frias. Invertir el orden lo movio al
otro brazo: **es artefacto de primera corrida, no del modo**. Descartar siempre la
primera medida tras un build cuando hay memoria del host de por medio.

**+2.2% de decode, prefill intacto.** Pero solo recupera 0.7 de los 11 t/s que
marcaba la sonda: el coste NO era el numero de despachos.

**Lo que queda es latencia de cola.** A 96% de acierto hay ~38 fallos por token
repartidos en 120 bancos; cada banco es un nodo del grafo y no termina hasta que
acaben sus hilos mas lentos, que son los que leen memoria del host a 13 GB/s. El
banco entero espera por un punado de lecturas. Por eso subir K de 112 a 120 no
ayuda (mismo acierto) y por eso quitar el trabajo frio entero valia 11 t/s.

Frente abierto: adelantar los fallos a VRAM antes de que corra el banco, o cortar
la cola de otra forma. No es coste de lanzamiento.

## FALLO GRAVE encontrado por la perplejidad: el camino de lotes ignoraba el deposito frio

`llama-perplexity` con depositos daba **PPL 63361.7** contra 9.4674 del camino normal
(OLMoE). La generacion parecia correcta porque los prompts cortos van por
`mul_mv_id`, que si estaba bien; la perplejidad usa lotes de 512 y va por
`mul_mm_id`, que **no conocia la tabla de los dos depositos**:

- caia en `encode_tier(bid_src0, 0)`, el reparto estatico, que descarta todo
  experto fuera del rango del deposito caliente;
- peor: la dimension Z del despacho y el kernel del mapa de tokens usaban
  `src0->ne[2]` = 114, asi que a los expertos 114-127 **ni se les asignaba grupo
  de hilos**;
- y los dos buffers de trabajo (`tpe`, `ids`) estaban dimensionados igual de corto.

Arreglado: `mul_mat_id_n_expert()` suma los dos depositos, el mapa y el despacho
recorren todos los expertos, y el kernel de lotes gana el modo 4 (una malla, base
elegida por experto) igual que el de decode.

**Verificado: PPL identica al camino normal en los dos modelos.** OLMoE 9.4674 y
Qwen3.6-35B 6.2339, cifra por cifra.

### Las medidas de prefill anteriores eran de un calculo incompleto

Con el arreglo, 35B, prompt de 1459 tokens de wikitext y 400 generados:

| | referencia ncmoe 24 | depositos K=112 |
|---|---:|---:|
| PPL | 6.2339 | **6.2339** |
| prompt | 230.61 t/s | **260.96** |
| generacion | 26.60 t/s | **30.86** |

**El 5.8x de prefill era falso**: salia de saltarse los expertos del deposito frio.
Lo real es +13%. RETIRADAS todas las cifras de prefill anteriores a este arreglo.

OJO con la varianza de la referencia: 29.30, 28.94 y 26.60 de tg en tres momentos
del dia. Comparar solo brazos medidos seguidos.

### Auditoria del resto del camino MoE

- analisis de dependencias (`ggml-metal-common.cpp`): recorre los `GGML_MAX_SRC`,
  ve el deposito frio. Limpio.
- optimizador de grafo (`ggml-metal-context.m`): mira la salida y `src[1]`, no la
  cuenta de expertos. Limpio.
- llamadas directas a `ggml_mul_mat_id` fuera del ayudante: son de LoRA
  (`granite-switch.cpp`, `build_lora_mm_id`), y esos tensores nunca se parten. Limpio.
- **`ggml_mul_mat_id` NO comprueba que los ids quepan en `as->ne[2]`**, por eso el
  fallo fue silencioso y no revento.
- cerrado el camino de corrupcion silenciosa: si el nodo dice que es un deposito
  partido y falta la tabla, ahora es un `GGML_ASSERT` en los dos caminos.
- puesta una guarda de colocacion: solo se parte si el tensor va a un buft de
  Metal. En CPU el nodo veria solo el deposito caliente. **Trampa evitada por poco**:
  `ggml_backend_dev_name` devuelve el nombre de la TARJETA, no "Metal"; hay que
  comparar el nombre del registro (`ggml_backend_reg_name` == "MTL").

## Contra la APP INSTALADA (0.85.7), mismo prompt y generacion larga

Prompt de 1459 tokens de wikitext, 1200 tokens generados, `ncmoe 24`, Qwen3.6-35B-A3B Q4_K_S.
La app por su `llama-server`; los otros dos por `llama-completion` del arbol.

| | app instalada | referencia del arbol | depositos K=112 |
|---|---:|---:|---:|
| prompt | 222.59 | 225.16 | **262.01** |
| generacion | 29.09 | 29.24 | **32.48** |

**La referencia del arbol y la app son el mismo numero** (dentro del ruido), asi que
el baseline no estaba lastrado. La ganancia real de los depositos sobre lo que hoy
corre el usuario es **+17.7% de prompt y +11.7% de generacion**, con la perplejidad
identica y 10.34 GiB de VRAM contra 10.11.

Con generacion corta (400) la referencia bajaba a 26.60 y la ganancia parecia mayor.
Medir con generacion larga: 1200 tokens.

## El techo real: el depósito frío no puede pasar del working set del dispositivo

Con K=75 el 35B tardaba 22 minutos sin terminar. **El motor lo avisaba en la carga y
nunca leí el log**, porque filtraba la salida por "eval time". Mi teoría del swap era
falsa: 1350 MB de swap en 32 GB no explica un 10x.

| K | frío en host | VRAM | resultado |
|---|---:|---:|---|
| 112 | ~1.5 GiB | 10.34 | 262.01 pp / 32.48 tg |
| 95 | 10.73 GiB | 9.31 | 264.35 pp / 28.79 tg, **va bien** |
| 75 | 12.07 GiB | 7.98 | **colapsa ~10x** |

**El límite es el host SOLO, no la suma.** A K=95 la suma es 20 GiB, muy por encima
del working set (11.98 GiB), y funciona. El precipicio está al cruzar el working set
con el depósito frío: 10.73 pasa, 12.07 no.

La curva ya estaba medida en un comentario del propio árbol y no la miré: 4.6 GiB de
banco → 22.3 t/s, 7.4 → 16.2, 12.6 → 0.31, en una tarjeta de 12.87 GiB.

### Arreglo

El límite vivía en `ggml_metal_buffer_init` como un aviso contra 7/8 del working set.
Ahora **rechaza la reserva** cuando el total de pesos residentes en host pasaría del
working set, así que el modelo falla al cargar con el número y la salida:

```
host-resident weights would reach 12.07 GiB, past what this device can map (11.98 GiB).
Keep more of the model in VRAM, or use --n-cpu-moe.
```

Verificado: K=112 y K=95 cargan y generan, K=75 se rechaza. PPL de OLMoE sigue en
9.4674, idéntica al camino normal.

**Consecuencia para el diseño**: el depósito frío nunca puede superar el tamaño de la
tarjeta. En una de 8 GB eso deja fuera al 35B, que necesitaría ~12 GiB de frío. El
método sirve para modelos que pasan poco de la VRAM, no para los que la doblan.

### Dos intentos de arreglo que estuvieron mal, y por qué

1. **Contar host + VRAM contra el working set**: refutado por K=95 antes de escribirlo.
2. **Recortar el depósito frío al presupuesto**: esos expertos se van a VRAM, la
   tarjeta se pasa y sale `command buffer failed with status 5`. Si el frío no cabe,
   la configuración no existe: lo correcto es rechazarla.

## REFUTADO: el techo no es lo asignado, es lo que se toca

Medido con `rset.swift` en la RX 6700 XT (working set 11.98 GiB):

| asignado en host | ventana leida | resultado |
|---:|---:|---|
| 4 GiB | 4 GiB | 1.92 GB/s, bien |
| **20 GiB** | 2 GiB | **3.91 GB/s, bien** |

Se pueden mapear **20 GiB** de memoria de host en una tarjeta de 11.98 de working set
sin colapso alguno, siempre que los kernels lean una ventana acotada. No hace falta ni
gestionar residencia.

**Queda refutada la conclusion anterior** de que el limite era la ASIGNACION. El colapso
del 35B (12.07 GiB de deposito frio) venia de que los kernels leian filas dispersas por
todo el deposito en cada token, no de tenerlo asignado.

**Consecuencia**: el diseno correcto es tener todos los expertos en memoria de host
direccionable y traer SOLO las filas que faltan a una cache acotada en VRAM. Eso mantiene
la ventana pequena y evita el techo.

## Lo que hace el motor de referencia, leido del codigo

1. **El LRU es un kernel de GPU.** La tabla vive en el dispositivo y el tensor de ids se
   reescribe in situ a numeros de ranura, o a -1 para lo que calcula la CPU. Nunca se
   vuelve al host, asi que no hay sincronizacion y es capturable en un grafo.
2. **Las copias son otro kernel**: un `index_copy` que trae solo las filas que faltan
   desde host fijado a las ranuras.
3. **El solape CPU/GPU lo hacen a mano**, no con el planificador: bajan las activaciones a
   memoria fijada, la GPU levanta una bandera con una operacion de stream (sin kernel), un
   coordinador de CPU persistente la ve, calcula y levanta otra bandera que la GPU espera.
   Descartaron los callbacks del host por medida: 30-50 us por llamada, ~6 ms por paso.

**Lo que porta**: 1 y 2, que eliminan casi toda la complejidad construida (tabla en el
host, promocion entre grafos, el problema de la reutilizacion de grafos).
**Lo que no se sabe**: 3. Su handshake necesita esperar sobre un valor en memoria desde el
stream; en Metal el equivalente son los eventos compartidos, y aqui esta medido que
`encodeWaitForEvent` bloquea el command buffer entero.

## Rediseno: la cache decide y trae en la GPU, sin volver al host

Piezas, cada una validada por separado antes de integrarla:

- **LRU como kernel de GPU** (`tosh-moe-lru.metal`): tres fases en un threadgroup. Validado
  contra un modelo de la misma politica: 2000 pasos, 5 configuraciones, cada decision
  coincide (`experiments/dynamic-moe/lrutest.swift`).
- **Traida selectiva**: un gather que mueve solo las filas que el LRU programo. Validado
  byte a byte, 360 pasos.
- Estado por CAPA, no por banco: los tres bancos de una capa comparten el ruteo.
- El LRU corre una vez por capa, detectando el cambio de pasada por el indice de nodo
  (los indices suben dentro de una pasada).

**Dos bugs reales encontrados al integrar**, que ninguna prueba aislada habria visto:

1. **El tensor de ids NO es contiguo**: `nbi1 = 256` con solo 8 enteros utiles por token.
   Leerlo en linea saca relleno a partir del segundo token.
2. **Los mismos ids los usa `ggml_get_rows` para los pesos del router**
   (`llama-graph.cpp:2047`). Reescribirlos in situ a ranuras da pesos de filas
   equivocadas. El motor de referencia no tiene este problema porque su capa MoE es suya.
   Solucion: el LRU escribe los slots en su propia area y el matmul lee de ahi.

**FUNCIONA de punta a punta** con la configuracion consistente
(`GGML_METAL_SHARED_BUFFERS_ENABLE=1`, todo compartido) y cache de 32 de 64 ranuras:
salida correcta.

### PENDIENTE: la configuracion MIXTA esta rota

Poner unos pocos tensores en el buffer type de host dejando el resto en VRAM da basura.
Reproduccion minima, sin MoE de por medio:

```
-ot 'attn_q\.weight=MTL0'   ->  "What is a what is a what a what..."
```

Con el interruptor de upstream (`GGML_METAL_SHARED_BUFFERS_ENABLE=1`, que hace compartido
el buffer type POR DEFECTO y por tanto todo) la salida es correcta. O sea el camino
compartido funciona en esta GPU discreta; lo que falla es MEZCLAR por defecto privado con
unos pocos tensores en host.

Descartado por lectura: `supports_buft` acepta el tipo, la resolucion de buffer es por
tensor, y `addr_virt` (0x400 en adelante) no colisiona con punteros reales del host.

La configuracion que funciona NO sirve de producto: pone tambien el KV y los buffers de
computo en host. Arreglar el modo mixto es el siguiente paso obligado.

### RESUELTO: por que la configuracion mixta daba basura

`ggml_metal_buffer_init` decidia como envolver la memoria mirando el flag del DISPOSITIVO:

```c
if (props_dev->use_shared_buffers && shared) {   // envolver la memoria de host
} else {
    newBufferWithLength(... StorageModePrivate)  // buffer nuevo en VRAM
}
```

Con el dispositivo en "no compartido" y un buffer que SI pide compartido (el buffer type de
host), reservaba memoria de host, marcaba el buffer como compartido, y despues creaba un
buffer PRIVADO en VRAM con basura. Las escrituras iban a la memoria de host y la GPU leia el
buffer privado.

Por eso el interruptor de upstream funcionaba (pone el dispositivo entero en compartido y las
dos condiciones coinciden) y la mezcla no. Arreglo: mirar `shared`, que ya lleva dentro la
respuesta del dispositivo mas la opcion de buffers de host.

**Verificado**: la reproduccion minima (`-ot 'attn_q\.weight=MTL0'`) responde bien, y la cache
con 63, 32 y 16 ranuras de 64 da salida IDENTICA a la referencia. Al no haber rama de CPU, no
hay ni diferencia numerica: la cache es transparente.

## PRIMER RESULTADO A FAVOR: gpt-oss-20B

Prompt de 1426 tokens de wikitext, 300 generados, cache de 16 de 32 expertos con el banco
completo en memoria de host.

| | referencia `ncmoe 10` | cache 16/32 |
|---|---:|---:|
| prompt | 280.50 t/s | **523.10** (1.9x) |
| generacion | 31.82 t/s | **35.85** (+12.7%) |
| VRAM | 7.67 GiB | **6.48 GiB** |
| host mapeado | 7.75 GiB | 10.16 GiB |
| RSS | 4.66 GiB | 10.78 GiB |

Mas rapida en las dos magnitudes y con 1.2 GiB MENOS de tarjeta. Con correctitud ya probada
(PPL identica, generacion byte a byte igual): al no haber rama de CPU no hay ni diferencia
numerica.

**El limite real del metodo es la RAM, no la VRAM.** La cache necesita TODOS los expertos en
memoria de host fijada; `ncmoe` solo guarda los de las capas que descarga. Aqui son ~6 GiB de
RAM extra a cambio de 1.2 GiB de VRAM y la velocidad.

**Por eso el 35B no se pudo medir**: 16 GiB de expertos fijados mas un modelo de 19.45 GiB en
una maquina de 32 GB manda al swap, con el proceso en estado UN. Abortado dos veces (con 100
ranuras y con 32; el numero de ranuras no cambia la presion, porque lo que pesa es el banco de
host completo).

**Sin optimizar nada todavia**: el LRU elige victima en un solo hilo con un bucle serie, la
traida es un threadgroup por fila, y se traen todos los fallos sin politica de tope.

### La comparacion a VRAM IGUALADA (gpt-oss)

Las 16 ranuras del apartado anterior NO igualaban la VRAM: usaban 1.2 GiB menos. Con 20
ranuras la VRAM coincide al gramo con `ncmoe 10`, y esa es la comparacion honesta:

| | `ncmoe 10` | **cache 20/32** |
|---|---:|---:|
| prompt | 280.50 t/s | **525.40** (1.9x) |
| generacion | 31.82 t/s | **49.06** (+54%) |
| VRAM | 7.67 GiB | **7.66 GiB** |
| RSS | 4.66 GiB | 10.77 GiB |

Curva de residencia, que sube muy rapido en esta zona:

| ranuras | VRAM | generacion |
|---|---:|---:|
| 16 | 6.48 GiB | 35.85 |
| **20** | **7.66** | **49.06** |
| 22 | 8.26 | 55.83 |

De 16 a 22 ranuras la generacion pasa de 35.9 a 55.8: el modelo concentra el uso en pocos
expertos y la cache los encuentra. Buscar ese codo solo (el modo `auto` del plan) vale mucho.

**El canje del metodo, ya bien medido: cambia RAM por velocidad sin gastar mas VRAM.** Son
~6 GiB de RAM extra (todos los expertos en host fijado, contra solo los de las capas que
`ncmoe` descarga) a cambio de casi doblar el prompt y +54% de generacion a igual tarjeta.

### Correccion: la cache llena NO supera al camino normal

Dije que si a partir de comparar contra 1005/75.78, que estaban medidos con OTRO binario (el
del release, no `build-moe`). Rehecho con el mismo binario y el mismo arnes:

| gpt-oss-20B | VRAM | RSS | prompt | generacion |
|---|---:|---:|---:|---:|
| todo apagado (cabe en VRAM) | 11.20 | **0.64** | 1251.51 | **87.17** |
| cache llena (32) | 11.20 | 10.74 | 1249.81 | 83.73 |
| **cache 20** | **7.66** | 10.75 | 530.05 | **49.17** |
| `ncmoe 10` | 7.67 | 4.66 | 280.50 | 31.82 |

- **Si el modelo cabe, la cache no sirve**: misma VRAM, -4% de generacion y 10 GiB mas de RAM.
  El mecanismo cuesta poco pero no es gratis.
- **Si no cabe, gana claro**: a 7.66 GiB de tarjeta da 1.9x de prompt y +54% de generacion
  sobre `ncmoe 10`, por unos 6 GiB de RAM.

**El prefill lee del banco de host cuando la residencia es parcial**: 530 contra 1250 de la
cache llena. Esos ~720 t/s son el precio, y es lo que atacaria el doble buffer (fase 7).

El ajuste que deja al prefill leer las ranuras solo se activa con residencia COMPLETA, que es
cuando ningun experto se queda sin sitio. Verificado que a 20 ranuras no cambia nada
(530.05/49.17 contra 525.40/49.06 antes del ajuste) y que la salida sigue siendo identica a la
referencia con 32 y con 20 ranuras.

## Optimizacion 1: repartir la traida de cada experto

Un experto son megabytes y se copiaba con UN threadgroup, dejando la GPU casi parada durante
la copia. Repartido entre varios (gpt-oss, 20 ranuras, misma VRAM):

| trozos por fila | generacion |
|---|---:|
| 1 (como estaba) | 48.91 t/s |
| 8 | 63.76 |
| **16** (por defecto) | **64.32** |
| 32 | 64.58 |

**+32% de generacion.** Satura a partir de 8. PPL identica (223.6723 contra 223.6723, mismo
binario y mismos fragmentos) y texto a temperatura 0 identico.

Contra `ncmoe 10` a igual VRAM la ganancia pasa de +54% a **+103% de generacion**.

## El 35B: medido por fin, en Q2_K_XL (11.45 GiB)

En esta tanda se asumio que el Q4_K_S (19.45 GiB) no cabia porque su banco host ronda 16 GiB.
La prueba posterior del 24 de agosto invalido esa explicacion: cargo con ~18.1 GiB de RSS y el
swap practicamente inmovil. El Q2_K_XL se uso aqui porque permitia completar el barrido, no
porque quedara demostrado que RAM era el limite del Q4.

Prompt de 1459 tokens de wikitext, 300 generados, VRAM igualada al gramo:

| | `ncmoe 24` | **cache 114/128** |
|---|---:|---:|
| prompt | **336.15** | 209.97 (-38%) |
| generacion | 26.93 | **51.46** (+91%) |
| VRAM | 6.53 GiB | 6.54 GiB |
| RSS | **6.78 GiB** | 10.96 GiB |

**La generacion casi se dobla a igual tarjeta.** Correctitud verificada: mismo texto a
temperatura 0 que la referencia.

Curva de residencia:

| ranuras | VRAM | prompt | generacion |
|---|---:|---:|---:|
| 55 | 4.28 | 206.95 | (parada a 2 tokens) |
| 85 | 5.43 | 208.22 | 48.65 |
| **114** | **6.54** | 209.97 | **51.46** |

**El prompt pierde un 38% y la causa esta identificada**: con residencia parcial el prefill lee
los expertos desde memoria de host (en gpt-oss el mismo efecto: 530 con cache parcial contra
1250 con cache llena). Eso es exactamente lo que resolveria el doble buffer de prefill.

Canje del metodo en el 35B: **casi el doble de generacion por un tercio menos de prompt y
4 GiB mas de RAM**. Para chat gana; para prompts largos con respuestas cortas, pierde.

Trampa del arnes: este modelo razona y cierra `</think>` enseguida, asi que con el prompt de
wikitext paraba a los 2 tokens y el tg no significaba nada. Hace falta `--ignore-eos`.

## Optimizacion 2 (LRU paralelo): NO aporta en gpt-oss

La busqueda de victima paso de un hilo a una reduccion del threadgroup. Validada correcta,
pero **64.45 t/s contra 64.58**: dentro del ruido. Con 20 ranuras y ~2 fallos por capa el bucle
serie eran 40 iteraciones, nada al lado de los 15.5 ms de un token. Deberia importar con 128
ranuras, pero eso no esta medido.

## Optimizacion 3: materializar el banco antes de un lote ancho (fase 7)

Un lote demasiado ancho para la cache leia los expertos dispersos desde memoria de host. Ahora
copia el banco de la capa de una vez a un scratch en VRAM (uno solo, del tamano del banco mas
grande, reutilizado por todas las capas) y calcula desde alli.

**No sirve para todos los modelos, y el gate es el tamano de fila:**

| | sin materializar | con materializacion |
|---|---:|---:|
| 35B Q2, 180 ranuras (filas ~300 KB) | 202.84 pp | **379.48** (+87%) |
| gpt-oss, 20 ranuras (filas 4.2 MB) | 521.42 pp | 410.22 (**-21%**) |

Un banco de muchas filas pequenas se lee fatal disperso y la copia contigua lo arregla. Uno de
pocas filas grandes ya se lee bien y la copia es trafico de mas. Gate por `nb02` (1 MiB por
defecto, `TOSH_MOE_STAGE_ROW_KIB`): con el, el 35B se lleva el +87% y gpt-oss vuelve a su
camino (499.98, contra 521.42, dentro de la variacion).

El motor de referencia tiene la misma distincion con su `_SMALL_BANK_FEAT_BYTES`.

Correctitud verificada en los dos modelos: mismo texto que la referencia, PPL identica en
gpt-oss (223.6723).

### El 35B a VRAM igualada, con todo lo de hoy

| | VRAM | prompt | generacion |
|---|---:|---:|---:|
| `ncmoe 24` | 6.53 | 336.15 | 26.93 |
| **cache 114** | 6.54 | **381.77** | **51.67** |

**Gana en las dos magnitudes**: +14% de prompt y +92% de generacion con la misma tarjeta.

### Dos fallos propios encontrados aqui

- El area de ids remapeados era una constante de 4096 puesta a ojo; este modelo necesita
  131072 (512 tokens x 256 expertos de paso). **Se desbordaba.** Ahora se dimensiona del modelo.
- El scratch se registraba desde un `static` local, que sobrevive a la carga de prueba que hace
  `common_fit_params`: en la segunda carga ya no registraba nada y el buffer quedaba en nulo.
  La fuente de verdad tiene que ser el registro, no un static.

**Y el modelo tiene 256 expertos, no 128**: todas las cifras de "cache 114/128" eran en realidad
114 de 256, un 45% de residencia.

---

# Auditoria de integracion — 24 de agosto de 2026

Esta revision mantiene Dynamic MoE fuera del producto: `TOSH_ENABLE_DYNAMIC_MOE` sigue en
`OFF`, no hay toggle en Swift ni cambios de interfaz. Todo lo siguiente pertenece solo al
build interno `build-moe`.

## Correcciones de integridad y reproducibilidad

- El estado compartido por los tres bancos de una capa era un `static` del loader e intentaba
  distinguir modelos truncando el puntero de `this` a `int`. Podia conservar un tensor de una
  carga anterior y asociarlo a una recarga posterior. Ahora el mapa pertenece a cada
  `llama_model_loader` y muere con la carga.
- El cambio que permitia respaldar el banco directamente con el GGUF existia solo en el vendor
  sucio. Ya esta capturado en `0070`, pero queda detras de `TOSH_MOE_MMAP_BANK=1` porque no es
  seguro como comportamiento normal.
- `0070` vuelve a aplicar limpiamente despues de toda la serie de parches de produccion.
- La prueba logica de LRU completa 20 000 pasos, 1 926 144 consultas y no crece en memoria.
- PPL en el mismo arnes de 2048 tokens: cache 20 y camino normal producen exactamente
  `1735.1274 +/- 172.06913`. El valor absoluto depende del fragmento; lo importante aqui es
  la igualdad exacta entre rutas.

## Medicion actual del 35B Q2, misma VRAM

Modelo: Qwen3.6-35B-A3B Q2_K_XL. Cache de 114/256 frente a `ncmoe 24`. Prompt real de
1456 tokens, 300 generados, `--ignore-eos`, contexto 4096 y el mismo binario.

| | `ncmoe 24` | cache 114 | diferencia |
|---|---:|---:|---:|
| prompt | **325.05 t/s** | 301.83 t/s | -7.1% |
| generacion | 21.20 t/s | **43.59 t/s** | **+105.6%** |
| huella fisica | 6.6–6.7 GiB | **10.5 GiB** | +3.8–3.9 GiB |

Con prompt corto y el sistema sin presion de swap, la generacion actual queda normalmente en
52–53 t/s. El contexto largo explica que ambos valores de generacion bajen en la tabla; la
comparacion valida es entre las dos rutas con el mismo contexto.

## Experimentos descartados

### Limitar la rejilla de fetch

Dos corridas alternadas dieron 52.76 t/s con la rejilla completa y 52.68 con el limite teorico.
Es ruido, no una mejora. El cambio se retiro.

### Cambiar el numero de fragmentos por experto

En el 35B actual: 8 = 49.38 t/s, 16 = **50.26**, 32 = 47.82 bajo la misma presion del
sistema. Se conserva 16. El optimo anterior de gpt-oss tambien era 16.

### Banco respaldado directamente por mmap

- mmap sin fijar: huella fisica aparente de solo ~533 MiB, pero entra en paginacion severa,
  supera 1 GiB de swap durante la primera prueba y no termina en tiempo razonable.
- mmap + mlock: tampoco consigue un camino util en esta GPU discreta; midio **0.39 t/s** y el
  swap total del sistema llego a ~2.7 GiB.

Por eso `TOSH_MOE_MMAP_BANK` queda solo como interruptor de laboratorio. No debe activarse en
la aplicacion.

### Arranque sin precargar ranuras

`TOSH_MOE_NO_SEED=1` ahorra alrededor de 2.5 s de carga porque evita copiar los 114 expertos
iniciales, pero puede perder rendimiento durante el calentamiento. No cambia el valor por
defecto. Agrupar esas copias en una sola transferencia tampoco redujo el tiempo y se retiro.

## Conclusion sobre RAM

La RAM adicional no es una fuga: es el banco completo, cuantizado, residente y estable que
permite resolver un fallo de la cache por PCIe sin tocar CPU ni disco. El mmap demuestra que
quitar esa residencia conserva direcciones virtuales, pero destruye el rendimiento fisico.

La siguiente reduccion seria una arquitectura nueva de tres niveles, no otro modo de carga:

1. ranuras LRU actuales en VRAM;
2. cache acotada y fijada en RAM para expertos frios recientes;
3. GGUF como respaldo para fallos raros, con lectura anticipada y sin sincronizar el router con
   CPU en cada token.

Antes de implementarla hacen falta contadores reales de fallos por capa y tiempo de fetch en
GPU. Sin ellos no se puede dimensionar el segundo nivel ni saber si el ahorro de 3.9 GiB
compensa los fallos a SSD. El diseno historico de dos depositos no es base valida: a K=75 tenia
errores y su generacion medida (~30.6 t/s) queda muy por debajo de la cache actual.

## Partes del plan aun no implementadas

- `hybrid` se acepta como nombre, pero ejecuta la misma ruta que `cache`; no hay rama CPU/GPU.
- No existen todavia `auto`, calibracion del equipo, presupuesto automatico de VRAM/RAM ni
  fallback por presion de memoria.
- No hay doble buffer asincrono entre RAM y VRAM ni solapamiento explicito con compute.
- No hay telemetria de aciertos/fallos de la cache GPU exportada al host.
- No hay soporte de producto, toggle, migracion ni exposicion a usuarios, de forma deliberada.

---

# Recuperacion de prefill con K8 — 24 de agosto de 2026

Objetivo corregido por la medicion del usuario: conservar la ventaja de generacion de la cache
minima frente a `ncmoe 24`, pero recuperar su prompt processing sin volver a gastar ~6 GiB.

## Piso de VRAM y cuello de botella

- K8 es el minimo exacto del modelo porque activa 8 expertos por token.
- K8 en las 40 capas usa ~1.99 GiB y da `30.74 +/- 0.18` tg128.
- El pp256 se queda en ~213 y no cambia al aumentar K: un lote de 256 tokens toca los 256
  expertos de cada capa, por lo que el prefill no puede beneficiarse de la localidad de la LRU.
- Microbatching, variar el tamano del grupo de staging y forzar los MoE a CPU fueron regresiones.

## Banco CPU y prefetch solapado

El banco completo puede pertenecer realmente a CPU durante el modo experimental. Asi el
scheduler reconoce transferencias CPU -> Metal y usa su cola secundaria de prefetch. Cuatro
slots de 136 MiB son el punto de saturacion:

| prefetch slots | pp256 |
|---:|---:|
| 2 | 250.76 |
| 3 | 248.14 |
| **4** | **296.97** |
| 6 | 294.35 |
| 8 | 293.98 |

Es +39% contra el K8 anterior, pero transferir todos los bancos (~10 GiB por prompt) fija un
techo fisico cercano a 300 pp en este enlace. Mas colas no lo rompen.

## Punto equilibrado encontrado: 8 capas residentes + K8 en 32

Se dejan completas en VRAM las capas 32-39 y se conserva K8 en las primeras 32. Los bancos
K8 usan el prefetch anterior. Mismo modelo, binario, `pp256/tg128`, tres repeticiones:

| modo | VRAM durante PP | pp256 | tg128 |
|---|---:|---:|---:|
| K8, 40 capas | ~2.78 GiB con 4 slots de prefetch | **299.28 +/- 1.09** | **32.44 +/- 0.45** |
| **8 residentes + K8 en 32** | **~4.67 GiB** | **346.56 +/- 1.57** | **37.14 +/- 0.54** |
| `ncmoe 24`, control local | ~6 GiB | 352.98 +/- 5.09 | ~22-24.5 |

El punto nuevo recupera ~98.2% del PP de `ncmoe 24` medido en el mismo entorno, usa alrededor
de 1.3 GiB menos y mejora TG en ~52% contra la referencia de 24.5. La referencia caliente del usuario (~600 PP) no se
reproduce en este arnes local; debe compararse caliente contra caliente antes de reclamarla.

La huella medida del punto equilibrado es:

- modelo Metal privado: 3918.06 MiB;
- KV + estado recurrente: 67.81 MiB;
- compute de pp256: 246.50 MiB;
- cuatro slots de prefetch: 544 MiB;
- total efectivo aproximado: 4776 MiB (4.66 GiB).

## Correctitud y seguridad

- PPL de un chunk de 512: `4.4753 +/- 0.62704` tanto en `ncmoe 24` como en el modo nuevo.
- Salida determinista de 64 tokens: 249 bytes identicos, comparacion byte por byte contra
  `ncmoe 24`.
- La prueba LRU: 20 064 pasos, 1 926 144 consultas, sin crecimiento de memoria.
- El banco CPU no se precarga en las ranuras: el primer token llena K8. No cambia pp256 ni
  tg128 medidos.
- La advertencia `ggml_metal_buffer_get_id ... buffer is nil` no era un falso positivo. El
  buffer CPU canonico no tiene puntero de dispositivo y el helper lo confundia con Metal; el
  kernel de fetch recibia `nil` durante el warmup y podia marcar una ranura sin copiarla.
  Corregido detectando `ggml_backend_buffer_is_host`: cero avisos, PPL y texto exactos.
- `0070` aplica limpiamente despues de toda la serie de parches.
- Sigue oculto: `TOSH_ENABLE_DYNAMIC_MOE=OFF`, sin toggle ni cambios de interfaz.

## Lo que falta para mantener ~2 GiB y alcanzar ~600 PP

No se resuelve con otro valor de K ni con mas prefetch. Requiere la pieza FreeToken que aun no
existe: dividir los expertos de una misma capa entre CPU y GPU, ejecutar ambas ramas en paralelo
y sumar la salida exacta. Sin esa ejecucion concurrente, el limite es elegir entre trafico PCIe
(~299 PP a K8) o mas residencia (~347 PP con 8 capas completas).

## Prototipo intra-capa CPU/GPU descartado — 24 de agosto de 2026

Se implemento temporalmente un prototipo oculto para medir la pieza anterior antes de
incorporarla a la serie de parches. La prueba:

- partia los expertos por un umbral `Q` dentro de cada capa;
- transferia a Metal solo el prefijo de `Q` expertos;
- enviaba el complemento al banco CPU;
- usaba un sentinel para no calcular en CPU las rutas asignadas a GPU;
- sumaba ambas salidas con mascaras derivadas del ruteo;
- probo tambien lanzar la rama CPU en un worker mientras Metal procesaba la suya.

Qwen3.6-35B-A3B Q2_K_XL, `pp256`, `mlock`, K8 y cuatro slots de prefetch:

| ruta | pp256 |
|---|---:|
| K8 normal restaurado | **297.84** |
| hibrido Q64 | 22.93 |
| hibrido Q128 | 22.10 |
| hibrido Q192 | **42.26** |
| hibrido Q224 | 41.25 |
| hibrido Q240 | 40.45 |
| hibrido Q248 | 38.71 |

La ruta no era correcta: el control PPL de K8 dio `4.4525 +/- 0.62416`, mientras el prototipo
hibrido termino en `PPL = nan`, con y sin el intento de concurrencia. El prototipo se retiro por
completo y el binario se recompilo; K8 volvio a `297.84 pp256`, sin variables, kernels sentinel
ni cambios de scheduler residuales.

La medicion demuestra que no basta con duplicar `MUL_MAT_ID` y unirlo dentro del scheduler
actual. Esa forma introduce tres fronteras CPU/Metal por capa y no posee una barrera de union
que represente correctamente las dependencias. La siguiente implementacion debe usar una
primitiva MoE hibrida de capa completa: empaquetado de rutas y pesos por backend, buffers de
salida separados, ejecucion concurrente propietaria y una unica union explicita al final de la
capa. Hasta que esa primitiva conserve PPL exacta, no se debe optimizar ni exponer.

## Barrido K114 correcto — 24 de agosto de 2026

Las primeras pruebas K114 que combinaban el modo con `-ncmoe` no activaban la cache dinamica:
los expertos quedaban en el buffer CPU canonico y no se creaban ranuras. Se descartaron. El
barrido valido usa el override Metal-host de los tensores MoE, banco CPU y cuatro ranuras de
prefetch, igual que las mediciones K8 anteriores.

| modo | VRAM efectiva aprox. | pp256 | tg128 |
|---|---:|---:|---:|
| K8, 40 capas | ~2.78 GiB | 299.28 +/- 1.09 | 32.44 +/- 0.45 |
| **K114, 40 capas** | **~6.98 GiB** | **296.47 +/- 1.38** | **50.51 +/- 2.17** |
| 8 residentes + K8 en 32 | ~4.67 GiB | 346.56 +/- 1.57 | 37.14 +/- 0.54 |
| **8 residentes + K114 en 32** | **~7.88 GiB** | **341.07 +/- 3.84** | **52.80 +/- 1.98** |

K114 aumenta TG un 55.7% en las 40 capas dinamicas y un 42.2% con ocho residentes, pero no
mejora PP: queda un 0.9% y 1.6% por debajo de K8, respectivamente. En pp256 se usan tantos
expertos que la ruta transfiere bancos completos; la capacidad LRU no evita ese trafico. En TG,
en cambio, hay localidad entre tokens y las 114 ranuras reducen fallos de cache.

La mejora TG no cumple el objetivo de menor VRAM que `ncmoe 24`: K114 puro ronda 6.98 GiB y
K114 con residentes 7.88 GiB. Por tanto K114 sirve para confirmar el comportamiento, no como
configuracion equilibrada. K8 + ocho capas residentes sigue siendo el mejor punto medido.

## PFlash pospuesto para exploracion posterior

Se identifico PFlash como la tecnica recordada para acelerar prompts mediante un GGUF auxiliar.
El equipo ya contiene `Qwen3-0.6B-BF16.gguf` (1.11 GiB), medido en Metal a 1943.83 pp256 y
2409.73 pp4096. El objetivo Qwen3.6-35B K8 puro midio 588.84 pp4096.

PFlash no mejora el PP exacto del objetivo: selecciona y elimina partes del prompt antes del
prefill, por lo que cambia la entrada y puede perder instrucciones, codigo o esquemas de
herramientas. Se deja explicitamente fuera de esta fase. Podra explorarse despues como modo
separado, opcional y solo para contexto largo; el trabajo actual continua buscando acelerar K8
puro con el prompt completo y resultados exactos.

Un barrido adicional de `GGML_METAL_NCB=4,8,12,16`, manteniendo K8 y cuatro slots de prefetch,
dio 297.14, 297.69, 297.95 y 297.58 pp256. La profundidad de command buffers no rompe el techo
actual y no justifica ningun cambio.

## Bancos mayores que el working set de Metal — 24 de agosto de 2026

La app publicada descubrio una asercion al probar GLM-4.7-Flash REAP Q4: el acumulado de buffers host visibles para Metal se limitaba a `7/8 × recommendedMaxWorkingSetSize`, como si esas paginas de RAM fueran VRAM privada. Esto contradecia el experimento `rset.swift`, que ya habia demostrado que 20 GiB de host pueden quedar direccionables en una RX 6700 XT mientras la ventana leida permanezca acotada.

Se conservo y completo el ajuste que estaba solo en el arbol generado: cada asignacion CPU estable se envuelve una vez, sus tensores reutilizan el mismo recurso y el limite estructural pasa a ser `maxBufferLength`. `GGML_METAL_HOST_WRAP_LIMIT_MB` queda como override manual de laboratorio, no como presupuesto derivado de VRAM. La serie `0070` se reconstruyo desde el commit upstream limpio y aplico sin depender del vendor sucio.

| modelo y modo | banco de expertos aprox. | resultado corto |
|---|---:|---:|
| GLM-4.7-Flash REAP Q4, Dynamic K4 | 12.58 GiB | 55.75 pp256 / 0.95 tg32 |
| GLM-4.7-Flash REAP Q4, `ncmoe 8` | colocacion normal | 112.10 pp16 / 40.56 tg16 |
| Qwen3.6-35B Q4_K_S, Dynamic K8 | 16.88 GiB | 53.66-57.38 pp16 / 0.20 tg32; carga correcta |
| Qwen3.6-35B Q2_K_XL, Dynamic K8 | 9.61 GiB | 297.75 pp256 / 29.92 tg32 |

Qwen Q4 alcanzo ~18.1 GiB de RSS y el swap del sistema solo paso de 260 a 263.5 MiB durante la carga corta. Por tanto, la hipotesis de presion de RAM queda descartada para esta ejecucion. Dividir el mapeo unico de 16.88 GiB en recursos por tensor tampoco cambio el resultado: 53.66 pp16 / 0.20 tg4. El perfil interno midio esperas de ~8.9 s por dos pasos en la frontera de salida, frente a ~0.11 s de trabajo GPU registrado; el bloqueo esta dentro de la ruta Metal que mantiene direccionable el banco grande, no en swap. GLM tampoco mejora al subir K: K8 y K16 permanecen cerca de 0.95 tg, y K24 cae a 0.32. La conclusion medida es que el cierre por capacidad quedo resuelto, pero los bancos Q4 grandes necesitan una ventana host acotada que exponga y transfiera solo los expertos seleccionados. No se debe confundir "carga" con "configuracion recomendada".

Como proteccion temporal, Auto vuelve a la ejecucion normal si el banco de expertos estimado supera el working set de la GPU seleccionada. Manual sigue disponible detras de `TOSH_MOE_UI=1` para pruebas controladas; la proteccion no se presenta como una solucion de rendimiento.

## Staging host acotado para decode — prototipo interno

Se implemento la primera version secuencial de la Fase 5 detras de `TOSH_MOE_BOUNDED_STAGE=1`, sin panel publico y fuera de Auto. El grafo se corta despues de cada router, copia a CPU solamente sus IDs I32, aplica el LRU ya probado, empaqueta en un buffer shared persistente solo las filas de los misses y hace blit a los slots privados. El matmul recibe IDs remapeados y nunca ve un `MTLBuffer` que represente el banco completo.

Los buffers persistentes adicionales son acotados por una capa: para Qwen Q4 K8 el staging de expertos es aproximadamente `K/E` del banco de una capa, unos 13.5 MiB, mas dos buffers de IDs de pocos KiB. No duplica los 16.88 GiB del banco RAM ni incrementa el swap de forma material. La RAM principal sigue siendo el GGUF completo y la VRAM principal sigue siendo la cache K.

| ruta, Qwen3.6-35B-A3B Q4_K_S K8 | PP | TG |
|---|---:|---:|
| banco host completo, medicion anterior | 53.66-57.38 pp16 | 0.20 tg32 |
| staging acotado inicial, sin LRU persistente | 6.15 pp1 | 6.10 tg8 |
| staging acotado + LRU + buffers persistentes | 13.70 pp1 | 9.22 tg16 |
| binario empaquetado en `dist/ToshLLM.app` | 10.73 pp1 | 6.78 tg4 |
| fallback de prefill ancho con el flag activo | 57.35 pp16 | — |

La mejora contra el bloqueo Q4 es de aproximadamente 46 veces en la medicion TG16, pero no cumple todavia el objetivo de acercarse a `ncmoe 24`. El limite actual es la sincronizacion CPU/GPU por cada una de las 40 capas. El prefill ancho conserva la ruta anterior y no regresa: pp16 queda dentro del rango historico.

La correccion se valido numericamente con Qwen Q2, contexto 16 y batch/ubatch 1. La ruta directa y la ruta acotada produjeron exactamente `PPL = 8.9828 +/- 9.52892`. La ruta acotada tardo 1.30 s y la directa 0.58 s, confirmando que debe seleccionarse solo cuando el banco completo provoca el bloqueo del driver.

## Alternativas exactas sin capas residentes

Se probaron tres caminos adicionales manteniendo el prompt y los pesos exactos:

| alternativa | resultado pp256 | veredicto |
|---|---:|---|
| staging de solo expertos seleccionados, sin prefetch | 215.64 +/- 7.82 | descartar; pierde el solape |
| 8/16/24/32 capas CPU + K8 en el resto | 296.44-298.41 | neutro; termina limitado por la misma copia |
| dos colas H2D Metal en paralelo | 12.48 GB/s | no supera una cola: 12.68 GB/s |

El staging selectivo fue un prototipo temporal: un threadgroup por experto comprobaba los IDs y
copiaba solo la fila usada. Fue correcto estructuralmente, pero la copia depende del router de
esa misma capa y no puede anticiparse; queda muy por debajo del prefetch lineal de ~298 pp. El
kernel y toda la telemetria temporal se retiraron y el binario se recompilo limpio.

La compresion sin perdida tampoco es viable. Zstd nivel 1 sobre cuatro regiones de 256 MiB del
GGUF Q2 dejo 99.47-99.64% del tamano original: el costo de descompresion no puede compensar un
ahorro inferior a 0.6%.

La frontera queda mas estrecha: una sola cola ya satura el enlace y los bancos Q2 son
incompresibles. Para superar ~298 pp sin residencia hay que solapar trabajo util con la copia
dentro de la capa, no ajustar colas. La siguiente ruta exacta es un operador MoE por franjas:
doble buffer de K expertos, transferencia de la franja siguiente mientras Metal calcula la
actual y acumulacion unica al final. Debe conservar PPL antes de cualquier optimizacion o
exposicion.

## Operador por franjas y doble buffer: correcto, pero descartado

Se implemento el operador exacto propuesto, oculto y sin interfaz de usuario. Dividia los 256
expertos en franjas K8, conservaba los indices originales de cada asignacion y escribia cada
salida una sola vez. Una segunda variante alternaba dos buffers K8: una cola Metal secundaria
cargaba la franja siguiente mientras la cola principal calculaba la actual, sincronizadas con
eventos GPU. El segundo buffer reutilizaba el scratch que ya existe para prefill, por lo que no
anadia otra reserva material de VRAM.

Qwen3.6-35B-A3B Q2_K_XL, `ncmoe 1`, `mlock`, pp256:

| ruta | prefetch del scheduler | pp256 |
|---|---:|---:|
| K8 normal | 4 | **295.37** |
| franjas K8 seriales | 0 | 203.41 |
| franjas K8 con doble buffer propio | 0 | 205.32 |

El doble buffer mejoro solo 1.91 pp (+0.9%) sobre las franjas seriales y quedo 30.5% por debajo
del K8 normal. La ruta fue correcta: K8 normal y doble buffer produjeron el mismo hash exacto
de logits (`a25e08713286e1af`) sobre el lote pp256. El problema es de rendimiento, no de
precision: 15 despachos por banco, mapas por franja y barreras A/B cuestan mas de lo que se
oculta, mientras copia y matmul siguen compitiendo en la misma GPU.

El prototipo, sus kernels, eventos y diagnostico de hash se retiraron. El motor se recompilo
con la ruta K8 anterior.

## Barrido complementario de prefetch en K8 normal

Se midieron valores no cubiertos por el barrido anterior, manteniendo `ncmoe 1`, `mlock`,
NCB8, pp256 y tres repeticiones:

| slots de prefetch | pp256 |
|---:|---:|
| 0 | 214.78 +/- 8.59 |
| 1 | 269.29 +/- 26.06 |
| **4 (control)** | **297.19 +/- 0.60** |
| 5 | 295.31 +/- 0.59 |
| 7 | 296.22 +/- 1.31 |
| 10 | 293.62 +/- 0.97 |
| 12 | 293.61 +/- 0.42 |
| 16 | 293.85 +/- 0.18 |

Junto con los valores ya medidos 2, 3, 6 y 8, el optimo sigue siendo cuatro. Menos de cuatro
no mantiene suficiente trabajo anticipado; por encima de cuatro el enlace ya esta saturado y
la administracion adicional resta aproximadamente 1-1.2% sin reducir VRAM.
