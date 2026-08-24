# Plan de integración progresiva de Dynamic MoE en ToshLLM

## 1. Objetivo

Implementar en ToshLLM un sistema experimental de ejecución dinámica para modelos Mixture of Experts, manteniendo:

- llama.cpp como motor principal.
- GGUF como formato de modelo.
- Compatibilidad con los modelos existentes.
- Compatibilidad con el sistema actual de offload.
- Soporte para Metal y GPU AMD.
- La ruta original de llama.cpp intacta.
- Un toggle experimental desactivado por defecto.
- Fallback automático al modo tradicional.

## 2. Resultado esperado

El sistema deberá utilizar los recursos del equipo como una plataforma conjunta:

```text
RAM:
Contiene el conjunto completo de expertos.

VRAM:
Contiene pesos no-MoE y una caché dinámica de expertos.

CPU:
Ejecuta algunos expertos ausentes cuando resulte más rápido que transferirlos.

GPU:
Ejecuta expertos residentes y expertos transferidos a la caché.

PCIe:
Transfiere únicamente los expertos necesarios.
```

La nueva modalidad deberá beneficiar principalmente a modelos MoE que no entran completamente en VRAM.

---

## 3. Principios de compatibilidad

La implementación deberá respetar las siguientes reglas.

### 3.1 El modo tradicional seguirá siendo el predeterminado

```text
Dynamic MoE desactivado:
llama.cpp funciona exactamente como antes.
```

No deberá modificarse:

- La asignación tradicional de tensores.
- `--n-gpu-layers`.
- `--n-cpu-moe`.
- El split entre múltiples GPU.
- La carga habitual de GGUF.
- La generación con modelos densos.

### 3.2 No modificar obligatoriamente GGUF

Los modelos existentes deberán funcionar sin conversión.

No se requerirá:

- Un formato de modelo nuevo.
- Cambiar metadatos del GGUF.
- Descargar pesos adicionales.
- Convertir previamente el modelo.

En una fase posterior podrá existir un archivo lateral opcional:

```text
modelo.gguf
modelo.tosh-moe-cache
```

Este archivo contendría expertos reorganizados para transferencias rápidas, pero nunca será obligatorio.

### 3.3 La RAM será la fuente de verdad

Todos los expertos deberán permanecer accesibles en RAM.

La caché de VRAM será únicamente una optimización. Expulsar un experto de la GPU no podrá causar pérdida de datos ni requerir recargarlo desde el disco.

### 3.4 Fallback obligatorio

Si el modo dinámico no es compatible, ToshLLM deberá volver al sistema tradicional.

Motivos posibles:

- Arquitectura no soportada.
- Cuantización no soportada.
- Memoria insuficiente.
- Error de asignación Metal.
- Error durante la calibración.
- Resultado numérico inválido.
- Fallo del kernel experimental.
- GPU no compatible.
- Configuración multi-GPU todavía no soportada.

---

## 4. Activación experimental

### Interfaz de ToshLLM

Añadir un toggle:

```text
Dynamic MoE experimental
```

Descripción:

```text
Utiliza la RAM, CPU y VRAM como una caché dinámica de expertos.
Puede mejorar modelos MoE mayores que la VRAM disponible.
No afecta modelos densos.
```

Estado predeterminado:

```text
Desactivado
```

### Acceso privado en la build pública

La build normal compila el motor con:

```text
TOSH_ENABLE_DYNAMIC_MOE=ON
```

Este flag solo incluye el código; sin variables de ejecución, `tosh_moe_mode()` devuelve
`off` y llama.cpp conserva su camino tradicional. El panel tampoco se muestra normalmente.

Para revelar la configuración experimental se escribe en **Argumentos extra**:

```text
TOSH_MOE_UI=1
```

El flag de acceso no activa por sí solo Dynamic MoE. Después hay que encender el toggle del
panel. Al hacerlo, servidor y Benchmarks comparten automáticamente la receta medida:

```text
TOSH_MOE_MODE=cache
TOSH_MOE_SLOTS=8            # configurable: 8, 16, 32, 64, 114
TOSH_MOE_CPU_BANK=1
GGML_SCHED_PREFETCH_EXPERTS=4  # configurable en el panel
GGML_METAL_NCB=8
--n-cpu-moe 1
--load-mode mlock
-ot \.ffn_(up|down|gate|gate_up)_(ch|)exps=MTL0
```

Quitar `TOSH_MOE_UI=1` o apagar el toggle elimina la receta experimental y devuelve el mismo
binario al camino normal. Router/multi-modelo queda excluido porque su preset todavía no puede
transportar el override por tensor. El panel privado ofrece `auto` y `cache` manual. La rama
`hybrid` CPU/GPU real sigue pendiente y no se expone.

En `auto`, ToshLLM selecciona la ruta antes de iniciar el proceso. Si el modelo ya cabe en la
VRAM útil, usa llama.cpp normal. Si es un MoE que no cabe, hay GPU dedicada compatible y la RAM
total puede alojar el banco con margen, activa la receta medida K8/prefetch4. Ante modelo denso,
RAM insuficiente, GPU no compatible, router o multi-GPU, conserva la ruta normal.

Dynamic MoE no calcula un número de "capas de expertos" equivalente a `ncmoe`. Participan todas
las capas MoE y cada capa dispone de K ranuras reutilizables en VRAM; el banco completo permanece
en RAM. El `--n-cpu-moe 1` de la receta es un requisito de colocación del cargador para habilitar
la fuente en RAM, no significa que solo una capa use Dynamic MoE.

### Estados visibles

Cuando se active, ToshLLM deberá mostrar uno de estos estados:

```text
Analizando modelo
Calibrando hardware
Dynamic MoE activo
Solo caché GPU
Ejecución híbrida CPU/GPU
Modo tradicional por compatibilidad
Desactivado por error
```

### Configuración interna

Para no interferir con los argumentos oficiales de llama.cpp, la primera implementación puede utilizar variables de entorno:

```text
TOSH_MOE_MODE=off
TOSH_MOE_MODE=trace
TOSH_MOE_MODE=cache
TOSH_MOE_MODE=hybrid
TOSH_MOE_MODE=auto
```

Opciones complementarias:

```text
TOSH_MOE_CACHE_MB=auto
TOSH_MOE_CPU_SHARE=auto
TOSH_MOE_PREFILL_PIPELINE=0
TOSH_MOE_TRACE=0
TOSH_MOE_VALIDATE=0
```

La aplicación Swift establecerá estas variables al iniciar `llama-server`.

### Modos internos

| Modo | Función |
|---|---|
| `off` | llama.cpp tradicional |
| `trace` | Registra expertos sin cambiar la ejecución |
| `cache` | Caché dinámica en VRAM, sin rama CPU/GPU concurrente |
| `hybrid` | Caché y ejecución simultánea CPU/GPU |
| `auto` | Política de la aplicación: selecciona `off` o la receta `cache` K8/prefetch4 |

El toggle de la interfaz activará inicialmente:

```text
TOSH_MOE_MODE=auto
```

Los demás modos podrán quedar disponibles solamente en ajustes avanzados.

---

# 5. Arquitectura propuesta

## 5.1 Componentes internos

Crear componentes aislados del resto de llama.cpp:

```text
ToshMoeModelInspector
ToshMoeHardwareProfiler
ToshMoeTraceRecorder
ToshMoeCacheSimulator
ToshMoeCacheManager
ToshMoeScheduler
ToshMoeCpuExecutor
ToshMoeMetalTransfer
ToshMoeValidator
ToshMoeFallbackManager
```

### ToshMoeModelInspector

Responsabilidades:

- Detectar si el modelo es MoE.
- Identificar cantidad de capas MoE.
- Identificar expertos por capa.
- Identificar expertos activos por token.
- Identificar tensores de cada experto.
- Detectar cuantización.
- Calcular tamaño completo de cada experto.
- Verificar si la arquitectura está soportada.

### ToshMoeHardwareProfiler

Responsabilidades:

- Medir RAM → VRAM.
- Medir VRAM → RAM.
- Medir kernels MoE en CPU.
- Medir kernels MoE en GPU.
- Determinar capacidad segura de caché.
- Crear un perfil por equipo.

### ToshMoeCacheManager

Responsabilidades:

- Mantener slots estables en VRAM.
- Asociar `(capa, experto)` con un slot.
- Implementar LRU.
- Reservar víctimas.
- Registrar aciertos y fallos.
- Transferir expertos completos.
- Evitar expulsar expertos en uso.

### ToshMoeScheduler

Responsabilidades:

- Recibir los expertos seleccionados por el router.
- Consultar cuáles están residentes.
- Separar hits y misses.
- Elegir qué misses transferir.
- Elegir cuáles ejecutar en CPU.
- Sincronizar y fusionar resultados.

---

## 5.2 Identificación de expertos

Cada experto deberá identificarse lógicamente mediante:

```cpp
struct ToshMoeExpertId {
    int32_t layer;
    int32_t expert;
};
```

La ubicación dinámica podrá representarse como:

```cpp
struct ToshMoeExpertLocation {
    int32_t layer;
    int32_t expert;
    int32_t gpu_slot;
    uint64_t last_used;
    bool resident;
    bool transfer_pending;
    bool executing;
};
```

Un slot deberá contener todos los tensores necesarios:

```text
gate projection
up projection
down projection
escalas de cuantización
metadatos de bloques
```

No deberán mantenerse partes incompletas de un experto salvo que una cuantización lo requiera expresamente.

---

# 6. Plan de implementación por fases

## Fase 0: aislar la implementación

### Objetivo

Preparar una base que pueda actualizarse junto con llama.cpp sin convertir cada actualización upstream en un conflicto grande.

### Tareas

1. Registrar el commit exacto de llama.cpp utilizado por ToshLLM.
2. Crear una serie separada de parches:

```text
patches/llama/experimental-moe/
```

3. Utilizar un flag de compilación:

```cpp
TOSH_ENABLE_DYNAMIC_MOE
```

4. Compilar dos configuraciones en CI:

```text
TOSH_ENABLE_DYNAMIC_MOE=OFF
TOSH_ENABLE_DYNAMIC_MOE=ON
```

5. Mantener todo el código experimental detrás del flag.
6. Evitar modificar operaciones existentes cuando pueda crearse una ruta alternativa.
7. Añadir pruebas que confirmen que `TOSH_MOE_MODE=off` utiliza la ruta original.

### Criterio de salida

- llama.cpp compila con y sin la funcionalidad.
- El modo desactivado mantiene los mismos resultados.
- El rendimiento del modo desactivado no cambia más de 1 %.
- Los modelos densos no entran en el código Dynamic MoE.

---

## Fase 1: inspección y trazado del router

### Objetivo

Comprobar que los modelos usados por Tosh presentan suficiente reutilización de expertos.

### Tareas

1. Interceptar la salida del router MoE.
2. Registrar los expertos `top-k` seleccionados.
3. No cambiar la ubicación ni la ejecución de los pesos.
4. Crear una traza compacta:

```text
token
capa
expertos seleccionados
pesos del router
duración de la capa
```

5. Añadir información del modelo:

```text
arquitectura
cuantización
cantidad de expertos
expertos activos
tamaño por experto
```

6. Limitar las trazas para evitar archivos gigantes.
7. Permitir activarlas mediante:

```text
TOSH_MOE_MODE=trace
```

### Modelos iniciales

Probar al menos:

- Qwen MoE.
- GPT-OSS.
- Un modelo MoE pequeño para validar corrección.
- Un modelo MoE que exceda claramente la VRAM.

### Entregable

Un archivo como:

```text
tosh-moe-trace.jsonl
```

### Criterio de salida

Las trazas deben permitir reconstruir exactamente qué expertos fueron utilizados en cada capa y token.

---

## Fase 2: simulador LRU offline

### Objetivo

Determinar si la caché dinámica ofrece ventaja antes de modificar Metal.

### Tareas

Reproducir las trazas con capacidades simuladas:

```text
256 MB
512 MB
1 GB
2 GB
4 GB
6 GB
8 GB
12 GB
```

Comparar:

- Distribución estática de llama.cpp.
- LRU global.
- LRU por capa.
- LRU ponderado por tamaño.
- LRU con protección temporal.
- Expertos fijos más caché dinámica.

### Métricas

```text
Tasa de aciertos
Fallos por token
Bytes transferidos
Evicciones
Reutilización entre tokens
Reutilización entre turnos
Working set por capa
```

### Criterio para continuar

Continuar si la caché dinámica produce al menos una de estas mejoras:

- 20 % menos bytes transferidos.
- 20 % menos fallos que la colocación estática.
- Working set estable que quepa parcialmente en VRAM.
- Reutilización clara entre tokens consecutivos.

Si no existe localidad suficiente, no debe construirse todavía la caché Metal para esa arquitectura.

---

## Fase 3: calibración del hardware

### Objetivo

Tomar decisiones utilizando el rendimiento real del equipo.

### Tareas

Crear un benchmark interno que mida:

```text
RAM → VRAM mediante Metal
VRAM → RAM
RAM disponible
VRAM disponible
Mayor asignación Metal segura
Rendimiento CPU por cuantización
Rendimiento GPU por cuantización
Latencia de transferir un experto
```

Calibrar utilizando formas reales de los expertos, no buffers genéricos pequeños.

### Política inicial

Calcular:

\[
q^\star \approx m \frac{B_P}{B_H}
\]

Donde:

```text
m   = expertos ausentes
BP  = ancho de banda efectivo RAM → VRAM
BH  = ancho de banda efectivo del kernel MoE en CPU
q*  = expertos que deben transferirse
```

Aplicar límites:

```text
0 ≤ q* ≤ m
```

Mientras la caché esté fría, transferir al menos un experto cuando haya espacio.

### Persistencia del perfil

Guardar el resultado utilizando una clave compuesta:

```text
GPU
CPU
versión de macOS
backend
commit del motor
modelo
cuantización
```

Invalidar el perfil si cambia cualquiera de esos elementos.

### Criterio de salida

El perfil debe predecir correctamente si resulta más rápido:

- Ejecutar en CPU.
- Transferir a GPU.
- Combinar ambas rutas.

---

## Fase 4: caché lógica sin concurrencia

### Objetivo

Validar la gestión de slots antes de implementar transferencias solapadas.

### Tareas

1. Crear un conjunto fijo de slots.
2. Implementar LRU.
3. Ejecutar las decisiones de manera secuencial.
4. Bloquear slots mientras estén en uso.
5. Mantener el conjunto completo en RAM.
6. Registrar cada asignación y expulsión.
7. Probar miles de cambios sintéticos de expertos.
8. Añadir detección de estados inválidos.

### Invariantes obligatorias

```text
Un experto residente tiene exactamente un slot.
Un slot contiene como máximo un experto.
Un slot en ejecución no puede expulsarse.
Un experto transferido debe contener todos sus tensores.
La tabla CPU y la tabla GPU deben coincidir.
```

### Criterio de salida

- No existen colisiones de slots.
- No se usan pesos incorrectos.
- La simulación soporta sesiones largas.
- No hay crecimiento progresivo de memoria.

---

## Fase 5: caché Metal durante decode

### Objetivo

Ejecutar expertos desde una caché real de VRAM.

### Alcance inicial

```text
Prefill:
ruta tradicional

Decode:
caché dinámica experimental

Concurrencia CPU/GPU:
todavía desactivada
```

### Diseño

1. Reservar un `MTLBuffer` privado para slots.
2. Mantener un buffer de staging compatible con la GPU.
3. Copiar expertos completos mediante blit.
4. Sincronizar la copia antes de ejecutar el experto.
5. Remapear el ID lógico al slot físico.
6. Ejecutar el experto desde el slot.
7. Actualizar la recencia solamente después de completar la operación.

### Integración con el grafo

La ruta MoE actual probablemente utiliza operaciones equivalentes a `mul_mat_id`.

No debe modificarse el comportamiento original. Crear una ruta alternativa:

```text
Router
↓
Clasificación de hits y misses
↓
Remapeo experto lógico → slot
↓
Ejecución desde caché
↓
Combinación de resultados
```

El primer prototipo puede dividir la ejecución después del router y sincronizar con CPU. Será más lento, pero permitirá comprobar la corrección.

La versión optimizada deberá evitar una sincronización CPU por cada capa. Para ello podrá utilizar una operación específica del backend:

```text
TOSH_MOE_CACHE_ROUTE
TOSH_MOE_CACHE_COPY
TOSH_MOE_CACHED_MUL_MAT_ID
```

Estas operaciones deberán existir solamente cuando el modo experimental esté activo.

### Criterio de salida

- Resultados correctos.
- Sin lecturas fuera de rango.
- Sin corrupción de command buffers.
- Caché estable durante al menos 1.000 tokens.
- Mejora medible frente a ejecutar todos los expertos en CPU.

---

## Fase 6: ejecución híbrida CPU/GPU

### Objetivo

Ejecutar simultáneamente los misses entre CPU y GPU.

### Flujo

```text
Router produce expertos activos
            |
            v
     Consultar caché
       /          \
     Hits        Misses
      |          /    \
     GPU    Transferir  CPU
              a GPU
       \          |     /
        \         |    /
         Fusionar resultados
```

### Tareas

1. Separar los expertos ausentes en dos grupos:

```text
Grupo F:
transferir y ejecutar en GPU

Grupo C:
ejecutar directamente en CPU
```

2. Calcular `q*` usando el perfil del hardware.
3. Ejecutar la rama CPU con un pool persistente de workers.
4. Ejecutar transferencias y GPU sin bloquear la rama CPU.
5. Aplicar los pesos del router.
6. Fusionar las salidas.
7. Preservar el orden numérico siempre que sea posible.

### Validación numérica

Comparar contra el modo tradicional:

```text
Error absoluto máximo
Error absoluto medio
RMSE
NaN
Inf
Diferencia de logits
Tokens greedy generados
```

Para generación greedy, exigir que la secuencia resultante sea idéntica durante una prueba definida.

Para sampling, comparar logits porque pequeñas diferencias flotantes pueden cambiar el token seleccionado.

### Criterio de salida

- No hay pérdida de calidad.
- La rama CPU y GPU trabajan realmente en paralelo.
- La latencia expuesta se aproxima a la rama más lenta, no a la suma.
- `hybrid` supera a `cache` en hardware donde el perfil lo recomienda.

---

## Fase 7: prefill con doble buffer

### Objetivo

Ocultar la transferencia de expertos durante el procesamiento del prompt.

### Diseño

```text
Buffer A:
GPU calcula capa N

Buffer B:
transfiere expertos de capa N+1
```

Después se intercambian.

### Restricción importante para AMD

Esta fase debe tratarse como separada y de alto riesgo debido a los problemas conocidos de Tosh con command buffers concurrentes en GPU AMD.

Debe implementarse con:

- Eventos Metal explícitos.
- Dependencias entre command buffers.
- Protección contra reutilización temprana.
- Límites de concurrencia.
- Fallback secuencial.
- Validación específica por familia de GPU.

### Familias mínimas

```text
GCN4
GCN5
RDNA1
RDNA2
```

### Condición de activación

El pipeline solo debe activarse si existen slots suficientes para dos capas completas.

Si no existen:

```text
Usar prefill tradicional
```

### Criterio de salida

- No hay corrupción.
- No hay bloqueos.
- El watchdog de macOS no se activa.
- El prefill mejora al menos 10 %.
- Tres ejecuciones consecutivas producen los mismos resultados.

---

## Fase 8: memoria elástica

### Objetivo

Redistribuir VRAM entre KV cache y caché de expertos.

### Estrategia

```text
Contexto corto:
más slots de expertos

Contexto largo:
más KV cache
menos slots de expertos
```

### Puntos seguros de redimensionamiento

- Antes de una solicitud.
- Entre turnos.
- Después de completar una evaluación.
- Al ampliar el contexto.
- Al detectar presión de memoria.

Nunca redimensionar:

- Durante un kernel.
- Mientras haya una transferencia pendiente.
- Mientras existan slots bloqueados.
- En mitad de la evaluación de un token.

### Criterio de salida

- La caché cambia de tamaño sin recargar el modelo.
- No se pierde el contexto.
- No se corrompen expertos.
- La generación puede continuar después de redimensionar.

---

## Fase 9: soporte multi-GPU

### Objetivo

Extender Dynamic MoE a configuraciones con varias GPU.

No debe formar parte del primer lanzamiento.

### Opciones que deben evaluarse

```text
Caché independiente por GPU
Expertos asignados por GPU
Caché principal y caché secundaria
Asignación según ancho de banda PCIe
```

Cada GPU deberá calibrarse por separado:

```text
GPU 0 RAM → VRAM
GPU 1 RAM → VRAM
GPU 0 capacidad disponible
GPU 1 capacidad disponible
```

En equipos donde una GPU utiliza PCIe x16 y otra x4, no debe asumirse que ambas ofrecen el mismo coste de transferencia.

### Primera política recomendada

- Mantener atención y pesos no-MoE en la GPU principal.
- Usar la GPU secundaria como banco adicional de expertos.
- Asignar expertos según coste medido.
- Evitar transferencias GPU → GPU si Metal debe pasar primero por RAM.

---

## Fase 10: caché semántica de estados

### Objetivo

Evitar prefills completos cuando una aplicación edite herramientas, razonamiento o historial.

Esta fase es independiente de la caché de expertos.

### Posibles anchors

```text
Inicio de turno
Fin de turno
Inicio de tool call
Fin de tool call
Resultado de herramienta
Bloque de razonamiento
Compacción de contexto
```

Debe conservarse:

- KV cache válida.
- Estado recurrente válido.
- Posición exacta del anchor.
- Hash del prefijo.

Esta función deberá implementarse después de estabilizar Dynamic MoE.

---

# 7. Selección automática del modo

La primera política automática implementada sigue este proceso conservador:

```text
1. Confirmar que el archivo existe y es MoE mediante metadatos GGUF.
2. Rechazar temporalmente router y reparto multi-GPU.
3. Confirmar una GPU dedicada seleccionada.
4. Restar a la VRAM la reserva del usuario y 512 MiB para cómputo/KV.
5. Si el GGUF cabe en esa VRAM útil, conservar la ruta normal.
6. Exigir en RAM el tamaño del GGUF más un margen del mayor entre 25 % y 4 GiB.
7. Si se cumplen las condiciones, activar `cache` con K8/prefetch4; si no, conservar la ruta normal.
```

Ejemplo:

```text
Modelo: Qwen MoE
Tamaño GGUF: 11.44 GiB
VRAM nominal: 12 GiB
Reserva: 1 GiB + 512 MiB de ejecución
RAM: suficiente para fijar el banco
Modo seleccionado: cache K8/prefetch4
```

Si el modelo entra completamente en VRAM:

```text
Dynamic MoE no es necesario.
Se utilizará la ruta GPU tradicional.
```

La calibración por ancho de banda y la selección de un futuro modo híbrido permanecen como una
fase posterior. No forman parte de esta primera política `auto` para evitar elegir una ruta aún
no validada.

---

# 8. Sistema de fallback

## Fallback preventivo

Antes de comenzar:

```text
Modelo no soportado → modo tradicional
Cuantización no soportada → modo tradicional
Caché demasiado pequeña → modo tradicional
Calibración inválida → modo tradicional
```

## Fallback durante ejecución

Si ocurre un error:

1. Detener la evaluación en un punto seguro.
2. Marcar Dynamic MoE como desactivado para la sesión.
3. Liberar la caché experimental.
4. Reconstruir el grafo tradicional.
5. Conservar los pesos residentes en RAM.
6. Repetir el token fallido si es seguro.
7. Informar en el log.

Mensaje:

```text
Dynamic MoE disabled.
Reason: Metal cache synchronization failure.
Falling back to standard llama.cpp execution.
```

Si no es posible repetir el token con seguridad, reiniciar el motor y conservar la configuración para abrir nuevamente el modelo en modo tradicional.

---

# 9. Observabilidad

El log debe indicar:

```text
Dynamic MoE mode
Cache capacity
Occupied slots
Hit rate
Miss rate
Evictions
Bytes transferred
CPU experts executed
GPU experts executed
q* selected
RAM usage
VRAM usage
Transfer time
CPU branch time
GPU branch time
Merge time
Tokens per second
```

Ejemplo:

```text
[TOSH-MOE]
mode=hybrid
cache=6144MB
slots=384/420
hits=82.4%
misses=17.6%
gpu_fills=2
cpu_experts=4
transfer=3.2ms
cpu=4.8ms
gpu=4.1ms
token=5.0ms
```

Los logs detallados estarán desactivados por defecto.

---

# 10. Compatibilidad de modelos

## Primera etapa

Soportar una arquitectura por vez.

Orden recomendado:

1. Qwen MoE.
2. GPT-OSS.
3. DeepSeek MoE.
4. Otras arquitecturas soportadas por llama.cpp.

## Cuantizaciones

Empezar con una ruta de referencia simple y después ampliar:

1. F16 o BF16 para corrección.
2. Q4_0.
3. Q4_K.
4. Q5_K.
5. Q8_0.
6. MXFP4/MFPX4.
7. Otros formatos utilizados por Tosh.

Cada cuantización necesita:

- Cálculo correcto del tamaño del experto.
- Copia alineada.
- Kernel CPU compatible.
- Kernel Metal compatible.
- Prueba numérica independiente.

---

# 11. Matriz mínima de pruebas

| Modelo | Cuantización | VRAM | Modo |
|---|---|---:|---|
| MoE pequeño | F16 | Suficiente | Referencia |
| Qwen MoE | Q4_K | Insuficiente | Off |
| Qwen MoE | Q4_K | Insuficiente | Cache |
| Qwen MoE | Q4_K | Insuficiente | Hybrid |
| GPT-OSS | MXFP4 | Insuficiente | Off |
| GPT-OSS | MXFP4 | Insuficiente | Hybrid |
| Modelo denso | Q4_K | Cualquiera | Auto |
| Wan/video | Aplicable | Cualquiera | Off |

Los motores de imagen y video no deberán entrar en esta ruta.

### Hardware

Probar:

```text
AMD GCN4
AMD GCN5
AMD RDNA1
AMD RDNA2
Una sola GPU
Dos GPU
CPU only
Apple Silicon como fallback
```

---

# 12. Criterios de aceptación

## Compatibilidad

- El modo `off` utiliza la ruta original.
- Los modelos densos no cambian.
- GGUF existente funciona sin conversión.
- `--n-gpu-layers` sigue funcionando.
- `--n-cpu-moe` sigue funcionando cuando Dynamic MoE está desactivado.
- Las actualizaciones de llama.cpp pueden aplicarse sin reescribir todo el parche.

## Corrección

- No aparecen NaN ni Inf.
- No hay corrupción de pesos.
- Los logits permanecen dentro de la tolerancia definida.
- La generación greedy coincide con la referencia.
- No hay degradación comprobable de calidad.

## Estabilidad

- Tres ejecuciones consecutivas terminan correctamente.
- Una sesión de al menos 1.000 tokens no presenta fallos.
- No existe crecimiento progresivo de RAM o VRAM.
- El fallback funciona.
- No se activa el watchdog de macOS.

## Rendimiento

- Modo `off`: máximo 1 % de diferencia.
- Modo `trace`: máximo 3 % de penalización.
- Modo `cache`: mejora mínima de 10 % en escenarios compatibles.
- Modo `hybrid`: debe superar al modo tradicional cuando `auto` lo seleccione.
- El sistema no activará Dynamic MoE si estima una regresión.

---

# 13. Lanzamiento progresivo

## Etapa A: desarrollo interno

```text
Toggle oculto
Solo logs
Solo modo trace
```

## Etapa B: usuarios avanzados

```text
Toggle experimental visible
Cache decode-only
Fallback automático
Advertencia de inestabilidad
```

## Etapa C: beta

```text
Modo auto
Calibración automática
Cache e Hybrid
Perfiles persistentes
Telemetría local exportable
```

## Etapa D: estable

Solo después de validar:

- Varias generaciones de AMD.
- Varias cuantizaciones.
- Diferentes modelos MoE.
- Contextos largos.
- Presión de memoria.
- Multi-GPU.

El modo podrá continuar siendo opcional incluso después de considerarse estable.

---

# 14. Organización sugerida del código

```text
vendor/llama.cpp/
    common/
    ggml/
    ggml/src/ggml-metal/

patches/llama/experimental-moe/
    0001-moe-tracing.patch
    0002-moe-hardware-profiler.patch
    0003-moe-cache-manager.patch
    0004-moe-metal-cache.patch
    0005-moe-hybrid-execution.patch
    0006-moe-prefill-pipeline.patch
    0007-moe-elastic-memory.patch

Sources/
    DynamicMoESettings.swift
    DynamicMoEStatus.swift
    DynamicMoEProfile.swift
```

Si es posible, conservar la implementación nativa en archivos separados:

```text
tosh-moe.h
tosh-moe.cpp
tosh-moe-cache.cpp
tosh-moe-profile.cpp
tosh-moe-metal.mm
```

Los cambios en archivos upstream deberán limitarse a:

- Hooks de inicialización.
- Detección del modo.
- Selección de la ruta MoE.
- Registro de estadísticas.
- Fallback.

---

# 15. Riesgos principales

## Riesgo 1: sincronización por capa

Leer los IDs del router en CPU después de cada capa puede eliminar cualquier mejora.

Solución:

- Usarlo solamente en el prototipo.
- Mover posteriormente clasificación y remapeo a Metal.
- Mantener buffers de tamaño fijo.

## Riesgo 2: command buffers concurrentes en AMD

Solución:

- Empezar secuencialmente.
- Añadir concurrencia por fases.
- Usar eventos explícitos.
- Mantener fallback.
- Validar cada familia de GPU.

## Riesgo 3: demasiadas modificaciones en ggml

Solución:

- Añadir operaciones nuevas y aisladas.
- No modificar semántica de operaciones actuales.
- Mantener una serie pequeña de parches.
- Añadir CI contra actualizaciones upstream.

## Riesgo 4: cuantizaciones diferentes

Solución:

- Habilitar cada formato individualmente.
- No asumir que todos los expertos tienen el mismo layout.
- Rechazar formatos sin validación.

## Riesgo 5: la caché no mejora ciertos modelos

Solución:

- Medir trazas antes de activar.
- Guardar perfiles por arquitectura.
- Permitir que `auto` seleccione la ruta tradicional.

## Riesgo 6: diferencias numéricas

Solución:

- Conservar el orden de combinación.
- Comparar logits.
- Validar CPU, GPU y Hybrid contra referencia.
- Desactivar la arquitectura si excede tolerancias.

---

# 16. Orden definitivo de trabajo

```text
[x] Crear flag de compilación (incluido en la build normal, inerte en runtime).
[x] Crear toggle experimental (oculto tras `TOSH_MOE_UI=1`).
[x] Implementar TOSH_MOE_MODE=off.
[x] Confirmar ruta original sin regresiones.
[x] Detectar modelos MoE.
[ ] Registrar expertos seleccionados.
[ ] Crear trazas.
[ ] Construir simulador LRU.
[ ] Medir localidad real.
[ ] Implementar benchmark CPU/RAM/PCIe/GPU.
[ ] Crear perfiles persistentes.
[x] Implementar slots lógicos.
[x] Implementar caché Metal secuencial.
[x] Validar decode en la configuración de referencia.
[ ] Implementar política q*.
[ ] Implementar rama CPU.
[ ] Fusionar CPU y GPU.
[ ] Validar logits.
[x] Añadir selección preventiva/fallback normal en `auto`.
[x] Exponer de forma privada a usuarios avanzados (`TOSH_MOE_UI=1`).
[ ] Implementar doble buffer de prefill.
[ ] Implementar memoria elástica.
[ ] Añadir arquitecturas y cuantizaciones.
[ ] Evaluar multi-GPU.
[ ] Evaluar caché semántica.
```

---

# 17. Primer milestone recomendado

El primer milestone no debe intentar acelerar todavía.

Debe entregar:

```text
Toggle experimental
Detección de modelos MoE
Modo trace
Registro de expertos
Simulador LRU
Reporte de viabilidad
```

El reporte deberá responder:

1. ¿Existe localidad temporal suficiente?
2. ¿Qué porcentaje del pool necesita permanecer en VRAM?
3. ¿Cuántos bytes se transferirían por token?
4. ¿Qué modelos presentan mayor beneficio?
5. ¿Qué cuantizaciones deben implementarse primero?
6. ¿Cuál sería la caché recomendada para 8, 12, 16 y 24 GB?
7. ¿La mejora teórica justifica modificar el backend Metal?

Solo después de responder estas preguntas deberá comenzar la caché real.

---

# 18. Decisión arquitectónica final

La integración deberá conservar esta separación:

```text
llama.cpp:
Carga GGUF, tokenización, grafo, contexto y ejecución tradicional.

Dynamic MoE de Tosh:
Observa el router, administra expertos, calibra el hardware y elige CPU/GPU.

Metal:
Mantiene slots, transfiere pesos y ejecuta expertos residentes.

Swift:
Expone el toggle, muestra estado y controla el lanzamiento.
```

Dynamic MoE será una capa experimental sobre llama.cpp, no un reemplazo de llama.cpp.

Esta separación permitirá:

- Seguir actualizando el motor upstream.
- Desactivar completamente la funcionalidad.
- Comparar ambos sistemas.
- Mantener compatibilidad con modelos existentes.
- Activar nuevas arquitecturas progresivamente.
- Evitar que una función experimental afecte la estabilidad general de ToshLLM.
