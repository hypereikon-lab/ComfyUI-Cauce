# Workflows de CAUCE

Esta suite convierte los nodos de CAUCE en recorridos de producción concretos.
Todos los graphs son workflows nativos de ComfyUI: no existe una interfaz
paralela, un editor externo ni una ontología de sujetos o acciones.

CAUCE organiza cuatro hechos: medios opacos, tiempo absoluto, relaciones entre
ventanas y artifacts versionados. La interpretación sigue ocurriendo en H3 a
partir de las imágenes, videos y prompts conectados. El soundtrack es un master
fijo: marca el reloj creativo y se monta al final, pero no se usa como target
generativo ni se reemplaza con audio sintetizado por H3.

## Preparación

1. Instalar o actualizar `ComfyUI-Cauce` desde Manager.
2. Reiniciar ComfyUI únicamente si cambió el código Python del pack.
3. Subir los archivos de `examples/assets/` al input de ComfyUI:
   - `cauce_forest_a.jpg`
   - `cauce_forest_b.jpg`
   - `cauce_forest_c.jpg`
   - `cauce_motion_reference.mp4`
4. Arrastrar un JSON de `examples/workflows/` sobre el canvas.
5. Ejecutar primero `00`, luego `10`; avanzar al resto cuando ese baseline esté
   estable.

Los workflows están fijados a los modelos existentes en la estación del lab:

```text
diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors
diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
vae/minimax_h3_video_vae_fp16.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
```

## Recorridos

| ID | Workflow | Ejecuta | Propósito |
|---|---|---:|---|
| 00 | `00_plate_sketch_handoff.json` | CPU | Componer un plate, asociarlo a tiempo/prompt y exportar PNG + sidecars. |
| 10 | `10_h3_fl2va_first_last.json` | 1 sample | Baseline first/last con MP4, parent latent y receipt. |
| 20 | `20_h3_ref2va_motion_reference.json` | 1 sample | Referencias visuales + video de movimiento; verificado landscape. |
| 30 | `30_h3_timed_guide.json` | 1 sample | Anclar una imagen intermedia; bloqueado en H3 0.33.1. |
| 40 | `40_h3_two_window_continuation.json` | 2 samples | Latent tail + endpoint visual; seam visual verificado. |
| 50 | `50_h3_latent_bridge.json` | 1 sample | Bridge experimental entre dos parents existentes. |
| 60 | `60_h3_confluence_seam_repair.json` | 1 sample | Reparación condicional; v1 rechazado, v2 LanPaint pendiente de prueba live. |

Todos fueron cargados y resueltos por el frontend real de la instancia del lab:
ningún graph contiene nodos desconocidos ni depende de los packs legacy. Los
resultados ejecutados, sus tiempos y los gates pendientes están en
[`LAB_RESULTS.md`](LAB_RESULTS.md). “Graph validado” no equivale a “calidad
visual aprobada”.

## 00 · Plate sketch y handoff

El graph de plates usa tres medios opacos como ejemplo. Pueden reemplazarse por
cualquier imagen, máscara o composición intermedia. Los nodos de layer exponen
posición, escala, rotación, opacidad, blend y feather; no intentan reconocer lo
que hay en cada capa.

`CAUCE · Export Plate` guarda:

```text
ComfyUI/output/cauce/plates/<point>_<timecode>_plate_*.png
ComfyUI/output/cauce/plates/<mismo_nombre>.prompt.txt
ComfyUI/output/cauce/plates/<mismo_nombre>.point.json
```

El PNG y el prompt son el handoff directo a Runway/GPT Image. Al regresar un
resultado, se reemplaza o adjunta la imagen en el mismo punto sin cambiar el
reloj ni el resto del workflow.

## 10 · FL2VA first/last

Éste es el baseline de producción. Hace explícitos:

```text
first IMAGE + last IMAGE + arbitrary prompt
  → CAUCE generation window
  → official MiniMaxH3ImageToVideo
  → H3 sigma shifts internos
  → res_multistep + simple / 20 steps
  → resolved phase-safe parent
  → video decode; audio H3 descartado
  → accepted temporal range
  → MP4 visual + latent estructural + receipt
```

Para iterar normalmente sólo se cambian imágenes, prompt y seed. El profile
`h3-5090-fl2va-640` mantiene el primer barrido dentro del envelope medido de la
5090 de 32 GB. No subir resolución o duración y número de referencias al mismo
tiempo: cambiar una sola dimensión deja receipts comparables.

## 20 · Ref2VA como campo de referencias

El demo conecta dos imágenes y un video de 56 frames a 24 fps. Los tags que H3
recibe son:

```text
<Picture 1>, <Picture 2>, <Video 1>
```

El prompt decide qué función cumple cada referencia. CAUCE conserva el orden,
valida cantidades y duración, y entrega los medios a Ref2VA; no almacena una
clasificación “identidad”, “estilo”, “movimiento” o “cámara”. El mismo socket de
video puede recibir primitivas, simulaciones, render 3D, optical flow
visualizado o material filmado.

El MP4 de demo dura 2,333 s y contiene 56 frames, una longitud válida `17k+5`.
Para reemplazarlo, el video debe estar a 24 fps y durar entre 2 y 15 s. El total
temporal de referencias video no puede exceder 15 s en este preset. Los sockets
de audio existen por compatibilidad upstream, pero no se usan en producción.

El workflow usa `h3-5090-ref2va-576x320`: mantiene un costo cercano al profile
cuadrado de 448, pero ofrece un canvas landscape más apropiado para referencias
panorámicas. El profile 448 permanece disponible para material realmente
cuadrado; cambiar de aspect ratio es una decisión de output, no una propiedad
semántica de las referencias.

## 30 · Guide en tiempo absoluto

`CAUCE · H3 Timed Guide` resuelve `master_seconds` contra el inicio real de la
ventana y llama al `MiniMaxH3AddGuide` oficial. Esto permite que un frame o clip
esté anclado a la música o a un punto de montaje sin codificar a mano un índice
local del latent. El runtime `ComfyUI 0.33.1` actualmente instalado en
el lab todavía no registra `MiniMaxH3AddGuide`; por eso este workflow carga y se
puede inspeccionar, pero queda bloqueado con un error explícito hasta que esa
clase oficial esté disponible. FL2VA y Ref2VA no dependen de ella.

Se pueden encadenar varios guides. Cada uno debe caber completamente en la
ventana renderizada. Para un clip guía, usar una longitud H3 válida. El master
de audio no se conecta a AddGuide.

## 40 · Continuación de dos ventanas

La segunda ventana declara `context_frames = 39`. El graph:

1. genera y resuelve el parent A;
2. decodifica A y selecciona su último frame aceptado de forma opaca;
3. usa ese endpoint como `first_frame` oficial de FL2VA para B;
4. copia exactamente el tail visual de A al head del target B;
5. coloca máscara cero sobre esa región heredada;
6. congela el stream de audio estructural y genera sólo video desconocido;
7. acepta y guarda A y B por separado.

La ventana B solicita `85` frames nuevos (`3,541666667 s`). Con los `39`
heredados, el render vuelve a sumar `124` frames: el mismo envelope ya verificado
para el baseline. Esto evita que la demostración de continuidad cambie duración
y algoritmo al mismo tiempo.

No se concatenan latents H3 independientes antes del VAE. Cada parent se
decodifica dentro de su propia fase causal y sólo las regiones aceptadas se
montan después sobre el reloj maestro.

`39` permanece como preset ya probado, pero ya no es una obligación AV. Los
boundaries visuales legales son `5`, `22`, `39`, `56`, etc.; esto permite elegir
contexto exclusivamente por continuidad visual.

En el runtime 0.33.1, el latent tail por sí solo ejecutó pero produjo un salto
visual fuerte. La guía de clip latente pertenece a un core H3 posterior y CAUCE
la bloquea explícitamente cuando no está disponible. El endpoint decodificado
es la estrategia compatible que ahora debe pasar el gate comparativo; no se
declara continuidad “resuelta” sólo porque el sampler termine. En la prueba
del lab, la variante híbrida conservó composición, perspectiva, cauce y árbol
central a través del join. El audio generado no forma parte de ese gate ni se
acepta en la pieza.

## 50 · Bridge

Este graph no es “run immediately”. Requiere dos `.safetensors` estructurales
producidos por `CAUCE · Save AV Latent`. Ajustar los dos paths antes de ejecutar.

El bridge copia 39 frames visuales desde cada extremo y conserva ambos bloques.
El stream de audio interno queda congelado y se descarta. La intersección
central permanece generable. Si los contexts se solapan, el nodo falla cerrado
en vez de inventar un join ambiguo.

## 60 · Confluence: reparación condicional entre dos gestos

Este recorrido parte de dos videos ya existentes, no de parents latentes. Los
normaliza al profile FL2VA 640, toma 2,5 s de contexto a cada lado y entrega el
tail de A y el head de B al VAE como un único dominio causal. El nodo de build
produce también la ventana H3 exacta, de modo que no existe un segundo widget
de duración que pueda quedar desincronizado.

Con los defaults:

```text
A: últimos 60 frames ┐
                     ├─ + 2 guards por lado → working domain de 124 frames
B: primeros 60 frames┘

corte working: frame 62
región generable: [38, 86) = 48 frames = 2 s
output: len(A) + len(B), sin cambio de duración
```

H3 ve first/last frames del working domain y el prompt arbitrario. `CAUCE ·
Confluence Fields` produce tres `MASK` comunes e inspeccionables:

```text
generation_strength  → fuerza continua del sampler
hard_acceptance      → intervalo binario permitido
output_opacity       → mezcla decodificada del parche
```

El preset usa una curva cosenoidal, 12 frames de transición latente y ocho de
blend decodificado. El valor parte exactamente en cero, alcanza uno en el
interior y vuelve a cero. La proyección a tokens H3 usa cobertura promedio para
conservar esa información gradual; `peak` queda sólo para comparación.

El sampler estándar anterior ejecutó correctamente, pero falló cualitativamente
con gestos reales. Este workflow requiere [LanPaint v2.1.0 o
posterior](https://github.com/scraed/LanPaint) y usa `LanPaint Sampler Custom
Advanced` con cinco iteraciones internas, lambda 5, step size 0,2 y modo `Image
First`. LanPaint realiza el muestreo condicional; CAUCE sigue controlando toda
la geometría temporal, las máscaras, el rango aceptado y el splice final.

Después del decode, CAUCE acepta sólo el parche central y repone el resto desde
los inputs originales. Modificar el exterior no es una solicitud al modelo,
sino una imposibilidad topológica.

El JSON contiene `gesture_a.mp4` y `gesture_b.mp4` como placeholders. Cada clip
debe tener al menos 60 frames y ambos deben ser exactamente 24 fps. La primera
validación cualitativa deberá cubrir gestos con distinta posición, velocidad y
aceleración en el corte; probar 0,5, 1 y 1,5 s por lado antes de adoptar 1 s como
preset definitivo. La resolución queda normalizada a 640×640 para el envelope
verificado de la 5090.

El recorrido estándar anterior se validó en la instancia del lab con dos
batches sintéticos de 60 frames: H3 recibió 124 frames, el sampler terminó, el
output unido devolvió 120 frames y el parche exactamente 48. Esa prueba confirmó
sockets, shapes, decode y splice, pero la posterior prueba perceptual fue mala.
La versión condicional debe volver a superar primero ese mismo par de gestos.

El workflow es intencionalmente video-only. El audio master no se conecta, no
se codifica y no se sustituye. H3 mantiene internamente un stream de audio vacío
y enmascarado sólo porque su latent es estructuralmente AV; ese resultado se
descarta. Para una pieza larga se corrigen clips locales y luego se colocan
sobre el reloj del master fijo.

### Instalar el sampler condicional

En ComfyUI Manager, instalar `LanPaint` o usar su Git URL:

```text
https://github.com/scraed/LanPaint.git
```

Reiniciar ComfyUI una vez. CAUCE no copia ni modifica LanPaint y no agrega sus
dependencias a la instalación propia.

## Runner: secuencia reiniciable

Los JSON visuales sirven para diseño e inspección. Para dejar varias ventanas
corriendo, CAUCE incluye dos plantillas API:

```text
examples/api/h3_fl2va_window.template.json
examples/api/h3_continuation_window.template.json
```

`examples/project.example.json` usa la primera para el parent inicial y la
segunda para una continuación. Cada ventana puede seleccionar su propia
`workflow_template`; el runner materializa el JSON exacto, escribe estado antes
y después de cada submit, y omite ventanas completas al reanudar.

```bash
python cauce_cli.py resume examples/project.example.json --dry-run
python cauce_cli.py resume examples/project.example.json
python cauce_cli.py status examples/project.example.json
```

La plantilla de continuación carga el parent guardado, lo decodifica y deriva
automáticamente su `first_frame`; no exige exportar manualmente un PNG de borde.

Los assets nombrados por una plantilla API deben existir previamente en
`ComfyUI/input/`. El modo recomendado es ejecutar el runner en la torre contra
`http://127.0.0.1:8188`; el login interactivo del túnel no se reutiliza.

## Ciclo operativo

```text
plate sketch
  → export PNG + prompt
  → generación de imagen en Runway
  → resultado vuelve a un CAUCE point
  → FL2VA / Ref2VA / guide
  → accepted MP4 + phase-safe parent + receipt
  → aprobar, rerollear o continuar
  → ensamblar regiones aceptadas en el master clock
```

El receipt fija seed, profile, sampler, scheduler, steps, hashes de modelos,
parents y hash del workflow. Un reroll puede reemplazar el índice del artifact
sin perder qué parent produjo la rama anterior.

## Fallos esperables

- **Modelo no aparece:** ejecutar `CAUCE · Preflight`; no instalar otra variante
  hasta revisar espacio y manifest.
- **Asset rojo o ausente:** subirlo primero a `ComfyUI/input/` y volver a
  seleccionarlo en `Load Image`/`Load Video`.
- **Out of memory:** volver al profile 640 FL2VA o 448 Ref2VA, cerrar previews
  pesados y reiniciar únicamente ComfyUI si la VRAM quedó fragmentada.
- **Ref2VA rechaza el video:** comprobar 24 fps, 2–15 s y duración acumulada.
- **Guide fuera de rango:** revisar el `render_range`, no sólo el tiempo visible
  solicitado.
- **Continuation antes de cero:** el tiempo aceptado debe dejar espacio para el
  context heredado.
- **`mask_plus_guide` no disponible:** usar el workflow 40 con endpoint
  decodificado; no actualizar ComfyUI automáticamente desde CAUCE.
- **Bridge sin archivo:** usar el path exacto entregado por `Save AV Latent`.
- **Confluence rechaza el input:** ambos videos deben ser 24 fps, tener al menos
  el contexto solicitado y normalizarse al mismo profile antes del VAE.
- **Confluence carga pero no corrige el gesto:** ampliar gradualmente la región
  reparada o cambiar el prompt/seed; no ampliar contexto, reparación y
  resolución simultáneamente.
- **El túnel cae:** el proceso de Comfy puede continuar en la torre; consultar
  history al reconectar antes de volver a encolar.
