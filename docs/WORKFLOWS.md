# Workflows de CAUCE

Esta suite convierte los nodos de CAUCE en recorridos de producción concretos.
Todos los graphs son workflows nativos de ComfyUI: no existe una interfaz
paralela, un editor externo ni una ontología de sujetos o acciones.

CAUCE organiza cuatro hechos: medios opacos, tiempo absoluto, relaciones entre
ventanas y artifacts versionados. La interpretación sigue ocurriendo en H3 a
partir de las imágenes, videos, audios y prompts conectados.

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
| 20 | `20_h3_ref2va_motion_reference.json` | 1 sample | Referencias visuales + video de movimiento sin semantizarlo. |
| 30 | `30_h3_timed_guide.json` | 1 sample | Anclar una imagen intermedia a tiempo absoluto. |
| 40 | `40_h3_two_window_continuation.json` | 2 samples | Heredar 39 frames AV y producir dos ventanas aceptadas. |
| 50 | `50_h3_latent_bridge.json` | 1 sample | Generar sólo el centro entre dos parents existentes. |

Todos fueron cargados y resueltos por el frontend real de la instancia del lab:
ningún graph contiene nodos desconocidos ni depende de los packs legacy.
“Graph validado” no equivale a “calidad visual aprobada”: Ref2VA, guide,
continuación y bridge todavía requieren comparación empírica sobre la GPU.

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
  → H3 sigma shifts (video 12 / audio 3)
  → res_multistep + simple / 20 steps
  → resolved phase-safe parent
  → video/audio decode
  → accepted temporal range
  → MP4 + AV latent + receipt
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
temporal de referencias video/audio no puede exceder 15 s.

## 30 · Guide en tiempo absoluto

`CAUCE · H3 Timed Guide` resuelve `master_seconds` contra el inicio real de la
ventana y llama al `MiniMaxH3AddGuide` oficial. Esto permite que un frame, clip
o audio esté anclado a la música o a un punto de montaje sin codificar a mano
un índice local del latent. El runtime `ComfyUI 0.33.1` actualmente instalado en
el lab todavía no registra `MiniMaxH3AddGuide`; por eso este workflow carga y se
puede inspeccionar, pero queda bloqueado con un error explícito hasta que esa
clase oficial esté disponible. FL2VA y Ref2VA no dependen de ella.

Se pueden encadenar varios guides. Cada uno debe caber completamente en la
ventana renderizada. Para un clip guía, usar una longitud H3 válida; para un
audio guide, conectar también el audio VAE.

## 40 · Continuación de dos ventanas

La segunda ventana declara `context_frames = 39`. El graph:

1. genera y resuelve el parent A;
2. copia exactamente su tail AV al head del target B;
3. coloca máscara cero sobre esa región heredada;
4. genera únicamente lo desconocido;
5. acepta y guarda A y B por separado.

No se concatenan latents H3 independientes antes del VAE. Cada parent se
decodifica dentro de su propia fase causal y sólo las regiones aceptadas se
montan después sobre el reloj maestro.

`39` es el primer boundary compartido entre la fase visual H3 y el grid de
audio. `90`, `141` y siguientes quedan disponibles para experimentos más
largos, pero aumentan el costo heredado y reducen el intervalo nuevo.

## 50 · Bridge

Este graph no es “run immediately”. Requiere dos `.safetensors` producidos por
`CAUCE · Save AV Latent`. Ajustar los dos paths antes de ejecutar.

El bridge copia 39 frames desde cada extremo y conserva ambos bloques. La
intersección central permanece generable. Si los contexts se solapan, el nodo
falla cerrado en vez de inventar un join ambiguo.

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
- **Bridge sin archivo:** usar el path exacto entregado por `Save AV Latent`.
- **El túnel cae:** el proceso de Comfy puede continuar en la torre; consultar
  history al reconectar antes de volver a encolar.
