# CAUCE examples

Los ejemplos están organizados por función:

- `assets/`: tres imágenes de plate y un video de movimiento de 56 frames.
- `workflows/`: graphs visuales para arrastrar sobre ComfyUI.
- `api/`: prompts de API materializables por el runner.
- `project.example.json`: parent FL2VA + continuación reiniciable.

Orden recomendado: `00 → 10 → 20/30 → 40 → 50`.

Estado real: `00` y `10` están verificados; `20` ejecutó y ahora usa un profile
landscape pendiente de confirmación; `30` requiere un AddGuide oficial más
nuevo; `40` es un candidato de continuidad híbrida; `50` sigue experimental.
Consultar [`docs/LAB_RESULTS.md`](../docs/LAB_RESULTS.md) antes de ejecutar jobs
costosos.

Antes de abrir un workflow, subir sus assets a `ComfyUI/input/`. Los JSON no
contienen modelos ni duplican los archivos de la torre. Consultar la guía
completa en [`docs/WORKFLOWS.md`](../docs/WORKFLOWS.md).
