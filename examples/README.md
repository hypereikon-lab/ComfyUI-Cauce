# CAUCE examples

Los ejemplos están organizados por función:

- `assets/`: tres imágenes de plate y un video de movimiento de 56 frames.
- `workflows/`: graphs visuales para arrastrar sobre ComfyUI.
- `api/`: prompts de API materializables por el runner.
- `project.example.json`: parent FL2VA + continuación reiniciable.

Orden recomendado: `00 → 10 → 20/30 → 40 → 50/60`.

Estado real: `00`, `10` y `20` están verificados; `30` requiere un AddGuide
oficial más nuevo; `40` pasó el primer gate visual de continuidad híbrida y
descarta el audio H3; `50` sigue experimental. La versión estándar de `60`
completó una ejecución sintética pero falló con gestos reales. El workflow
actual usa campos continuos + LanPaint y está pendiente de validación live con
los mismos pares rechazados.
Consultar [`docs/LAB_RESULTS.md`](../docs/LAB_RESULTS.md) antes de ejecutar jobs
costosos.

Antes de abrir un workflow, subir sus assets a `ComfyUI/input/`. Los JSON no
contienen modelos ni duplican los archivos de la torre. Consultar la guía
completa en [`docs/WORKFLOWS.md`](../docs/WORKFLOWS.md).
