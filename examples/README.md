# CAUCE examples

Los ejemplos están organizados por función:

- `assets/`: tres imágenes de plate y un video de movimiento de 56 frames.
- `workflows/`: graphs visuales para arrastrar sobre ComfyUI.
- `api/`: prompts de API materializables por el runner.
- `project.example.json`: parent FL2VA + continuación reiniciable.

Orden recomendado: `00 → 10 → 20/30 → 40 → 50/60`.

Estado real: `00`, `10` y `20` están verificados; `30` requiere un AddGuide
oficial más nuevo; `40` pasó el primer gate visual de continuidad híbrida y
descarta el audio H3; `50` sigue experimental. Las versiones v1–v3 de `60`
ejecutaron pero fallaron con gestos reales. La auditoría identificó un core H3
anterior a las máscaras temporales correctas y un intervalo generado tres veces
mayor que el solicitado. La v4 usa máscara oficial por token, un centro de 22
frames y clips guía bidireccionales de 22 frames; se niega a ejecutar sobre el
runtime antiguo y está pendiente de validación live tras actualizar ComfyUI.
Consultar [`docs/LAB_RESULTS.md`](../docs/LAB_RESULTS.md) antes de ejecutar jobs
costosos.

Antes de abrir un workflow, subir sus assets a `ComfyUI/input/`. Los JSON no
contienen modelos ni duplican los archivos de la torre. Consultar la guía
completa en [`docs/WORKFLOWS.md`](../docs/WORKFLOWS.md).
