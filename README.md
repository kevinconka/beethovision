# <div align="center">beethovision</div>

<div align="center">
Markerless Moction Capture in Piano Applications
</div>

</details>

## <div align="center">Documentation</div>

<details open>
<summary>📤 Import dataset to FiftyOne</summary>

```bash
python3 beethovision/fo_utils/import_dataset.py --dataset-name rach3 --dataset-dir /path/to/dataset --keyboard-bboxes rach3_bounding_boxes.json
```

</details>

<details open>
<summary>🏃‍♂️ Run MediaPipe Hand Landmark model</summary>

```bash
python3 beethovision/fo_utils/run_mediapipe.py --dataset-name rach3
```

</details>

## TODOs

- [ ] Hacer midi de la _misma_ resolution que los videos
  - jugar con el tempo
  - idealmente cada cuadro del video corresponde a una columna del `piano_roll`
  - una idea es tener doble resolucion en el midi y luego hacer una interpolacion lineal para tener la misma resolucion que el video
- [ ] Encontrar casos en donde no hay detección de manos pero si hay una nota en el midi
