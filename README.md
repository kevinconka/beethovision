# <div align="center">beethovision</div>

<div align="center">
Markerless Moction Capture in Piano Applications
</div>

https://github.com/kevinconka/beethovision/assets/35779409/9c1b86a8-ee68-47b2-9651-bd0dca04941e

</details>

## <div align="center">Documentation</div>

<details open>
<summary>📥📤 Download and Import Metadata to FiftyOne</summary>

Download the metadata using `gdown`:
```bash
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1UxbJs-YEuI9rhUygtZf5mtxH5iXOAWX0', 'fiftyone.tgz', quiet=False)"
```

Assuming you have the following structure:
```
rach3_s
├── rach3_bounding_boxes.json
├── test   # mp4, midi, flac
└── train  # mp4, midi, flac
```

Set up an environment variable with the path to the dataset:
```bash
export DATASET_DIR=/path/to/rach3_s
```

Unzip the metadata:
```bash
tar -xvzf fiftyone.tgz -C $DATASET_DIR
```

Import the metadata to FiftyOne:
```bash
fiftyone datasets create --name rach3 -d $DATASET_DIR --type fiftyone.types.FiftyOneDataset
```

Alternatively, you can run the `fetch_unzip_fiftyone.sh` under `scripts`:
```console
$ DATASET_DIR=rach3_s bash scripts/fetch_unzip_fiftyone.sh 
Downloading...
From (original): https://drive.google.com/uc?id=1UxbJs-YEuI9rhUygtZf5mtxH5iXOAWX0
From (redirected): https://drive.google.com/uc?id=1UxbJs-YEuI9rhUygtZf5mtxH5iXOAWX0&confirm=t&uuid=125a5697-528a-4619-bbd4-30cf18fbff49
To: /Users/kevinserrano/GitHub/JKU-AI/beethovision/fiftyone.tgz
100%|█████████████████████████████████| 1.68G/1.68G [02:08<00:00, 13.0MB/s]
x frames.json
x metadata.json
x samples.json
Importing samples...
 100% |███████████████████████████████| 293/293 [3.9ms elapsed, 0s remaining, 75.4K samples/s]       
Importing frames...
Importing frames...
 100% |███████████████████████████████| 2263220/2263220 [57.0s elapsed, 0s remaining, 37.6K samples/s]      
Dataset 'rach3' created
```

Launch fiftyone:
```bash
fiftyone app launch
```

</details>

<details>
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
