"""
Example usage:
>>> python custom_export.py /path/to/export/dir rach3 hand_landmarker_mp

A JSON file will be created for each sample in the dataset. The JSON file will
contain the following structure:
{
    "filename": "sample_name.mp4",
    "frames": [
        {
            "frame_number": 1,
            "keypoints": [
                {
                    "label": "Left",
                    "points": [[x, y], [x, y], ...]
                },
                ...
            ]
        },
        ...
    ]
}
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import fiftyone as fo


def export(export_dir, dataset_name, field):
    dataset = fo.load_dataset(dataset_name)
    for sample in tqdm(dataset):
        dump = {
            "filename": str(Path(sample.filepath).name),
            "frames": [
                {
                    "frame_number": frame_number,
                    "keypoints": [
                        {"label": kp.label, "points": kp.points}
                        for kp in frame[field].keypoints
                    ],
                }
                for frame_number, frame in sample.frames.items()
            ],
        }

        Path(export_dir).mkdir(exist_ok=True, parents=True)
        fname = Path(sample.filepath).stem
        fpath = Path(export_dir) / f"{fname}.json"
        with open(str(fpath), "w", encoding="utf-8") as f:
            json.dump(dump, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("export_dir", type=str)
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("field", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export(**vars(args))
