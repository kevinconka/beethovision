"""
Imports data into FiftyOne by creating a dataset from video files and adding
keyboard bounding box detections to each frame.

Usage:
    python3 fo_import_data.py \
        --dataset-name rach3 \
        --dataset-dir /path/to/dataset \
        --keyboard-bboxes rach3_bounding_boxes.json
"""


import re
import argparse
import glob
import json
from pathlib import Path
from datetime import datetime
import fiftyone as fo


def create_dataset(name: str, glob_pattern: str = "./**/*.mp4"):
    """
    Creates a FiftyOne dataset by adding samples from video files.

    Args:
        name (str): The name of the dataset.
        glob_pattern (str, optional): The glob pattern to search for video files.
            Defaults to "./**/*.mp4".

    Returns:
        fiftyone.core.dataset.Dataset: The created dataset.
    """

    def sort_key(s):
        pattern = r"(\d{4}-\d{2}-\d{2})_a(\d+)_.*_split(\d+)"
        # Use re.search to find the first match
        match = re.search(pattern, Path(s).stem)
        if match:
            # Extract the date, n, and s values from the match
            date_str, n_str, s_str = match.groups()
            # Convert date to a sortable format (days since year 1)
            date = datetime.strptime(date_str, "%Y-%m-%d").toordinal()
            # Return a tuple for sorting: first by date, then by n, then by s
            return (date, int(n_str), int(s_str))
        else:
            raise ValueError(f"Could not extract date, n, and s from '{s}'")

    video_fps = sorted(glob.glob(glob_pattern, recursive=True), key=sort_key)

    ds = fo.Dataset(name=name)
    ds.persistent = True

    print(f"Adding {len(video_fps)} samples to dataset '{name}'")
    for video_fp in video_fps:
        sample = fo.Sample(filepath=video_fp)
        # tag train or test depending on the file path
        sample.tags.append("train" if "train" in video_fp else "test")
        ds.add_sample(sample)

    ds.compute_metadata()
    return ds


def add_session_field(dataset: fo.Dataset, field_name: str = "session"):
    """
    Adds a session field to each sample in the dataset based on the file path.

    Args:
        dataset (fiftyone.core.dataset.Dataset): The dataset to modify.
        field_name (str, optional): The name of the session field.
            Defaults to "session".
    """
    session_pattern = r"(\d{4}-\d{2}-\d{2}_a\d+)"
    for sample in dataset.iter_samples(progress=True, autosave=True):
        match = re.search(session_pattern, sample.filepath)
        if not match:
            raise ValueError(f"Could not extract session from '{sample.filepath}'")
        session = match.group(1)
        sample[field_name] = session


def add_keyboard_bboxes(
    dataset: fo.Dataset, json_fp: str, field_name: str = "keyboard"
):
    """
    Adds keyboard bounding box detections to each frame in the dataset.

    Args:
        dataset (fiftyone.core.dataset.Dataset): The dataset to modify.
        json_fp (str): The file path to the JSON file containing the keyboard
            bounding box predictions.
        field_name (str, optional): The name of the field to store the detections.
            Defaults to "keyboard".
    """
    with open(json_fp, "r", encoding="utf-8") as f:
        keyboard_preds = json.load(f)

    for sample in dataset.iter_samples(progress=True):
        # get the keyboard predictions for the current sample
        preds = [e["box"] for e in keyboard_preds if e["session_id"] == sample.session]
        assert len(preds) == 1  # one entry per session

        height, width = sample.metadata.frame_height, sample.metadata.frame_width
        detections = []
        for pred in preds[0]:
            x1 = pred["box"]["x1"] / width
            y1 = pred["box"]["y1"] / height
            x2 = pred["box"]["x2"] / width
            y2 = pred["box"]["y2"] / height
            rel_box = [x1, y1, x2 - x1, y2 - y1]  # fiftyone format
            detections.append(
                fo.Detection(
                    label=pred["name"],
                    bounding_box=rel_box,
                    confidence=pred["confidence"],
                    cls=pred["class"],
                )
            )

        for i in range(sample.metadata.total_frame_count):
            # fiftyone frames are 1-indexed
            sample.frames[i + 1][field_name] = fo.Detections(detections=detections)

        sample.save()


def parse_args():
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="rach3")
    parser.add_argument("--dataset-dir", type=str, default=".")
    parser.add_argument(
        "--keyboard-bboxes", type=str, default="rach3_bounding_boxes.json"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main(dataset_name: str, dataset_dir: str, keyboard_bboxes: str, overwrite: bool):
    """
    Main function to create and modify a dataset.

    Args:
        dataset_name (str): The name of the dataset.
        dataset_dir (str): The directory containing the video files.
        keyboard_bboxes (str): The file path to the JSON file containing the keyboard
            bounding box predictions.
        overwrite (bool): Whether to overwrite an existing dataset with the same name.
    """
    if overwrite and dataset_name in fo.list_datasets():
        print(f"Overwriting dataset {dataset_name}")
        fo.delete_dataset(dataset_name)

    if dataset_name not in fo.list_datasets():
        print(f"📼 Creating dataset '{dataset_name}'")
        dataset = create_dataset(
            dataset_name, glob_pattern=str(Path(dataset_dir) / "**" / "*.mp4")
        )
        add_session_field(dataset)
        print("🎹 Adding keyboard bounding boxes to dataset")
        add_keyboard_bboxes(dataset, json_fp=str(Path(dataset_dir) / keyboard_bboxes))
    else:
        print(f"Dataset {dataset_name} already exists")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
