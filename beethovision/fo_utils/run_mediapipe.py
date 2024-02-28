"""
Runs the MediaPipe Hand Landmarker on a FiftyOne video dataset.
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

Usage:
    python3 fo_run_mediapipe.py --dataset-name rach3

    python3 fo_run_mediapipe.py \
        --dataset-name rach3 \
        --model-asset-path hand_landmarker.task \
        --keypoints-field hand_landmarker_mp \
        --num-samples 10 \
        --seed 0x5EED
"""

import urllib.request
import argparse
from tqdm import tqdm
import cv2
import fiftyone as fo
import mediapipe as mp
from mediapipe import solutions


class DownloadProgressBar(tqdm):
    """
    A progress bar for downloads.
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """Update the progress bar."""
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def add_default_keypoint_skeleton(dataset: fo.Dataset):
    """
    Adds a default skeleton to the dataset.

    Args:
        dataset (fo.Dataset): The dataset to add the default skeleton to.
    """
    dataset.default_skeleton = fo.KeypointSkeleton(
        labels=[e.name for e in solutions.hands.HandLandmark],
        edges=solutions.hands_connections.HAND_CONNECTIONS,
    )
    dataset.save()  # must save after edits


def download_model_asset():
    """
    Downloads the model asset for the hand landmarker.

    Returns:
        str: The path to the model asset.
    """
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    output_path = "hand_landmarker.task"
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    return output_path


def get_landmarker_options(model_asset_path):
    """
    Returns the options for the hand landmarker.

    Args:
        model_asset_path (str): The path to the model asset.

    Returns:
        mp.tasks.vision.HandLandmarkerOptions: The options for the hand landmarker.
    """
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a hand landmarker instance with the video mode:
    return HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_asset_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )


def detect_hands(video_frame, frame_timestamp_ms, hand_landmarker):
    """
    Detects hands in a video frame using the hand landmarker.

    Args:
        video_frame: The video frame to detect hands in.
        frame_timestamp_ms: The timestamp of the video frame in milliseconds.
        hand_landmarker: The hand landmarker instance.

    Returns:
        list: A list of keypoints representing the detected hands.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=video_frame)
    hand_landmarker_result = hand_landmarker.detect_for_video(
        mp_image, frame_timestamp_ms
    )

    hand_landmarks_list = hand_landmarker_result.hand_landmarks
    handedness_list = hand_landmarker_result.handedness

    keypoints = []
    for handedness, landmarks in zip(handedness_list, hand_landmarks_list):
        # FiftyOne expects [top-left-x, top-left-y, width, height] normalized to [0, 1]
        keypoints.append(
            fo.Keypoint(
                label=handedness[0].category_name,
                points=[(landmark.x, landmark.y) for landmark in landmarks],
            )
        )
    return keypoints


def run_mp(
    dataset: fo.Dataset, options: mp.tasks.vision.HandLandmarkerOptions, field_name: str
):
    """
    Runs the hand landmarker on the dataset.

    Args:
        dataset (fo.Dataset): The dataset to run the hand landmarker on.
        options (mp.tasks.vision.HandLandmarkerOptions): hand landmarker options.
        field_name (str): The name of the field to store the keypoints in.
    """
    HandLandmarker = mp.tasks.vision.HandLandmarker

    for sample in tqdm(dataset):
        cap = cv2.VideoCapture(sample.filepath)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        with HandLandmarker.create_from_options(options) as landmarker:
            for frame_number, frame in tqdm(
                sample.frames.items(), total=n_frames, leave=False
            ):
                success, video_frame = cap.read()
                if not success:
                    break
                video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                frame_timestamp_ms = int(1000 / fps * frame_number)

                keypoints = detect_hands(video_frame, frame_timestamp_ms, landmarker)
                frame[field_name] = fo.Keypoints(keypoints=keypoints)
            sample.save()


def parse_args():
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="rach3")
    parser.add_argument("--model-asset-path", type=str, default="")
    parser.add_argument("--keypoints-field", type=str, default="hand_landmarker_mp")
    parser.add_argument(
        "--num-samples", type=int, default=-1, help="If -1, use all samples."
    )
    parser.add_argument(
        "--seed", type=int, default=0x5EED, help="Random seed for sampling."
    )
    return parser.parse_args()


def main(
    dataset_name: str,
    model_asset_path: str,
    keypoints_field: str,
    num_samples: int,
    seed: int,
):
    """
    The main function of the script.

    Args:
        dataset_name (str): The name of the dataset.
        model_asset_path (str): The path to the model asset.
        keypoints_field (str): The name of the field to store the keypoints in.
        num_samples (int): The number of samples to process.
        seed (int): The random seed for sampling.
    """
    print(f"📂 Loading dataset '{dataset_name}'")
    dataset: fo.Dataset = fo.load_dataset(dataset_name)

    print("💀 Adding default keypoint skeleton to dataset")
    add_default_keypoint_skeleton(dataset)

    if not model_asset_path:
        model_asset_path = download_model_asset()

    if num_samples > 0:
        print(f"Sampling {num_samples} samples with seed {seed}")
        dataset = dataset.take(num_samples, seed=seed)

    print(f"🙌 Running MediaPipe Hand Landmarker on {len(dataset)} samples")
    print(f"   - Model asset path: {model_asset_path}")
    print(f"   - Keypoints field: {keypoints_field}")
    print("💫 This may take a while...")
    run_mp(
        dataset,
        get_landmarker_options(model_asset_path),
        field_name=keypoints_field,
    )


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
