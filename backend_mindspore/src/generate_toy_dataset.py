import os
import numpy as np

import config
from core_preprocess import to_double_relative_with_velocity


RNG = np.random.default_rng(42)
NUM_CLASSES = 3
SAMPLES_PER_SPLIT = {
    "train": 6,
    "val": 2,
    "test": 2,
}


POSE_TEMPLATE = np.array([
    [0.00, 0.00], [-0.04, 0.06], [0.04, 0.06], [-0.08, 0.12], [0.08, 0.12],
    [-0.10, 0.20], [0.10, 0.20], [-0.11, 0.30], [0.11, 0.30], [-0.12, 0.40],
    [0.12, 0.40], [-0.06, 0.18], [0.06, 0.18], [-0.05, 0.28], [0.05, 0.28],
    [-0.04, 0.38], [0.04, 0.38], [-0.05, 0.50], [0.05, 0.50], [-0.04, 0.62],
    [0.04, 0.62], [-0.03, 0.74], [0.03, 0.74], [-0.02, 0.86], [0.02, 0.86],
], dtype=np.float32)

HAND_TEMPLATE = np.array([
    [0.00, 0.00],
    [-0.02, -0.01], [-0.04, -0.05], [-0.05, -0.10], [-0.06, -0.16],
    [-0.01, -0.02], [-0.01, -0.08], [-0.01, -0.15], [-0.01, -0.22],
    [0.01, -0.02], [0.02, -0.09], [0.03, -0.17], [0.04, -0.25],
    [0.03, -0.02], [0.05, -0.08], [0.07, -0.14], [0.09, -0.20],
    [0.05, -0.01], [0.08, -0.05], [0.11, -0.09], [0.14, -0.13],
], dtype=np.float32)


def rotate(points: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    mat = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points @ mat.T


def hand_shape(class_id: int, side: str, phase: float) -> np.ndarray:
    pts = HAND_TEMPLATE.copy()

    if class_id == 0:
        # 左手打开，右手收拢
        scale = 1.25 if side == "left" else 0.65
        angle = 0.35 if side == "left" else -0.10
        wobble = np.array([0.010 * np.sin(phase), 0.012 * np.cos(phase)], dtype=np.float32)
    elif class_id == 1:
        # 右手打开并旋转，左手收拢
        scale = 1.25 if side == "right" else 0.65
        angle = -0.35 if side == "right" else 0.10
        wobble = np.array([0.012 * np.cos(phase), 0.010 * np.sin(phase)], dtype=np.float32)
    else:
        # 双手同时张开，但方向相反
        scale = 0.95
        angle = 0.28 if side == "left" else -0.28
        wobble = np.array([0.016 * np.sin(phase), 0.016 * np.sin(phase + np.pi / 2)], dtype=np.float32)

    pts = rotate(pts * scale, angle)
    pts[1:] += wobble
    pts[1:, 1] += 0.01 * np.sin(phase * 1.5 + np.linspace(0.0, 1.0, 20))
    return pts.astype(np.float32)


def make_sequence(class_id: int, sample_id: int, split: str) -> np.ndarray:
    length = int(RNG.integers(52, 78))
    frames = []

    base_nose = np.array([0.50, 0.22], dtype=np.float32)
    left_wrist_base = np.array([0.36, 0.58], dtype=np.float32)
    right_wrist_base = np.array([0.64, 0.58], dtype=np.float32)

    split_jitter = {"train": 0.0, "val": 0.015, "test": 0.03}[split]
    sample_phase = 0.35 * sample_id + class_id * 0.6

    for t in range(length):
        phase = (2 * np.pi * t / max(length - 1, 1)) + sample_phase

        nose = base_nose + np.array([
            0.010 * np.sin(phase * 0.5),
            0.008 * np.cos(phase * 0.4),
        ], dtype=np.float32)

        pose = nose + POSE_TEMPLATE.copy()
        pose[:, 0] += 0.006 * np.sin(phase + np.linspace(0.0, 1.2, 25))
        pose[:, 1] += 0.006 * np.cos(phase * 0.8 + np.linspace(0.0, 1.0, 25))

        if class_id == 0:
            left_wrist = left_wrist_base + np.array([0.05 * np.sin(phase), -0.03 * np.cos(phase)], dtype=np.float32)
            right_wrist = right_wrist_base + np.array([0.01 * np.sin(phase * 0.5), 0.0], dtype=np.float32)
        elif class_id == 1:
            left_wrist = left_wrist_base + np.array([0.01 * np.cos(phase * 0.5), 0.0], dtype=np.float32)
            right_wrist = right_wrist_base + np.array([-0.05 * np.sin(phase), -0.03 * np.cos(phase)], dtype=np.float32)
        else:
            left_wrist = left_wrist_base + np.array([0.04 * np.sin(phase), -0.02 * np.cos(phase)], dtype=np.float32)
            right_wrist = right_wrist_base + np.array([-0.04 * np.sin(phase), -0.02 * np.cos(phase)], dtype=np.float32)

        left_hand = left_wrist + hand_shape(class_id, "left", phase)
        right_hand = right_wrist + hand_shape(class_id, "right", phase)

        frame = np.concatenate([
            pose.reshape(-1),
            left_hand.reshape(-1),
            right_hand.reshape(-1),
        ]).astype(np.float32)

        noise_scale = 0.002 + split_jitter
        frame += RNG.normal(0.0, noise_scale, size=frame.shape).astype(np.float32)
        frames.append(frame)

    return np.stack(frames).astype(np.float32)


def main() -> None:
    os.makedirs(config.DATA_ROOT, exist_ok=True)
    os.makedirs(config.PROCESSED_FEATURE_DIR, exist_ok=True)

    split_lines = {"train": [], "val": [], "test": []}
    transformed_train = []

    print(f"Generating toy dataset under: {config.DATA_ROOT}")

    for class_id in range(NUM_CLASSES):
        for split, count in SAMPLES_PER_SPLIT.items():
            for sample_idx in range(count):
                name = f"toy_c{class_id}_{split}_{sample_idx:02d}.npy"
                path = os.path.join(config.PROCESSED_FEATURE_DIR, name)
                raw = make_sequence(class_id, sample_idx, split)
                np.save(path, raw)
                split_lines[split].append(f"{path},{class_id}")

                if split == "train":
                    transformed_train.append(to_double_relative_with_velocity(raw))

    for split, lines in split_lines.items():
        map_path = getattr(config, f"{split.upper()}_MAP_PATH")
        with open(map_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Saved {split} map: {map_path} ({len(lines)} samples)")

    stacked = np.concatenate(transformed_train, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    np.save(config.GLOBAL_MEAN_PATH, mean)
    np.save(config.GLOBAL_STD_PATH, std)
    print(f"Saved mean: {config.GLOBAL_MEAN_PATH}")
    print(f"Saved std:  {config.GLOBAL_STD_PATH}")
    print("Done. Toy dataset is ready for training.")


if __name__ == "__main__":
    main()
