from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from scipy.signal import savgol_filter
import cv2
import numpy as np
from tqdm import tqdm

import mediapipe as mp
mp_hands = mp.solutions.hands


@dataclass
class HandLandmarkData:
    id: int
    x: float
    y: float
    z: float


def compute_bound_area(landmarks: List[HandLandmarkData]) -> float:
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    if not xs or not ys:
        return 0.0
    return float((max(xs) - min(xs)) * (max(ys) - min(ys)))


def _clone_landmarks(landmarks: List[HandLandmarkData]) -> List[HandLandmarkData]:
    return [HandLandmarkData(id=p.id, x=p.x, y=p.y, z=p.z) for p in landmarks]


def _clone_hand_payload(hand: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "hand_index": int(hand["hand_index"]),
        "hand_type": str(hand["hand_type"]),
        "bound_area": float(hand["bound_area"]),
        "hand_gesture": str(hand.get("hand_gesture", "unknown")),
        "landmarks": _clone_landmarks(hand["landmarks"]),
    }


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def _interpolate_hand_payload(prev_hand: Dict[str, Any], next_hand: Dict[str, Any], t: float) -> Dict[str, Any]:
    prev_landmarks: List[HandLandmarkData] = prev_hand["landmarks"]
    next_landmarks: List[HandLandmarkData] = next_hand["landmarks"]
    landmarks: List[HandLandmarkData] = []

    for prev_lm, next_lm in zip(prev_landmarks, next_landmarks):
        landmarks.append(
            HandLandmarkData(
                id=prev_lm.id,
                x=_lerp(prev_lm.x, next_lm.x, t),
                y=_lerp(prev_lm.y, next_lm.y, t),
                z=_lerp(prev_lm.z, next_lm.z, t),
            )
        )

    return {
        "hand_index": int(prev_hand["hand_index"]),
        "hand_type": str(prev_hand["hand_type"]),
        "bound_area": _lerp(float(prev_hand["bound_area"]), float(next_hand["bound_area"]), t),
        "hand_gesture": str(prev_hand.get("hand_gesture", "unknown")),
        "landmarks": landmarks,
    }


def _sort_and_reindex_hands(hands_payload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    order = {"Left": 0, "Right": 1}
    hands_sorted = sorted(hands_payload, key=lambda h: (order.get(h["hand_type"], 99), h["hand_index"]))
    for idx, hand in enumerate(hands_sorted):
        hand["hand_index"] = idx
    return hands_sorted


def _interpolate_missing_hands(frames: List[Dict[str, Any]], max_gap: int) -> List[Dict[str, Any]]:
    if not frames:
        return frames

    for hand_type in ("Left", "Right"):
        frame_to_hand: Dict[int, Dict[str, Any]] = {}
        for frame_idx, packet in enumerate(frames):
            for hand in packet["hands"]:
                if hand["hand_type"] == hand_type:
                    frame_to_hand[frame_idx] = _clone_hand_payload(hand)
                    break

        available = sorted(frame_to_hand.keys())
        for start_idx, end_idx in zip(available, available[1:]):
            gap = end_idx - start_idx - 1
            if gap <= 0 or gap > max_gap:
                continue

            start_hand = frame_to_hand[start_idx]
            end_hand = frame_to_hand[end_idx]
            for missing_idx in range(start_idx + 1, end_idx):
                if any(h["hand_type"] == hand_type for h in frames[missing_idx]["hands"]):
                    continue
                t = (missing_idx - start_idx) / float(end_idx - start_idx)
                frames[missing_idx]["hands"].append(_interpolate_hand_payload(start_hand, end_hand, t))

    for packet in frames:
        packet["hands"] = _sort_and_reindex_hands(packet["hands"])
        packet["hand_count"] = len(packet["hands"])

    return frames


def _smooth_landmarks_savgol(frames: List[Dict[str, Any]], window: int = 7, poly: int = 2) -> List[Dict[str, Any]]:

    total = len(frames)
    win = min(window, total if total % 2 == 1 else total - 1)
    if win < poly + 2:
        return frames

    for hand_type in ("Left", "Right"):
        hand_frames: List[Optional[List[Any]]] = []
        frame_indices: List[int] = []
        for fi, packet in enumerate(frames):
            found = None
            for hand in packet["hands"]:
                if hand["hand_type"] == hand_type:
                    found = hand["landmarks"]
                    break
            hand_frames.append(found)
            if found is not None:
                frame_indices.append(fi)

        if len(frame_indices) < win:
            continue

        first_landmarks = hand_frames[frame_indices[0]]
        if first_landmarks is None:
            continue

        n_lm = len(first_landmarks)
        for lm_id in range(n_lm):
            for axis in ("x", "y", "z"):
                vals = np.array(
                    [
                        hand_frames[fi][lm_id][axis]
                        if hand_frames[fi] is not None and isinstance(hand_frames[fi][lm_id], dict)
                        else getattr(hand_frames[fi][lm_id], axis)
                        for fi in frame_indices
                    ],
                    dtype=float,
                )
                smoothed = savgol_filter(vals, window_length=win, polyorder=poly)

                for i, fi in enumerate(frame_indices):
                    landmarks = hand_frames[fi]
                    if landmarks is None:
                        continue
                    lm = landmarks[lm_id]
                    if isinstance(lm, dict):
                        lm[axis] = float(smoothed[i])
                    else:
                        setattr(lm, axis, float(smoothed[i]))

    return frames


def extract_frames_from_video(
    video_path: Path,
    max_hands: int,
    interpolate_missing: bool,
    interpolate_max_gap: int,
    swap_handedness: bool,
) -> Dict[str, Any]:

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_index = 0
    frames: List[Dict[str, Any]] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result: Any = hands.process(rgb)

        hands_payload: List[Dict[str, Any]] = []
        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_index, (hand_landmarks, handedness) in enumerate(
                zip(result.multi_hand_landmarks, result.multi_handedness)
            ):
                landmarks: List[HandLandmarkData] = []
                for landmark_id, lm in enumerate(hand_landmarks.landmark):
                    landmarks.append(
                        HandLandmarkData(
                            id=landmark_id,
                            x=float(lm.x),
                            y=float(lm.y),
                            z=float(lm.z),
                        )
                    )

                hand_type = handedness.classification[0].label
                if swap_handedness:
                    hand_type = "Right" if hand_type == "Left" else "Left" if hand_type == "Right" else hand_type

                hands_payload.append(
                    {
                        "hand_index": hand_index,
                        "hand_type": hand_type,
                        "bound_area": compute_bound_area(landmarks),
                        "hand_gesture": "unknown",
                        "landmarks": landmarks,
                    }
                )

        hands_payload = _sort_and_reindex_hands(hands_payload)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms <= 0:
            timestamp_ms = int(round(frame_index * 1000.0 / fps))

        frames.append(
            {
                "frame_index": frame_index,
                "timestamp_ms": timestamp_ms,
                "frame_time_sec": frame_index / fps,
                "hand_count": len(hands_payload),
                "hands": hands_payload,
            }
        )
        frame_index += 1

    hands.close()
    cap.release()

    if interpolate_missing:
        frames = _interpolate_missing_hands(frames, max_gap=interpolate_max_gap)

    frames = _smooth_landmarks_savgol(frames, window=7, poly=2)

    return {
        "meta": {
            "video_id": video_path.stem,
            "video_path": str(video_path),
            "fps": float(fps),
            "total_frames": len(frames),
            "duration_ms": frames[-1]["timestamp_ms"] if frames else 0,
            "frame_interval_ms": float(1000.0 / fps),
            "sync_source": "video_frame_time",
        },
        "frames": frames,
    }


def export_unity_json(video_path: Path, output_json: Path, max_hands: int, interpolate_missing: bool, interpolate_max_gap: int, swap_handedness: bool) -> None:
    payload = extract_frames_from_video(
        video_path=video_path,
        max_hands=max_hands,
        interpolate_missing=interpolate_missing,
        interpolate_max_gap=interpolate_max_gap,
        swap_handedness=swap_handedness,
    )

    serializable_frames = []
    for frame in payload["frames"]:
        serializable_frames.append(
            {
                "frame_index": frame["frame_index"],
                "timestamp_ms": frame["timestamp_ms"],
                "frame_time_sec": frame["frame_time_sec"],
                "hand_count": frame["hand_count"],
                "hands": [
                    {
                        "hand_index": hand["hand_index"],
                        "hand_type": hand["hand_type"],
                        "bound_area": hand["bound_area"],
                        "hand_gesture": hand["hand_gesture"],
                        "landmarks": [p.__dict__ for p in hand["landmarks"]],
                    }
                    for hand in frame["hands"]
                ],
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps({"meta": payload["meta"], "frames": serializable_frames}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[export-json] video={video_path.name}, frames={len(serializable_frames)}, out={output_json}")


def export_unity_gesture_stream(video_path: Path, output_jsonl: Path, max_hands: int, interpolate_missing: bool, interpolate_max_gap: int, swap_handedness: bool) -> None:
    payload = extract_frames_from_video(
        video_path=video_path,
        max_hands=max_hands,
        interpolate_missing=interpolate_missing,
        interpolate_max_gap=interpolate_max_gap,
        swap_handedness=swap_handedness,
    )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for frame in payload["frames"]:
            packet = {
                "frame_index": frame["frame_index"],
                "timestamp_ms": frame["timestamp_ms"],
                "hand_count": frame["hand_count"],
                "hands": [
                    {
                        "hand_index": hand["hand_index"],
                        "hand_type": hand["hand_type"],
                        "bound_area": hand["bound_area"],
                        "hand_gesture": hand["hand_gesture"],
                        "landmarks": [p.__dict__ for p in hand["landmarks"]],
                    }
                    for hand in frame["hands"]
                ],
            }
            f.write(json.dumps(packet, ensure_ascii=False) + "\n")

    print(f"[export-gesture-stream] video={video_path.name}, frames={len(payload['frames'])}, out={output_jsonl}")


def _iter_video_files(input_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
        ]
    )


def _normalize_word_label(word: str) -> str:
    normalized = word.strip().upper()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+\d+$", "", normalized)
    return normalized.strip()


def _extract_word_key(video_path: Path) -> str:
    stem = video_path.stem.strip()
    if "-" in stem:
        candidate = stem.split("-", 1)[1].strip()
        if candidate:
            return _normalize_word_label(candidate)
    return _normalize_word_label(stem)


def _select_unique_word_videos(
    video_files: List[Path],
    sample_size: int,
    random_seed: int,
) -> Tuple[List[Path], Dict[str, str]]:
    grouped: Dict[str, List[Path]] = {}
    for video_path in video_files:
        word_key = _extract_word_key(video_path)
        grouped.setdefault(word_key, []).append(video_path)

    if len(grouped) < sample_size:
        raise RuntimeError(f"Only found {len(grouped)} unique words, cannot sample {sample_size}.")

    rng = random.Random(random_seed)
    selected: Dict[str, Path] = {}
    words = list(grouped.keys())
    rng.shuffle(words)
    for word in words:
        if len(selected) >= sample_size:
            break
        selected[word] = rng.choice(grouped[word])

    if len(selected) < sample_size:
        raise RuntimeError(f"After sampling, only selected {len(selected)} unique words.")

    selected_words = sorted(selected.keys())[:sample_size]
    selected_videos = [selected[word] for word in selected_words]
    dictionary = {
        word: selected[word].name
        for word in selected_words
    }
    return selected_videos, dictionary


def _write_dictionary_files(output_dir: Path, dictionary_name: str, dictionary: Dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{dictionary_name}.json"
    txt_path = output_dir / f"{dictionary_name}.txt"

    json_path.write_text(
        json.dumps(
            {
                "dictionary_name": dictionary_name,
                "word_count": len(dictionary),
                "words": dictionary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    txt_lines = [f"{word}\t{video_name}" for word, video_name in sorted(dictionary.items())]
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")


def batch_export_unity_gesture_stream(
    input_dir: Path,
    json_output_dir: Path,
    video_output_dir: Path,
    max_hands: int,
    interpolate_missing: bool,
    interpolate_max_gap: int,
    swap_handedness: bool,
    sample_size: int,
    random_seed: int,
    dictionary_name: str,
) -> None:
    if not input_dir.exists():
        raise RuntimeError(f"Input directory not found: {input_dir}")

    video_files = _iter_video_files(input_dir)
    if not video_files:
        raise RuntimeError(f"No video files found in: {input_dir}")

    selected_videos, dictionary = _select_unique_word_videos(
        video_files=video_files,
        sample_size=sample_size,
        random_seed=random_seed,
    )

    if json_output_dir.exists():
        shutil.rmtree(json_output_dir)
    json_output_dir.mkdir(parents=True, exist_ok=True)

    if video_output_dir.exists():
        shutil.rmtree(video_output_dir)
    video_output_dir.mkdir(parents=True, exist_ok=True)

    for video_path in tqdm(selected_videos, desc="Exporting JSONL", unit="video"):
        output_jsonl = json_output_dir / f"unity_gesture_stream_{video_path.stem}.jsonl"
        export_unity_gesture_stream(
            video_path=video_path,
            output_jsonl=output_jsonl,
            max_hands=max_hands,
            interpolate_missing=interpolate_missing,
            interpolate_max_gap=interpolate_max_gap,
            swap_handedness=swap_handedness,
        )
        shutil.copy2(video_path, video_output_dir / video_path.name)

    _write_dictionary_files(output_dir=json_output_dir, dictionary_name=dictionary_name, dictionary=dictionary)

    print(
        f"[batch-export-gesture-stream] videos={len(selected_videos)}, "
        f"unique_words={len(dictionary)}, in={input_dir}, json_out={json_output_dir}, video_out={video_output_dir}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline hand pipeline without MySQL.")
    parser.add_argument(
        "--mode",
        choices=["export-gesture-stream", "batch-export-gesture-stream"],
        default="batch-export-gesture-stream",
    )
    parser.add_argument("--video", help="Input video path for single-file export")
    parser.add_argument(
        "--input-dir",
        default="/home/jm802/sign_language/data/ASL_Citizen/videos",
        help="Input folder for batch export",
    )
    parser.add_argument(
        "--json-output-dir",
        default="/home/jm802/sign_language/new_sign_python/ASL_300_JSON",
        help="JSON output folder for batch export",
    )
    parser.add_argument(
        "--video-output-dir",
        default="/home/jm802/sign_language/new_sign_python/ALS_300_VIDEO",
        help="Video output folder for batch export",
    )
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--interpolate-missing", action="store_true", default=True)
    parser.add_argument("--no-interpolate-missing", action="store_false", dest="interpolate_missing")
    parser.add_argument("--interpolate-max-gap", type=int, default=6)
    parser.add_argument("--swap-handedness", action="store_true", default=True)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dictionary-name", default="asl_300")
    parser.add_argument("--output-jsonl", default="new_sign_python/unity_gesture_stream.jsonl")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "batch-export-gesture-stream":
        batch_export_unity_gesture_stream(
            input_dir=Path(args.input_dir).resolve(),
            json_output_dir=Path(args.json_output_dir).resolve(),
            video_output_dir=Path(args.video_output_dir).resolve(),
            max_hands=args.max_hands,
            interpolate_missing=args.interpolate_missing,
            interpolate_max_gap=args.interpolate_max_gap,
            swap_handedness=args.swap_handedness,
            sample_size=args.sample_size,
            random_seed=args.random_seed,
            dictionary_name=args.dictionary_name,
        )
        return

    if not args.video:
        raise RuntimeError("--video is required in export-gesture-stream mode")

    export_unity_gesture_stream(
        video_path=Path(args.video).resolve(),
        output_jsonl=Path(args.output_jsonl).resolve(),
        max_hands=args.max_hands,
        interpolate_missing=args.interpolate_missing,
        interpolate_max_gap=args.interpolate_max_gap,
        swap_handedness=args.swap_handedness,
    )


if __name__ == "__main__":
    main()
