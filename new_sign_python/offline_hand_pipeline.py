
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2


@dataclass
class HandLandmarkData:
    id: int
    x: float
    y: float
    z: float


@dataclass
class MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


def get_mysql_connection(cfg: MySQLConfig, with_database: bool = True):
    try:
        import pymysql
    except ImportError as exc:
        raise RuntimeError("pymysql is required: pip install pymysql") from exc

    kwargs = {
        "host": cfg.host,
        "port": cfg.port,
        "user": cfg.user,
        "password": cfg.password,
        "charset": "utf8mb4",
        "autocommit": False,
    }
    if with_database:
        kwargs["database"] = cfg.database
    return pymysql.connect(**kwargs)


def init_db(cfg: MySQLConfig) -> None:
    root_conn = get_mysql_connection(cfg, with_database=False)
    root_cur = root_conn.cursor()
    root_cur.execute(
        f"CREATE DATABASE IF NOT EXISTS `{cfg.database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
    )
    root_conn.commit()
    root_cur.close()
    root_conn.close()

    conn = get_mysql_connection(cfg, with_database=True)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS video_meta (
            video_id VARCHAR(255) PRIMARY KEY,
            video_path TEXT NOT NULL,
            fps DOUBLE NOT NULL,
            total_frames INT NOT NULL,
            duration_ms BIGINT NOT NULL,
            frame_interval_ms DOUBLE NOT NULL,
            sync_source VARCHAR(32) NOT NULL,
            z_scale DOUBLE NOT NULL,
            xy_divisor DOUBLE NOT NULL,
            y_offset DOUBLE NOT NULL,
            left_x_offset DOUBLE NOT NULL,
            right_x_offset DOUBLE NOT NULL,
            left_model_path TEXT,
            right_model_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS video_frames (
            video_id VARCHAR(255) NOT NULL,
            frame_index INT NOT NULL,
            timestamp_ms BIGINT NOT NULL,
            frame_time_sec DOUBLE NOT NULL,
            PRIMARY KEY(video_id, frame_index),
            INDEX idx_video_time(video_id, timestamp_ms)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS hand_landmarks (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            video_id VARCHAR(255) NOT NULL,
            frame_index INT NOT NULL,
            timestamp_ms BIGINT NOT NULL,
            hand_type VARCHAR(16) NOT NULL,
            hand_index INT NOT NULL,
            bound_area DOUBLE NOT NULL,
            hand_gesture VARCHAR(64) NOT NULL,
            landmarks_json LONGTEXT NOT NULL,
            INDEX idx_video_frame(video_id, frame_index),
            INDEX idx_video_time(video_id, timestamp_ms)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
    )

    conn.commit()
    cur.close()
    conn.close()


def compute_bound_area(landmarks: List[HandLandmarkData]) -> float:
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    if not xs or not ys:
        return 0.0
    return float((max(xs) - min(xs)) * (max(ys) - min(ys)))


def _clone_landmarks(landmarks: List[HandLandmarkData]) -> List[HandLandmarkData]:
    return [HandLandmarkData(id=p.id, x=p.x, y=p.y, z=p.z) for p in landmarks]


def _clone_hand_payload(hand: Dict) -> Dict:
    return {
        "hand_index": int(hand["hand_index"]),
        "hand_type": str(hand["hand_type"]),
        "bound_area": float(hand["bound_area"]),
        "hand_gesture": str(hand.get("hand_gesture", "unknown")),
        "landmarks": _clone_landmarks(hand["landmarks"]),
    }


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def _interpolate_hand_payload(prev_hand: Dict, next_hand: Dict, t: float) -> Dict:
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


def _sort_and_reindex_hands(hands_payload: List[Dict]) -> List[Dict]:
    order = {"Left": 0, "Right": 1}
    hands_sorted = sorted(hands_payload, key=lambda h: (order.get(h["hand_type"], 99), h["hand_index"]))
    for idx, hand in enumerate(hands_sorted):
        hand["hand_index"] = idx
    return hands_sorted


def _interpolate_missing_hands(frames: List[Dict], max_gap: int) -> List[Dict]:
    if not frames:
        return frames

    hand_types = ("Left", "Right")
    total_frames = len(frames)

    for hand_type in hand_types:
        frame_to_hand: Dict[int, Dict] = {}
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


def extract_direct_to_unity_gesture_stream(
    video_path: Path,
    output_jsonl: Path,
    max_hands: int,
    interpolate_missing: bool,
    interpolate_max_gap: int,
    swap_handedness: bool,
) -> None:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("mediapipe is required for direct-export mode: pip install mediapipe") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    frames: List[Dict] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        hands_payload: List[Dict] = []
        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_index, (hand_landmarks, handedness) in enumerate(
                zip(result.multi_hand_landmarks, result.multi_handedness)
            ):
                landmarks: List[HandLandmarkData] = []
                for landmark_id, lm in enumerate(hand_landmarks.landmark):
                    # Keep MediaPipe's raw normalized coordinates so they match the
                    # existing Unity playback math in Test.cs.
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

                bound_area = compute_bound_area(landmarks)
                hands_payload.append(
                    {
                        "hand_index": hand_index,
                        "hand_type": hand_type,
                        "bound_area": bound_area,
                        "hand_gesture": "unknown",
                        "landmarks": landmarks,
                    }
                )

        hands_payload = _sort_and_reindex_hands(hands_payload)
        frames.append(
            {
                "hand_count": len(hands_payload),
                "hands": hands_payload,
            }
        )
        frame_index += 1

    if interpolate_missing:
        frames = _interpolate_missing_hands(frames, max_gap=interpolate_max_gap)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for packet in frames:
            serializable_packet = {
                "hand_count": packet["hand_count"],
                "hands": [
                    {
                        "hand_index": hand["hand_index"],
                        "hand_type": hand["hand_type"],
                        "bound_area": hand["bound_area"],
                        "hand_gesture": hand["hand_gesture"],
                        "landmarks": [p.__dict__ for p in hand["landmarks"]],
                    }
                    for hand in packet["hands"]
                ],
            }
            f.write(json.dumps(serializable_packet, ensure_ascii=False) + "\n")

    hands.close()
    cap.release()
    print(
        f"[direct-export-gesture-stream] video={video_path.name}, frames={frame_index}, "
        f"interpolate_missing={interpolate_missing}, max_gap={interpolate_max_gap}, out={output_jsonl}"
    )


def upsert_video_meta(
    conn,
    video_id: str,
    video_path: Path,
    fps: float,
    total_frames: int,
    duration_ms: int,
    frame_interval_ms: float,
    sync_source: str,
    z_scale: float,
    xy_divisor: float,
    y_offset: float,
    left_x_offset: float,
    right_x_offset: float,
    left_model_path: Optional[Path],
    right_model_path: Optional[Path],
) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO video_meta(
            video_id, video_path, fps, total_frames, duration_ms, frame_interval_ms,
            sync_source, z_scale, xy_divisor, y_offset, left_x_offset, right_x_offset,
            left_model_path, right_model_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            video_path=VALUES(video_path),
            fps=VALUES(fps),
            total_frames=VALUES(total_frames),
            duration_ms=VALUES(duration_ms),
            frame_interval_ms=VALUES(frame_interval_ms),
            sync_source=VALUES(sync_source),
            z_scale=VALUES(z_scale),
            xy_divisor=VALUES(xy_divisor),
            y_offset=VALUES(y_offset),
            left_x_offset=VALUES(left_x_offset),
            right_x_offset=VALUES(right_x_offset),
            left_model_path=VALUES(left_model_path),
            right_model_path=VALUES(right_model_path)
        """,
        (
            video_id,
            str(video_path),
            float(fps),
            int(total_frames),
            int(duration_ms),
            float(frame_interval_ms),
            sync_source,
            float(z_scale),
            float(xy_divisor),
            float(y_offset),
            float(left_x_offset),
            float(right_x_offset),
            str(left_model_path) if left_model_path else None,
            str(right_model_path) if right_model_path else None,
        ),
    )
    conn.commit()
    cur.close()


def clear_video_rows(conn, video_id: str) -> None:
    cur = conn.cursor()
    cur.execute("DELETE FROM hand_landmarks WHERE video_id=%s", (video_id,))
    cur.execute("DELETE FROM video_frames WHERE video_id=%s", (video_id,))
    conn.commit()
    cur.close()


def extract_to_db(
    video_path: Path,
    mysql_cfg: MySQLConfig,
    video_id: str,
    max_hands: int,
    sync_source: str,
    z_scale: float,
    xy_divisor: float,
    y_offset: float,
    left_x_offset: float,
    right_x_offset: float,
    left_model_path: Optional[Path],
    right_model_path: Optional[Path],
) -> None:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("mediapipe is required for extract mode: pip install mediapipe") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    init_db(mysql_cfg)
    conn = get_mysql_connection(mysql_cfg, with_database=True)
    clear_video_rows(conn, video_id)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_cur = conn.cursor()
    hand_cur = conn.cursor()

    frame_index = 0
    total_hand_rows = 0
    last_ts = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms <= 0:
            timestamp_ms = int(round(frame_index * 1000.0 / fps))
        last_ts = timestamp_ms

        frame_cur.execute(
            """
            INSERT INTO video_frames(video_id, frame_index, timestamp_ms, frame_time_sec)
            VALUES (%s, %s, %s, %s)
            """,
            (video_id, frame_index, timestamp_ms, frame_index / fps),
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_index, (hand_lms, handedness) in enumerate(
                zip(result.multi_hand_landmarks, result.multi_handedness)
            ):
                hand_type = handedness.classification[0].label
                landmarks: List[HandLandmarkData] = []
                for i, lm in enumerate(hand_lms.landmark):
                    landmarks.append(
                        HandLandmarkData(id=i, x=float(lm.x), y=float(lm.y), z=float(lm.z))
                    )

                bound_area = compute_bound_area(landmarks)
                hand_gesture = "unknown"
                landmarks_json = json.dumps([p.__dict__ for p in landmarks], ensure_ascii=False)

                hand_cur.execute(
                    """
                    INSERT INTO hand_landmarks(
                        video_id, frame_index, timestamp_ms, hand_type, hand_index,
                        bound_area, hand_gesture, landmarks_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        video_id,
                        frame_index,
                        timestamp_ms,
                        hand_type,
                        hand_index,
                        bound_area,
                        hand_gesture,
                        landmarks_json,
                    ),
                )
                total_hand_rows += 1

        if frame_index % 200 == 0:
            conn.commit()
        frame_index += 1

    conn.commit()

    upsert_video_meta(
        conn=conn,
        video_id=video_id,
        video_path=video_path,
        fps=fps,
        total_frames=frame_index,
        duration_ms=last_ts,
        frame_interval_ms=(1000.0 / fps),
        sync_source=sync_source,
        z_scale=z_scale,
        xy_divisor=xy_divisor,
        y_offset=y_offset,
        left_x_offset=left_x_offset,
        right_x_offset=right_x_offset,
        left_model_path=left_model_path,
        right_model_path=right_model_path,
    )

    frame_cur.close()
    hand_cur.close()
    conn.close()
    hands.close()
    cap.release()

    print(
        f"[extract] video_id={video_id}, total_frames={frame_index}, "
        f"frame_rows={frame_index}, hand_rows={total_hand_rows}, duration_ms={last_ts}"
    )


def export_unity_json(mysql_cfg: MySQLConfig, video_id: str, output_json: Path) -> None:
    conn = get_mysql_connection(mysql_cfg, with_database=True)

    cur = conn.cursor()
    cur.execute(
        """
        SELECT video_id, video_path, fps, total_frames, duration_ms, frame_interval_ms,
               sync_source, z_scale, xy_divisor, y_offset, left_x_offset, right_x_offset,
               left_model_path, right_model_path
        FROM video_meta
        WHERE video_id = %s
        """,
        (video_id,),
    )
    meta_row = cur.fetchone()
    if not meta_row:
        raise RuntimeError(f"video_id not found in video_meta: {video_id}")

    meta = {
        "video_id": meta_row[0],
        "video_path": meta_row[1],
        "fps": float(meta_row[2]),
        "total_frames": int(meta_row[3]),
        "duration_ms": int(meta_row[4]),
        "frame_interval_ms": float(meta_row[5]),
        "sync_source": meta_row[6],
        "render_params": {
            "z_scale": float(meta_row[7]),
            "xy_divisor": float(meta_row[8]),
            "y_offset": float(meta_row[9]),
            "left_x_offset": float(meta_row[10]),
            "right_x_offset": float(meta_row[11]),
        },
        "base_models": {
            "left_hand": meta_row[12],
            "right_hand": meta_row[13],
        },
    }

    cur.execute(
        """
        SELECT frame_index, timestamp_ms, frame_time_sec
        FROM video_frames
        WHERE video_id=%s
        ORDER BY frame_index ASC
        """,
        (video_id,),
    )
    frames: Dict[int, Dict] = {}
    for frame_index, timestamp_ms, frame_time_sec in cur.fetchall():
        frames[int(frame_index)] = {
            "frame_index": int(frame_index),
            "timestamp_ms": int(timestamp_ms),
            "frame_time_sec": float(frame_time_sec),
            "hands": [],
        }

    cur.execute(
        """
        SELECT frame_index, timestamp_ms, hand_type, hand_index, bound_area, hand_gesture, landmarks_json
        FROM hand_landmarks
        WHERE video_id=%s
        ORDER BY frame_index ASC, hand_index ASC
        """,
        (video_id,),
    )

    for frame_index, timestamp_ms, hand_type, hand_index, bound_area, hand_gesture, landmarks_json in cur.fetchall():
        frame = frames.get(int(frame_index))
        if frame is None:
            continue
        frame["hands"].append(
            {
                "timestamp_ms": int(timestamp_ms),
                "hand_type": str(hand_type),
                "hand_index": int(hand_index),
                "bound_area": float(bound_area),
                "hand_gesture": str(hand_gesture),
                "landmarks": json.loads(landmarks_json),
            }
        )

    payload = {
        "meta": meta,
        "frames": [frames[idx] for idx in sorted(frames.keys())],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    conn.close()
    print(f"[export] video_id={video_id}, frames={len(payload['frames'])}, out={output_json}")


def export_unity_gesture_stream(mysql_cfg: MySQLConfig, video_id: str, output_jsonl: Path) -> None:
    """
    Export per-frame GestureData packets compatible with Unity Test.cs expectation:
    {
      "hand_count": N,
      "hands": [...]
    }
    One JSON object per line (JSONL), ordered by frame_index.
    """
    conn = get_mysql_connection(mysql_cfg, with_database=True)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT frame_index
        FROM video_frames
        WHERE video_id=%s
        ORDER BY frame_index ASC
        """,
        (video_id,),
    )
    frame_indexes = [int(row[0]) for row in cur.fetchall()]

    cur.execute(
        """
        SELECT frame_index, hand_type, hand_index, bound_area, hand_gesture, landmarks_json
        FROM hand_landmarks
        WHERE video_id=%s
        ORDER BY frame_index ASC, hand_index ASC
        """,
        (video_id,),
    )

    hands_by_frame: Dict[int, List[Dict]] = {}
    for frame_index, hand_type, hand_index, bound_area, hand_gesture, landmarks_json in cur.fetchall():
        hands_by_frame.setdefault(int(frame_index), []).append(
            {
                "hand_index": int(hand_index),
                "hand_type": str(hand_type),
                "bound_area": float(bound_area),
                "hand_gesture": str(hand_gesture),
                "landmarks": json.loads(landmarks_json),
            }
        )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for idx in frame_indexes:
            hands = hands_by_frame.get(idx, [])
            pkt = {
                "hand_count": len(hands),
                "hands": hands,
            }
            f.write(json.dumps(pkt, ensure_ascii=False) + "\n")

    cur.close()
    conn.close()
    print(f"[export-gesture-stream] video_id={video_id}, frames={len(frame_indexes)}, out={output_jsonl}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backend-only hand timestamp tagging pipeline (MySQL)."
    )
    parser.add_argument(
        "--mode",
        choices=["extract", "export-json", "export-gesture-stream", "direct-export-gesture-stream"],
        default="extract",
    )
    parser.add_argument("--video", help="Input video path for extract mode")
    parser.add_argument("--video-id", default=None, help="Video id; default uses input video stem")
    parser.add_argument("--max-hands", type=int, default=2)

    parser.add_argument("--mysql-host", default="127.0.0.1")
    parser.add_argument("--mysql-port", type=int, default=3306)
    parser.add_argument("--mysql-user", default="root")
    parser.add_argument("--mysql-password", default="")
    parser.add_argument("--mysql-database", default="sign_language")

    parser.add_argument("--sync-source", default="video_frame_time")

    parser.add_argument("--z-scale", type=float, default=8.0)
    parser.add_argument("--xy-divisor", type=float, default=1.5)
    parser.add_argument("--y-offset", type=float, default=0.3)
    parser.add_argument("--left-x-offset", type=float, default=-0.15)
    parser.add_argument("--right-x-offset", type=float, default=0.15)

    parser.add_argument("--left-hand-model", default=None)
    parser.add_argument("--right-hand-model", default=None)
    parser.add_argument("--interpolate-missing", action="store_true", default=True)
    parser.add_argument("--no-interpolate-missing", action="store_false", dest="interpolate_missing")
    parser.add_argument("--interpolate-max-gap", type=int, default=6)
    parser.add_argument("--swap-handedness", action="store_true",default=True)

    parser.add_argument("--output-json", default="new_sign_python/unity_playback_data.json")
    parser.add_argument("--output-jsonl", default="new_sign_python/unity_gesture_stream.jsonl")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    mysql_cfg = MySQLConfig(
        host=args.mysql_host,
        port=args.mysql_port,
        user=args.mysql_user,
        password=args.mysql_password,
        database=args.mysql_database,
    )

    if args.mode == "extract":
        if not args.video:
            raise RuntimeError("--video is required in extract mode")

        video_path = Path(args.video).resolve()
        video_id = args.video_id or video_path.stem

        extract_to_db(
            video_path=video_path,
            mysql_cfg=mysql_cfg,
            video_id=video_id,
            max_hands=args.max_hands,
            sync_source=args.sync_source,
            z_scale=args.z_scale,
            xy_divisor=args.xy_divisor,
            y_offset=args.y_offset,
            left_x_offset=args.left_x_offset,
            right_x_offset=args.right_x_offset,
            left_model_path=Path(args.left_hand_model).resolve() if args.left_hand_model else None,
            right_model_path=Path(args.right_hand_model).resolve() if args.right_hand_model else None,
        )
        return

    if args.mode == "export-json":
        if not args.video_id:
            raise RuntimeError("--video-id is required in export-json mode")
        output_json = Path(args.output_json).resolve()
        export_unity_json(mysql_cfg=mysql_cfg, video_id=args.video_id, output_json=output_json)
        return

    if args.mode == "export-gesture-stream":
        if not args.video_id:
            raise RuntimeError("--video-id is required in export-gesture-stream mode")
        output_jsonl = Path(args.output_jsonl).resolve()
        export_unity_gesture_stream(mysql_cfg=mysql_cfg, video_id=args.video_id, output_jsonl=output_jsonl)
        return

    if args.mode == "direct-export-gesture-stream":
        if not args.video:
            raise RuntimeError("--video is required in direct-export-gesture-stream mode")
        video_path = Path(args.video).resolve()
        output_jsonl = Path(args.output_jsonl).resolve()
        extract_direct_to_unity_gesture_stream(
            video_path=video_path,
            output_jsonl=output_jsonl,
            max_hands=args.max_hands,
            interpolate_missing=args.interpolate_missing,
            interpolate_max_gap=args.interpolate_max_gap,
            swap_handedness=args.swap_handedness,
        )
        return


if __name__ == "__main__":
    main()
