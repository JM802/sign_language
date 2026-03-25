import os
import re
import cv2
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS 
import config
from inference_camera import SignLanguageInferencePipeline


app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:18081",
            "http://127.0.0.1:18081",
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# 配置
ALS_JSON_PATH = config.ASL_JSON_ROOT
ASL_VIDEO_PATH = config.ASL_300_VIDEO
ASL_JSON_ROUTE = config.ASL_JSON_ROUTE
ASL_VIDEO_ROUTE = config.ASL_VIDEO_ROUTE

# 复用 inference_camera.py 的完整推理流水线
# 包含：模型加载、归一化统计量、标签映射、平滑器
pipeline = SignLanguageInferencePipeline(model_path=config.MODEL_PATH)


def _strip_prefix(filename: str) -> str:
    if filename.startswith("unity_gesture_stream_"):
        return filename[len("unity_gesture_stream_"):]
    return filename


def _extract_resource_stem(filename: str) -> str:
    return os.path.splitext(_strip_prefix(filename))[0]


def _extract_word_from_stem(stem: str) -> str:
    if "-" in stem:
        stem = stem.split("-", 1)[1]
    word = stem.strip().upper()
    word = re.sub(r"\s+", " ", word)
    word = re.sub(r"\s+\d+$", "", word)
    return word.strip()


def _build_asl_resource_index():
    json_index = {}
    video_index = {}

    if os.path.isdir(ALS_JSON_PATH):
        for filename in os.listdir(ALS_JSON_PATH):
            if filename.lower().endswith(".jsonl"):
                json_index[_extract_resource_stem(filename)] = filename

    if os.path.isdir(ASL_VIDEO_PATH):
        for filename in os.listdir(ASL_VIDEO_PATH):
            if filename.lower().endswith((".mp4")):
                video_index[_extract_resource_stem(filename)] = filename

    pairs = []
    for stem in sorted(set(json_index.keys()) & set(video_index.keys())):
        pairs.append(
            {
                "stem": stem,
                "word": _extract_word_from_stem(stem),
                "json_filename": json_index[stem],
                "video_filename": video_index[stem],
            }
        )
    return pairs


@app.route('/api/sign/predict', methods=['POST'])
def predict():
    """
    POST /api/sign/predict  (multipart/form-data)
    字段: video — 用户录制的手语视频 (mp4/webm/avi)

    后端处理流程:
      1. OpenCV 逐帧解码视频
      2. pipeline.extract_features(frame) → 134 维坐标
      3. pipeline.smoother.smooth(raw_feat) → 平滑
      4. pipeline.predict(frame_buffer)
         内部: preprocess_sequence → 双重相对坐标 + 线性抽样 → 模型推理
    """
    # 接收视频文件
    if 'video' not in request.files:
        return jsonify({"code": 400, "message": "Missing 'video' field"}), 400

    video_file = request.files['video']

    # 保存到临时文件 (OpenCV 需要文件路径)
    suffix = os.path.splitext(video_file.filename)[1] or '.mp4'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    video_file.save(tmp_path)
    tmp.close()

    try:
        # OpenCV 逐帧解码
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({"code": 400, "message": "Cannot decode video file"}), 400

        frame_buffer = []
        pipeline.smoother.prev_landmarks = None  # 每次请求重置平滑器

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 提取 134 维特征 + 平滑 (与摄像头推理完全一致)
            raw_feat = pipeline.extract_features(frame)
            smooth_feat = pipeline.smoother.smooth(raw_feat)
            if smooth_feat is not None:
                frame_buffer.append(smooth_feat)

        cap.release()

        # 帧数检查
        if len(frame_buffer) < 5:
            return jsonify({"code": 400, "message": "Video too short, minimum 5 frames required"}), 400

        # 推理 (内部: 双重相对坐标 + 归一化 + 线性插值到64帧 + 模型前向)
        word_name, confidence = pipeline.predict(frame_buffer)

        return jsonify({
            "code": 200,
            "data": {
                "word_name": word_name.lower(),
                "confidence": round(float(confidence), 4)
            }
        })

    finally:
        os.unlink(tmp_path)

@app.route('/api/sign/resources', methods=['GET'])
def get_resources():
    word = request.args.get('name')
    if not word:
        return jsonify({"code": 400, "msg": "Missing query parameter: name"}), 400

    word_query = word.strip().upper()
    matched_pairs = []

    for item in _build_asl_resource_index():
        if item["word"] != word_query:
            continue

        json_stem = _extract_resource_stem(item["json_filename"])
        video_stem = _extract_resource_stem(item["video_filename"])
        if json_stem != video_stem:
            continue

        matched_pairs.append(
            {
                "stem": item["stem"],
                "word": item["word"],
                "json_url": f"http://{request.host}{ASL_JSON_ROUTE}/{item['json_filename']}",
                "video_url": f"http://{request.host}{ASL_VIDEO_ROUTE}/{item['video_filename']}",
            }
        )

    if not matched_pairs:
        return jsonify({"code": 404, "msg": "Resource not found"}), 404

    return jsonify(
        {
            "code": 200,
            "data": {
                "word": word_query,
                "count": len(matched_pairs),
                "items": matched_pairs,
            },
        }
    )


@app.route(f'{ASL_JSON_ROUTE}/<path:filename>')
def serve_asl_json(filename):
    return send_from_directory(ALS_JSON_PATH, filename)


@app.route(f'{ASL_VIDEO_ROUTE}/<path:filename>')
def serve_asl_video(filename):
    return send_from_directory(ASL_VIDEO_PATH, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
