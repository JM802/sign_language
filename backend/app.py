import os
import sys
import cv2
import tempfile
import numpy as np
import pymysql
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS 
import config
from inference_camera import SignLanguageInferencePipeline


app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# 配置
ALS_JSON_PATH = config.ASL_JSON_ROOT

# 复用 inference_camera.py 的完整推理流水线
# 包含：模型加载、归一化统计量、标签映射、平滑器
pipeline = SignLanguageInferencePipeline(model_path=config.MODEL_PATH)


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
        os.unlink(tmp_path)  # 清理临时文件

@app.route('/api/sign/downloads', methods=['GET'])
def downloads():
    word = request.args.get('name')

    # 在这里转为大写，去匹配数据库
    word_query = word.strip().upper()

    # 从 MySQL 查文件夹名
    conn = pymysql.connect(**DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)
    with conn.cursor() as cursor:
        cursor.execute("SELECT folder_name FROM sign_assets WHERE word_name = %s", (word_query,))
        row = cursor.fetchone()
    conn.close()

    if not row:
        return jsonify({"code": 404, "msg": "Resource not found"}), 404

    folder = row['folder_name']
    abs_path = os.path.join(GLB_ROOT, folder)
    
    # 递归/遍历所有 .glb
    urls = []
    for root, _, files in os.walk(abs_path):
        for f in files:
            if f.endswith('.glb'):
                rel = os.path.relpath(os.path.join(root, f), GLB_ROOT)
                urls.append(f"http://{request.host}/result_3d/glb_models/{rel}")
    
    urls.sort() # 确保帧顺序
    return jsonify({"code": 200, "data": {"urls": urls}})

@app.route('/result_3d/glb_models/<path:filename>')
def serve_glb(filename):
    return send_from_directory(ALS_JSON_PATH, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)