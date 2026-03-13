import os
import json
import cv2
import csv
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from multiprocessing import Pool
import logging
from datetime import datetime
from scipy.ndimage import gaussian_filter1d

# 导入配置
import config_3d

# 日志配置
os.makedirs(config_3d.LOG_DIR, exist_ok=True)
log_filename = os.path.join(config_3d.LOG_DIR, f"preprocess_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局变量（用于多进程重用对象）
global_hands = None

def init_worker():
    """每个 Worker 进程启动时初始化一次 MediaPipe 对象"""
    global global_hands
    mp_hands = mp.solutions.hands
    global_hands = mp_hands.Hands(
        static_image_mode=False, 
        max_num_hands=2,
        min_detection_confidence=config_3d.MP_DETECTION_CONFIDENCE,
        min_tracking_confidence=config_3d.MP_TRACKING_CONFIDENCE
    )

# 辅助函数
def extract_bbox_from_landmarks(landmarks, image_width, image_height):
    """
    从 MediaPipe 手部关键点提取单手 BBox (x_min, y_min, x_max, y_max)
    坐标为归一化后的像素值
    """
    if not landmarks:
        return None
        
    x_coords = [np.clip(lm.x * image_width, 0, image_width-1) for lm in landmarks.landmark]
    y_coords = [np.clip(lm.y * image_height, 0, image_height-1) for lm in landmarks.landmark]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    return [x_min, y_min, x_max, y_max]


def expand_and_clamp_bbox(bbox, expand_ratio, image_width, image_height):
    """
    不仅返回裁剪框，还返回 scale 和 offset 用于后续 3D 还原
    """
    if bbox is None:
        return None, None
        
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    
    # S2HAND 喜欢正方形输入，取最大边
    max_side = max(width, height) * expand_ratio
    
    # 计算最终裁剪区域
    new_x_min = int(center_x - max_side / 2)
    new_y_min = int(center_y - max_side / 2)
    
    # 原始坐标 = (裁剪后坐标 / IMAGE_SIZE) * max_side + new_x_min
    meta_params = {
        "offset": [new_x_min, new_y_min],
        "scale": max_side,
        "original_size": [image_width, image_height]
    }
    
    # 实际裁剪边界需限制在图像内
    crop_box = [
        max(0, new_x_min),
        max(0, new_y_min),
        min(image_width - 1, int(center_x + max_side / 2)),
        min(image_height - 1, int(center_y + max_side / 2))
    ]
    
    return crop_box, meta_params

def interpolate_missing_bboxes(bboxes):
    """
    对序列中丢失的 BBox 进行线性插值补全
    """
    bboxes_arr = np.array([b if b is not None else [np.nan]*4 for b in bboxes], dtype=np.float64)
    n = len(bboxes_arr)
    
    valid_indices = np.where(~np.isnan(bboxes_arr[:, 0]))[0]
    
    if len(valid_indices) == 0:
        return None # 一帧都没检测到
    
    # 如果只有一帧或者前面有缺失，进行外推（直接复制最近的有效帧）
    for i in range(4):
        # 内部缺失使用线性插值
        bboxes_arr[:, i] = np.interp(
            np.arange(n), 
            valid_indices, 
            bboxes_arr[valid_indices, i]
        )
        
    return bboxes_arr.astype(int).tolist()

def smooth_bboxes(bboxes, sigma=config_3d.SIGMA):
    """
    对 BBox 序列进行高斯平滑，减少检测框的震颤
    bboxes: shape (N, 4) 的列表或数组
    sigma: 标准差，越大越平滑，但过大会产生延迟感 (1.0-2.0 适合手语)
    """
    if bboxes is None or len(bboxes) < 3:
        return bboxes
    
    bboxes_arr = np.array(bboxes, dtype=np.float64)
    smoothed_bboxes = np.zeros_like(bboxes_arr)
    
    # 分别对 x_min, y_min, x_max, y_max 进行平滑
    for i in range(4):
        smoothed_bboxes[:, i] = gaussian_filter1d(bboxes_arr[:, i], sigma=sigma)
    
    return smoothed_bboxes.tolist()

def save_hand_sequence(frames, bboxes_and_metas, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metadata = []
    
    for i, (image, item) in enumerate(zip(frames, bboxes_and_metas)):
        bbox, meta = item
        x_min, y_min, x_max, y_max = bbox
        crop_img = image[y_min:y_max+1, x_min:x_max+1]
        
        if crop_img.size == 0:
            continue

        # 缩放
        resized_img = cv2.resize(crop_img, (config_3d.IMAGE_SIZE, config_3d.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        
        save_path = f"frame_{i:04d}.jpg"
        cv2.imwrite(os.path.join(output_dir, save_path), cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))
        
        # 存入本帧元数据
        meta["frame_idx"] = i
        meta["file_name"] = save_path
        metadata.append(meta)
    
    # 保存该手部序列的完整索引信息
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    return len(metadata) > 0

# 核心处理流程
def process_video_task(args):
    """
    单个视频处理任务：提取、插值、裁剪并保存左右手序列及元数据
    增加了自适应采样与首尾无效帧过滤
    """
    global global_hands
    video_file, gloss_label = args
    
    vid_path = os.path.join(config_3d.VIDEO_DIR, video_file)
    video_id = os.path.splitext(video_file)[0]
    # 输出路径: HAND_CROP_DIR / Gloss / VideoID
    output_dir_base = os.path.join(config_3d.HAND_CROP_DIR, gloss_label, video_id)
    
    if not os.path.exists(vid_path):
        return False, video_id, f"Video not found: {vid_path}"
        
    try:
        # 1. 完整读取视频所有帧 (1-2秒短视频直接全读进内存更高效)
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            return False, video_id, "Failed to open video"
            
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        total_frames = len(all_frames)
        if total_frames < 8: # 极度过短的视频直接放弃
            return False, video_id, f"Video too short ({total_frames} frames)"        
        
        raw_bboxes_R, raw_bboxes_L = [], []
        
        #初始化身份记录本（存储上一帧的中心点）
        last_pos = {"R": None, "L": None}
        lost_frames = {"R": 0, "L": 0}  # 新增

        # 3. 逐帧检测左右手，获取完整轨迹
        for frame_idx, image in enumerate(all_frames):
            results = global_hands.process(image)
            image_h, image_w = image.shape[:2]
            bbox_R, bbox_L = None, None

            if results.multi_hand_landmarks:
                # 提取当前帧检测到的所有 BBox
                current_detected_hands = []
                for hl in results.multi_hand_landmarks:
                    box = extract_bbox_from_landmarks(hl, image_w, image_h)

                    cx, cy = box[0] + (box[2]-box[0])/2, box[1] + (box[3]-box[1])/2

                    current_detected_hands.append({"box": box, "center": (cx, cy)})
                
                # 身份初始化（第一帧或丢失后重新找回）
                if last_pos["R"] is None and last_pos["L"] is None:
                    # 按 X 坐标排序，左边给 R，右边给 L
                    current_detected_hands.sort(key=lambda h: h["center"][0])

                    if len(current_detected_hands) == 1:
                        if current_detected_hands[0]["center"][0] < image_w / 2:
                            bbox_R = current_detected_hands[0]["box"]
                        else:
                            bbox_L = current_detected_hands[0]["box"]
                    else:
                        bbox_R = current_detected_hands[0]["box"]
                        bbox_L = current_detected_hands[-1]["box"]
                
                # 实时追踪（身份继承逻辑）
                else:
                    if len(current_detected_hands) == 1:
                        hand=current_detected_hands[0]
                        dist_to_R = np.linalg.norm(np.array(hand["center"]) - np.array(last_pos["R"])) if last_pos["R"] else 9999

                        dist_to_L = np.linalg.norm(np.array(hand["center"]) - np.array(last_pos["L"])) if last_pos["L"] else 9999
                    
                        if dist_to_R < dist_to_L:
                            if bbox_R is None: 
                                bbox_R = hand["box"] # 贴上 R 标签
                        else:
                            if bbox_L is None: 
                                bbox_L = hand["box"] # 贴上 L 标签

                    elif len(current_detected_hands) >= 2:
                        # 两只手都有，直接按距离各自匹配最近的
                        for hand in current_detected_hands:
                            dist_to_R = np.linalg.norm(np.array(hand["center"]) - np.array(last_pos["R"])) if last_pos["R"] else 9999

                            dist_to_L = np.linalg.norm(np.array(hand["center"]) - np.array(last_pos["L"])) if last_pos["L"] else 9999

                            if dist_to_R < dist_to_L:
                                if bbox_R is None:
                                    bbox_R = hand["box"]
                                else:
                                    bbox_L=hand["box"]
                                    
                            else:
                                if bbox_L is None: 
                                    bbox_L = hand["box"]
                                else:
                                    bbox_R=hand["box"]

            # 更新记录本，供下一帧使用
            if bbox_R: 
                last_pos["R"] = (bbox_R[0]+(bbox_R[2]-bbox_R[0])/2, bbox_R[1]+(bbox_R[3]-bbox_R[1])/2)

                lost_frames["R"] = 0
            else:
                lost_frames["R"] += 1

                if lost_frames["R"] > 2:  # 连续丢失2帧则重置
                    last_pos["R"] = None

            if bbox_L: 
                last_pos["L"] = (bbox_L[0]+(bbox_L[2]-bbox_L[0])/2, bbox_L[1]+(bbox_L[3]-bbox_L[1])/2)

                lost_frames["L"] = 0
            else:
                lost_frames["L"] += 1

                if lost_frames["L"] > 2:
                    last_pos["L"] = None

                        
            raw_bboxes_R.append(bbox_R)
            raw_bboxes_L.append(bbox_L)
            
        
        # 4. 后处理与自适应保存
        results_report = []
        for side, raw_bboxes in [("R", raw_bboxes_R), ("L", raw_bboxes_L)]:
            valid_indices = [idx for idx, b in enumerate(raw_bboxes) if b is not None]
            if not valid_indices:
                results_report.append(f"{side}:No_Detection")
                continue
                
            start_f, end_f = valid_indices[0], valid_indices[-1]
            active_len = end_f - start_f + 1
            if active_len < 8: 
                continue

            target_count = max(16, min(active_len, 64))
            sample_indices = np.round(np.linspace(start_f, end_f, target_count)).astype(int).tolist()
            
            sampled_frames = [all_frames[idx] for idx in sample_indices]
            sampled_bboxes = [raw_bboxes[idx] for idx in sample_indices]
            
            interp_bboxes = interpolate_missing_bboxes(sampled_bboxes)
            if interp_bboxes:
                smoothed_bboxes = [[int(v) for v in row] for row in smooth_bboxes(interp_bboxes, sigma=config_3d.SIGMA)]

                combined_data = []
                for i, b in enumerate(smoothed_bboxes):
                    h, w = sampled_frames[i].shape[:2]
                    c_box, c_meta = expand_and_clamp_bbox(b, config_3d.BBOX_EXPAND_RATIO, w, h)
                    if c_box is None: c_box = [0, 0, 223, 223]
                    c_meta["original_start_frame"] = start_f
                    combined_data.append((c_box, c_meta))
                
                success = save_hand_sequence(sampled_frames, combined_data, os.path.join(output_dir_base, side))
                results_report.append(f"{side}:Success({target_count}f)")

        return True, video_id, f"Details: {', '.join(results_report)}"
        
    except Exception as e:
        return False, video_id, str(e)
    
# 主控入口

def process_dataset():
    logger.info("Starting ASL Citizen 3D Hand ROI Preprocessing...")
    
    # 读取 ASL Citizen 表格 (TSV 或 CSV 格式)
    # 获取 CSV 路径（确保 config_3d.py 里的 CSV_PATH 指向了 splits/ 下的文件）
    csv_path = getattr(config_3d, 'CSV_PATH', None)
    
    if not csv_path or not os.path.exists(csv_path):
        logger.error(f"Cannot find CSV file: {csv_path}")
        return
        
    tasks = []
    # 使用','分隔符读取
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            video_file = row.get('Video file')
            gloss = row.get('Gloss')
            if video_file and gloss:
                tasks.append((video_file, gloss))
                
    logger.info(f"Total videos to process: {len(tasks)}")

    # 🌟【开启测试模式】：只取前 1 个视频处理
    tasks = tasks[:1]
    logger.info(f"🚀 TEST MODE: Processing only {len(tasks)} video.")
    
    success_count, fail_count = 0, 0
    
    #记得这里改回来
    with Pool(processes=1,initializer=init_worker) as pool:
        results = list(tqdm(pool.imap_unordered(process_video_task, tasks), total=len(tasks), desc="Processing Videos"))

        with open(os.path.join(config_3d.LOG_DIR, "failed_videos.txt"), "w") as f_fail:
            for success, vid, msg in results:
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    f_fail.write(f"{vid}\t{msg}\n")
                    logger.warning(f"Failed: {vid} - {msg}")
                    
    logger.info("=" * 40)
    logger.info(f"Total Attempted: {len(tasks)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info("Preprocessing completed.")

if __name__ == "__main__":
    process_dataset()
