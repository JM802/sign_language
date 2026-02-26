import os
import json
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from multiprocessing import Pool
import logging
from datetime import datetime

# 导入配置
import config_3d

# =============================
# 日志配置
# =============================
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

# =============================
# 辅助函数
# =============================

def get_uniform_frame_indices(total_frames, target_frames):
    """
    均匀采样帧索引
    """
    if total_frames <= 0:
        return []
    
    if total_frames == target_frames:
        return list(range(total_frames))
        
    # linspace 包含首尾
    indices = np.linspace(0, total_frames - 1, target_frames)
    indices = np.round(indices).astype(int)
    
    # 确保不越界
    indices = np.clip(indices, 0, total_frames - 1)
    return indices.tolist()

def extract_bbox_from_landmarks(landmarks, image_width, image_height):
    """
    从 MediaPipe 手部关键点提取单手 BBox (x_min, y_min, x_max, y_max)
    坐标为归一化后的像素值
    """
    if not landmarks:
        return None
        
    x_coords = [lm.x * image_width for lm in landmarks.landmark]
    y_coords = [lm.y * image_height for lm in landmarks.landmark]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    return [x_min, y_min, x_max, y_max]

def merge_bboxes(bbox1, bbox2):
    """
    合并两个 BBox，取得它们的最小外接矩形
    """
    if bbox1 is None:
        return bbox2
    if bbox2 is None:
        return bbox1
        
    x_min = min(bbox1[0], bbox2[0])
    y_min = min(bbox1[1], bbox2[1])
    x_max = max(bbox1[2], bbox2[2])
    y_max = max(bbox1[3], bbox2[3])
    
    return [x_min, y_min, x_max, y_max]

def expand_and_clamp_bbox(bbox, expand_ratio, image_width, image_height):
    """
    向外扩展 BBox，并确保不超出图像边界，同时使其尽量为正方形以便于缩放
    """
    if bbox is None:
        return None
        
    x_min, y_min, x_max, y_max = bbox
    
    # 计算当前中心和宽高
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    
    # 取最大边以产生正方形 BBox
    max_side = max(width, height)
    
    # 应用扩展比例
    new_side = max_side * expand_ratio
    
    # 计算新的边界
    new_x_min = center_x - new_side / 2
    new_y_min = center_y - new_side / 2
    new_x_max = center_x + new_side / 2
    new_y_max = center_y + new_side / 2
    
    # 限制在图像边界内
    new_x_min = max(0, int(new_x_min))
    new_y_min = max(0, int(new_y_min))
    new_x_max = min(image_width - 1, int(new_x_max))
    new_y_max = min(image_height - 1, int(new_y_max))
    
    # 确保宽度和高度大于0
    if new_x_max <= new_x_min or new_y_max <= new_y_min:
        return None
        
    return [new_x_min, new_y_min, new_x_max, new_y_max]

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

# =============================
# 核心处理流程
# =============================

def process_video_task(args):
    """
    单个视频处理任务（用于多进程调度）
    """
    video_id, subset, gloss_label, start_frame, end_frame = args
    
    vid_path = os.path.join(config_3d.VIDEO_DIR, f"{video_id}.mp4")
    bbox_cache_path = os.path.join(config_3d.BBOX_CACHE_DIR, f"{video_id}.npy")
    output_dir = os.path.join(config_3d.HAND_CROP_DIR, gloss_label, subset, video_id)
    
    # 1. 检查断点续传
    if config_3d.SKIP_EXISTING and os.path.exists(bbox_cache_path) and os.path.exists(output_dir):
        # 简单检查输出图片数量
        if len(os.listdir(output_dir)) == config_3d.SEQ_LEN:
            return True, video_id, "Skipped (Already exists)"
            
    if not os.path.exists(vid_path):
        return False, video_id, f"Video not found: {vid_path}"
        
    try:
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            return False, video_id, "Failed to open video"
            
        # 统计有效帧范围
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame == -1: start_frame = 0
        if end_frame == -1 or end_frame >= total_video_frames: end_frame = total_video_frames - 1
        
        valid_frame_count = end_frame - start_frame + 1
        
        if valid_frame_count < config_3d.SEQ_LEN // 2:
            cap.release()
            return False, video_id, f"Video too short ({valid_frame_count} valid frames)"
            
        # 2. 采样帧索引
        sample_indices = get_uniform_frame_indices(valid_frame_count, config_3d.SEQ_LEN)
        # 映射回视频的绝对帧号
        absolute_indices = [idx + start_frame for idx in sample_indices]
        
        # 读取需要的帧
        sampled_frames = []
        sampled_frames_raw_h_w = []
        for frame_idx in absolute_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                # 容错：如果读取失败，复用上一帧
                if sampled_frames:
                    sampled_frames.append(sampled_frames[-1].copy())
                    sampled_frames_raw_h_w.append(sampled_frames_raw_h_w[-1])
                continue
                
            sampled_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            sampled_frames_raw_h_w.append(frame.shape[:2])
            
        cap.release()
        
        # 应对末尾由于某些原因没有读够的情况
        while len(sampled_frames) < config_3d.SEQ_LEN:
            sampled_frames.append(sampled_frames[-1].copy())
            sampled_frames_raw_h_w.append(sampled_frames_raw_h_w[-1])
            
            
        # 3. 初始化 MediaPipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True, # 处理采样的离散帧，推荐 True
            max_num_hands=config_3d.MP_MAX_NUM_HANDS,
            min_detection_confidence=config_3d.MP_DETECTION_CONFIDENCE,
            min_tracking_confidence=config_3d.MP_TRACKING_CONFIDENCE
        )
        
        raw_bboxes = []
        # 4. 执行检测
        for i, image in enumerate(sampled_frames):
            results = hands.process(image)
            image_h, image_w = sampled_frames_raw_h_w[i]
            
            frame_bbox = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_bbox = extract_bbox_from_landmarks(hand_landmarks, image_w, image_h)
                    frame_bbox = merge_bboxes(frame_bbox, hand_bbox)
                    
            raw_bboxes.append(frame_bbox)
            
        hands.close()
        
        # 5. 插值与拓展
        interpolated_bboxes = interpolate_missing_bboxes(raw_bboxes)
        
        if interpolated_bboxes is None:
            return False, video_id, "No hands detected in the entire sequence"
            
        final_bboxes = []
        for i, bbox in enumerate(interpolated_bboxes):
            image_h, image_w = sampled_frames_raw_h_w[i]
            expanded = expand_and_clamp_bbox(bbox, config_3d.BBOX_EXPAND_RATIO, image_w, image_h)
            if expanded is None:
                # 极端异常情况回退
                expanded = [0, 0, image_w-1, image_h-1]
            final_bboxes.append(expanded)
            
        # 6. 保存 BBox 缓存
        os.makedirs(os.path.dirname(bbox_cache_path), exist_ok=True)
        np.save(bbox_cache_path, np.array(final_bboxes, dtype=np.int32))
        
        # 7. 裁剪图像并保存
        os.makedirs(output_dir, exist_ok=True)
        for i, (image, bbox) in enumerate(zip(sampled_frames, final_bboxes)):
            x_min, y_min, x_max, y_max = bbox
            crop_img = image[y_min:y_max+1, x_min:x_max+1]
            
            # 容错
            if crop_img.size == 0:
                 crop_img = image # 容错
                 
            resized_img = cv2.resize(crop_img, (config_3d.IMAGE_SIZE, config_3d.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            resized_img_bgr = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)
            
            save_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            
            # 也可以在这里传参做 jpeg 压缩控制，这里为了简便使用 opencv 默认
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), config_3d.JPEG_QUALITY]
            cv2.imwrite(save_path, resized_img_bgr, encode_param)
            
        return True, video_id, "Success"
        
    except Exception as e:
         return False, video_id, f"Exception: {str(e)}"

# =============================
# 主控入口
# =============================

def process_dataset():
    logger.info("Starting 3D Hand ROI Preprocessing...")
    logger.info(f"JSON Config: {config_3d.SPLIT_JSON_PATH}")
    
    if not os.path.exists(config_3d.SPLIT_JSON_PATH):
        logger.error(f"Cannot find split JSON file: {config_3d.SPLIT_JSON_PATH}")
        return
        
    with open(config_3d.SPLIT_JSON_PATH, "r") as f:
        split_data = json.load(f)
        
    # 构建任务列表
    # split_data 格式: video_id: {subset: 'train', action: [label_idx, start, end]}
    # 需要把 label_idx 转成分类名作为目录存储更好，或者直接用 index 字符串
    
    tasks = []
    for video_id, info in split_data.items():
        subset = info["subset"]
        label_idx = info["action"][0]
        start_frame = info["action"][1]
        end_frame = info["action"][2]
        
        # 将 glossary ID 作为字符串
        gloss_label = f"{label_idx:03d}" 
        
        tasks.append((video_id, subset, gloss_label, start_frame, end_frame))
        
    logger.info(f"Total videos to process: {len(tasks)}")
    logger.info(f"Num Workers: {config_3d.NUM_WORKERS if config_3d.NUM_WORKERS > 0 else 'Auto'}")
    
    num_workers = config_3d.NUM_WORKERS if config_3d.NUM_WORKERS > 0 else None
    
    success_count = 0
    fail_count = 0
    fail_details = []
    
    # 使用多进程池加速
    with Pool(processes=num_workers) as pool:
        # 使用 tqdm 显示进度
        results = list(tqdm(pool.imap_unordered(process_video_task, tasks), total=len(tasks), desc="Processing Videos"))
        
        for success, vid, msg in results:
            if success:
                success_count += 1
            else:
                fail_count += 1
                fail_details.append(f"{vid}: {msg}")
                logger.warning(f"Failed: {vid} - {msg}")
                
    logger.info("=" * 40)
    logger.info("Processing Summary:")
    logger.info(f"Total Attempted: {len(tasks)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {fail_count}")
    
    if fail_count > 0:
        logger.info("\nFirst 10 Failures:")
        for fd in fail_details[:10]:
            logger.info(f" - {fd}")
            
    logger.info("Data preprocessing completed.")

if __name__ == "__main__":
    process_dataset()
