import os
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN  # 需安装：pip install mtcnn
from config import CONFIG
from utils.common_utils import create_dir_if_not_exists, logger

class DataPreprocessor:
    def __init__(self, img_size=CONFIG.IMG_SIZE, frame_rate=CONFIG.FRAME_RATE, frames_per_video=CONFIG.FRAMES_PER_VIDEO):
        self.img_size = img_size
        self.frame_rate = frame_rate
        self.frames_per_video = frames_per_video
        self.face_detector = MTCNN()  # 人脸检测器
    
    def extract_frames_from_video(self, video_path):
        """从视频中提取指定帧率的帧"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频：{video_path}")
            return None
        
        # 获取视频原始帧率
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        # 计算帧采样间隔
        frame_interval = max(1, int(original_fps / self.frame_rate))
        
        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 按间隔采样帧
            if frame_count % frame_interval == 0:
                # 转换为RGB（OpenCV默认BGR）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            frame_count += 1
        
        cap.release()
        
        # 如果帧数量不足，补最后一帧；如果过多，均匀采样
        if len(frames) < self.frames_per_video:
            frames += [frames[-1]] * (self.frames_per_video - len(frames))
        else:
            indices = np.linspace(0, len(frames)-1, self.frames_per_video, dtype=int)
            frames = [frames[i] for i in indices]
        
        return frames[:self.frames_per_video]
    
    def detect_and_crop_face(self, image):
        """检测并裁剪人脸（仅保留最大的人脸）"""
        results = self.face_detector.detect_faces(image)
        if not results:
            logger.warning("未检测到人脸，返回原图缩放结果")
            return cv2.resize(image, self.img_size)
        
        # 取最大的人脸
        max_face = max(results, key=lambda x: x['box'][2] * x['box'][3])
        x1, y1, width, height = max_face['box']
        x2, y2 = x1 + width, y1 + height
        
        # 边界检查
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        # 裁剪人脸并缩放
        face = image[y1:y2, x1:x2]
        face_resized = cv2.resize(face, self.img_size)
        return face_resized
    
    def apply_data_augmentation(self, image):
        """应用数据增强（训练集使用）"""
        # 转换为TensorFlow张量
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # 随机水平翻转
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_left_right(image)
        
        # 随机亮度调整
        image = tf.image.random_brightness(image, max_delta=0.2)
        
        # 随机对比度调整
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        
        # 归一化到[0, 1]
        image = image / 255.0
        
        return image.numpy()
    
    def preprocess_single_video(self, video_path, is_training=True):
        """预处理单个视频：提取帧→检测人脸→增强（可选）→标准化"""
        # 提取帧
        frames = self.extract_frames_from_video(video_path)
        if frames is None:
            return None
        
        # 处理每一帧
        processed_frames = []
        for frame in frames:
            # 检测并裁剪人脸
            face = self.detect_and_crop_face(frame)
            # 数据增强（仅训练集）
            if is_training:
                face = self.apply_data_augmentation(face)
            else:
                face = face / 255.0  # 仅归一化
            processed_frames.append(face)
        
        # 转换为numpy数组
        processed_frames = np.array(processed_frames, dtype=np.float32)
        return processed_frames
    
    def preprocess_video_batch(self, video_paths, labels, domain_labels, is_training=True):
        """预处理视频批次（适配模型输入）"""
        batch_frames = []
        batch_labels = []
        batch_domains = []
        
        for video_path, label, domain in zip(video_paths, labels, domain_labels):
            frames = self.preprocess_single_video(video_path, is_training)
            if frames is not None:
                batch_frames.append(frames)
                batch_labels.append(label)
                batch_domains.append(domain)
        
        # 转换为tensor
        batch_frames = tf.convert_to_tensor(batch_frames, dtype=tf.float32)
        batch_labels = tf.convert_to_tensor(batch_labels, dtype=tf.int32)
        batch_domains = tf.convert_to_tensor(batch_domains, dtype=tf.int32)
        
        # 标签one-hot编码
        batch_labels = tf.one_hot(batch_labels, depth=2)
        batch_domains = tf.one_hot(batch_domains, depth=CONFIG.LDABN_DCL_CONFIG["num_domains"])
        
        return batch_frames, batch_labels, batch_domains