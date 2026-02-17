import os
import logging
import random
import numpy as np
import tensorflow as tf
from datetime import datetime

def set_random_seed(seed=42):
    """设置随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_logger(log_dir, log_name="deepfake_detector.log"):
    """配置日志系统，输出到文件和控制台"""
    # 创建日志器
    logger = logging.getLogger("DeepfakeDetector")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 避免重复添加处理器
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器（保存到日志文件）
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name), encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器（输出到终端）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def create_dir_if_not_exists(dir_path):
    """创建目录（不存在则创建）"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return True
    return False

def save_model_weights(model, save_dir, epoch, metric_value, metric_name="val_accuracy"):
    """保存模型权重（按指标命名，方便后续加载最佳模型）"""
    create_dir_if_not_exists(save_dir)
    weight_name = f"model_epoch_{epoch}_{metric_name}_{metric_value:.4f}.h5"
    weight_path = os.path.join(save_dir, weight_name)
    model.save_weights(weight_path)
    return weight_path

def load_best_model_weights(model, weight_dir, metric_name="val_accuracy", mode="max"):
    """加载最优模型权重（按指标值筛选）"""
    weight_files = [f for f in os.listdir(weight_dir) if f.endswith(".h5") and metric_name in f]
    if not weight_files:
        raise FileNotFoundError(f"未在{weight_dir}找到包含{metric_name}的权重文件")
    
    # 提取指标值并排序
    weight_metrics = []
    for f in weight_files:
        # 解析文件名中的指标值（如 "model_epoch_10_val_accuracy_0.9123.h5" → 0.9123）
        metric_str = [s for s in f.split("_") if "." in s][0]
        metric_val = float(metric_str)
        weight_metrics.append((f, metric_val))
    
    # 按mode筛选最优权重（max：准确率；min：损失）
    if mode == "max":
        best_weight = max(weight_metrics, key=lambda x: x[1])
    else:
        best_weight = min(weight_metrics, key=lambda x: x[1])
    
    # 加载权重
    best_weight_path = os.path.join(weight_dir, best_weight[0])
    model.load_weights(best_weight_path)
    return model, best_weight_path, best_weight[1]

def calculate_inference_time(model, input_data, repeat_times=10):
    """计算模型推理时间（多次运行取平均，避免偶然误差）"""
    # 预热（避免首次运行包含初始化耗时）
    _ = model.predict(input_data, verbose=0)
    
    # 多次运行取平均
    total_time = 0.0
    for _ in range(repeat_times):
        start_time = datetime.now()
        _ = model.predict(input_data, verbose=0)
        end_time = datetime.now()
        total_time += (end_time - start_time).total_seconds()
    
    avg_time = total_time / repeat_times
    return avg_time