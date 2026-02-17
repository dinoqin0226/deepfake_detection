import os
import numpy as np
import tensorflow as tf
from config import CONFIG
from data.dataset_loader import DatasetLoader
from model_builder import build_inference_model
from utils.common_utils import set_random_seed, setup_logger, calculate_inference_time, logger

# 初始化配置
set_random_seed(CONFIG.SEED)
logger = setup_logger(CONFIG.LOG_DIR)

class InferenceSpeedTester:
    def __init__(self):
        self.inference_model = build_inference_model()
        self.dataset_loader = DatasetLoader()
        # 加载最优权重
        self.inference_model.load_weights(os.path.join(CONFIG.CHECKPOINT_DIR, "best_model.h5"))
        # 准备测试数据（模拟10秒视频，对应CONFIG.FRAMES_PER_VIDEO=150帧）
        self.test_video_frames = self._prepare_test_video_frames()
    
    def _prepare_test_video_frames(self):
        """准备测试视频帧（模拟10秒视频）"""
        # 随机生成测试帧（匹配模型输入尺寸）
        test_frames = np.random.rand(1, CONFIG.FRAMES_PER_VIDEO, *CONFIG.IMG_SIZE, 3).astype(np.float32)
        logger.info(f"测试数据准备完成：形状={test_frames.shape}")
        return test_frames
    
    def test_raw_model_speed(self):
        """测试原始推理模型速度"""
        logger.info("测试原始推理模型速度（10秒视频）")
        avg_time = calculate_inference_time(self.inference_model, self.test_video_frames, repeat_times=10)
        logger.info(f"原始模型平均推理时间：{avg_time:.2f}秒（要求≤{CONFIG.MAX_INFERENCE_TIME}秒）")
        
        # 验证是否满足PID要求
        if avg_time <= CONFIG.MAX_INFERENCE_TIME:
            logger.info("✓ 原始模型推理速度满足PID要求")
        else:
            logger.warning(f"✗ 原始模型推理速度不满足要求，需量化优化")
        
        return avg_time
    
    def test_quantized_model_speed(self, quantized_model_path):
        """测试量化后模型速度"""
        logger.info("测试量化模型速度（10秒视频）")
        # 加载量化模型
        interpreter = tf.lite.Interpreter(model_path=quantized_model_path)
        interpreter.allocate_tensors()
        
        # 获取输入输出张量
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # 预热
        interpreter.set_tensor(input_details[0]['index'], self.test_video_frames)
        interpreter.invoke()
        
        # 测试推理时间
        total_time = 0.0
        repeat_times = 10
        for _ in range(repeat_times):
            start_time = tf.timestamp()
            interpreter.set_tensor(input_details[0]['index'], self.test_video_frames)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
            total_time += (tf.timestamp() - start_time).numpy()
        
        avg_time = total_time / repeat_times
        logger.info(f"量化模型平均推理时间：{avg_time:.2f}秒（要求≤{CONFIG.MAX_INFERENCE_TIME}秒）")
        
        if avg_time <= CONFIG.MAX_INFERENCE_TIME:
            logger.info("✓ 量化模型推理速度满足PID要求")
        else:
            logger.warning(f"✗ 量化模型推理速度仍不满足要求，需进一步轻量化")
        
        return avg_time
    
    def save_speed_report(self, raw_time, quantized_time=None):
        """保存速度测试报告"""
        report_path = os.path.join(CONFIG.EVAL_DIR, "inference_speed_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("推理速度测试报告\n")
            f.write(f"测试条件：10秒视频，重复{10}次取平均\n")
            f.write(f"PID要求：≤{CONFIG.MAX_INFERENCE_TIME}秒\n")
            f.write(f"原始模型推理时间：{raw_time:.2f}秒\n")
            if quantized_time:
                f.write(f"量化模型推理时间：{quantized_time:.2f}秒\n")
                f.write(f"速度提升：{((raw_time - quantized_time)/raw_time)*100:.2f}%\n")
        logger.info(f"推理速度报告已保存至：{report_path}")

# 测试推理速度
if __name__ == "__main__":
    speed_tester = InferenceSpeedTester()
    raw_time = speed_tester.test_raw_model_speed()
    # 若有量化模型，测试量化速度
    quant_model_path = os.path.join(CONFIG.CHECKPOINT_DIR, "quantized_model.tflite")
    if os.path.exists(quant_model_path):
        quant_time = speed_tester.test_quantized_model_speed(quant_model_path)
        speed_tester.save_speed_report(raw_time, quant_time)
    else:
        speed_tester.save_speed_report(raw_time)