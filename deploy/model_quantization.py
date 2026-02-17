import os
import tensorflow as tf
from config import CONFIG
from model_builder import build_inference_model
from utils.common_utils import set_random_seed, setup_logger, logger

# 初始化配置
set_random_seed(CONFIG.SEED)
logger = setup_logger(CONFIG.LOG_DIR)

class ModelQuantizer:
    def __init__(self):
        self.inference_model = build_inference_model()
        # 加载最优权重
        self.inference_model.load_weights(os.path.join(CONFIG.CHECKPOINT_DIR, "best_model.h5"))
        logger.info("推理模型加载完成，开始量化")
    
    def _get_calibration_dataset(self):
        """获取校准数据集（用于后训练量化）"""
        # 生成模拟校准数据（匹配模型输入）
        calibration_data = tf.random.rand(
            CONFIG.QUANTIZATION_CONFIG["calibration_dataset_size"],
            CONFIG.FRAMES_PER_VIDEO,
            *CONFIG.IMG_SIZE,
            3
        ).astype(np.float32)
        return calibration_data
    
    def quantize_model(self):
        """模型量化（后训练量化，int8）"""
        logger.info("开始模型量化（int8）")
        
        # 转换为TensorFlow Lite模型
        converter = tf.lite.TFLiteConverter.from_keras_model(self.inference_model)
        
        # 配置量化参数
        if CONFIG.QUANTIZATION_CONFIG["enable_post_training_quant"]:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
            
            # 校准数据集
            calibration_data = self._get_calibration_dataset()
            def representative_dataset():
                for i in range(len(calibration_data)):
                    yield [calibration_data[i:i+1]]
            
            converter.representative_dataset = representative_dataset
        
        # 转换模型
        tflite_model = converter.convert()
        
        # 保存量化模型
        quant_model_path = os.path.join(CONFIG.CHECKPOINT_DIR, "quantized_model.tflite")
        with open(quant_model_path, "wb") as f:
            f.write(tflite_model)
        
        logger.info(f"量化模型已保存至：{quant_model_path}")
        
        # 验证量化模型
        self._validate_quantized_model(quant_model_path)
        
        return quant_model_path
    
    def _validate_quantized_model(self, quant_model_path):
        """验证量化模型是否可运行"""
        logger.info("验证量化模型")
        # 加载量化模型
        interpreter = tf.lite.Interpreter(model_path=quant_model_path)
        interpreter.allocate_tensors()
        
        # 检查输入输出形状
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info(f"量化模型输入形状：{input_details[0]['shape']}")
        logger.info(f"量化模型输出形状：{output_details[0]['shape']}")
        
        # 测试推理
        test_input = tf.random.rand(1, CONFIG.FRAMES_PER_VIDEO, *CONFIG.IMG_SIZE, 3).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        logger.info(f"量化模型推理测试完成，输出形状：{output.shape}")

# 测试模型量化
if __name__ == "__main__":
    quantizer = ModelQuantizer()
    quantizer.quantize_model()