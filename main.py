import os
import argparse
from config import CONFIG
from utils.common_utils import set_random_seed, setup_logger, logger

# 初始化配置
set_random_seed(CONFIG.SEED)
logger = setup_logger(CONFIG.LOG_DIR)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Deepfake Detection Project - Main Entry")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "da_train", "evaluate", "quantize", "speed_test", "detect"],
                        help="Running mode: train (base training), da_train (domain adaptive training), evaluate (model evaluation), quantize (model quantization), speed_test (inference speed test), detect (GUI detection)")
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"启动Deepfake Detection工具，模式：{args.mode}")
    
    try:
        if args.mode == "train":
            # 基础训练
            from trainers.trainer import BaseTrainer
            trainer = BaseTrainer()
            trainer.train()
        
        elif args.mode == "da_train":
            # 域自适应训练
            from trainers.domain_adaptive_trainer import DomainAdaptiveTrainer
            da_trainer = DomainAdaptiveTrainer()
            da_trainer.train()
        
        elif args.mode == "evaluate":
            # 模型评估
            from evaluators.evaluator import ModelEvaluator
            evaluator = ModelEvaluator()
            evaluator.evaluate_complete_model()
            evaluator.evaluate_inference_model()
            evaluator.evaluate_robustness()
        
        elif args.mode == "quantize":
            # 模型量化
            from deploy.model_quantization import ModelQuantizer
            quantizer = ModelQuantizer()
            quantizer.quantize_model()
        
        elif args.mode == "speed_test":
            # 推理速度测试
            from evaluators.inference_speed_test import InferenceSpeedTester
            speed_tester = InferenceSpeedTester()
            raw_time = speed_tester.test_raw_model_speed()
            quant_model_path = os.path.join(CONFIG.CHECKPOINT_DIR, "quantized_model.tflite")
            if os.path.exists(quant_model_path):
                quant_time = speed_tester.test_quantized_model_speed(quant_model_path)
                speed_tester.save_speed_report(raw_time, quant_time)
            else:
                speed_tester.save_speed_report(raw_time)
        
        elif args.mode == "detect":
            # GUI检测工具
            from deploy.deepfake_detector import DeepfakeDetectorGUI
            import tkinter as tk
            root = tk.Tk()
            app = DeepfakeDetectorGUI(root)
            root.mainloop()
        
        logger.info(f"{args.mode}模式执行完成")
    
    except Exception as e:
        logger.error(f"{args.mode}模式执行失败：{str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()