import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from config import CONFIG
from data.dataset_loader import DatasetLoader
from model_builder import build_complete_model, build_inference_model
from utils.common_utils import set_random_seed, setup_logger, load_best_model_weights, logger
from utils.visualization_utils import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, plot_grad_cam_heatmap

# 初始化配置
set_random_seed(CONFIG.SEED)
logger = setup_logger(CONFIG.LOG_DIR)

class ModelEvaluator:
    def __init__(self):
        self.complete_model = build_complete_model()
        self.inference_model = build_inference_model(self.complete_model)
        self.dataset_loader = DatasetLoader()
        # 加载最优权重
        self.complete_model, self.best_weight_path, self.best_metric = load_best_model_weights(
            self.complete_model,
            CONFIG.CHECKPOINT_DIR,
            metric_name="val_accuracy",
            mode="max"
        )
        logger.info(f"加载最优模型权重：{self.best_weight_path}（{self.best_metric:.4f}）")
        # 同步推理模型权重
        self.inference_model.set_weights([w for i, w in enumerate(self.complete_model.get_weights()) 
                                          if i < len(self.inference_model.get_weights())])
        
        # 获取测试集
        self.test_data = self.dataset_loader.split_dataset(*self.dataset_loader.load_dataset_metadata())[2]
        self.test_paths, self.test_labels, self.test_domains = self.test_data
    
    def _preprocess_test_data(self):
        """预处理测试数据"""
        logger.info("预处理测试数据")
        test_frames = []
        test_labels_onehot = []
        test_domains_onehot = []
        
        for path, label, domain in zip(self.test_paths, self.test_labels, self.test_domains):
            frames = self.dataset_loader.preprocessor.preprocess_single_video(path, is_training=False)
            if frames is not None:
                test_frames.append(frames)
                test_labels_onehot.append(tf.one_hot(label, depth=2).numpy())
                test_domains_onehot.append(tf.one_hot(domain, depth=3).numpy())
        
        self.test_frames = tf.convert_to_tensor(test_frames, dtype=tf.float32)
        self.test_labels_onehot = np.array(test_labels_onehot)
        self.test_domains_onehot = np.array(test_domains_onehot)
        self.y_true = np.argmax(self.test_labels_onehot, axis=1)
        
        logger.info(f"测试数据预处理完成：有效样本数={len(self.test_frames)}")
    
    def evaluate_complete_model(self):
        """评估完整模型（含域分类）"""
        logger.info("开始评估完整模型")
        self._preprocess_test_data()
        
        # 模型预测
        fake_pred, domain_pred = self.complete_model.predict(
            [self.test_frames, self.test_domains],
            batch_size=CONFIG.BATCH_SIZE,
            verbose=1
        )
        self.fake_pred = fake_pred
        self.domain_pred = domain_pred
        
        # 计算伪造检测指标
        y_pred = np.argmax(fake_pred, axis=1)
        accuracy = np.mean(self.y_true == y_pred)
        report = classification_report(self.y_true, y_pred, target_names=["Real", "Fake"])
        
        # 计算域分类指标
        domain_true = self.test_domains
        domain_pred_arg = np.argmax(domain_pred, axis=1)
        domain_accuracy = np.mean(domain_true == domain_pred_arg)
        
        # 保存评估结果
        self._save_evaluation_report(accuracy, domain_accuracy, report)
        
        # 可视化
        self._plot_evaluation_results()
        
        logger.info(f"完整模型评估完成：伪造检测准确率={accuracy:.4f}，域分类准确率={domain_accuracy:.4f}")
        return {
            "fake_detection_accuracy": accuracy,
            "domain_classification_accuracy": domain_accuracy,
            "classification_report": report
        }
    
    def evaluate_inference_model(self):
        """评估推理模型（轻量化）"""
        logger.info("开始评估推理模型")
        self._preprocess_test_data()
        
        # 模型预测
        fake_pred = self.inference_model.predict(
            self.test_frames,
            batch_size=CONFIG.BATCH_SIZE,
            verbose=1
        )
        y_pred = np.argmax(fake_pred, axis=1)
        accuracy = np.mean(self.y_true == y_pred)
        report = classification_report(self.y_true, y_pred, target_names=["Real", "Fake"])
        
        # 保存推理模型评估结果
        with open(os.path.join(CONFIG.EVAL_DIR, "inference_model_report.txt"), "w", encoding="utf-8") as f:
            f.write(f"推理模型评估报告\n")
            f.write(f"准确率：{accuracy:.4f}\n")
            f.write(f"分类报告：\n{report}")
        
        logger.info(f"推理模型评估完成：准确率={accuracy:.4f}")
        return {
            "inference_accuracy": accuracy,
            "classification_report": report
        }
    
    def evaluate_robustness(self):
        """评估模型鲁棒性（添加噪声/模糊）"""
        logger.info("开始评估模型鲁棒性")
        self._preprocess_test_data()
        
        # 定义扰动类型
        perturbations = {
            "gaussian_noise": self._add_gaussian_noise,
            "blur": self._add_blur,
            "compression": self._add_compression
        }
        
        robustness_results = {}
        for pert_name, pert_func in perturbations.items():
            # 应用扰动
            perturbed_frames = pert_func(self.test_frames)
            # 预测
            fake_pred = self.inference_model.predict(perturbed_frames, verbose=0)
            y_pred = np.argmax(fake_pred, axis=1)
            accuracy = np.mean(self.y_true == y_pred)
            robustness_results[pert_name] = accuracy
            logger.info(f"鲁棒性测试 - {pert_name}：准确率={accuracy:.4f}")
        
        # 保存鲁棒性结果
        with open(os.path.join(CONFIG.EVAL_DIR, "robustness_report.txt"), "w", encoding="utf-8") as f:
            f.write("模型鲁棒性评估报告\n")
            for pert_name, acc in robustness_results.items():
                f.write(f"{pert_name}：{acc:.4f}\n")
        
        return robustness_results
    
    def _add_gaussian_noise(self, frames, mean=0, std=0.05):
        """添加高斯噪声"""
        noise = tf.random.normal(shape=frames.shape, mean=mean, stddev=std)
        return tf.clip_by_value(frames + noise, 0.0, 1.0)
    
    def _add_blur(self, frames, kernel_size=(3, 3)):
        """添加高斯模糊"""
        return tf.image.gaussian_blur(frames, kernel_size)
    
    def _add_compression(self, frames, quality=50):
        """模拟视频压缩"""
        def compress_frame(frame):
            frame = tf.cast(frame * 255, tf.uint8)
            frame_jpeg = tf.image.encode_jpeg(frame, quality=quality)
            frame_decoded = tf.image.decode_jpeg(frame_jpeg)
            return tf.cast(frame_decoded, tf.float32) / 255.0
        
        return tf.map_fn(lambda x: tf.map_fn(compress_frame, x), frames)
    
    def _save_evaluation_report(self, fake_acc, domain_acc, report):
        """保存评估报告"""
        report_path = os.path.join(CONFIG.EVAL_DIR, "complete_model_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("完整模型评估报告\n")
            f.write(f"最优权重路径：{self.best_weight_path}\n")
            f.write(f"最优验证集指标：{self.best_metric:.4f}\n")
            f.write(f"伪造检测准确率：{fake_acc:.4f}\n")
            f.write(f"域分类准确率：{domain_acc:.4f}\n")
            f.write(f"分类报告：\n{report}")
        logger.info(f"评估报告已保存至：{report_path}")
    
    def _plot_evaluation_results(self):
        """绘制评估结果可视化图表"""
        # 混淆矩阵
        plot_confusion_matrix(self.y_true, self.fake_pred, CONFIG.EVAL_DIR, title="Confusion Matrix - Complete Model")
        # ROC曲线
        plot_roc_curve(self.y_true, self.fake_pred, CONFIG.EVAL_DIR, title="ROC Curve - Complete Model")
        # 精确率-召回率曲线
        plot_precision_recall_curve(self.y_true, self.fake_pred, CONFIG.EVAL_DIR, title="Precision-Recall Curve - Complete Model")
        # Grad-CAM热力图（选第一个样本）
        plot_grad_cam_heatmap(
            self.inference_model,
            self.test_frames[:1],
            layer_name="top_activation",  # EfficientNet-B4的顶层激活层
            save_path=CONFIG.EVAL_DIR
        )
        logger.info("评估可视化图表已保存")

# 测试模型评估
if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.evaluate_complete_model()
    evaluator.evaluate_inference_model()
    evaluator.evaluate_robustness()