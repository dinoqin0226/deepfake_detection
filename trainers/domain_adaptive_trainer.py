import os
import tensorflow as tf
import numpy as np
from config import CONFIG
from trainers.trainer import BaseTrainer
from utils.common_utils import set_random_seed, setup_logger, logger
from utils.visualization_utils import plot_confusion_matrix, plot_roc_curve

# 初始化配置
set_random_seed(CONFIG.SEED)
logger = setup_logger(CONFIG.LOG_DIR)

class DomainAdaptiveTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        # 域自适应训练额外配置
        self.domain_weights = self._get_domain_balance_weights()
    
    def _get_domain_balance_weights(self):
        """计算域平衡权重（解决不同域样本量不平衡问题）"""
        _, _, all_domains = self.dataset_loader.load_dataset_metadata()
        domain_counts = np.bincount(all_domains)
        total_samples = len(all_domains)
        domain_weights = total_samples / (len(domain_counts) * domain_counts)
        logger.info(f"域平衡权重：{domain_weights}")
        return domain_weights
    
    def _adapt_domain_loss(self, y_true, y_pred):
        """自定义域损失（加入平衡权重）"""
        # 解析域标签
        domain_true = tf.cast(y_true, tf.int32)
        # 应用平衡权重
        weights = tf.gather(self.domain_weights, domain_true)
        # 计算交叉熵损失
        cross_entropy = tf.keras.losses.categorical_crossentropy(domain_true, y_pred)
        weighted_loss = cross_entropy * weights
        return tf.reduce_mean(weighted_loss)
    
    def train(self):
        """执行域自适应训练（在基础训练上增强域泛化）"""
        logger.info("开始域自适应训练")
        
        # 重新编译模型（替换域损失为加权损失）
        self.model.compile(
            optimizer=self.model.optimizer,
            loss={
                "fake_detection_output": self.model.loss["fake_detection_output"],
                "domain_classification_output": self._adapt_domain_loss
            },
            loss_weights=self.model.loss_weights,
            metrics=self.model.metrics
        )
        
        # 适配数据集（加入域权重）
        def adapt_domain_dataset(dataset):
            return dataset.map(lambda x, y, z: ([x, z], [y, z]))
        
        train_dataset_adapted = adapt_domain_dataset(self.train_dataset)
        val_dataset_adapted = adapt_domain_dataset(self.val_dataset)
        
        # 开始训练
        history = self.model.fit(
            train_dataset_adapted,
            validation_data=val_dataset_adapted,
            epochs=CONFIG.EPOCHS,
            steps_per_epoch=self.train_steps_per_epoch,
            validation_steps=self.val_steps_per_epoch,
            callbacks=self.callbacks,
            verbose=1
        )
        
        # 跨域验证
        self._cross_domain_evaluation()
        
        # 保存训练结果
        plot_training_history(history, CONFIG.LOG_DIR)
        final_weight_path = os.path.join(CONFIG.CHECKPOINT_DIR, "domain_adaptive_best_model.h5")
        self.model.save_weights(final_weight_path)
        logger.info(f"域自适应模型权重已保存至：{final_weight_path}")
        
        return history
    
    def _cross_domain_evaluation(self):
        """跨域验证（测试不同域的模型性能）"""
        logger.info("开始跨域性能评估")
        test_data = self.dataset_loader.split_dataset(*self.dataset_loader.load_dataset_metadata())[2]
        test_paths, test_labels, test_domains = test_data
        
        # 按域分组评估
        domain_names = ["FaceForensics++", "OpenFake", "DFDC"]
        for domain_id in range(3):
            domain_mask = test_domains == domain_id
            domain_paths = test_paths[domain_mask]
            domain_labels = test_labels[domain_mask]
            
            if len(domain_paths) == 0:
                logger.warning(f"域{domain_names[domain_id]}无测试样本，跳过")
                continue
            
            # 预处理测试数据
            test_frames = []
            test_labels_onehot = []
            for path, label in zip(domain_paths, domain_labels):
                frames = self.dataset_loader.preprocessor.preprocess_single_video(path, is_training=False)
                if frames is not None:
                    test_frames.append(frames)
                    test_labels_onehot.append(tf.one_hot(label, depth=2).numpy())
            
            if len(test_frames) == 0:
                continue
            
            # 模型预测
            test_frames = tf.convert_to_tensor(test_frames, dtype=tf.float32)
            domain_input = tf.convert_to_tensor([domain_id]*len(test_frames), dtype=tf.int32)
            fake_pred, _ = self.model.predict([test_frames, domain_input], verbose=0)
            
            # 计算指标
            y_true = np.argmax(test_labels_onehot, axis=1)
            y_pred = np.argmax(fake_pred, axis=1)
            accuracy = np.mean(y_true == y_pred)
            
            # 可视化
            plot_confusion_matrix(
                y_true, fake_pred,
                os.path.join(CONFIG.EVAL_DIR, f"domain_{domain_names[domain_id]}"),
                title=f"Confusion Matrix - {domain_names[domain_id]}"
            )
            plot_roc_curve(
                y_true, fake_pred,
                os.path.join(CONFIG.EVAL_DIR, f"domain_{domain_names[domain_id]}"),
                title=f"ROC Curve - {domain_names[domain_id]}"
            )
            
            logger.info(f"域{domain_names[domain_id]}评估结果：准确率={accuracy:.4f}")

# 测试域自适应训练
if __name__ == "__main__":
    da_trainer = DomainAdaptiveTrainer()
    da_trainer.train()