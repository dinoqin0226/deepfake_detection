import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from config import CONFIG
from data.dataset_loader import DatasetLoader
from model_builder import build_complete_model
from utils.common_utils import set_random_seed, setup_logger, save_model_weights, logger
from utils.visualization_utils import plot_training_history

# 初始化配置
set_random_seed(CONFIG.SEED)
logger = setup_logger(CONFIG.LOG_DIR)

class BaseTrainer:
    def __init__(self):
        self.model = build_complete_model()
        self.dataset_loader = DatasetLoader()
        self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_loader.load_all_datasets()
        
        # 计算训练步数（按epoch数）
        self.train_steps_per_epoch = len(self.dataset_loader.load_dataset_metadata()[0]) // CONFIG.BATCH_SIZE
        self.val_steps_per_epoch = len(self.dataset_loader.split_dataset(*self.dataset_loader.load_dataset_metadata())[1][0]) // CONFIG.BATCH_SIZE
        
        # 构建回调函数
        self.callbacks = self._build_callbacks()
    
    def _build_callbacks(self):
        """构建训练回调函数（早停、学习率调度、日志保存等）"""
        callbacks = []
        
        # 早停（避免过拟合）
        early_stopping = EarlyStopping(
            monitor=CONFIG.EARLY_STOPPING_CONFIG["monitor"],
            patience=CONFIG.EARLY_STOPPING_CONFIG["patience"],
            mode=CONFIG.EARLY_STOPPING_CONFIG["mode"],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # 学习率调度
        lr_scheduler = ReduceLROnPlateau(
            monitor="val_loss",
            factor=CONFIG.LR_SCHEDULER_CONFIG["factor"],
            patience=CONFIG.LR_SCHEDULER_CONFIG["patience"],
            min_lr=CONFIG.LR_SCHEDULER_CONFIG["min_lr"],
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # 模型权重保存（仅保存最优）
        checkpoint = ModelCheckpoint(
            os.path.join(CONFIG.CHECKPOINT_DIR, "best_model.h5"),
            monitor=CONFIG.EARLY_STOPPING_CONFIG["monitor"],
            save_best_only=True,
            save_weights_only=True,
            mode=CONFIG.EARLY_STOPPING_CONFIG["mode"],
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # 训练日志保存
        csv_logger = CSVLogger(
            os.path.join(CONFIG.LOG_DIR, "training_log.csv"),
            separator=",",
            append=False
        )
        callbacks.append(csv_logger)
        
        logger.info("训练回调函数构建完成")
        return callbacks
    
    def train(self):
        """执行基础训练流程"""
        logger.info("开始模型训练")
        logger.info(f"训练参数：epochs={CONFIG.EPOCHS}, batch_size={CONFIG.BATCH_SIZE}, lr={CONFIG.OPTIMIZER_CONFIG['learning_rate']}")
        
        # 适配多输出输入格式
        def adapt_dataset(dataset):
            return dataset.map(lambda x, y, z: ([x, z], [y, z]))
        
        train_dataset_adapted = adapt_dataset(self.train_dataset)
        val_dataset_adapted = adapt_dataset(self.val_dataset)
        
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
        
        # 保存训练历史可视化
        plot_training_history(history, CONFIG.LOG_DIR)
        logger.info("训练历史曲线已保存")
        
        # 保存最终权重
        final_weight_path = save_model_weights(
            self.model,
            CONFIG.CHECKPOINT_DIR,
            CONFIG.EPOCHS,
            history.history[CONFIG.EARLY_STOPPING_CONFIG["monitor"]][-1],
            CONFIG.EARLY_STOPPING_CONFIG["monitor"]
        )
        logger.info(f"最终模型权重已保存至：{final_weight_path}")
        
        return history

# 测试基础训练
if __name__ == "__main__":
    trainer = BaseTrainer()
    trainer.train()