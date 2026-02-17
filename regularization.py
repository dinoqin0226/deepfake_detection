import tensorflow as tf
from tensorflow.keras import losses

class FocalLoss(losses.Loss):
    """Focal Loss（解决类别不平衡）"""
    def __init__(self, gamma=2.0, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        """
        计算Focal Loss
        :param y_true: 真实标签 (batch_size, num_classes)（one-hot）
        :param y_pred: 预测概率 (batch_size, num_classes)
        :return: Focal Loss值
        """
        # 标签平滑
        y_true = tf.one_hot(
            tf.argmax(y_true, axis=1),
            depth=tf.shape(y_true)[-1],
            on_value=1.0 - self.label_smoothing,
            off_value=self.label_smoothing / (tf.shape(y_true)[-1] - 1)
        )
        
        # 计算交叉熵
        cross_entropy = losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
        
        # 计算调制因子（1 - p_t)^gamma
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        modulating_factor = tf.pow(1 - p_t, self.gamma)
        
        # Focal Loss = -α * (1 - p_t)^gamma * log(p_t)
        # 这里α=1（简化版）
        focal_loss = modulating_factor * cross_entropy
        focal_loss = tf.reduce_mean(focal_loss)
        
        return focal_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "label_smoothing": self.label_smoothing
        })
        return config

# 测试Focal Loss
if __name__ == "__main__":
    focal_loss = FocalLoss(gamma=2.0)
    y_true = tf.one_hot([0, 1, 0, 1], depth=2)
    y_pred = tf.constant([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])
    loss = focal_loss(y_true, y_pred)
    print(f"Focal Loss: {loss.numpy():.4f}")