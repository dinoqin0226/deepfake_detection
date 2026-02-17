import tensorflow as tf
from tensorflow.keras import layers

class LDABNLayer(layers.Layer):
    """LDABN（Label-Domain Adaptive Batch Normalization）层"""
    def __init__(self, num_domains=3, momentum=0.99, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.num_domains = num_domains
        self.momentum = momentum
        self.epsilon = epsilon
        
        # 为每个域初始化gamma和beta
        self.gamma = []
        self.beta = []
        for _ in range(num_domains):
            self.gamma.append(self.add_weight(
                shape=(1,),
                initializer="ones",
                trainable=True,
                name=f"gamma_domain_{_}"
            ))
            self.beta.append(self.add_weight(
                shape=(1,),
                initializer="zeros",
                trainable=True,
                name=f"beta_domain_{_}"
            ))
        
        # 共享moving_mean和moving_variance
        self.moving_mean = None
        self.moving_variance = None
    
    def build(self, input_shape):
        """构建层（初始化移动均值和方差）"""
        feature_dim = input_shape[0][-1]
        self.moving_mean = self.add_weight(
            shape=(feature_dim,),
            initializer="zeros",
            trainable=False,
            name="moving_mean"
        )
        self.moving_variance = self.add_weight(
            shape=(feature_dim,),
            initializer="ones",
            trainable=False,
            name="moving_variance"
        )
        super().build(input_shape)
    
    def call(self, inputs, training=False):
        """
        前向传播
        :param inputs: [特征张量, 域标签]
               - 特征张量: (batch_size, feature_dim)
               - 域标签: (batch_size,) （整数，0~num_domains-1）
        :param training: 是否训练模式
        :return: 域自适应归一化后的特征
        """
        features, domain_labels = inputs
        batch_size = tf.shape(features)[0]
        
        # Batch Normalization
        if training:
            mean, variance = tf.nn.moments(features, axes=[0])
            # 更新移动均值和方差
            self.moving_mean.assign(self.momentum * self.moving_mean + (1 - self.momentum) * mean)
            self.moving_variance.assign(self.momentum * self.moving_variance + (1 - self.momentum) * variance)
        else:
            mean = self.moving_mean
            variance = self.moving_variance
        
        # 归一化
        normalized = tf.nn.batch_normalization(
            features, mean, variance, None, None, self.epsilon
        )
        
        # 根据域标签选择gamma和beta
        gamma = tf.gather(tf.concat(self.gamma, axis=0), domain_labels)
        beta = tf.gather(tf.concat(self.beta, axis=0), domain_labels)
        
        # 重塑维度以匹配特征
        gamma = tf.reshape(gamma, (batch_size, 1))
        beta = tf.reshape(beta, (batch_size, 1))
        
        # 缩放和平移
        output = normalized * gamma + beta
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_domains": self.num_domains,
            "momentum": self.momentum,
            "epsilon": self.epsilon
        })
        return config

class DomainContrastiveLoss(tf.keras.losses.Loss):
    """域对比损失（DCL）"""
    def __init__(self, temperature=0.07, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
    
    def call(self, y_true, y_pred):
        """
        计算域对比损失
        :param y_true: 真实域标签 (batch_size, num_domains)（one-hot）
        :param y_pred: 预测域特征/概率 (batch_size, num_domains)
        :return: 对比损失值
        """
        # 归一化特征
        y_pred = tf.math.l2_normalize(y_pred, axis=1)
        
        # 计算相似度矩阵
        similarity_matrix = tf.matmul(y_pred, y_pred, transpose_b=True) / self.temperature
        
        # 掩码：同一域为1，不同域为0
        domain_labels = tf.argmax(y_true, axis=1)
        mask = tf.equal(tf.expand_dims(domain_labels, 0), tf.expand_dims(domain_labels, 1))
        mask = tf.cast(mask, tf.float32)
        
        # 排除自身相似度
        mask_self = tf.eye(tf.shape(domain_labels)[0], dtype=tf.float32)
        mask = mask - mask_self
        
        # 计算对比损失
        exp_sim = tf.exp(similarity_matrix) * (1 - mask_self)
        log_prob = similarity_matrix - tf.math.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True))
        mean_log_prob_pos = tf.reduce_sum((mask * log_prob), axis=1) / tf.reduce_sum(mask, axis=1)
        loss = -tf.reduce_mean(mean_log_prob_pos)
        
        return loss
    
    def get_config(self):
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config

# 测试LDABN和DCL
if __name__ == "__main__":
    # 测试LDABN
    features = layers.Input(shape=(512,))
    domain_labels = layers.Input(shape=(1,), dtype=tf.int32)
    ldabn = LDABNLayer(num_domains=3)
    ldabn_output = ldabn([features, domain_labels])
    ldabn_model = tf.keras.Model(inputs=[features, domain_labels], outputs=ldabn_output)
    
    # 测试DCL损失
    dcl_loss = DomainContrastiveLoss()
    y_true = tf.one_hot([0, 0, 1, 1], depth=3)
    y_pred = tf.random.normal((4, 3))
    loss = dcl_loss(y_true, y_pred)
    print(f"DCL Loss: {loss.numpy():.4f}")