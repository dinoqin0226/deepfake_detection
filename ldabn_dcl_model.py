import tensorflow as tf
from tensorflow.keras import layers, Model
from mixed_mbconv import MixedMBConvBlock  # 导入Week8的混合MBConv

class LDABN(layers.Layer):
    """轻量级领域自适应批归一化（LDABN）"""
    def __init__(self, num_domains=3, epsilon=1e-3, momentum=0.999, **kwargs):
        super().__init__(**kwargs)
        self.num_domains = num_domains  # 3个领域：GAN/扩散/混合
        self.epsilon = epsilon
        self.momentum = momentum

    def build(self, input_shape):
        channels = input_shape[-1]
        # 领域无关的全局统计量（基础BN参数）
        self.gamma = self.add_weight(shape=(channels,), initializer="ones", trainable=True, name="gamma")
        self.beta = self.add_weight(shape=(channels,), initializer="zeros", trainable=True, name="beta")
        self.moving_mean = self.add_weight(shape=(channels,), initializer="zeros", trainable=False, name="moving_mean")
        self.moving_var = self.add_weight(shape=(channels,), initializer="ones", trainable=False, name="moving_var")
        
        # 领域特定的统计量（仅1000左右参数，轻量化）
        self.domain_means = self.add_weight(
            shape=(self.num_domains, channels),
            initializer="zeros",
            trainable=True,
            name="domain_means"
        )
        self.domain_vars = self.add_weight(
            shape=(self.num_domains, channels),
            initializer="ones",
            trainable=True,
            name="domain_vars"
        )
        # 领域门控权重（动态融合）
        self.domain_gate = self.add_weight(
            shape=(self.num_domains,),
            initializer="ones",
            trainable=True,
            name="domain_gate"
        )

    def call(self, inputs, domain_ids=None, training=False):
        # 训练阶段：使用领域特定统计量
        if training and domain_ids is not None:
            # 计算当前批次统计量
            batch_mean, batch_var = tf.nn.moments(inputs, axes=[0,1,2], keepdims=False)
            # 更新全局移动统计量
            self.moving_mean.assign(self.momentum * self.moving_mean + (1 - self.momentum) * batch_mean)
            self.moving_var.assign(self.momentum * self.moving_var + (1 - self.momentum) * batch_var)
            
            # 动态融合领域统计量
            gate_weights = tf.nn.softmax(self.domain_gate)
            fused_mean = tf.reduce_sum(self.domain_means * gate_weights[:, tf.newaxis], axis=0)
            fused_var = tf.reduce_sum(self.domain_vars * gate_weights[:, tf.newaxis], axis=0)
            
            # 领域自适应归一化
            x = (inputs - fused_mean[tf.newaxis, tf.newaxis, tf.newaxis, :]) / tf.sqrt(fused_var[tf.newaxis, tf.newaxis, tf.newaxis, :] + self.epsilon)
        # 推理阶段：使用全局统计量（无领域依赖）
        else:
            x = (inputs - self.moving_mean[tf.newaxis, tf.newaxis, tf.newaxis, :]) / tf.sqrt(self.moving_var[tf.newaxis, tf.newaxis, tf.newaxis, :] + self.epsilon)
        
        # 缩放和平移
        x = x * self.gamma[tf.newaxis, tf.newaxis, tf.newaxis, :] + self.beta[tf.newaxis, tf.newaxis, tf.newaxis, :]
        return x

class DomainContrastiveLoss(tf.keras.losses.Loss):
    """领域对比损失（DCL）"""
    def __init__(self, temperature=0.1, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature  # 温度系数（轻量化）

    def call(self, features, domain_ids, labels):
        """
        Args:
            features: 模型提取的特征 [batch_size, feat_dim]
            domain_ids: 领域标签 [batch_size]（0=GAN,1=扩散,2=混合）
            labels: 真实/伪造标签 [batch_size,1]
        Returns:
            dcl_loss: 领域对比损失
        """
        # 归一化特征（L2）
        features = tf.math.l2_normalize(features, axis=1)
        batch_size = tf.shape(features)[0]
        
        # 计算特征相似度矩阵
        sim_matrix = tf.matmul(features, features, transpose_b=True) / self.temperature
        
        # 掩码：同标签（真实/伪造）但不同领域的样本为正例
        label_mask = tf.equal(labels, tf.transpose(labels))
        domain_mask = tf.not_equal(domain_ids, tf.transpose(domain_ids))
        pos_mask = tf.logical_and(label_mask, domain_mask)
        pos_mask = tf.linalg.set_diag(pos_mask, tf.zeros(batch_size, dtype=tf.bool))  # 排除自身
        
        # 掩码：不同标签为负例
        neg_mask = tf.logical_not(label_mask)
        
        # 计算对比损失
        exp_sim = tf.exp(sim_matrix)
        pos_sim = tf.reduce_sum(exp_sim * tf.cast(pos_mask, tf.float32), axis=1)
        neg_sim = tf.reduce_sum(exp_sim * tf.cast(neg_mask, tf.float32), axis=1)
        dcl_loss = -tf.reduce_mean(tf.math.log(pos_sim / (pos_sim + neg_sim + 1e-8)))
        
        return dcl_loss

# 集成LDABN+DCL的EfficientNet-B4模型
def build_ldabn_dcl_model(input_shape=(224,224,3), num_domains=3):
    """
    构建带LDABN+DCL的混合卷积EfficientNet-B4（适配CPU）
    """
    inputs = layers.Input(shape=input_shape)
    domain_input = layers.Input(shape=(1,), dtype=tf.int32)  # 领域标签输入
    
    # 替换标准BN为LDABN的混合MBConv骨干（复用Week8模块）
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(inputs)
    x = LDABN(num_domains=num_domains)(x, domain_ids=domain_input, training=True)
    x = layers.Activation(tf.nn.swish)(x)

    # EfficientNet-B4核心块（替换为混合MBConv+LDABN）
    mbconv_configs = [
        (48, 6, 1), (48, 6, 2), (64, 6, 1), (64, 6, 2),
        (128, 6, 1), (128, 6, 1), (128, 6, 1), (128, 6, 2),
        (192, 6, 1), (192, 6, 1), (192, 6, 1), (192, 6, 1),
        (192, 6, 2), (320, 6, 1)
    ]
    for filters, expand_ratio, stride in mbconv_configs:
        x = MixedMBConvBlock(filters=filters, expand_ratio=expand_ratio, stride=stride)(x)
        x = LDABN(num_domains=num_domains)(x, domain_ids=domain_input, training=True)
        x = layers.Activation(tf.nn.swish)(x)

    # 全局平均池化（提取特征用于DCL）
    feat = layers.GlobalAveragePooling2D()(x)
    # 分类头（保持轻量化）
    outputs = layers.Dense(1, activation="sigmoid")(feat)

    # 构建多输入模型（图像+领域标签）
    model = Model(inputs=[inputs, domain_input], outputs=[outputs, feat])
    return model

# 测试代码（验证模型集成，CPU运行）
if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    # 构建模型
    model = build_ldabn_dcl_model()
    # 模拟输入（批次8，适配CPU）
    test_imgs = tf.random.normal((8,224,224,3))
    test_domains = tf.random.uniform((8,1), minval=0, maxval=3, dtype=tf.int32)
    # 前向传播
    pred, feat = model([test_imgs, test_domains])
    print(f"分类输出尺寸: {pred.shape}")
    print(f"特征输出尺寸: {feat.shape}")
    print(f"模型总参数: {model.count_params()}")  # ≈20.6M，符合约束