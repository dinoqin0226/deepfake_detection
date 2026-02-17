import tensorflow as tf
from tensorflow.keras import layers

class StALiteModule(layers.Layer):
    """StA-Lite时空融合模块（轻量级时空注意力）"""
    def __init__(self, hidden_dim=512, num_heads=8, gru_units=256, dropout_rate=0.1, use_attention_fusion=True, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.use_attention_fusion = use_attention_fusion
        
        # 时间注意力层
        self.time_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            dropout=dropout_rate
        )
        # GRU层（捕获时间依赖）
        self.gru = layers.GRU(
            units=gru_units,
            return_sequences=True,
            return_state=True,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate
        )
        # 特征投影层
        self.projection = layers.Dense(hidden_dim, activation="swish")
        # Dropout层
        self.dropout = layers.Dropout(dropout_rate)
        # 层归一化
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs, training=False):
        """
        前向传播
        :param inputs: 输入张量 (batch_size, frames, feature_dim)
        :param training: 是否训练模式
        :return: 融合后的特征 (batch_size, feature_dim)
        """
        # 时间注意力
        time_attn_output = self.time_attention(
            query=inputs,
            value=inputs,
            key=inputs,
            training=training
        )
        # 残差连接 + 层归一化
        x = self.layer_norm(inputs + time_attn_output)
        
        # GRU处理时间序列
        gru_output, gru_state = self.gru(x, training=training)
        
        if self.use_attention_fusion:
            # 注意力加权融合所有帧特征
            attn_weights = tf.nn.softmax(self.projection(gru_output), axis=1)
            fused_features = tf.reduce_sum(gru_output * attn_weights, axis=1)
        else:
            # 直接取最后一帧GRU状态
            fused_features = gru_state
        
        # Dropout + 投影
        fused_features = self.dropout(fused_features, training=training)
        fused_features = self.projection(fused_features)
        
        return fused_features
    
    def get_config(self):
        """保存配置（用于模型序列化）"""
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "gru_units": self.gru_units,
            "dropout_rate": self.dropout_rate,
            "use_attention_fusion": self.use_attention_fusion
        })
        return config

# 测试StA-Lite模块
if __name__ == "__main__":
    input_tensor = layers.Input(shape=(150, 1792))  # (帧数, EfficientNet-B4特征维度)
    sta_lite = StALiteModule()
    output_tensor = sta_lite(input_tensor)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()