import tensorflow as tf
from tensorflow.keras import layers, Model
from backbone.efficientnet_b4 import build_efficientnet_b4_backbone
from modules.sta_lite import StALiteModule
from config import CONFIG

def build_efficientnet_sta_model(input_shape=(CONFIG.FRAMES_PER_VIDEO, *CONFIG.IMG_SIZE, 3), num_classes=2):
    """
    构建EfficientNet+StA-Lite基础模型（历史版本）
    :param input_shape: 输入形状 (frames, H, W, C)
    :param num_classes: 分类数
    :return: 模型
    """
    # 输入层
    inputs = layers.Input(shape=input_shape)
    
    # 展开帧维度
    batch_size = tf.shape(inputs)[0]
    frames_flat = layers.Reshape((-1, *CONFIG.IMG_SIZE, 3))(inputs)
    
    # EfficientNet-B4特征提取
    backbone = build_efficientnet_b4_backbone(
        input_shape=(*CONFIG.IMG_SIZE, 3),
        weights="imagenet",
        include_top=False
    )
    frame_features = backbone(frames_flat)
    frame_features = layers.GlobalAveragePooling2D()(frame_features)
    
    # 恢复时间维度
    feature_dim = frame_features.shape[-1]
    video_features = layers.Reshape((CONFIG.FRAMES_PER_VIDEO, feature_dim))(frame_features)
    
    # StA-Lite时空融合
    sta_lite = StALiteModule(
        hidden_dim=CONFIG.STA_LITE_CONFIG["hidden_dim"],
        num_heads=CONFIG.STA_LITE_CONFIG["num_heads"],
        gru_units=CONFIG.STA_LITE_CONFIG["gru_units"]
    )
    fused_features = sta_lite(video_features)
    
    # 分类头
    x = layers.Dropout(0.2)(fused_features)
    x = layers.Dense(256, activation="swish")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    # 构建模型
    model = Model(inputs=inputs, outputs=outputs, name="EfficientNet_STA_Model")
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    
    return model

# 测试模型构建
if __name__ == "__main__":
    model = build_efficientnet_sta_model()
    model.summary()