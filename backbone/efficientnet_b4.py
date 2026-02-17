import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.efficientnet import EfficientNetB4 as KerasEfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input

def build_efficientnet_b4_backbone(input_shape=(224, 224, 3), weights="imagenet", include_top=False, dropout_rate=0.2):
    """
    构建EfficientNet-B4主干网络
    :param input_shape: 输入形状 (H, W, C)
    :param weights: 预训练权重（imagenet/None）
    :param include_top: 是否包含顶层分类器
    :param dropout_rate: Dropout率
    :return: EfficientNet-B4模型
    """
    # 输入层
    inputs = layers.Input(shape=input_shape)
    # 预处理（适配EfficientNet）
    x = preprocess_input(inputs)
    
    # 加载Keras官方EfficientNet-B4
    base_model = KerasEfficientNetB4(
        input_tensor=x,
        weights=weights,
        include_top=include_top,
        dropout_rate=dropout_rate
    )
    
    # 冻结特征提取层（后续在model_builder中调整）
    for layer in base_model.layers:
        layer.trainable = False
    
    # 构建主干模型
    backbone = Model(inputs=inputs, outputs=base_model.output, name="EfficientNetB4_Backbone")
    
    return backbone

# 测试主干网络构建
if __name__ == "__main__":
    backbone = build_efficientnet_b4_backbone()
    backbone.summary()