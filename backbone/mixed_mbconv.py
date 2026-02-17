import tensorflow as tf
from tensorflow.keras import layers

def mbconv_block(input_tensor, expand_ratio, kernel_size, se_ratio, strides=1):
    """
    基础MBConv块（Mobile Inverted Residual Block）
    :param input_tensor: 输入张量
    :param expand_ratio: 扩展率
    :param kernel_size: 卷积核尺寸
    :param se_ratio: SE模块压缩率
    :param strides: 步长
    :return: 输出张量
    """
    input_channels = input_tensor.shape[-1]
    expanded_channels = input_channels * expand_ratio
    
    # 1x1扩展卷积
    x = layers.Conv2D(
        expanded_channels,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.swish)(x)
    
    # 深度卷积
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.swish)(x)
    
    # SE模块
    if se_ratio > 0:
        se_channels = int(input_channels * se_ratio)
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Reshape((1, 1, expanded_channels))(se)
        se = layers.Conv2D(
            se_channels,
            kernel_size=1,
            padding="same",
            activation=tf.nn.swish
        )(se)
        se = layers.Conv2D(
            expanded_channels,
            kernel_size=1,
            padding="same",
            activation=tf.nn.sigmoid
        )(se)
        x = layers.Multiply()([x, se])
    
    # 1x1投影卷积
    x = layers.Conv2D(
        input_channels,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    
    # 残差连接（步长为1且通道数匹配）
    if strides == 1 and input_channels == input_tensor.shape[-1]:
        x = layers.Add()([x, input_tensor])
    
    return x

def build_mixed_mbconv_block(expand_ratio=6, kernel_sizes=[3,5,7], se_ratio=0.25, input_tensor=None):
    """
    混合MBConv块（多卷积核并行）
    :param expand_ratio: 扩展率
    :param kernel_sizes: 混合卷积核尺寸列表
    :param se_ratio: SE模块压缩率
    :param input_tensor: 输入张量
    :return: 输出张量
    """
    if input_tensor is None:
        raise ValueError("input_tensor must be provided")
    
    # 并行多卷积核MBConv
    mbconv_branches = []
    for kernel_size in kernel_sizes:
        branch = mbconv_block(
            input_tensor=input_tensor,
            expand_ratio=expand_ratio,
            kernel_size=kernel_size,
            se_ratio=se_ratio,
            strides=1
        )
        mbconv_branches.append(branch)
    
    # 拼接所有分支
    x = layers.Concatenate(axis=-1)(mbconv_branches)
    
    # 融合卷积（降维至原通道数）
    input_channels = input_tensor.shape[-1]
    x = layers.Conv2D(
        input_channels,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.nn.swish)(x)
    
    # 残差连接
    x = layers.Add()([x, input_tensor])
    
    return x

# 测试混合MBConv块
if __name__ == "__main__":
    input_tensor = layers.Input(shape=(224, 224, 128))
    output_tensor = build_mixed_mbconv_block(input_tensor=input_tensor)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()