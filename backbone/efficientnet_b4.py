import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses
import numpy as np

# --------------------------
# 1. SE Attention Module
# --------------------------
def se_block(input_tensor, reduction=4):
    """
    Squeeze-and-Excitation attention module (EfficientNet paper specification)
    :param input_tensor: Input feature tensor (batch, h, w, channels)
    :param reduction: Channel reduction ratio (default=4 as per paper)
    :return: Attention-weighted feature tensor
    """
    channels = input_tensor.shape[-1]
    # Squeeze: Global average pooling to aggregate spatial information
    x = layers.GlobalAveragePooling2D()(input_tensor)
    x = layers.Reshape((1, 1, channels))(x)
    # Excitation: Learn channel-wise attention weights
    x = layers.Dense(channels // reduction, activation='swish')(x)
    x = layers.Dense(channels, activation='sigmoid')(x)
    # Scale input features with attention weights
    output = layers.Multiply()([input_tensor, x])
    return output

# --------------------------
# 2. MBConv Module
# --------------------------
def mbconv_block(input_tensor, filters, kernel_size=3, strides=1, expansion_factor=6):
    """
    Mobile Inverted Residual Block (MBConv) with SE attention
    :param input_tensor: Input tensor (batch, h, w, in_channels)
    :param filters: Number of output channels
    :param kernel_size: Convolution kernel size (3/5 as per EfficientNet-B4)
    :param strides: Stride for depthwise convolution (1/2 for downsampling)
    :param expansion_factor: Channel expansion ratio (6 for MBConv6 in B4)
    :return: Output tensor of MBConv block
    """
    in_channels = input_tensor.shape[-1]
    expanded_channels = in_channels * expansion_factor

    # Step 1: Expansion convolution (1x1 pointwise conv to upscale channels)
    x = layers.Conv2D(expanded_channels, kernel_size=1, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    # Step 2: Depthwise convolution (reduce computation)
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    # Step 3: SE attention module
    x = se_block(x)

    # Step 4: Projection convolution (1x1 pointwise conv to downscale channels)
    x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Step 5: Residual connection (only if stride=1 and channel matches)
    if strides == 1 and in_channels == filters:
        x = layers.Add()([x, input_tensor])
    
    return x

# --------------------------
# 3. EfficientNet-B4 Architecture
# --------------------------
def build_efficientnet_b4(input_shape=(224, 224, 3), num_classes=1):
    """
    Build EfficientNet-B4 model (compound scaling from B0: α=1.2, β=1.1, γ=1.15)
    :param input_shape: Input tensor shape (default: 224×224×3 for preprocessed deepfake data)
    :param num_classes: Number of output classes (1 for binary classification: real/fake)
    :return: EfficientNet-B4 model
    """
    # Scaling coefficients for B4 (from EfficientNet paper)
    depth_coeff = 1.2
    width_coeff = 1.1
    # Resolution is 224 (instead of 380) for CPU efficiency

    # Define channel sizes for each stage (scaled from B0)
    channels = [32, 16, 24, 40, 80, 112, 192, 320]
    channels = [int(c * width_coeff) for c in channels]
    
    # Define depth (number of MBConv blocks per stage) (scaled from B0)
    depths = [1, 2, 2, 3, 3, 4, 1]
    depths = [int(d * depth_coeff) for d in depths]

    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Stage 1: Initial convolution
    x = layers.Conv2D(channels[0], kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    # Stage 2: MBConv1 (expansion factor=1)
    x = mbconv_block(x, filters=channels[1], kernel_size=3, strides=1, expansion_factor=1)

    # Stage 3: MBConv6
    for _ in range(depths[1]):
        x = mbconv_block(x, filters=channels[2], kernel_size=3, strides=2, expansion_factor=6)

    # Stage 4: MBConv6
    for _ in range(depths[2]):
        x = mbconv_block(x, filters=channels[3], kernel_size=5, strides=2, expansion_factor=6)

    # Stage 5: MBConv6
    for _ in range(depths[3]):
        x = mbconv_block(x, filters=channels[4], kernel_size=3, strides=2, expansion_factor=6)

    # Stage 6: MBConv6
    for _ in range(depths[4]):
        x = mbconv_block(x, filters=channels[5], kernel_size=5, strides=1, expansion_factor=6)

    # Stage 7: MBConv6
    for _ in range(depths[5]):
        x = mbconv_block(x, filters=channels[6], kernel_size=5, strides=2, expansion_factor=6)

    # Stage 8: MBConv6
    for _ in range(depths[6]):
        x = mbconv_block(x, filters=channels[7], kernel_size=3, strides=1, expansion_factor=6)

    # Stage 9: Final convolution + pooling + output
    x = layers.Conv2D(1280, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dropout (0.3 for B4 as per paper)
    x = layers.Dropout(0.3)(x)
    
    # Output layer (sigmoid for binary classification)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    # Build model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# --------------------------
# 4. Training Configuration (CPU-friendly)
# --------------------------
def get_training_config(model):
    """
    Configure training parameters for laptop CPU deployment
    :param model: EfficientNet-B4 model
    :return: Compiled model
    """
    # Optimizer: RMSProp (as per EfficientNet paper)
    optimizer = optimizers.RMSprop(
        learning_rate=0.001,  # Low LR for CPU stability
        rho=0.9,
        momentum=0.9,
        weight_decay=1e-5
    )

    # Loss function: Binary crossentropy with label smoothing (improve generalization)
    loss_fn = losses.BinaryCrossentropy(label_smoothing=0.1)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )

    return model

# --------------------------
# 5. Validate Forward Pass
# --------------------------
if __name__ == "__main__":
    # Build and compile model
    effnet_b4 = build_efficientnet_b4(input_shape=(224, 224, 3), num_classes=1)
    effnet_b4 = get_training_config(effnet_b4)

    # Print model summary (check parameter count: ~19M)
    effnet_b4.summary()

    # Test forward pass with random sample (simulate preprocessed deepfake data)
    sample_input = np.random.rand(8, 224, 224, 3)  # Batch size=8 (CPU-friendly)
    sample_output = effnet_b4.predict(sample_input)

    # Verify output shape (batch_size, 1) and value range (0-1)
    print(f"\nSample input shape: {sample_input.shape}")
    print(f"Sample output shape: {sample_output.shape}")
    print(f"Output value range: [{np.min(sample_output):.4f}, {np.max(sample_output):.4f}]")