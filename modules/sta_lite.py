import tensorflow as tf
from tensorflow.keras import layers

def sta_lite_block(input_tensor, num_frames=5, output_channels=128):
    """
    Lightweight Spatiotemporal Adapter (simplified from StA in the paper)
    Args:
        input_tensor: Feature sequence from EfficientNet-B4 (batch, num_frames, h, w, channels)
        num_frames: Number of input frames (default=5, consistent with video frame extraction strategy)
        output_channels: Output channel number (consistent with EfficientNet-B4 feature channels)
    Returns:
        Fused spatiotemporal feature tensor
    """
    # Spatial stream: 3D conv (focus on spatial features within single frame)
    spatial_stream = layers.Conv3D(
        filters=output_channels // 2,
        kernel_size=(3, 3, 1),  # 3x3 spatial kernel, 1x temporal kernel
        padding='same',
        activation='swish'
    )(input_tensor)
    
    # Temporal stream: 3D conv (focus on temporal dependencies between frames)
    temporal_stream = layers.Conv3D(
        filters=output_channels // 2,
        kernel_size=(1, 1, 3),  # 1x1 spatial kernel, 3x temporal kernel
        padding='same',
        activation='swish'
    )(input_tensor)
    
    # Feature fusion (concatenation + 1x1 conv to adjust channels)
    fused_features = layers.Concatenate(axis=-1)([spatial_stream, temporal_stream])
    fused_features = layers.Conv3D(
        filters=output_channels,
        kernel_size=(1, 1, 1),
        padding='same'
    )(fused_features)
    
    # Residual connection (maintain feature stability)
    return layers.Add()([input_tensor, fused_features])