import tensorflow as tf
from tensorflow.keras import layers, Model
from efficientnet_b4_model import build_efficientnet_b4
from sta_lite import sta_lite_block

def build_spatiotemporal_model(input_shape=(224, 224, 3), num_frames=5, num_classes=1):
    """
    Build integrated model: EfficientNet-B4 + StA-Lite
    Args:
        input_shape: Single frame shape (224, 224, 3)
        num_frames: Number of frames in sequence (default=5)
        num_classes: Output classes (1 for binary classification)
    Returns:
        Integrated spatiotemporal detection model
    """
    # Step 1: Build EfficientNet-B4 (freeze bottom layers for feature extraction)
    effnet = build_efficientnet_b4(input_shape=input_shape, num_classes=num_classes)
    effnet.trainable = True  # Unfreeze for joint training
    single_frame_input = effnet.input
    
    # Step 2: Extract single-frame features
    single_frame_features = effnet.layers[-3].output  # Output before dropout (shape: (batch, 128))
    single_frame_features = layers.Reshape((1, 1, 1, 128))(single_frame_features)  # (batch, 1, 1, 1, 128)
    
    # Step 3: Adapt to frame sequence (for image input: repeat to pseudo-sequence; for video: direct input)
    def process_sequence(input_tensor):
        if len(input_tensor.shape) == 4:  # Single image input (batch, 224, 224, 3)
            frame_features = tf.keras.backend.repeat_elements(single_frame_features, num_frames, axis=1)
        else:  # Video sequence input (batch, num_frames, 224, 224, 3)
            frame_features = []
            for i in range(num_frames):
                frame = input_tensor[:, i, :, :, :]
                feat = effnet(frame)
                frame_features.append(layers.Reshape((1, 1, 1, 128))(feat))
            frame_features = layers.Concatenate(axis=1)(frame_features)  # (batch, num_frames, 1, 1, 128)
        return frame_features
    
    # Step 4: StA-Lite for spatiotemporal fusion
    sequence_input = layers.Input(shape=(num_frames, 224, 224, 3)) if num_frames > 1 else single_frame_input
    if num_frames > 1:
        frame_features = process_sequence(sequence_input)
        fused_features = sta_lite_block(frame_features, num_frames=num_frames)
        # Global pooling across frames
        pooled_features = layers.GlobalAveragePooling3D()(fused_features)
    else:
        pooled_features = single_frame_features
    
    # Step 5: Classification layer
    dropout = layers.Dropout(0.3)(pooled_features)
    outputs = layers.Dense(num_classes, activation='sigmoid', name='confidence_score')(dropout)
    
    # Build model
    model = Model(inputs=sequence_input, outputs=outputs, name='EfficientNet-B4-StA-Lite')
    return model

# Test model construction
if __name__ == "__main__":
    model = build_spatiotemporal_model(num_frames=5)
    model.summary()  # Verify total parameters ≤21M