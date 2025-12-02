import tensorflow as tf
import numpy as np
from efficientnet_sta_model import build_spatiotemporal_model
from train_config import get_training_params

def load_video_sequence_data(data_path='./preprocessed_data', num_frames=5):
    """
    Load preprocessed video frame sequences (simulated function, replace with actual data loading logic)
    Returns:
        train_sequences: Training data (batch, num_frames, 224, 224, 3)
        train_labels: Training labels (batch, 1)
        val_sequences: Validation data (batch, num_frames, 224, 224, 3)
        val_labels: Validation labels (batch, 1)
    """
    # Simulate data (replace with actual Pandas/NumPy loading code)
    train_sequences = np.random.rand(1000, num_frames, 224, 224, 3)
    train_labels = np.random.randint(0, 2, (1000, 1))
    val_sequences = np.random.rand(200, num_frames, 224, 224, 3)
    val_labels = np.random.randint(0, 2, (200, 1))
    return train_sequences, train_labels, val_sequences, val_labels

if __name__ == "__main__":
    # 1. Load data
    train_seq, train_labels, val_seq, val_labels = load_video_sequence_data(num_frames=5)
    
    # 2. Build and compile model
    model = build_spatiotemporal_model(num_frames=5)
    train_params = get_training_params()
    # Adjust learning rate for spatiotemporal training
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0008, rho=0.9, momentum=0.9, weight_decay=1e-5)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # 3. Initial training (10 epochs)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    history = model.fit(
        train_seq, train_labels,
        batch_size=train_params['batch_size'],
        epochs=10,
        validation_data=(val_seq, val_labels),
        shuffle=train_params['shuffle'],
        verbose=train_params['verbose'],
        callbacks=[early_stopping]
    )
    
    # 4. Save training history and model checkpoint
    np.save('./training_history/week7_initial_train_history.npy', history.history)
    model.save('./model_checkpoints/efficientnet_sta_lite_initial.h5')
    
    # 5. Test inference speed
    test_video_seq = np.random.rand(1, 5, 224, 224, 3)  # Simulate 10-second video (5 frames)
    import time
    start_time = time.time()
    model.predict(test_video_seq)
    inference_time = time.time() - start_time
    print(f"Inference time for 5-frame video sequence: {inference_time:.2f}s (target ≤30s)")