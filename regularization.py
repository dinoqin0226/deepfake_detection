import tensorflow as tf
import numpy as np

def cutmix_video_frames(images, labels, alpha=1.0):
    """
    针对视频帧的CutMix数据增强（适配5帧序列，CPU友好）
    Args:
        images: 输入张量 [batch_size, height, width, channels]（批次8）
        labels: 标签张量 [batch_size, 1]（0=真实，1=伪造）
        alpha: CutMix超参数（默认1.0，轻量级增强）
    Returns:
        mixed_images: 增强后的帧
        mixed_labels: 加权后的标签
    """
    batch_size = tf.shape(images)[0]
    height = tf.shape(images)[1]
    width = tf.shape(images)[2]

    # 生成CutMix的裁剪区域（CPU友好的numpy操作）
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)

    # 随机裁剪位置
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    # 随机打乱批次索引（CPU友好）
    shuffled_idx = np.random.permutation(batch_size)

    # 执行CutMix（张量操作，适配CPU）
    mixed_images = tf.convert_to_tensor(images.numpy())
    mixed_images = tf.tensor_scatter_nd_update(
        mixed_images,
        indices=tf.stack([
            tf.repeat(tf.range(batch_size), (bby2 - bby1) * (bbx2 - bbx1)),
            tf.tile(tf.range(bby1, bby2), [batch_size, bbx2 - bbx1]),
            tf.tile(tf.range(bbx1, bbx2), [batch_size, bby2 - bby1]),
            tf.repeat(tf.range(tf.shape(images)[3]), (bby2 - bby1) * (bbx2 - bbx1) * batch_size)
        ], axis=1),
        updates=tf.gather(images, shuffled_idx)[
            :, bby1:bby2, bbx1:bbx2, :
        ].reshape(-1)
    )

    # 标签加权（平衡真实/伪造标签）
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (width * height))
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, shuffled_idx)

    return mixed_images, mixed_labels

class LabelSmoothingV2(tf.keras.losses.Loss):
    """Label Smoothing V2（缓解过拟合，适配二分类）"""
    def __init__(self, smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        # 软化标签：0→smoothing/2，1→1 - smoothing/2
        y_true = tf.cast(y_true, tf.float32)
        y_true_smooth = y_true * (1.0 - self.smoothing) + self.smoothing / 2.0
        # 二元交叉熵（保持Week7损失逻辑）
        loss = tf.keras.losses.binary_crossentropy(y_true_smooth, y_pred)
        return tf.reduce_mean(loss)

# 数据生成器（集成正则化，适配CPU训练）
def create_regularized_data_generator(dataset, batch_size=8, augment=True):
    """
    带正则化的数据生成器（适配笔记本CPU）
    Args:
        dataset: 预处理后的数据集（Week4-5输出）
        batch_size: 批次8（CPU内存友好）
        augment: 是否启用CutMix
    Returns:
        生成器：(images, labels)
    """
    def generator():
        for images, labels in dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE):
            # 启用CutMix（仅训练阶段）
            if augment:
                images, labels = cutmix_video_frames(images, labels)
            yield images, labels
    # 返回CPU友好的数据集
    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, 1), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

# 测试代码（验证正则化逻辑，CPU运行）
if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    # 模拟Week4-5预处理后的数据集（批次8）
    test_images = tf.random.normal((8, 224, 224, 3))
    test_labels = tf.random.uniform((8, 1), minval=0, maxval=2, dtype=tf.int32)
    # 测试CutMix
    mixed_imgs, mixed_labs = cutmix_video_frames(test_images, test_labels)
    print(f"CutMix后图像尺寸: {mixed_imgs.shape}")
    print(f"加权标签示例: {mixed_labs[:2]}")
    # 测试Label Smoothing V2
    ls_loss = LabelSmoothingV2(smoothing=0.1)
    loss = ls_loss(test_labels, tf.random.uniform((8,1)))
    print(f"Label Smoothing损失值: {loss.numpy()}")