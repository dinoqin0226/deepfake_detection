import tensorflow as tf
import pandas as pd
import numpy as np
from mixed_mbconv import MixedMBConvBlock
from advanced_regularization import create_regularized_data_generator, LabelSmoothingV2
from ldabn_dcl_model import build_ldabn_dcl_model, DomainContrastiveLoss

# 全局配置（适配笔记本CPU）
CONFIG = {
    "batch_size": 8,
    "epochs": 20,
    "learning_rate": 0.0006,  # 适配DCL的低学习率
    "patience": 3,  # 早停策略
    "num_domains": 3,  # GAN/扩散/混合
    "dcl_weight": 0.1,  # DCL损失权重（不主导主任务）
    "data_paths": {
        "train": "./data/train_preprocessed",  # Week4-5预处理后的训练集
        "test_gan": "./data/test_gan",        # GAN领域测试集
        "test_diffusion": "./data/test_diffusion",  # 扩散领域测试集
        "test_mixed": "./data/test_mixed"     # 混合领域测试集
    }
}

# 加载预处理数据集（Week4-5输出，CPU友好）
def load_preprocessed_data(data_path):
    """加载numpy格式的预处理数据（避免GPU依赖）"""
    images = np.load(f"{data_path}/images_224x224.npy")
    labels = np.load(f"{data_path}/labels.npy")
    domains = np.load(f"{data_path}/domain_ids.npy")  # 0=GAN,1=扩散,2=混合
    # 转换为TF数据集（CPU友好）
    dataset = tf.data.Dataset.from_tensor_slices(((images, domains), labels))
    return dataset

# 定义总损失函数（二元交叉熵+DCL）
class TotalLoss(tf.keras.losses.Loss):
    def __init__(self, ls_loss, dcl_loss, dcl_weight=0.1):
        super().__init__()
        self.ls_loss = ls_loss  # Label Smoothing V2
        self.dcl_loss = dcl_loss  # DCL
        self.dcl_weight = dcl_weight

    def call(self, y_true, y_pred, features, domain_ids):
        # 主损失：Label Smoothing
        ls_loss = self.ls_loss(y_true, y_pred)
        # 辅助损失：DCL
        dcl_loss = self.dcl_loss(features, domain_ids, y_true)
        # 总损失
        total_loss = ls_loss + self.dcl_weight * dcl_loss
        return total_loss, ls_loss, dcl_loss

# 训练循环（适配CPU）
def train_domain_adaptive_model():
    # 1. 禁用GPU，强制CPU训练
    tf.config.set_visible_devices([], 'GPU')
    tf.config.optimizer.set_jit(False)  # 禁用XLA，适配CPU

    # 2. 加载数据
    train_data = load_preprocessed_data(CONFIG["data_paths"]["train"])
    train_generator = create_regularized_data_generator(train_data, batch_size=CONFIG["batch_size"], augment=True)

    # 3. 构建模型
    model = build_ldabn_dcl_model(input_shape=(224,224,3), num_domains=CONFIG["num_domains"])
    
    # 4. 定义损失和优化器
    ls_loss = LabelSmoothingV2(smoothing=0.1)
    dcl_loss = DomainContrastiveLoss(temperature=0.1)
    total_loss_fn = TotalLoss(ls_loss, dcl_loss, dcl_weight=CONFIG["dcl_weight"])
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=CONFIG["learning_rate"],
        rho=0.9,
        momentum=0.9,
        epsilon=1e-7
    )

    # 5. 早停回调（CPU友好）
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=CONFIG["patience"],
        restore_best_weights=True
    )

    # 6. 训练循环（自定义，适配CPU）
    @tf.function(reduce_retracing=True)  # 减少重编译，适配CPU
    def train_step(batch_imgs, batch_domains, batch_labels):
        with tf.GradientTape() as tape:
            pred, feat = model([batch_imgs, batch_domains], training=True)
            total_loss, ls_loss_val, dcl_loss_val = total_loss_fn(batch_labels, pred, feat, batch_domains)
        # 梯度更新（CPU友好）
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, ls_loss_val, dcl_loss_val

    # 7. 开始训练
    history = {"loss": [], "ls_loss": [], "dcl_loss": []}
    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        epoch_loss = 0.0
        epoch_ls_loss = 0.0
        epoch_dcl_loss = 0.0
        batch_count = 0

        # 遍历训练批次
        for (batch_imgs, batch_domains), batch_labels in train_generator:
            loss, ls_loss_val, dcl_loss_val = train_step(batch_imgs, batch_domains, batch_labels)
            epoch_loss += loss.numpy()
            epoch_ls_loss += ls_loss_val.numpy()
            epoch_dcl_loss += dcl_loss_val.numpy()
            batch_count += 1

            # 打印批次进度（CPU友好）
            if batch_count % 10 == 0:
                print(f"Batch {batch_count} - Loss: {loss.numpy():.4f}, LS Loss: {ls_loss_val.numpy():.4f}, DCL Loss: {dcl_loss_val.numpy():.4f}")

        # 记录epoch损失
        avg_loss = epoch_loss / batch_count
        avg_ls_loss = epoch_ls_loss / batch_count
        avg_dcl_loss = epoch_dcl_loss / batch_count
        history["loss"].append(avg_loss)
        history["ls_loss"].append(avg_ls_loss)
        history["dcl_loss"].append(avg_dcl_loss)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg LS Loss: {avg_ls_loss:.4f}, Avg DCL Loss: {avg_dcl_loss:.4f}")

    # 8. 保存模型（CPU友好格式）
    model.save("./models/domain_adaptive_efficientnet.h5", save_format="h5")
    print("模型保存完成！")

    return model, history

# 领域特定评估（核心：验证每个领域的准确率）
def evaluate_domain_specific(model):
    """分领域评估模型性能"""
    domains = ["gan", "diffusion", "mixed"]
    metrics = {"accuracy": [], "precision": [], "recall": []}

    for domain in domains:
        # 加载领域测试集
        test_data = load_preprocessed_data(CONFIG["data_paths"][f"test_{domain}"])
        test_generator = create_regularized_data_generator(test_data, batch_size=CONFIG["batch_size"], augment=False)

        # 评估指标
        correct = 0
        total = 0
        tp = 0  # 真阳性（伪造检测正确）
        fp = 0  # 假阳性（真实误判为伪造）
        fn = 0  # 假阴性（伪造误判为真实）

        # 遍历测试批次
        for (batch_imgs, batch_domains), batch_labels in test_generator:
            pred, _ = model([batch_imgs, batch_domains], training=False)
            pred_labels = tf.cast(pred > 0.5, tf.int32)
            batch_labels = tf.cast(batch_labels, tf.int32)

            # 统计指标
            correct += tf.reduce_sum(tf.cast(tf.equal(pred_labels, batch_labels), tf.int32)).numpy()
            total += len(batch_labels)
            tp += tf.reduce_sum(tf.cast(tf.logical_and(pred_labels == 1, batch_labels == 1), tf.int32)).numpy()
            fp += tf.reduce_sum(tf.cast(tf.logical_and(pred_labels == 1, batch_labels == 0), tf.int32)).numpy()
            fn += tf.reduce_sum(tf.cast(tf.logical_and(pred_labels == 0, batch_labels == 1), tf.int32)).numpy()

        # 计算指标
        accuracy = correct / total
        precision = tp / (tp + fp + 1e-8)  # 避免除零
        recall = tp / (tp + fn + 1e-8)
        metrics["accuracy"].append(accuracy)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)

        print(f"\n{domain.upper()}领域评估结果：")
        print(f"准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}")

    # 打印平均指标
    avg_acc = np.mean(metrics["accuracy"])
    avg_prec = np.mean(metrics["precision"])
    avg_recall = np.mean(metrics["recall"])
    print(f"\n所有领域平均：")
    print(f"平均准确率: {avg_acc:.4f}, 平均精确率: {avg_prec:.4f}, 平均召回率: {avg_recall:.4f}")

    return metrics

# 推理速度测试（验证≤30秒/10秒视频）
def test_inference_speed(model):
    """测试10秒720p视频的推理速度（5帧）"""
    tf.config.set_visible_devices([], 'GPU')
    # 模拟10秒720p视频的5帧（224×224缩放后）
    test_frames = tf.random.normal((5, 224, 224, 3))
    test_domains = tf.constant([[0]]*5)  # 模拟领域标签

    # 计时推理（CPU）
    import time
    start_time = time.time()
    pred, _ = model([test_frames, test_domains], training=False)
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"\n10秒视频（5帧）推理时间: {inference_time:.2f}秒")
    if inference_time <= 30:
        print("✅ 推理速度达标！")
    else:
        print("❌ 推理速度超标，建议优化！")

    return inference_time

# 主函数
if __name__ == "__main__":
    # 训练模型
    model, history = train_domain_adaptive_model()
    # 分领域评估
    metrics = evaluate_domain_specific(model)
    # 测试推理速度
    inference_time = test_inference_speed(model)