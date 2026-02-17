import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import tensorflow as tf

# 设置中文字体（如需展示中文，根据系统调整）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_confusion_matrix(y_true, y_pred, save_path, title="Confusion Matrix"):
    """绘制混淆矩阵"""
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, np.argmax(y_pred, axis=1))
    # 归一化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    # 保存
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"), dpi=300)
    plt.close()
    return cm

def plot_roc_curve(y_true, y_pred_proba, save_path, title="ROC Curve"):
    """绘制ROC曲线"""
    # 计算ROC曲线和AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right")
    plt.tight_layout()
    # 保存
    plt.savefig(os.path.join(save_path, "roc_curve.png"), dpi=300)
    plt.close()
    return roc_auc

def plot_precision_recall_curve(y_true, y_pred_proba, save_path, title="Precision-Recall Curve"):
    """绘制精确率-召回率曲线"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
    
    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    # 保存
    plt.savefig(os.path.join(save_path, "precision_recall_curve.png"), dpi=300)
    plt.close()

def plot_training_history(history, save_path):
    """绘制训练历史曲线（损失/准确率）"""
    # 提取训练指标
    metrics = list(history.history.keys())
    loss_metrics = [m for m in metrics if 'loss' in m and not 'val' in m]
    val_loss_metrics = [m for m in metrics if 'loss' in m and 'val' in m]
    acc_metrics = [m for m in metrics if 'accuracy' in m and not 'val' in m]
    val_acc_metrics = [m for m in metrics if 'accuracy' in m and 'val' in m]
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    for loss in loss_metrics:
        plt.plot(history.history[loss], label=f'Train {loss}')
    for val_loss in val_loss_metrics:
        plt.plot(history.history[val_loss], label=f'Val {val_loss}')
    plt.title('Training Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    for acc in acc_metrics:
        plt.plot(history.history[acc], label=f'Train {acc}')
    for val_acc in val_acc_metrics:
        plt.plot(history.history[val_acc], label=f'Val {val_acc}')
    plt.title('Training Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_history.png"), dpi=300)
    plt.close()

def plot_grad_cam_heatmap(model, img_array, layer_name, save_path, alpha=0.4):
    """绘制Grad-CAM热力图（可视化模型关注区域）"""
    # 创建Grad-CAM模型
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    
    # 计算梯度
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    # 调整热力图尺寸匹配原图
    img = img_array[0]
    heatmap = tf.image.resize(heatmap.numpy()[..., tf.newaxis], (img.shape[0], img.shape[1]))
    heatmap = heatmap.numpy().squeeze()
    
    # 叠加热力图和原图
    plt.figure(figsize=(8, 8))
    plt.imshow(img.astype(np.uint8))
    plt.imshow(heatmap, cmap='jet', alpha=alpha)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "grad_cam_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()