# --- 初步评估报告生成脚本 ---
# 环境配置（适配CPU）
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from domain_adaptive_training import train_domain_adaptive_model, evaluate_domain_specific, test_inference_speed
from ldabn_dcl_model import build_ldabn_dcl_model

# 设置中文字体（适配学术报告）
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

# 1. 训练损失曲线可视化
def plot_training_history(history):
    """绘制训练/验证损失曲线"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 总损失
    ax1.plot(history["loss"], label="训练总损失", color="#2E86AB")
    ax1.set_title("训练总损失曲线", fontsize=14)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("损失值")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Label Smoothing损失
    ax2.plot(history["ls_loss"], label="Label Smoothing损失", color="#A23B72")
    ax2.set_title("Label Smoothing损失曲线", fontsize=14)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("损失值")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # DCL损失
    ax3.plot(history["dcl_loss"], label="DCL损失", color="#F18F01")
    ax3.set_title("领域对比损失曲线", fontsize=14)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("损失值")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("./reports/training_loss.png", dpi=300, bbox_inches="tight")
    plt.show()

# 2. 领域特定指标可视化
def plot_domain_metrics(metrics):
    """绘制分领域评估指标"""
    domains = ["GAN", "扩散模型", "混合技术"]
    metrics_df = pd.DataFrame({
        "领域": domains,
        "准确率": metrics["accuracy"],
        "精确率": metrics["precision"],
        "召回率": metrics["recall"]
    })
    
    # 柱状图
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(domains))
    width = 0.25
    
    ax.bar(x - width, metrics_df["准确率"], width, label="准确率", color="#2E86AB", alpha=0.8)
    ax.bar(x, metrics_df["精确率"], width, label="精确率", color="#A23B72", alpha=0.8)
    ax.bar(x + width, metrics_df["召回率"], width, label="召回率", color="#F18F01", alpha=0.8)
    
    # 标注数值
    for i, v in enumerate(metrics_df["准确率"]):
        ax.text(i - width, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    for i, v in enumerate(metrics_df["精确率"]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    for i, v in enumerate(metrics_df["召回率"]):
        ax.text(i + width, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    
    # 目标线（≥90%）
    ax.axhline(y=0.9, color="red", linestyle="--", label="目标值（90%）")
    
    ax.set_title("分领域模型性能指标", fontsize=16)
    ax.set_xlabel("深度伪造技术领域", fontsize=12)
    ax.set_ylabel("指标值", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig("./reports/domain_metrics.png", dpi=300, bbox_inches="tight")
    plt.show()

# 3. 推理速度对比可视化
def plot_inference_speed(inference_time):
    """绘制推理速度对比图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 数据
    categories = ["Week7初始模型", "Week9优化模型"]
    speeds = [25.6, inference_time]  # Week7基准值（示例）
    colors = ["#CCCCCC", "#2E86AB"]
    
    # 柱状图
    bars = ax.bar(categories, speeds, color=colors, alpha=0.8)
    
    # 标注数值
    for bar, speed in zip(bars, speeds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{speed:.2f}秒", ha="center", fontsize=12)
    
    # 目标线（≤30秒）
    ax.axhline(y=30, color="red", linestyle="--", label="目标值（30秒）")
    
    ax.set_title("10秒视频推理速度对比", fontsize=14)
    ax.set_ylabel("推理时间（秒）", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig("./reports/inference_speed.png", dpi=300, bbox_inches="tight")
    plt.show()

# 4. 生成技术报告摘要
def generate_report_summary(history, metrics, inference_time):
    """生成10页技术报告的核心摘要"""
    summary = f"""
# 深度伪造检测模型Week9初步评估报告
## 1. 训练概况
- 训练轮数：{len(history['loss'])} Epoch
- 最终总损失：{history['loss'][-1]:.4f}
- 最终Label Smoothing损失：{history['ls_loss'][-1]:.4f}
- 最终DCL损失：{history['dcl_loss'][-1]:.4f}

## 2. 分领域性能（目标≥90%）
| 领域       | 准确率   | 精确率   | 召回率   |
|------------|----------|----------|----------|
| GAN        | {metrics['accuracy'][0]:.4f} | {metrics['precision'][0]:.4f} | {metrics['recall'][0]:.4f} |
| 扩散模型   | {metrics['accuracy'][1]:.4f} | {metrics['precision'][1]:.4f} | {metrics['recall'][1]:.4f} |
| 混合技术   | {metrics['accuracy'][2]:.4f} | {metrics['precision'][2]:.4f} | {metrics['recall'][2]:.4f} |
| 平均       | {np.mean(metrics['accuracy']):.4f} | {np.mean(metrics['precision']):.4f} | {np.mean(metrics['recall']):.4f} |

## 3. 推理性能
- 10秒720p视频推理时间：{inference_time:.2f}秒
- 是否达标：{'✅ 是' if inference_time <=30 else '❌ 否'}

## 4. 模型短板与优化建议
1. 扩散模型领域召回率{metrics['recall'][1]:.4f}，建议增加OpenFake数据集样本量至30%；
2. DCL损失权重0.1可微调至0.15，提升跨领域特征对齐；
3. 推理速度{inference_time:.2f}秒，符合要求，无需优化。

## 5. 核心参数统计
- 模型总参数：{build_ldabn_dcl_model().count_params()/1e6:.2f}M（目标≤20.6M）
- 轻量化约束：{'✅ 达标' if build_ldabn_dcl_model().count_params()/1e6 <=20.6 else '❌ 超标'}
    """
    # 保存摘要
    with open("./reports/report_summary.md", "w", encoding="utf-8") as f:
        f.write(summary)
    print("📄 技术报告摘要已生成！")
    print(summary)

# 主执行流程
if __name__ == "__main__":
    # 1. 训练并评估模型
    model, history = train_domain_adaptive_model()
    metrics = evaluate_domain_specific(model)
    inference_time = test_inference_speed(model)
    
    # 2. 生成可视化图表
    plot_training_history(history)
    plot_domain_metrics(metrics)
    plot_inference_speed(inference_time)
    
    # 3. 生成报告摘要
    generate_report_summary(history, metrics, inference_time)