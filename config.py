import os
from datetime import datetime

# ===================== 基础配置 =====================
class BaseConfig:
    # 项目根路径（请替换为你的GitLab仓库本地路径）
    ROOT_DIR = "/home/your-username/deepfake_detection"
    # 实验名称（用于保存日志/权重，自动带时间戳）
    EXP_NAME = f"deepfake_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # 随机种子（保证实验可复现）
    SEED = 42
    # 设备配置（自动检测GPU，无则用CPU）
    DEVICE = "GPU" if os.environ.get("CUDA_VISIBLE_DEVICES") else "CPU"
    # 日志保存路径
    LOG_DIR = os.path.join(ROOT_DIR, "logs", EXP_NAME)
    # 模型权重保存路径
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints", EXP_NAME)
    # 评估结果保存路径
    EVAL_DIR = os.path.join(ROOT_DIR, "eval_results", EXP_NAME)

    # 创建目录（避免运行时报错）
    @classmethod
    def create_dirs(cls):
        for dir_path in [cls.LOG_DIR, cls.CHECKPOINT_DIR, cls.EVAL_DIR]:
            os.makedirs(dir_path, exist_ok=True)

# ===================== 数据集配置 =====================
class DatasetConfig(BaseConfig):
    # 多数据集路径（请替换为你的本地数据集路径）
    DATA_PATHS = {
        "FaceForensics++": os.path.join(BaseConfig.ROOT_DIR, "datasets/faceforensicspp"),
        "OpenFake": os.path.join(BaseConfig.ROOT_DIR, "datasets/openfake"),
        "DFDC": os.path.join(BaseConfig.ROOT_DIR, "datasets/dfdc")
    }
    # 数据集划分比例
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1
    # 输入图像尺寸（适配EfficientNet-B4）
    IMG_SIZE = (224, 224)
    # 视频帧提取速率（统一为15fps）
    FRAME_RATE = 15
    # 每视频提取帧数量（平衡时序信息和推理速度）
    FRAMES_PER_VIDEO = 10
    # 批量大小（根据你的笔记本显存调整，建议8/16）
    BATCH_SIZE = 8
    # 数据加载线程数
    NUM_WORKERS = 4

# ===================== 模型配置 =====================
class ModelConfig(DatasetConfig):
    # EfficientNet-B4配置
    EFF_NET_CONFIG = {
        "width_coefficient": 1.4,
        "depth_coefficient": 1.8,
        "dropout_rate": 0.4,
        "num_classes": 2,  # 伪造/真实二分类
        "include_top": False,  # 不包含顶层分类器（后续拼接自定义模块）
        "weights": "imagenet"  # 预训练权重
    }
    # StA-Lite时空模块配置
    STA_LITE_CONFIG = {
        "hidden_dim": 512,
        "num_heads": 8,
        "gru_units": 256,
        "dropout_rate": 0.3,
        "use_attention_fusion": True  # 启用注意力融合
    }
    # LDABN+DCL域自适应配置
    LDABN_DCL_CONFIG = {
        "num_domains": 3,  # FaceForensics++/OpenFake/DFDC
        "temperature": 0.8,  # DCL损失温度系数
        "lambda_dcl": 0.1,  # DCL损失权重
        "ldabn_momentum": 0.99
    }
    # 混合卷积MBConv配置
    MIXED_MBCONV_CONFIG = {
        "use_mixed_conv": True,  # 是否启用混合卷积
        "expand_ratio": 6,
        "kernel_sizes": [3, 5, 7],  # 多尺度卷积核
        "se_ratio": 0.25
    }

# ===================== 训练配置 =====================
class TrainConfig(ModelConfig):
    # 优化器配置
    OPTIMIZER_CONFIG = {
        "name": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "beta_1": 0.9,
        "beta_2": 0.999
    }
    # 学习率调度器
    LR_SCHEDULER_CONFIG = {
        "name": "ReduceLROnPlateau",
        "patience": 5,
        "factor": 0.5,
        "min_lr": 1e-6
    }
    # 损失函数配置
    LOSS_CONFIG = {
        "main_loss": "FocalLoss",  # 解决样本不平衡
        "label_smoothing": 0.2,  # 标签平滑系数
        "focal_gamma": 2.0
    }
    # 正则化配置
    REGULARIZATION_CONFIG = {
        "cutmix_prob": 0.5,
        "dropblock_prob": 0.1,
        "dropblock_size": 7
    }
    # 训练参数
    EPOCHS = 50
    # 早停配置（避免过拟合）
    EARLY_STOPPING_CONFIG = {
        "monitor": "val_accuracy",
        "patience": 8,
        "mode": "max"
    }
    # 每N轮保存一次权重
    SAVE_EVERY_N_EPOCHS = 2
    # 每N轮验证一次
    VALIDATE_EVERY_N_STEPS = 100

# ===================== 推理/量化配置 =====================
class InferenceConfig(TrainConfig):
    # 推理速度要求（PID指标）
    MAX_INFERENCE_TIME = 30  # 10秒视频推理≤30秒
    # 量化配置
    QUANTIZATION_CONFIG = {
        "quantize_type": "int8",  # int8量化（平衡精度和速度）
        "enable_post_training_quant": True,
        "calibration_dataset_size": 1000  # 校准数据集大小
    }
    # 可视化配置
    VIS_CONFIG = {
        "show_heatmap": True,  # 显示Grad-CAM热力图
        "save_results": True,
        "result_format": "pdf"
    }

# 导出最终配置（所有模块统一使用这个配置）
CONFIG = InferenceConfig()
# 初始化目录
CONFIG.create_dirs()