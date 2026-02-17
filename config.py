import os

# ===================== 基础配置 =====================
class CONFIG:
    # 随机种子
    SEED = 42
    # 图像尺寸 (高, 宽)
    IMG_SIZE = (224, 224)
    # 视频帧率（每秒提取帧数）
    FRAME_RATE = 15
    # 每个视频提取的总帧数（10秒视频）
    FRAMES_PER_VIDEO = 150
    # 批次大小
    BATCH_SIZE = 8
    # 训练轮数
    EPOCHS = 50
    # 数据集划分比例
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    # 最大推理时间（PID要求：≤30秒）
    MAX_INFERENCE_TIME = 30.0

    # ===================== 路径配置 =====================
    # 项目根目录
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # 数据集路径（替换为你的本地路径）
    DATA_PATHS = {
        "FaceForensics++": os.path.join(ROOT_DIR, "data/datasets/FaceForensics++"),
        "OpenFake": os.path.join(ROOT_DIR, "data/datasets/OpenFake"),
        "DFDC": os.path.join(ROOT_DIR, "data/datasets/DFDC")
    }
    # 日志目录
    LOG_DIR = os.path.join(ROOT_DIR, "logs")
    # 模型权重保存目录
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
    # 评估结果保存目录
    EVAL_DIR = os.path.join(ROOT_DIR, "eval_results")

    # ===================== EfficientNet-B4配置 =====================
    EFF_NET_CONFIG = {
        "weights": "imagenet",  # 预训练权重
        "include_top": False,   # 不包含顶层分类器
        "dropout_rate": 0.2     # Dropout率
    }

    # ===================== 混合MBConv配置 =====================
    MIXED_MBCONV_CONFIG = {
        "use_mixed_conv": True,        # 是否使用混合卷积
        "expand_ratio": 6,              # 扩展率
        "kernel_sizes": [3, 5, 7],      # 混合卷积核尺寸
        "se_ratio": 0.25                # SE模块压缩率
    }

    # ===================== StA-Lite配置 =====================
    STA_LITE_CONFIG = {
        "hidden_dim": 512,              # 隐藏层维度
        "num_heads": 8,                 # 注意力头数
        "gru_units": 256,               # GRU单元数
        "dropout_rate": 0.1,            # Dropout率
        "use_attention_fusion": True    # 是否使用注意力融合
    }

    # ===================== LDABN+DCL配置 =====================
    LDABN_DCL_CONFIG = {
        "num_domains": 3,               # 域数量（FaceForensics++/OpenFake/DFDC）
        "ldabn_momentum": 0.99,         # LDABN动量
        "temperature": 0.07,            # DCL温度系数
        "lambda_dcl": 0.1               # DCL损失权重
    }

    # ===================== 优化器配置 =====================
    OPTIMIZER_CONFIG = {
        "learning_rate": 1e-4,          # 初始学习率
        "weight_decay": 1e-5,           # 权重衰减
        "beta_1": 0.9,                  # AdamW beta1
        "beta_2": 0.999                 # AdamW beta2
    }

    # ===================== 损失函数配置 =====================
    LOSS_CONFIG = {
        "focal_gamma": 2.0,             # Focal Loss gamma
        "label_smoothing": 0.1          # 标签平滑系数
    }

    # ===================== 早停配置 =====================
    EARLY_STOPPING_CONFIG = {
        "monitor": "val_fake_detection_output_accuracy",  # 监控指标
        "patience": 10,                                   # 耐心值
        "mode": "max"                                     # 最大化指标
    }

    # ===================== 学习率调度配置 =====================
    LR_SCHEDULER_CONFIG = {
        "factor": 0.5,      # 学习率衰减因子
        "patience": 5,      # 耐心值
        "min_lr": 1e-6      # 最小学习率
    }

    # ===================== 量化配置 =====================
    QUANTIZATION_CONFIG = {
        "enable_post_training_quant": True,  # 启用后训练量化
        "calibration_dataset_size": 100      # 校准数据集大小
    }

# 创建必要目录
for dir_path in [CONFIG.LOG_DIR, CONFIG.CHECKPOINT_DIR, CONFIG.EVAL_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)