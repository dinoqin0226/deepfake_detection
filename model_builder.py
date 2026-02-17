import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4

# 导入你的自定义模块（确保文件路径正确）
from config import CONFIG
from backbone.efficientnet_b4 import build_mixed_mbconv_block  # 你的混合卷积MBConv
from modules.sta_lite import StALiteModule  # 你的StA-Lite模块
from modules.ldabn_dcl import LDABNLayer, DomainContrastiveLoss  # 你的LDABN+DCL模块
from regularization import FocalLoss  # 你的Focal Loss

# ===================== 构建主干网络（EfficientNet-B4 + 混合卷积） =====================
def build_backbone(config=CONFIG):
    """
    构建带混合卷积的EfficientNet-B4主干网络，输出图像特征
    """
    # 加载基础EfficientNet-B4
    base_model = EfficientNetB4(
        input_shape=(*config.IMG_SIZE, 3),
        weights=config.EFF_NET_CONFIG["weights"],
        include_top=config.EFF_NET_CONFIG["include_top"],
        dropout_rate=config.EFF_NET_CONFIG["dropout_rate"]
    )
    
    # 替换后3层MBConv为混合卷积MBConv（轻量化+多尺度特征）
    if config.MIXED_MBCONV_CONFIG["use_mixed_conv"]:
        # 获取原模型层列表
        layer_list = list(base_model.layers)
        # 替换最后3个MBConv块（EfficientNet-B4共27个MBConv块）
        for i in range(-3, 0):
            if "mbconv" in layer_list[i].name:
                # 构建混合卷积MBConv块
                mixed_mbconv = build_mixed_mbconv_block(
                    expand_ratio=config.MIXED_MBCONV_CONFIG["expand_ratio"],
                    kernel_sizes=config.MIXED_MBCONV_CONFIG["kernel_sizes"],
                    se_ratio=config.MIXED_MBCONV_CONFIG["se_ratio"],
                    input_tensor=layer_list[i-1].output
                )
                # 替换层
                layer_list[i] = mixed_mbconv
        
        # 重新构建主干模型
        backbone = Model(inputs=base_model.input, outputs=layer_list[-1].output)
    else:
        backbone = base_model
    
    # 冻结前80%层权重（减少训练参数，避免过拟合）
    num_layers = len(backbone.layers)
    freeze_layers = int(num_layers * 0.8)
    for layer in backbone.layers[:freeze_layers]:
        layer.trainable = False
    
    return backbone

# ===================== 构建完整模型（主干+时空+域自适应） =====================
def build_complete_model(config=CONFIG):
    """
    构建端到端完整模型：
    EfficientNet-B4（混合卷积）→ StA-Lite（时空融合）→ LDABN（域自适应）→ 分类头
    """
    # 1. 输入层（视频帧序列：[batch_size, frames_per_video, 224, 224, 3]）
    frame_input = layers.Input(
        shape=(config.FRAMES_PER_VIDEO, *config.IMG_SIZE, 3),
        name="video_frames_input"
    )
    # 域标签输入（用于域自适应训练）
    domain_input = layers.Input(shape=(1,), dtype=tf.int32, name="domain_label_input")
    
    # 2. 提取单帧图像特征（时间维度展开）
    batch_size = tf.shape(frame_input)[0]
    # 展开：(B, T, H, W, C) → (B*T, H, W, C)
    frames_flat = layers.Reshape((-1, *config.IMG_SIZE, 3))(frame_input)
    frames_flat = layers.Reshape((*config.IMG_SIZE, 3))(frames_flat)
    
    # 主干网络提取特征
    backbone = build_backbone(config)
    frame_features = backbone(frames_flat)
    # 全局平均池化
    frame_features = layers.GlobalAveragePooling2D()(frame_features)
    # 恢复时间维度：(B*T, D) → (B, T, D)
    feature_dim = frame_features.shape[-1]
    video_features = layers.Reshape((config.FRAMES_PER_VIDEO, feature_dim))(frame_features)
    
    # 3. StA-Lite时空融合模块
    sta_lite = StALiteModule(
        hidden_dim=config.STA_LITE_CONFIG["hidden_dim"],
        num_heads=config.STA_LITE_CONFIG["num_heads"],
        gru_units=config.STA_LITE_CONFIG["gru_units"],
        dropout_rate=config.STA_LITE_CONFIG["dropout_rate"],
        use_attention_fusion=config.STA_LITE_CONFIG["use_attention_fusion"]
    )
    fused_features = sta_lite(video_features)
    
    # 4. LDABN域自适应层
    ldabn = LDABNLayer(
        num_domains=config.LDABN_DCL_CONFIG["num_domains"],
        momentum=config.LDABN_DCL_CONFIG["ldabn_momentum"]
    )
    adaptive_features = ldabn([fused_features, domain_input])
    
    # 5. 分类头（伪造/真实二分类）
    # 特征dropout
    x = layers.Dropout(config.EFF_NET_CONFIG["dropout_rate"])(adaptive_features)
    # 全连接层
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    # 输出层
    fake_pred = layers.Dense(2, activation="softmax", name="fake_detection_output")(x)
    
    # 6. 域分类头（用于DCL损失）
    domain_pred = layers.Dense(
        config.LDABN_DCL_CONFIG["num_domains"],
        activation="softmax",
        name="domain_classification_output"
    )(adaptive_features)
    
    # 7. 构建完整模型
    model = Model(
        inputs=[frame_input, domain_input],
        outputs=[fake_pred, domain_pred],
        name="DeepfakeDetector_Complete"
    )
    
    # 8. 编译模型
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config.OPTIMIZER_CONFIG["learning_rate"],
        weight_decay=config.OPTIMIZER_CONFIG["weight_decay"],
        beta_1=config.OPTIMIZER_CONFIG["beta_1"],
        beta_2=config.OPTIMIZER_CONFIG["beta_2"]
    )
    
    # 损失函数：主损失（Focal Loss） + DCL损失
    focal_loss = FocalLoss(
        gamma=config.LOSS_CONFIG["focal_gamma"],
        label_smoothing=config.LOSS_CONFIG["label_smoothing"]
    )
    dcl_loss = DomainContrastiveLoss(
        temperature=config.LDABN_DCL_CONFIG["temperature"]
    )
    
    # 多输出损失权重
    model.compile(
        optimizer=optimizer,
        loss={
            "fake_detection_output": focal_loss,
            "domain_classification_output": dcl_loss
        },
        loss_weights={
            "fake_detection_output": 1.0,
            "domain_classification_output": config.LDABN_DCL_CONFIG["lambda_dcl"]
        },
        metrics={
            "fake_detection_output": ["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
            "domain_classification_output": ["accuracy"]
        }
    )
    
    return model

# ===================== 构建推理模型（去除域分类头，轻量化） =====================
def build_inference_model(complete_model=None, config=CONFIG):
    """
    构建推理专用模型（仅保留伪造检测功能，适配笔记本端推理）
    """
    if complete_model is None:
        complete_model = build_complete_model(config)
    
    # 提取推理所需的输入和输出
    inference_input = complete_model.input[0]  # 仅保留视频帧输入
    inference_output = complete_model.output[0]  # 仅保留伪造检测输出
    
    # 构建推理模型
    inference_model = Model(
        inputs=inference_input,
        outputs=inference_output,
        name="DeepfakeDetector_Inference"
    )
    
    # 复制权重
    inference_model.set_weights([w for i, w in enumerate(complete_model.get_weights()) 
                                 if i < len(inference_model.get_weights())])
    
    return inference_model

# 测试模型构建（运行该文件时验证）
if __name__ == "__main__":
    # 构建完整模型
    complete_model = build_complete_model()
    print("完整模型结构：")
    complete_model.summary()
    
    # 构建推理模型
    inference_model = build_inference_model(complete_model)
    print("\n推理模型结构：")
    inference_model.summary()
    
    # 保存模型结构图（方便答辩展示）
    tf.keras.utils.plot_model(
        complete_model,
        to_file=os.path.join(CONFIG.LOG_DIR, "complete_model.png"),
        show_shapes=True,
        show_layer_names=True
    )
    print(f"\n模型结构图已保存至：{CONFIG.LOG_DIR}/complete_model.png")