import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4
from config import CONFIG
from backbone.mixed_mbconv import build_mixed_mbconv_block  # 你的混合卷积模块
from modules.sta_lite import StALiteModule  # 你的StA-Lite模块
from modules.ldabn_dcl import LDABNLayer, DomainContrastiveLoss  # 你的LDABN+DCL模块
from regularization import FocalLoss  # 你的Focal Loss
from utils.common_utils import set_random_seed, setup_logger, logger

# 设置随机种子
set_random_seed(CONFIG.SEED)
# 初始化日志
logger = setup_logger(CONFIG.LOG_DIR)

def build_backbone(config=CONFIG):
    """构建带混合卷积的EfficientNet-B4主干网络"""
    logger.info("开始构建主干网络（EfficientNet-B4 + 混合卷积）")
    # 加载基础模型
    base_model = EfficientNetB4(
        input_shape=(*config.IMG_SIZE, 3),
        weights=config.EFF_NET_CONFIG["weights"],
        include_top=config.EFF_NET_CONFIG["include_top"],
        dropout_rate=config.EFF_NET_CONFIG["dropout_rate"]
    )
    
    # 替换混合卷积MBConv
    if config.MIXED_MBCONV_CONFIG["use_mixed_conv"]:
        logger.info("替换EfficientNet-B4后3层MBConv为混合卷积")
        # 重建模型（保留前层，替换后3层）
        inputs = base_model.input
        x = inputs
        
        # 遍历基础模型层，替换后3个MBConv
        mbconv_count = 0
        for layer in base_model.layers[1:-1]:  # 跳过输入层和输出层
            if "mbconv" in layer.name:
                mbconv_count += 1
                # 替换最后3个MBConv
                if mbconv_count > (27 - 3):  # EfficientNet-B4共27个MBConv
                    x = build_mixed_mbconv_block(
                        expand_ratio=config.MIXED_MBCONV_CONFIG["expand_ratio"],
                        kernel_sizes=config.MIXED_MBCONV_CONFIG["kernel_sizes"],
                        se_ratio=config.MIXED_MBCONV_CONFIG["se_ratio"],
                        input_tensor=x
                    )
                else:
                    x = layer(x)
            else:
                x = layer(x)
        
        # 输出层
        x = base_model.layers[-1](x)
        backbone = Model(inputs=inputs, outputs=x, name="EfficientNetB4_MixedConv")
    else:
        backbone = base_model
    
    # 冻结前80%层
    num_layers = len(backbone.layers)
    freeze_layers = int(num_layers * 0.8)
    for layer in backbone.layers[:freeze_layers]:
        layer.trainable = False
    logger.info(f"冻结主干网络前{freeze_layers}层（共{num_layers}层）")
    
    return backbone

def build_complete_model(config=CONFIG):
    """构建端到端完整模型（主干+时空+域自适应）"""
    logger.info("开始构建完整模型")
    # 输入层
    frame_input = layers.Input(
        shape=(config.FRAMES_PER_VIDEO, *config.IMG_SIZE, 3),
        name="video_frames_input"
    )
    domain_input = layers.Input(shape=(1,), dtype=tf.int32, name="domain_label_input")
    
    # 展开帧维度
    batch_size = tf.shape(frame_input)[0]
    frames_flat = layers.Reshape((-1, *config.IMG_SIZE, 3))(frame_input)
    frames_flat = layers.Reshape((*config.IMG_SIZE, 3))(frames_flat)
    
    # 主干特征提取
    backbone = build_backbone(config)
    frame_features = backbone(frames_flat)
    frame_features = layers.GlobalAveragePooling2D()(frame_features)
    
    # 恢复时间维度
    feature_dim = frame_features.shape[-1]
    video_features = layers.Reshape((config.FRAMES_PER_VIDEO, feature_dim))(frame_features)
    
    # StA-Lite时空融合
    logger.info("添加StA-Lite时空融合模块")
    sta_lite = StALiteModule(
        hidden_dim=config.STA_LITE_CONFIG["hidden_dim"],
        num_heads=config.STA_LITE_CONFIG["num_heads"],
        gru_units=config.STA_LITE_CONFIG["gru_units"],
        dropout_rate=config.STA_LITE_CONFIG["dropout_rate"],
        use_attention_fusion=config.STA_LITE_CONFIG["use_attention_fusion"]
    )
    fused_features = sta_lite(video_features)
    
    # LDABN域自适应
    logger.info("添加LDABN域自适应层")
    ldabn = LDABNLayer(
        num_domains=config.LDABN_DCL_CONFIG["num_domains"],
        momentum=config.LDABN_DCL_CONFIG["ldabn_momentum"]
    )
    adaptive_features = ldabn([fused_features, domain_input])
    
    # 伪造检测分类头
    x = layers.Dropout(config.EFF_NET_CONFIG["dropout_rate"])(adaptive_features)
    x = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.3)(x)
    fake_pred = layers.Dense(2, activation="softmax", name="fake_detection_output")(x)
    
    # 域分类头
    domain_pred = layers.Dense(
        config.LDABN_DCL_CONFIG["num_domains"],
        activation="softmax",
        name="domain_classification_output"
    )(adaptive_features)
    
    # 构建模型
    model = Model(
        inputs=[frame_input, domain_input],
        outputs=[fake_pred, domain_pred],
        name="DeepfakeDetector_Complete"
    )
    
    # 编译模型
    logger.info("编译模型（Focal Loss + DCL Loss）")
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config.OPTIMIZER_CONFIG["learning_rate"],
        weight_decay=config.OPTIMIZER_CONFIG["weight_decay"],
        beta_1=config.OPTIMIZER_CONFIG["beta_1"],
        beta_2=config.OPTIMIZER_CONFIG["beta_2"]
    )
    
    focal_loss = FocalLoss(
        gamma=config.LOSS_CONFIG["focal_gamma"],
        label_smoothing=config.LOSS_CONFIG["label_smoothing"]
    )
    dcl_loss = DomainContrastiveLoss(
        temperature=config.LDABN_DCL_CONFIG["temperature"]
    )
    
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
            "fake_detection_output": ["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
            "domain_classification_output": ["accuracy"]
        }
    )
    
    # 打印模型结构
    model.summary(print_fn=logger.info)
    
    # 保存模型结构图
    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(CONFIG.LOG_DIR, "complete_model.png"),
        show_shapes=True,
        show_layer_names=True
    )
    logger.info(f"模型结构图已保存至：{CONFIG.LOG_DIR}/complete_model.png")
    
    return model

def build_inference_model(complete_model=None, config=CONFIG):
    """构建轻量化推理模型"""
    logger.info("构建推理专用模型（去除域分类头）")
    if complete_model is None:
        complete_model = build_complete_model(config)
    
    # 提取推理输入输出
    inference_input = complete_model.input[0]
    inference_output = complete_model.output[0]
    
    # 构建推理模型
    inference_model = Model(
        inputs=inference_input,
        outputs=inference_output,
        name="DeepfakeDetector_Inference"
    )
    
    # 复制权重
    inference_model.set_weights([w for i, w in enumerate(complete_model.get_weights()) 
                                 if i < len(inference_model.get_weights())])
    
    # 打印推理模型结构
    inference_model.summary(print_fn=logger.info)
    
    # 保存推理模型结构图
    tf.keras.utils.plot_model(
        inference_model,
        to_file=os.path.join(CONFIG.LOG_DIR, "inference_model.png"),
        show_shapes=True,
        show_layer_names=True
    )
    logger.info(f"推理模型结构图已保存至：{CONFIG.LOG_DIR}/inference_model.png")
    
    return inference_model

# 测试模型构建
if __name__ == "__main__":
    complete_model = build_complete_model()
    inference_model = build_inference_model(complete_model)
    logger.info("模型构建测试完成")