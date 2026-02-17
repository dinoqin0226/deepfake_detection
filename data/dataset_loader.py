import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import CONFIG
from data.data_preprocessor import DataPreprocessor
from utils.common_utils import logger

class DatasetLoader:
    def __init__(self):
        self.data_paths = CONFIG.DATA_PATHS
        self.batch_size = CONFIG.BATCH_SIZE
        self.train_ratio = CONFIG.TRAIN_RATIO
        self.val_ratio = CONFIG.VAL_RATIO
        self.test_ratio = CONFIG.TEST_RATIO
        self.preprocessor = DataPreprocessor()
        
        # 域标签映射（FaceForensics++:0, OpenFake:1, DFDC:2）
        self.domain_mapping = {
            "FaceForensics++": 0,
            "OpenFake": 1,
            "DFDC": 2
        }
    
    def load_dataset_metadata(self):
        """加载所有数据集的元数据（视频路径、标签、域标签）"""
        all_video_paths = []
        all_labels = []
        all_domains = []
        
        for domain_name, domain_path in self.data_paths.items():
            domain_label = self.domain_mapping[domain_name]
            logger.info(f"加载数据集：{domain_name}（域标签：{domain_label}）")
            
            # 遍历真实/伪造文件夹（假设数据集按real/fake划分）
            for label, subdir in enumerate(["real", "fake"]):
                subdir_path = os.path.join(domain_path, subdir)
                if not os.path.exists(subdir_path):
                    logger.warning(f"文件夹不存在：{subdir_path}，跳过")
                    continue
                
                # 遍历所有视频文件
                video_files = [f for f in os.listdir(subdir_path) if f.endswith((".mp4", ".avi", ".mov"))]
                for video_file in video_files:
                    video_path = os.path.join(subdir_path, video_file)
                    all_video_paths.append(video_path)
                    all_labels.append(label)
                    all_domains.append(domain_label)
        
        # 转换为numpy数组
        all_video_paths = np.array(all_video_paths)
        all_labels = np.array(all_labels)
        all_domains = np.array(all_domains)
        
        logger.info(f"数据集加载完成：总视频数={len(all_video_paths)}，域数={len(np.unique(all_domains))}")
        return all_video_paths, all_labels, all_domains
    
    def split_dataset(self, video_paths, labels, domains):
        """划分训练/验证/测试集"""
        # 先划分训练集和临时集（验证+测试）
        train_paths, temp_paths, train_labels, temp_labels, train_domains, temp_domains = train_test_split(
            video_paths, labels, domains,
            test_size=1 - self.train_ratio,
            stratify=labels,  # 按标签分层，保证分布一致
            random_state=CONFIG.SEED
        )
        
        # 划分验证集和测试集
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_paths, test_paths, val_labels, test_labels, val_domains, test_domains = train_test_split(
            temp_paths, temp_labels, temp_domains,
            test_size=1 - val_ratio_adjusted,
            stratify=temp_labels,
            random_state=CONFIG.SEED
        )
        
        logger.info(f"数据集划分完成：训练集={len(train_paths)}，验证集={len(val_paths)}，测试集={len(test_paths)}")
        return (train_paths, train_labels, train_domains), (val_paths, val_labels, val_domains), (test_paths, test_labels, test_domains)
    
    def create_tf_dataset(self, paths, labels, domains, is_training=True):
        """创建TensorFlow Dataset（支持批量加载和预处理）"""
        # 创建数据集
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels, domains))
        
        # 训练集打乱+重复
        if is_training:
            dataset = dataset.shuffle(buffer_size=1000, seed=CONFIG.SEED)
            dataset = dataset.repeat()
        
        # 批量预处理
        def preprocess_fn(video_path, label, domain):
            # 转换为numpy（tf.py_function需要）
            video_path = video_path.numpy().decode('utf-8')
            label = label.numpy()
            domain = domain.numpy()
            
            # 预处理视频
            frames = self.preprocessor.preprocess_single_video(video_path, is_training)
            if frames is None:
                return None, None, None
            
            return frames, label, domain
        
        # 使用tf.py_function包装Python预处理函数
        dataset = dataset.map(
            lambda x, y, z: tf.py_function(
                preprocess_fn,
                inp=[x, y, z],
                Tout=[tf.float32, tf.int32, tf.int32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # 过滤空数据
        dataset = dataset.filter(lambda x, y, z: x is not None)
        
        # 批次化
        dataset = dataset.batch(self.batch_size)
        # 预取（提升性能）
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def load_all_datasets(self):
        """加载并返回训练/验证/测试集"""
        # 加载元数据
        all_paths, all_labels, all_domains = self.load_dataset_metadata()
        # 划分数据集
        train_data, val_data, test_data = self.split_dataset(all_paths, all_labels, all_domains)
        # 创建TF Dataset
        train_dataset = self.create_tf_dataset(*train_data, is_training=True)
        val_dataset = self.create_tf_dataset(*val_data, is_training=False)
        test_dataset = self.create_tf_dataset(*test_data, is_training=False)
        
        return train_dataset, val_dataset, test_dataset