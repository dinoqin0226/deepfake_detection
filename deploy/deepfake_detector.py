import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tensorflow as tf
from config import CONFIG
from data.data_preprocessor import DataPreprocessor
from utils.common_utils import set_random_seed, setup_logger, logger
from utils.visualization_utils import plot_grad_cam_heatmap

# 初始化配置
set_random_seed(CONFIG.SEED)
logger = setup_logger(CONFIG.LOG_DIR)

class DeepfakeDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Deepfake Detection Tool")
        self.root.geometry("800x600")
        
        # 初始化组件
        self.preprocessor = DataPreprocessor()
        self.model = self._load_model()
        self.selected_video_path = None
        
        # 构建GUI
        self._build_gui()
    
    def _load_model(self):
        """加载推理模型（优先加载量化模型）"""
        quant_model_path = os.path.join(CONFIG.CHECKPOINT_DIR, "quantized_model.tflite")
        if os.path.exists(quant_model_path):
            logger.info("加载量化推理模型")
            interpreter = tf.lite.Interpreter(model_path=quant_model_path)
            interpreter.allocate_tensors()
            self.input_details = interpreter.get_input_details()
            self.output_details = interpreter.get_output_details()
            return interpreter
        else:
            logger.info("加载原始推理模型")
            from model_builder import build_inference_model
            model = build_inference_model()
            model.load_weights(os.path.join(CONFIG.CHECKPOINT_DIR, "best_model.h5"))
            return model
    
    def _build_gui(self):
        """构建GUI界面"""
        # 标题
        title_label = ttk.Label(self.root, text="Deepfake Detection Tool", font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # 选择视频按钮
        select_btn = ttk.Button(self.root, text="Select Video File", command=self._select_video)
        select_btn.pack(pady=10)
        
        # 检测按钮
        detect_btn = ttk.Button(self.root, text="Start Detection", command=self._detect, state=tk.DISABLED)
        self.detect_btn = detect_btn
        detect_btn.pack(pady=10)
        
        # 结果显示区域
        self.result_label = ttk.Label(self.root, text="Detection Result: ", font=("Arial", 12))
        self.result_label.pack(pady=20)
        
        # 进度条
        self.progress_bar = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress_bar.pack(pady=10)
        
        # 状态标签
        self.status_label = ttk.Label(self.root, text="Status: Ready", font=("Arial", 10))
        self.status_label.pack(pady=10)
    
    def _select_video(self):
        """选择视频文件"""
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*"))
        )
        if video_path:
            self.selected_video_path = video_path
            self.status_label.config(text=f"Selected: {os.path.basename(video_path)}")
            self.detect_btn.config(state=tk.NORMAL)
    
    def _detect(self):
        """执行检测"""
        if not self.selected_video_path:
            messagebox.showerror("Error", "Please select a video file first!")
            return
        
        try:
            self.progress_bar['value'] = 0
            self.root.update_idletasks()
            
            # 1. 预处理视频
            self.status_label.config(text="Preprocessing video...")
            self.progress_bar['value'] = 20
            self.root.update_idletasks()
            frames = self.preprocessor.preprocess_single_video(self.selected_video_path, is_training=False)
            if frames is None:
                raise ValueError("Video preprocessing failed!")
            
            # 2. 模型推理
            self.status_label.config(text="Running detection...")
            self.progress_bar['value'] = 60
            self.root.update_idletasks()
            frames_tensor = tf.convert_to_tensor([frames], dtype=np.float32)
            
            if isinstance(self.model, tf.lite.Interpreter):
                # 量化模型推理
                self.model.set_tensor(self.input_details[0]['index'], frames_tensor)
                self.model.invoke()
                pred = self.model.get_tensor(self.output_details[0]['index'])[0]
            else:
                # 原始模型推理
                pred = self.model.predict(frames_tensor, verbose=0)[0]
            
            # 3. 解析结果
            self.progress_bar['value'] = 80
            self.root.update_idletasks()
            fake_prob = pred[1] * 100
            real_prob = pred[0] * 100
            result = "Fake" if fake_prob > 50 else "Real"
            
            # 4. 保存热力图
            plot_grad_cam_heatmap(
                self.model if not isinstance(self.model, tf.lite.Interpreter) else None,
                frames_tensor,
                layer_name="top_activation",
                save_path=CONFIG.EVAL_DIR
            )
            
            # 5. 显示结果
            self.progress_bar['value'] = 100
            self.status_label.config(text="Detection completed!")
            self.result_label.config(
                text=f"Detection Result: {result}\nReal Probability: {real_prob:.2f}%\nFake Probability: {fake_prob:.2f}%"
            )
            
            logger.info(f"检测完成：{self.selected_video_path} → {result} (Fake: {fake_prob:.2f}%)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            logger.error(f"检测失败：{str(e)}")
            self.status_label.config(text="Status: Error")

# 运行GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = DeepfakeDetectorGUI(root)
    root.mainloop()