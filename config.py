import os
import sys

# ==================== 1. 基础路径配置 ====================

# 获取项目根目录的绝对路径
# 逻辑：当前文件 config.py 就在根目录下，所以它的目录就是 PROJECT_ROOT
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据存储路径
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "dataset_manual_wp")  # 原始采集数据
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")    # 预处理后数据

# 模型保存路径
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# 日志保存路径
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# 确保必要的目录存在
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_SAVE_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== 2. 数据处理配置 ====================

# 流量预处理参数
TRAFFIC_MAX_BYTES = 32*32        # 截取流量前多少字节
TRAFFIC_IMG_SIZE = (224, 224)   # ViT 输入尺寸

# 日志预处理参数
LOG_MAX_LEN = 128               # BERT 输入的最大 Token 长度

# 滑动窗口切片参数
WINDOW_SIZE = 40.0              # 窗口大小 (秒)
STRIDE = 5.0                    # 滑动步长 (秒)

# ==================== 3. 模型超参数 ====================

# 标签映射 (Label Mapping)
# 将行为阶段映射为数字 ID
LABEL_MAP = {
    "Phase0_Normal_Browsing": 0,
    "Phase1_Recon_Scanning": 1,
    "Phase2_Attack_BruteForce": 2,
    "Phase3_Exploit_Action": 3
}
NUM_CLASSES = len(LABEL_MAP)

# 训练参数
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
DEVICE = "cuda"  # 或者 "cpu"

# ==================== 4. 调试辅助 ====================

def print_config():
    """打印当前配置信息"""
    print("-" * 30)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Raw Data:     {RAW_DATA_DIR}")
    print(f"Num Classes:  {NUM_CLASSES}")
    print("-" * 30)

if __name__ == "__main__":
    print_config()