import os
import sys
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader

# --- 1. 路径修复与 Config 导入 (关键修改) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    import config
except ImportError:
    # 如果找不到 config，定义一个临时的配置类 (保底策略)
    class Config:
        DATA_DIR = os.path.join(project_root, "data")


    config = Config()


# -------------------------------------------

class MultiModalSlidingWindowDataset(Dataset):
    def __init__(self, index_path, window_size=5, stride=1):
        """
        Args:
            index_path: train_index.json 的绝对路径
            window_size: 滑动窗口大小
            stride: 步长
        """
        self.window_size = window_size

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"[Error] Index file not found at: {index_path}")

        with open(index_path, 'r') as f:
            self.all_samples = json.load(f)

        # --- 构建滑动窗口 ---
        self.windows = []

        # 按 Session + Phase 分组
        session_groups = {}
        for sample in self.all_samples:
            # 唯一键: SessionID_PhaseName
            key = f"{sample['session']}_{sample['phase']}"
            if key not in session_groups:
                session_groups[key] = []
            session_groups[key].append(sample)

        # 组内切片
        for key, samples in session_groups.items():
            # 按样本ID排序 (确保时序正确)
            samples.sort(key=lambda x: x['sample_id'])

            if len(samples) < window_size:
                continue

            for i in range(0, len(samples) - window_size + 1, stride):
                window_indices = samples[i: i + window_size]
                self.windows.append(window_indices)

        print(f"[Dataset] Loaded {len(self.all_samples)} events.")
        print(f"[Dataset] Generated {len(self.windows)} sliding windows (Size={window_size}).")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window_meta = self.windows[idx]

        images_list = []
        texts_list = []

        for meta in window_meta:
            # 获取 .pt 文件的路径
            pt_path = meta['path']

            # --- 路径兼容性处理 ---
            # 如果 json 里存的是相对路径，或者换了机器运行，需要重新拼接
            if not os.path.isabs(pt_path):
                pt_path = os.path.join(project_root, pt_path)

            if not os.path.exists(pt_path):
                raise FileNotFoundError(f"Pt file not found: {pt_path}")

            # 加载数据
            try:
                data = torch.load(pt_path)
            except Exception as e:
                print(f"[Error] Failed to load {pt_path}: {e}")
                # 遇到坏数据，生成一个全黑的 dummy 数据防止训练崩溃
                # 注意：这里需要知道图片形状，假设是 (6, 3, 224, 224)
                # 更好的做法是在 preprocess 阶段保证数据完好
                raise e

            images_list.append(data['image'])  # Shape: (6, 3, 224, 224)
            texts_list.append(data['text'])  # String

        # 堆叠: (Window, Seq, C, H, W) -> (5, 6, 3, 224, 224)
        images_tensor = torch.stack(images_list)

        # 标签: 取窗口最后一个事件的标签
        last_data = torch.load(
            meta['path'] if os.path.isabs(meta['path']) else os.path.join(project_root, meta['path']))
        label_phase = last_data['label']

        # 暂时还没做应用分类，先给个占位符 -1，或者从 metadata 读
        # 如果 preprocess.py 里存了 'dataset' 字段，可以根据 dataset 名字判断 app label
        label_app = 0
        if 'dataset' in meta:
            if 'tomcat' in meta['dataset'].lower():
                label_app = 1
            elif 'manual_wp' in meta['dataset'].lower():
                label_app = 0
            # else: 0 (default WP)

        return {
            "images": images_tensor,
            "texts": texts_list,
            "label_phase": torch.tensor(label_phase, dtype=torch.long),
            "label_app": torch.tensor(label_app, dtype=torch.long)
        }


# --- 测试代码 ---
if __name__ == "__main__":
    # 使用 config 中的绝对路径，彻底解决找不到文件的问题
    index_json_path = os.path.join(config.DATA_DIR, "train_index.json")

    print(f"Checking index file at: {index_json_path}")

    if os.path.exists(index_json_path):
        ds = MultiModalSlidingWindowDataset(index_json_path, window_size=5)

        if len(ds) > 0:
            sample = ds[0]
            print("\n>>> Sample 0 Data Check:")
            print(f"  Images Shape: {sample['images'].shape}")  # 预期: (5, 6, 3, 224, 224)
            print(f"  Texts Count:  {len(sample['texts'])}")  # 预期: 5
            print(f"  Phase Label:  {sample['label_phase']}")
            print(f"  App Label:    {sample['label_app']}")
            print("\n>>> Success! Data loader is working.")
        else:
            print("[Warning] Dataset is empty. Check your preprocess logic or window size.")
    else:
        print(f"[Error] File not found: {index_json_path}")
        print("Tip: Please run 'python utils/preprocess.py' first.")