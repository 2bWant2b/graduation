import torch
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import config


def inspect():
    processed_dir = config.PROCESSED_DATA_DIR
    print("-" * 60)
    print(f"Inspecting Data from: {processed_dir}")
    print("-" * 60)

    all_files = [f for f in os.listdir(processed_dir) if f.endswith(".pt")]

    if not all_files:
        print("[Error] No processed data found.")
        return

    print(f"Total samples found: {len(all_files)}")

    # 随机抽取 5 个样本 (因为每个样本现在有6张图，少抽点以免弹窗太多)
    samples_to_inspect = random.sample(all_files, min(5, len(all_files)))

    # 按文件名排序一下，方便对比同一阶段的数据
    samples_to_inspect.sort()

    for idx, filename in enumerate(samples_to_inspect):
        file_path = os.path.join(processed_dir, filename)
        data = torch.load(file_path)

        # 获取数据
        img_seq = data['image']  # Shape 应该是 (Seq_Len, 3, 224, 224)
        text = data['text']
        label = data['label']
        duration = data.get('duration', 0.0)  # 获取新增的持续时间字段

        print(f"\n{'=' * 20} Sample [{idx + 1}/{len(samples_to_inspect)}] {'=' * 20}")
        print(f"File:      {filename}")
        print(f"Label ID:  {label}")
        print(f"Duration:  {duration:.4f} sec")
        print(f"Tensor:    {img_seq.shape}")  # 检查维度是否为 (6, 3, 224, 224)

        # 打印日志预览
        if text == "[NO_LOG]":
            print(f"Log Text:  \033[91m[NO_LOG]\033[0m")
        else:
            # 简单清理换行符
            clean_text = text.replace('\n', ' ').replace('\r', '')
            print(f"Log Text:  {clean_text[:80]}...")

            # --- 统计整个序列的信息 ---
        total_non_zero = torch.count_nonzero(img_seq).item()
        total_elements = img_seq.numel()
        print(
            f"Seq Stat:  Non-zero Pixels: {total_non_zero} / {total_elements} ({total_non_zero / total_elements:.2%})")

        # --- 可视化 (多帧并排显示) ---
        num_frames = img_seq.shape[0]

        # 创建画布：1行 N列
        fig, axes = plt.subplots(1, num_frames, figsize=(15, 3))
        fig.suptitle(f"Label: {label} | Duration: {duration:.2f}s | {filename}", fontsize=10)

        # 如果只有1帧，axes 可能不是列表，强制转换
        if num_frames == 1:
            axes = [axes]

        for i in range(num_frames):
            # 取出单帧 (3, H, W) -> (H, W, 3) 用作显示
            img_tensor = img_seq[i]
            img_display = img_tensor.permute(1, 2, 0).numpy()

            # 取单通道看灰度 (因为是灰度图转RGB，三个通道一样)
            gray_img = img_display[:, :, 0]

            # 统计单帧非零比率
            frame_non_zero = np.count_nonzero(gray_img)
            frame_ratio = frame_non_zero / gray_img.size

            ax = axes[i]
            ax.imshow(gray_img, cmap='gray', vmin=0.0, vmax=1.0)

            # 设置子标题：显示这是第几帧，以及是否有内容
            status = "DATA" if frame_ratio > 0 else "PAD"
            color = "black" if frame_ratio > 0 else "red"
            ax.set_title(f"Frame {i + 1}\n{status} ({frame_ratio:.1%})", color=color, fontsize=8)
            ax.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    inspect()