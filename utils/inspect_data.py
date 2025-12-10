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
    print(f"Scanning Data from: {processed_dir} (Recursive)")
    print("-" * 60)

    # 1. 递归搜索所有 .pt 文件
    all_files_info = []

    # os.walk 会自动遍历所有层级: processed -> dataset -> session -> phase
    for root, dirs, files in os.walk(processed_dir):
        for file in files:
            if file.endswith(".pt"):
                full_path = os.path.join(root, file)

                # 智能推断 Phase (从父目录名)
                parent_dir_name = os.path.basename(root)

                # 如果目录名是 PhaseX_...，直接用；否则尝试从文件名提取
                phase_guess = parent_dir_name if "Phase" in parent_dir_name else "Unknown"
                if phase_guess == "Unknown" and "Phase" in file:
                    # 简单的文件名解析尝试
                    parts = file.split('_')
                    for p in parts:
                        if p.startswith("Phase"):
                            phase_guess = p
                            break

                all_files_info.append({
                    "path": full_path,
                    "name": file,
                    "phase": phase_guess
                })

    if not all_files_info:
        print(f"[Error] No processed data found in {processed_dir}")
        print("Tip: Did you run preprocess.py after deleting old data?")
        return

    # 2. 统计类别
    unique_phases = sorted(list(set([x['phase'] for x in all_files_info])))
    print(f"Found {len(all_files_info)} samples across {len(unique_phases)} categories:")
    for p in unique_phases:
        count = sum(1 for x in all_files_info if x['phase'] == p)
        print(f"  - {p}: {count} samples")

    # 3. 交互式筛选
    print("\n[Input] Enter a category name to filter (e.g., 'Phase2'), or press Enter for ALL:")
    target_filter = input(">>> ").strip()

    filtered_samples = []
    if target_filter:
        # 支持模糊搜索，比如输入 "Phase2" 就能匹配 "Phase2_Attack_BruteForce"
        filtered_samples = [x for x in all_files_info if target_filter in x['phase'] or target_filter in x['name']]
        print(f"Filtered count: {len(filtered_samples)}")
    else:
        filtered_samples = all_files_info

    if not filtered_samples:
        print("[Warning] No samples match your filter.")
        return

    # 4. 可视化检查
    samples_to_inspect = random.sample(filtered_samples, min(5, len(filtered_samples)))

    for idx, item in enumerate(samples_to_inspect):
        file_path = item['path']
        filename = item['name']

        try:
            data = torch.load(file_path)
        except Exception as e:
            print(f"[Error] Failed to load {filename}: {e}")
            continue

        img_seq = data['image']  # Shape (6, 3, 224, 224)
        text = data['text']
        label = data['label']
        duration = data.get('duration', 0.0)

        # 计算相对路径以便打印 (去除 project_root)
        rel_path = os.path.relpath(file_path, project_root)

        print(f"\n{'=' * 20} Sample [{idx + 1}/{len(samples_to_inspect)}] {'=' * 20}")
        print(f"Path:      {rel_path}")  # 这里应该能看到 dataset_manual_wp
        print(f"Label ID:  {label} ({item['phase']})")
        print(f"Duration:  {duration:.4f} sec")
        print(f"Tensor:    {img_seq.shape}")

        if text == "[NO_LOG]":
            print(f"Log Text:  \033[91m[NO_LOG]\033[0m")
        else:
            clean_text = text.replace('\n', ' ').replace('\r', '')
            print(f"Log Text:  {clean_text[:80]}...")

            # Matplotlib 绘图
        num_frames = img_seq.shape[0]
        fig, axes = plt.subplots(1, num_frames, figsize=(15, 3))
        fig.suptitle(f"{rel_path}\nDuration: {duration:.2f}s", fontsize=9)

        if num_frames == 1: axes = [axes]

        for i in range(num_frames):
            img_tensor = img_seq[i]
            img_display = img_tensor.permute(1, 2, 0).numpy()
            gray_img = img_display[:, :, 0]

            frame_non_zero = np.count_nonzero(gray_img)
            frame_ratio = frame_non_zero / gray_img.size

            ax = axes[i]
            ax.imshow(gray_img, cmap='gray', vmin=0.0, vmax=1.0)

            status = "DATA" if frame_ratio > 0 else "PAD"
            color = "black" if frame_ratio > 0 else "red"
            ax.set_title(f"Fr{i + 1}: {status}\n({frame_ratio:.1%})", color=color, fontsize=8)
            ax.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    inspect()