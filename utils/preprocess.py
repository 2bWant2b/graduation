import numpy as np
import os
import sys
import re
import json
import torch
from datetime import datetime
from PIL import Image
from tqdm import tqdm

# [Scapy 导入]
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

# [路径修复逻辑]
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 尝试导入 config
try:
    import config
except ImportError:
    class Config:
        # 默认配置（如果没有 config.py）
        RAW_DATA_DIR = os.path.join(project_root, "data", "dataset_manual_wp")
        PROCESSED_DATA_DIR = os.path.join(project_root, "data", "processed")
        DATA_DIR = os.path.join(project_root, "data")
        LABEL_MAP = {
            "Phase0_Normal_Browsing": 0,
            "Phase1_Recon_Scanning": 1,
            "Phase2_Attack_BruteForce": 2,
            "Phase3_Exploit_Action": 3
        }
        # 新增参数：每个样本的最大图片序列长度
        TRAFFIC_MAX_BYTES = 224 * 224
        MAX_IMG_SEQ_LEN = 6  # 两个日志之间最多截取6张图（特征聚合用）


    config = Config()

# 确保 config 中有 MAX_IMG_SEQ_LEN，防止旧配置文件报错
if not hasattr(config, 'MAX_IMG_SEQ_LEN'):
    config.MAX_IMG_SEQ_LEN = 6


class TrafficPreprocessor:
    def __init__(self, max_bytes=1024, target_size=(224, 224), max_seq_len=6):
        self.max_bytes = max_bytes
        self.target_size = target_size
        self.max_seq_len = max_seq_len

    def _bytes_to_single_tensor(self, raw_bytes):
        """内部辅助函数：将单段字节流转为一张图片 Tensor"""
        if len(raw_bytes) == 0:
            return torch.zeros((3, *self.target_size), dtype=torch.float32)

        byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)

        # 截断与补零 (针对单张图的标准大小)
        if len(byte_array) < self.max_bytes:
            padding = np.zeros(self.max_bytes - len(byte_array), dtype=np.uint8)
            byte_array = np.concatenate((byte_array, padding))
        else:
            byte_array = byte_array[:self.max_bytes]

        # 重塑为 2D 矩阵
        side_len = int(np.sqrt(self.max_bytes))
        img_array = byte_array.reshape((side_len, side_len))

        # 转换为图像并 Resize
        img = Image.fromarray(img_array, mode='L')
        img = img.resize(self.target_size, Image.Resampling.NEAREST)

        # 归一化并转为 3 通道
        img_np = np.array(img).astype(np.float32) / 255.0
        # (3, H, W)
        img_tensor = torch.from_numpy(np.stack([img_np, img_np, img_np], axis=0))

        return img_tensor

    def bytes_to_image_sequence(self, raw_bytes):
        """
        核心修改：将长字节流切割为图片序列
        返回形状: (Seq_Len, 3, H, W), 例如 (6, 3, 224, 224)
        """
        # 1. 如果没有流量，生成全黑序列
        if len(raw_bytes) == 0:
            return torch.zeros((self.max_seq_len, 3, *self.target_size), dtype=torch.float32)

        # 2. 按 max_bytes 切分流量块
        chunks = [raw_bytes[i: i + self.max_bytes] for i in range(0, len(raw_bytes), self.max_bytes)]

        # 3. 截断（如果切出的图片超过设定长度）
        if len(chunks) > self.max_seq_len:
            chunks = chunks[:self.max_seq_len]

        # 4. 生成 Tensor 序列
        image_list = []
        for i in range(self.max_seq_len):
            if i < len(chunks):
                # 有内容，转换
                img_tensor = self._bytes_to_single_tensor(chunks[i])
            else:
                # 没内容，补全黑图 (Padding)
                img_tensor = torch.zeros((3, *self.target_size), dtype=torch.float32)

            image_list.append(img_tensor)

        # 堆叠为 (Seq, C, H, W)
        return torch.stack(image_list)


class LogPreprocessor:
    def __init__(self):
        self.config = TemplateMinerConfig()
        self.miner = TemplateMiner(persistence_handler=None, config=self.config)

    def parse_timestamp(self, log_line):
        """解析日志时间戳"""
        # 匹配标准 Apache/Nginx 格式: [06/Dec/2025:17:26:55 +0000]
        match = re.search(r'\[(\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2} \+\d{4})\]', log_line)
        if match:
            time_str = match.group(1)
            try:
                dt = datetime.strptime(time_str, "%d/%b/%Y:%H:%M:%S %z")
                return dt.timestamp()
            except ValueError:
                pass
        return None

    def process_single_log(self, log_line):
        """处理单条日志，返回模板"""
        log_line = log_line.strip()
        if not log_line or log_line.startswith("---"):
            return "[NO_LOG]"

        result = self.miner.add_log_message(log_line)
        return result['template_mined']


class DataProcessor:
    def __init__(self):
        # 初始化预处理器
        self.traffic_prep = TrafficPreprocessor(
            max_bytes=config.TRAFFIC_MAX_BYTES,
            max_seq_len=config.MAX_IMG_SEQ_LEN
        )
        self.log_prep = LogPreprocessor()

        self.output_dir = config.PROCESSED_DATA_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def process_session(self, session_id, phase_name, pcap_path, log_path):
        # 1. 加载 PCAP
        try:
            packets = rdpcap(pcap_path)
            if not packets: return []
            # 简单验证包时间
            # print(f"  Pcap loaded: {len(packets)} packets")
        except Exception as e:
            print(f"    [Error] Pcap read failed: {e}")
            return []

        # 2. 加载 Log 并提取时间戳
        # 格式: list of (timestamp, raw_line_string)
        log_events = []
        try:
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        ts = self.log_prep.parse_timestamp(line)
                        if ts:
                            log_events.append((ts, line.strip()))
        except Exception as e:
            print(f"    [Error] Log read failed: {e}")
            return []

        # 按时间排序日志
        log_events.sort(key=lambda x: x[0])

        if len(log_events) < 2:
            print(f"    [Skip] Not enough logs to form intervals in {session_id}")
            return []

        # 3. 基于日志间隔切片 (Event-driven Slicing)
        saved_samples = []

        # 遍历每一对相邻的日志 (L_i, L_{i+1})
        for i in range(len(log_events) - 1):
            curr_ts, curr_line = log_events[i]
            next_ts, next_line = log_events[i + 1]

            # 定义时间窗口: [当前日志时间, 下一条日志时间]
            start_time = curr_ts
            end_time = next_ts

            # 异常检查：如果间隔过长（例如超过5分钟），可能是会话中断，可以选择跳过或截断
            # 这里为了简化，我们设定一个硬性上限，比如60秒，超过60秒只取前60秒的流量
            if end_time - start_time > 60.0:
                end_time = start_time + 60.0

            # --- 提取该时间段内的流量 ---
            # 优化：不需要每次都遍历整个 packets 列表，可以使用指针，但为保安全这里先遍历
            # (如果数据量极大，建议优化为指针滑动)
            window_raw_bytes = b''

            # 快速筛选包
            slice_packets = [p for p in packets if start_time <= float(p.time) < end_time]

            for pkt in slice_packets:
                if pkt.haslayer(IP):
                    window_raw_bytes += bytes(pkt[IP])
                elif pkt.haslayer(IPv6):
                    window_raw_bytes += bytes(pkt[IPv6])
                else:
                    window_raw_bytes += bytes(pkt)

            # --- 核心改变：生成图片序列 (Feature Aggregation Ready) ---
            # 返回 shape: (MAX_SEQ_LEN, 3, 224, 224)
            traffic_img_seq = self.traffic_prep.bytes_to_image_sequence(window_raw_bytes)

            # --- 处理日志 ---
            # 语义上，这段流量是由 curr_line 触发的，或者是为了达成 next_line
            # 这里我们取 curr_line 的语义作为这段行为的标签
            log_template = self.log_prep.process_single_log(curr_line)

            # --- 保存 ---
            # 即使流量为空，但日志发生了，也要记录（图片全黑）
            label_id = config.LABEL_MAP.get(phase_name, -1)

            data_item = {
                "image": traffic_img_seq,  # 注意：现在是4D Tensor
                "text": log_template,
                "label": label_id,
                "timestamp": start_time,
                "duration": end_time - start_time
            }

            file_name = f"{session_id}_{phase_name}_evt{i:04d}.pt"
            save_path = os.path.join(self.output_dir, file_name)
            torch.save(data_item, save_path)

            saved_samples.append({
                "sample_id": file_name.replace(".pt", ""),
                "path": save_path,
                "label": label_id,
                "log_template": log_template
            })

        return saved_samples

    def run_all(self):
        """主入口"""
        print("-" * 50)
        print(f"Reading from: {config.RAW_DATA_DIR}")
        print(f"Saving to:    {config.PROCESSED_DATA_DIR}")
        print(f"Mode:         Event-driven (Log Interval Alignment)")
        print(f"Image Seq:    Max {config.MAX_IMG_SEQ_LEN} frames per sample")
        print("-" * 50)

        all_indices = []
        raw_root = config.RAW_DATA_DIR

        if not os.path.exists(raw_root):
            print(f"[Error] Directory not found: {raw_root}")
            return

        session_dirs = [d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d))]

        for session_id in tqdm(session_dirs, desc="Processing Sessions"):
            session_path = os.path.join(raw_root, session_id)
            for phase_name in os.listdir(session_path):
                phase_path = os.path.join(session_path, phase_name)
                if not os.path.isdir(phase_path): continue

                pcap = os.path.join(phase_path, "traffic.pcap")
                log = os.path.join(phase_path, "app.log")

                if os.path.exists(pcap) and os.path.exists(log):
                    samples = self.process_session(session_id, phase_name, pcap, log)
                    all_indices.extend(samples)

        index_path = os.path.join(config.DATA_DIR, "train_index.json")
        with open(index_path, "w") as f:
            json.dump(all_indices, f, indent=4)

        print(f"\n>>> Done! Total event samples: {len(all_indices)}")


if __name__ == "__main__":
    processor = DataProcessor()
    processor.run_all()