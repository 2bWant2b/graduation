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
        RAW_DATA_DIR = os.path.join(project_root, "data", "dataset_manual_wp")
        PROCESSED_DATA_DIR = os.path.join(project_root, "data", "processed")
        DATA_DIR = os.path.join(project_root, "data")
        LABEL_MAP = {
            "Phase0_Normal_Browsing": 0,
            "Phase1_Recon_Scanning": 1,
            "Phase2_Attack_BruteForce": 2,
            "Phase3_Exploit_Action": 3
        }
        TRAFFIC_MAX_BYTES = 1024
        MAX_IMG_SEQ_LEN = 6


    config = Config()

if not hasattr(config, 'MAX_IMG_SEQ_LEN'):
    config.MAX_IMG_SEQ_LEN = 6


class TrafficPreprocessor:
    def __init__(self, max_bytes=1024, target_size=(224, 224), max_seq_len=6):
        self.max_bytes = max_bytes
        self.target_size = target_size
        self.max_seq_len = max_seq_len

    def _bytes_to_single_tensor(self, raw_bytes):
        if len(raw_bytes) == 0:
            return torch.zeros((3, *self.target_size), dtype=torch.float32)

        byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)

        if len(byte_array) < self.max_bytes:
            padding = np.zeros(self.max_bytes - len(byte_array), dtype=np.uint8)
            byte_array = np.concatenate((byte_array, padding))
        else:
            byte_array = byte_array[:self.max_bytes]

        side_len = int(np.ceil(np.sqrt(self.max_bytes)))
        if len(byte_array) < side_len * side_len:
            padding = np.zeros(side_len * side_len - len(byte_array), dtype=np.uint8)
            byte_array = np.concatenate((byte_array, padding))

        img_array = byte_array.reshape((side_len, side_len))

        img = Image.fromarray(img_array, mode='L')
        img = img.resize(self.target_size, Image.Resampling.NEAREST)

        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(np.stack([img_np, img_np, img_np], axis=0))
        return img_tensor

    def bytes_to_image_sequence(self, raw_bytes):
        if len(raw_bytes) == 0:
            return torch.zeros((self.max_seq_len, 3, *self.target_size), dtype=torch.float32)

        chunks = [raw_bytes[i: i + self.max_bytes] for i in range(0, len(raw_bytes), self.max_bytes)]

        if len(chunks) > self.max_seq_len:
            chunks = chunks[:self.max_seq_len]

        image_list = []
        for i in range(self.max_seq_len):
            if i < len(chunks):
                img_tensor = self._bytes_to_single_tensor(chunks[i])
            else:
                img_tensor = torch.zeros((3, *self.target_size), dtype=torch.float32)
            image_list.append(img_tensor)

        return torch.stack(image_list)


class LogPreprocessor:
    def __init__(self):
        # 移除 Drain3 初始化，改用正则预编译
        # 1. 匹配 IPv4 地址
        self.ip_pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
        # 2. 匹配中括号内的时间戳，例如 [06/Dec/2025:17:26:55 +0000]
        self.time_pattern = re.compile(r'\[\d{2}/[A-Za-z]{3}/\d{4}.*?\]')
        # 3. 匹配部分 Hex 编码的 Payload (可选，视情况而定，这里先不加太重，以免误伤)

    def parse_timestamp(self, log_line):
        """保持原有的时间解析逻辑不变，用于切片"""
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
        """
        使用正则清洗代替 Drain3 模板挖掘。
        目标：去除 IP 和 时间 噪声，保留 Method, URL, Status, UserAgent 语义。
        """
        log_line = log_line.strip()
        if not log_line or log_line.startswith("---"):
            return "[NO_LOG]"

        # 1. 替换 IP 地址为 [IP]
        clean_line = self.ip_pattern.sub("[IP]", log_line)

        # 2. 替换时间戳为 [TIME]
        clean_line = self.time_pattern.sub("[TIME]", clean_line)

        # 3. 截断超长日志 (BERT 最大长度限制)
        # 很多 User-Agent 或 Payload 会非常长，截取前 256 字符通常足够包含关键攻击语义
        if len(clean_line) > 256:
            clean_line = clean_line[:256]

        return clean_line

class DataProcessor:
    def __init__(self):
        self.traffic_prep = TrafficPreprocessor(
            max_bytes=config.TRAFFIC_MAX_BYTES,
            max_seq_len=config.MAX_IMG_SEQ_LEN
        )
        self.log_prep = LogPreprocessor()
        self.output_dir = config.PROCESSED_DATA_DIR

    def process_session(self, session_id, phase_name, pcap_path, log_path):
        try:
            packets = rdpcap(pcap_path)
            if not packets: return []
        except Exception as e:
            print(f"    [Error] Pcap read failed: {e}")
            return []

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

        log_events.sort(key=lambda x: x[0])
        if (len(log_events) < 2): return []

        # === 核心修改：增加 dataset 名称层级 ===
        # 获取 raw_data 目录的最后一级名称，例如 "dataset_manual_wp"
        dataset_name = os.path.basename(config.RAW_DATA_DIR.rstrip(os.sep))

        # 路径结构: data/processed/<dataset_name>/<session_id>/<phase_name>/
        save_dir = os.path.join(self.output_dir, dataset_name, session_id, phase_name)
        os.makedirs(save_dir, exist_ok=True)
        # =================================

        saved_samples = []

        for i in range(len(log_events) - 1):
            curr_ts, curr_line = log_events[i]
            next_ts, next_line = log_events[i + 1]

            start_time = curr_ts
            end_time = next_ts

            if end_time - start_time > 60.0:
                end_time = start_time + 60.0

            slice_packets = [p for p in packets if start_time <= float(p.time) < end_time]

            window_raw_bytes = b''
            for pkt in slice_packets:
                if pkt.haslayer(IP):
                    window_raw_bytes += bytes(pkt[IP])
                elif pkt.haslayer(IPv6):
                    window_raw_bytes += bytes(pkt[IPv6])
                else:
                    window_raw_bytes += bytes(pkt)

            traffic_img_seq = self.traffic_prep.bytes_to_image_sequence(window_raw_bytes)
            log_template = self.log_prep.process_single_log(curr_line)
            label_id = config.LABEL_MAP.get(phase_name, -1)

            data_item = {
                "image": traffic_img_seq,
                "text": log_template,
                "label": label_id,
                "timestamp": start_time,
                "duration": end_time - start_time
            }

            file_name = f"{session_id}_{phase_name}_evt{i:04d}.pt"
            save_path = os.path.join(save_dir, file_name)
            torch.save(data_item, save_path)

            # 计算相对路径 (例如: "data/processed/dataset_manual_wp/...")
            relative_path = os.path.relpath(save_path, project_root)

            # 为了兼容 Windows/Linux 路径分隔符差异 (反斜杠 vs 斜杠)
            # 统一强制替换为 "/"，这样 Linux 读 Windows 生成的 JSON 也不会错
            if "\\" in relative_path:
                relative_path = relative_path.replace("\\", "/")
            # ----------------
            print(relative_path)

            saved_samples.append({
                "sample_id": file_name.replace(".pt", ""),
                "path": relative_path,  # <--- 变成了灵活的相对路径
                "label": label_id,
                "phase": phase_name,
                "session": session_id,
                "dataset": dataset_name
            })

        return saved_samples

    def run_all(self):
        dataset_name = os.path.basename(config.RAW_DATA_DIR.rstrip(os.sep))
        print("-" * 60)
        print(f"Reading from: {config.RAW_DATA_DIR}")
        print(f"Saving to:    {config.PROCESSED_DATA_DIR}/{dataset_name}/<session>/<phase>/")
        print(f"Params:       MaxBytes={config.TRAFFIC_MAX_BYTES}, SeqLen={config.MAX_IMG_SEQ_LEN}")
        print("-" * 60)

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