"""改进版延迟相位编码脚本。

主要改进：
1. 多通道编码（亮度 + 边缘）。
2. 自适应延迟相位编码（依据输入强度动态调整时间窗）。
3. 事件驱动门控（低变化输入不触发脉冲）。
4. 可选 STDP 突触更新（用于后续 BNN/SNN 联合训练）。

示例：
python improved_delayed_phase_encoding.py \
  --image ./dataset/ETH3x100/apple/apple_65.png \
  --model ./modelpara.pth \
  --save-spikes ./artifacts/apple_65_spikes.pkl
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# 允许某些环境下 MKL 重复加载
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"





def _load_matplotlib():
    import matplotlib
    from matplotlib import pyplot as plt

    font = {"family": "MicroSoft YaHei", "weight": "bold", "size": "10"}
    matplotlib.rc("font", **font)
    return plt

class ETH_Network(nn.Module):
    """ETH-80 数据集的小型 CNN 特征提取器。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=5, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(12, 14, kernel_size=5, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(14, 8, kernel_size=5, padding=0)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        return x


class PhotoReceptor:
    """光感受器：将归一化强度映射为脉冲时间。"""

    def __init__(self, t_max: float, alpha: float) -> None:
        self.t_max = float(t_max)
        self.alpha = float(alpha)

    def get_spike_time(self, intensity: float) -> float:
        intensity = float(np.clip(intensity, 0.0, 1.0))
        return self.t_max * (1.0 - np.arctan(1.557 * self.alpha * intensity))


@dataclass
class EncoderConfig:
    n_rf: int = 25
    t_max: float = 1.0
    alpha: float = 1.0
    base_delay_ms: float = 10.0
    # 动态编码参数
    adaptive_min_scale: float = 0.7
    adaptive_max_scale: float = 1.4
    # 事件驱动参数
    event_threshold: float = 0.08
    # 离散时间网格
    time_grid_ms: Tuple[int, ...] = (0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800)


class AdaptiveDelayPhaseEncoder:
    """多通道 + 动态延迟相位编码器。"""

    def __init__(self, config: EncoderConfig) -> None:
        self.cfg = config

    def _encode_receptive_field(self, stimulation: np.ndarray, t_scale: float) -> np.ndarray:
        receptor = [PhotoReceptor(t_max=self.cfg.t_max * t_scale, alpha=self.cfg.alpha) for _ in range(self.cfg.n_rf)]
        spike_ms = np.zeros(self.cfg.n_rf, dtype=np.float64)
        for idx, intensity in enumerate(stimulation):
            spike_ms[idx] = receptor[idx].get_spike_time(float(intensity)) * 1000.0

        # 延迟相位偏置
        for idx in range(1, spike_ms.size):
            spike_ms[idx] += self.cfg.base_delay_ms / self.cfg.n_rf * idx
        return spike_ms

    def _normalize_and_discretize(self, encoded: List[np.ndarray]) -> Tuple[List[List[float]], List[List[int]]]:
        stacked = np.vstack(encoded)
        min_v = float(stacked.min())
        max_v = float(stacked.max())
        if np.isclose(max_v, min_v):
            max_v = min_v + 1e-6

        scaled = [((arr - min_v) / (max_v - min_v)) * max(self.cfg.time_grid_ms) for arr in encoded]

        spike_lists: List[List[float]] = []
        channel_status: List[List[int]] = []
        for arr in scaled:
            snapped = [float(min(self.cfg.time_grid_ms, key=lambda x: abs(x - t))) for t in arr]
            uniq = sorted(set(snapped))
            spike_lists.append(uniq)
            channel_status.append([1 if t in uniq else 0 for t in self.cfg.time_grid_ms])

        return spike_lists, channel_status

    def encode(self, input_array: np.ndarray) -> Tuple[List[List[float]], List[List[int]], Dict[str, float]]:
        if input_array.ndim != 1:
            raise ValueError("input_array must be a 1-D vector")
        if input_array.size % self.cfg.n_rf != 0:
            raise ValueError("input_array length should be divisible by n_rf")

        n_fields = input_array.size // self.cfg.n_rf
        fields = np.split(input_array.astype(np.float64), n_fields)

        encoded: List[np.ndarray] = []
        active_fields = 0
        for field in fields:
            # 事件驱动门控：变化不足时跳过
            if float(np.std(field)) < self.cfg.event_threshold:
                encoded.append(np.zeros(self.cfg.n_rf, dtype=np.float64))
                continue

            active_fields += 1
            # 动态时间窗：平均强度越高，编码越快（时间窗更短）
            mean_intensity = float(np.mean(field))
            scale = self.cfg.adaptive_max_scale - (
                self.cfg.adaptive_max_scale - self.cfg.adaptive_min_scale
            ) * mean_intensity
            encoded.append(self._encode_receptive_field(field, t_scale=scale))

        spikes, channel_status = self._normalize_and_discretize(encoded)
        stats = {
            "active_fields": float(active_fields),
            "active_ratio": float(active_fields / len(fields)),
            "event_threshold": float(self.cfg.event_threshold),
        }
        return spikes, channel_status, stats


def extract_channels(image: Image.Image, target_size: Tuple[int, int] = (68, 68)) -> Dict[str, np.ndarray]:
    """生成多通道输入：亮度 + 边缘。"""
    img_rgb = image.convert("RGB").resize(target_size)
    arr = np.asarray(img_rgb).astype(np.float64) / 255.0

    gray = np.mean(arr, axis=2)
    # 简化 Sobel（不依赖 cv2）
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    edge = np.sqrt(gx ** 2 + gy ** 2)
    edge = edge / (edge.max() + 1e-8)

    return {"intensity": gray, "edge": edge}


def flatten_by_receptive_field(channel_2d: np.ndarray, m: int = 5, n: int = 5) -> np.ndarray:
    if channel_2d.shape[0] % m != 0 or channel_2d.shape[1] % n != 0:
        raise ValueError("image size must be divisible by receptive field size")

    out: List[float] = []
    for i in range(0, channel_2d.shape[0], m):
        for j in range(0, channel_2d.shape[1], n):
            block = channel_2d[i : i + m, j : j + n]
            out.extend(block.reshape(-1).tolist())
    return np.asarray(out, dtype=np.float64)


def stdp_update(
    weights: np.ndarray,
    pre_spike_ms: np.ndarray,
    post_spike_ms: np.ndarray,
    a_plus: float = 0.01,
    a_minus: float = 0.012,
    tau_plus: float = 20.0,
    tau_minus: float = 20.0,
) -> np.ndarray:
    """离线 STDP 更新（论文常用指数窗形式）。"""
    if weights.shape != (pre_spike_ms.size, post_spike_ms.size):
        raise ValueError("weights shape must match [n_pre, n_post]")

    updated = weights.copy().astype(np.float64)
    for i, t_pre in enumerate(pre_spike_ms):
        for j, t_post in enumerate(post_spike_ms):
            dt = float(t_post - t_pre)
            if dt >= 0:
                dw = a_plus * np.exp(-dt / tau_plus)
            else:
                dw = -a_minus * np.exp(dt / tau_minus)
            updated[i, j] += dw
    return np.clip(updated, 0.0, 1.0)


def maybe_plot(spikes: Dict[str, List[List[float]]], title: str = "多通道脉冲编码") -> None:
    plt = _load_matplotlib()
    plt.figure(figsize=(11, 5))
    plt.style.use("ggplot")

    offset = 0
    y_ticks = []
    y_labels = []
    for channel_name, spike_list in spikes.items():
        adjusted = [[t for t in neuron_spikes] for neuron_spikes in spike_list]
        plt.eventplot(adjusted, lineoffsets=np.arange(offset, offset + len(adjusted)), linelengths=0.8, linewidths=1.8)
        y_ticks.extend(range(offset, offset + len(adjusted), max(1, len(adjusted) // 4)))
        y_labels.extend([channel_name] * len(range(offset, offset + len(adjusted), max(1, len(adjusted) // 4))))
        offset += len(adjusted) + 1

    plt.xticks(np.arange(0, 2001, 200))
    plt.xlabel("time(ms)")
    plt.ylabel("neuron index")
    plt.title(title)
    plt.grid(color="black", axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


def run_pipeline(args: argparse.Namespace) -> None:
    image_path = Path(args.image)
    model_path = Path(args.model)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    net = ETH_Network().eval()
    state_dict = torch.load(model_path, map_location="cpu")
    net.load_state_dict(state_dict)

    image = Image.open(image_path)
    tensor = transforms.ToTensor()(image.resize((68, 68))).unsqueeze(0)
    features = net(tensor).detach().numpy().reshape(-1).astype(np.float64)

    cfg = EncoderConfig(n_rf=args.n_rf, event_threshold=args.event_threshold)
    encoder = AdaptiveDelayPhaseEncoder(cfg)

    channels = extract_channels(image)
    multi_channel_spikes: Dict[str, List[List[float]]] = {}
    multi_channel_status: Dict[str, List[List[int]]] = {}

    # 1) CNN 特征编码
    cnn_spikes, cnn_status, cnn_stats = encoder.encode(features)
    multi_channel_spikes["cnn_feature"] = cnn_spikes
    multi_channel_status["cnn_feature"] = cnn_status

    # 2) 原图强度 + 边缘编码
    for name, channel in channels.items():
        flattened = flatten_by_receptive_field(channel, m=args.rf_h, n=args.rf_w)
        spikes, status, stats = encoder.encode(flattened)
        multi_channel_spikes[name] = spikes
        multi_channel_status[name] = status
        print(f"[{name}] active_ratio={stats['active_ratio']:.3f}")

    print(f"[cnn_feature] active_ratio={cnn_stats['active_ratio']:.3f}")

    if args.plot:
        maybe_plot(multi_channel_spikes)

    if args.save_spikes:
        save_path = Path(args.save_spikes)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as f:
            pickle.dump(
                {
                    "spikes": multi_channel_spikes,
                    "channel_status": multi_channel_status,
                    "cnn_stats": cnn_stats,
                    "config": cfg,
                },
                f,
            )
        print(f"saved spikes to: {save_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BNN/SNN 多通道自适应延迟相位编码")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--model", type=str, default="./modelpara.pth", help="CNN 权重路径")
    parser.add_argument("--n-rf", type=int, default=25, help="每个感受野像素数")
    parser.add_argument("--rf-h", type=int, default=5, help="感受野高")
    parser.add_argument("--rf-w", type=int, default=5, help="感受野宽")
    parser.add_argument("--event-threshold", type=float, default=0.08, help="事件驱动门控阈值（标准差）")
    parser.add_argument("--plot", action="store_true", help="绘制 eventplot")
    parser.add_argument("--save-spikes", type=str, default="", help="保存编码结果 pkl 路径")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_pipeline(args)
