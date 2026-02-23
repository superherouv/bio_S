"""改进版延迟相位编码与基线解码评估。

实现目标：
1) 多通道编码（CNN 特征 + 强度 + 边缘）。
2) 自适应延迟相位编码 + 事件驱动门控。
3) STDP 离线更新函数。
4) 批量评估：对 annotation CSV 执行编码 -> 特征聚合 -> 原型分类器解码。

说明：
- 单图模式：输出脉冲与统计信息。
- 数据集模式：输出 accuracy / macro-f1 / spike 稀疏度 / 处理时延。
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def _load_matplotlib():
    import matplotlib
    from matplotlib import pyplot as plt

    font = {"family": "MicroSoft YaHei", "weight": "bold", "size": "10"}
    matplotlib.rc("font", **font)
    return plt


class ETH_Network(nn.Module):
    """ETH-80 小型 CNN 特征提取器（仅卷积部分）。"""

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
    adaptive_min_scale: float = 0.7
    adaptive_max_scale: float = 1.4
    event_threshold: float = 0.08
    time_grid_ms: Tuple[int, ...] = (0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800)


class AdaptiveDelayPhaseEncoder:
    def __init__(self, config: EncoderConfig) -> None:
        self.cfg = config

    def _encode_receptive_field(self, stimulation: np.ndarray, t_scale: float) -> np.ndarray:
        receptors = [PhotoReceptor(t_max=self.cfg.t_max * t_scale, alpha=self.cfg.alpha) for _ in range(self.cfg.n_rf)]
        spike_ms = np.zeros(self.cfg.n_rf, dtype=np.float64)
        for idx, intensity in enumerate(stimulation):
            spike_ms[idx] = receptors[idx].get_spike_time(float(intensity)) * 1000.0
        spike_ms += np.arange(self.cfg.n_rf, dtype=np.float64) * (self.cfg.base_delay_ms / self.cfg.n_rf)
        return spike_ms

    def _normalize_and_discretize(self, encoded: List[np.ndarray]) -> Tuple[List[List[float]], List[List[int]]]:
        stacked = np.vstack(encoded)
        min_v, max_v = float(stacked.min()), float(stacked.max())
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
            raise ValueError("input_array must be 1-D")
        if input_array.size % self.cfg.n_rf != 0:
            raise ValueError("input length must be divisible by n_rf")

        fields = np.split(input_array.astype(np.float64), input_array.size // self.cfg.n_rf)
        encoded: List[np.ndarray] = []
        active_fields = 0
        for field in fields:
            if float(np.std(field)) < self.cfg.event_threshold:
                encoded.append(np.zeros(self.cfg.n_rf, dtype=np.float64))
                continue
            active_fields += 1
            mean_intensity = float(np.mean(field))
            scale = self.cfg.adaptive_max_scale - (
                self.cfg.adaptive_max_scale - self.cfg.adaptive_min_scale
            ) * mean_intensity
            encoded.append(self._encode_receptive_field(field, scale))

        spikes, status = self._normalize_and_discretize(encoded)
        stats = {
            "active_fields": float(active_fields),
            "active_ratio": float(active_fields / len(fields)),
            "event_threshold": float(self.cfg.event_threshold),
        }
        return spikes, status, stats


def extract_channels(image: Image.Image, target_size: Tuple[int, int] = (68, 68)) -> Dict[str, np.ndarray]:
    img_rgb = image.convert("RGB").resize(target_size)
    arr = np.asarray(img_rgb).astype(np.float64) / 255.0

    gray = np.mean(arr, axis=2)
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
            out.extend(channel_2d[i : i + m, j : j + n].reshape(-1).tolist())
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
    if weights.shape != (pre_spike_ms.size, post_spike_ms.size):
        raise ValueError("weights shape must match [n_pre, n_post]")
    updated = weights.copy().astype(np.float64)
    for i, t_pre in enumerate(pre_spike_ms):
        for j, t_post in enumerate(post_spike_ms):
            dt = float(t_post - t_pre)
            dw = a_plus * np.exp(-dt / tau_plus) if dt >= 0 else -a_minus * np.exp(dt / tau_minus)
            updated[i, j] += dw
    return np.clip(updated, 0.0, 1.0)


def maybe_plot(spikes: Dict[str, List[List[float]]], title: str = "多通道脉冲编码") -> None:
    plt = _load_matplotlib()
    plt.figure(figsize=(11, 5))
    plt.style.use("ggplot")
    offset = 0
    for _, spike_list in spikes.items():
        plt.eventplot(
            spike_list,
            lineoffsets=np.arange(offset, offset + len(spike_list)),
            linelengths=0.8,
            linewidths=1.5,
        )
        offset += len(spike_list) + 1
    plt.xticks(np.arange(0, 2001, 200))
    plt.xlabel("time(ms)")
    plt.ylabel("neuron index")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def load_model(model_path: Path) -> ETH_Network:
    net = ETH_Network().eval()
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    return net


def encode_single_image(
    image_path: Path,
    net: ETH_Network,
    encoder: AdaptiveDelayPhaseEncoder,
    rf_h: int,
    rf_w: int,
) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[List[int]]], Dict[str, Dict[str, float]]]:
    image = Image.open(image_path)
    tensor = transforms.ToTensor()(image.resize((68, 68))).unsqueeze(0)
    features = net(tensor).detach().numpy().reshape(-1).astype(np.float64)

    spikes: Dict[str, List[List[float]]] = {}
    status: Dict[str, List[List[int]]] = {}
    stats: Dict[str, Dict[str, float]] = {}

    cnn_spikes, cnn_status, cnn_stats = encoder.encode(features)
    spikes["cnn_feature"], status["cnn_feature"], stats["cnn_feature"] = cnn_spikes, cnn_status, cnn_stats

    channels = extract_channels(image)
    for name, channel in channels.items():
        flat = flatten_by_receptive_field(channel, m=rf_h, n=rf_w)
        ch_spikes, ch_status, ch_stats = encoder.encode(flat)
        spikes[name], status[name], stats[name] = ch_spikes, ch_status, ch_stats

    return spikes, status, stats


def aggregate_spike_features(spikes: Dict[str, List[List[float]]], time_grid: Sequence[int]) -> np.ndarray:
    """将脉冲序列聚合成可用于解码器的定长向量。

    每个通道提取：
    - mean spike count
    - mean first-spike latency
    - mean last-spike latency
    - active neuron ratio
    """
    feats: List[float] = []
    max_t = float(max(time_grid))
    for _, neurons in sorted(spikes.items(), key=lambda kv: kv[0]):
        counts = np.asarray([len(n) for n in neurons], dtype=np.float64)
        first = np.asarray([n[0] if len(n) else max_t for n in neurons], dtype=np.float64)
        last = np.asarray([n[-1] if len(n) else max_t for n in neurons], dtype=np.float64)
        active = np.asarray([1.0 if len(n) else 0.0 for n in neurons], dtype=np.float64)

        feats.extend([
            float(np.mean(counts)),
            float(np.mean(first)),
            float(np.mean(last)),
            float(np.mean(active)),
        ])
    return np.asarray(feats, dtype=np.float64)


def fit_prototype_decoder(x_train: np.ndarray, y_train: np.ndarray) -> Dict[int, np.ndarray]:
    prototypes: Dict[int, np.ndarray] = {}
    for cls in sorted(set(y_train.tolist())):
        prototypes[int(cls)] = np.mean(x_train[y_train == cls], axis=0)
    return prototypes


def predict_prototype_decoder(x_test: np.ndarray, prototypes: Dict[int, np.ndarray]) -> np.ndarray:
    classes = sorted(prototypes.keys())
    pred = []
    for x in x_test:
        dists = [float(np.linalg.norm(x - prototypes[c])) for c in classes]
        pred.append(classes[int(np.argmin(dists))])
    return np.asarray(pred, dtype=np.int64)


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = sorted(set(y_true.tolist()))
    f1s: List[float] = []
    for c in classes:
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))


def load_annotation_csv(csv_path: Path) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["path"], int(row["species"])))
    return rows


def run_dataset_eval(args: argparse.Namespace, encoder: AdaptiveDelayPhaseEncoder, net: ETH_Network) -> None:
    train_rows = load_annotation_csv(Path(args.train_csv))
    test_rows = load_annotation_csv(Path(args.test_csv))

    def encode_rows(rows: List[Tuple[str, int]]) -> Tuple[np.ndarray, np.ndarray, float, float]:
        xs: List[np.ndarray] = []
        ys: List[int] = []
        active_ratios: List[float] = []
        elapsed: List[float] = []
        for p, y in rows:
            t0 = time.perf_counter()
            spikes, _, stats = encode_single_image(Path(p), net, encoder, args.rf_h, args.rf_w)
            elapsed.append((time.perf_counter() - t0) * 1000.0)
            xs.append(aggregate_spike_features(spikes, encoder.cfg.time_grid_ms))
            ys.append(y)
            active_ratios.append(float(np.mean([v["active_ratio"] for v in stats.values()])))
        return np.vstack(xs), np.asarray(ys, dtype=np.int64), float(np.mean(active_ratios)), float(np.mean(elapsed))

    x_train, y_train, train_active, train_ms = encode_rows(train_rows)
    x_test, y_test, test_active, test_ms = encode_rows(test_rows)

    prototypes = fit_prototype_decoder(x_train, y_train)
    y_pred = predict_prototype_decoder(x_test, prototypes)

    acc = float(np.mean(y_pred == y_test))
    macro_f1 = compute_macro_f1(y_test, y_pred)

    result = {
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
        "train_mean_active_ratio": train_active,
        "test_mean_active_ratio": test_active,
        "train_mean_encode_ms": train_ms,
        "test_mean_encode_ms": test_ms,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    print("\n=== Dataset evaluation summary ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    if args.save_metrics:
        out = Path(args.save_metrics)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            pickle.dump(result, f)
        print(f"saved metrics to: {out}")


def run_single(args: argparse.Namespace, encoder: AdaptiveDelayPhaseEncoder, net: ETH_Network) -> None:
    spikes, status, stats = encode_single_image(Path(args.image), net, encoder, args.rf_h, args.rf_w)
    for channel_name, channel_stats in stats.items():
        print(f"[{channel_name}] active_ratio={channel_stats['active_ratio']:.3f}")

    if args.plot:
        maybe_plot(spikes)

    if args.save_spikes:
        out = Path(args.save_spikes)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as f:
            pickle.dump({"spikes": spikes, "channel_status": status, "stats": stats, "config": encoder.cfg}, f)
        print(f"saved spikes to: {out}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BNN/SNN 多通道自适应延迟相位编码 + 基线解码")
    parser.add_argument("--model", type=str, default="./modelpara.pth", help="CNN 权重路径")
    parser.add_argument("--n-rf", type=int, default=25)
    parser.add_argument("--rf-h", type=int, default=5)
    parser.add_argument("--rf-w", type=int, default=5)
    parser.add_argument("--event-threshold", type=float, default=0.08)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-spikes", type=str, default="")
    parser.add_argument("--save-metrics", type=str, default="")

    parser.add_argument("--image", type=str, default="", help="单图模式输入")
    parser.add_argument("--train-csv", type=str, default="", help="数据集模式 train annotation csv")
    parser.add_argument("--test-csv", type=str, default="", help="数据集模式 test annotation csv")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    cfg = EncoderConfig(n_rf=args.n_rf, event_threshold=args.event_threshold)
    encoder = AdaptiveDelayPhaseEncoder(cfg)
    net = load_model(model_path)

    if args.train_csv and args.test_csv:
        run_dataset_eval(args, encoder, net)
        return

    if not args.image:
        raise ValueError("请提供 --image（单图模式）或同时提供 --train-csv / --test-csv（数据集模式）")
    if not Path(args.image).exists():
        raise FileNotFoundError(f"image not found: {args.image}")
    run_single(args, encoder, net)


if __name__ == "__main__":
    main()
