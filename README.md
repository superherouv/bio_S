# Biomimetic Visual Information Spatiotemporal Encoding Method for In Vitro Biological Neural Networks

该仓库用于 BNN/SNN 视觉编码与解码实验，当前包含：

- `cnn_ETH__training.py`：ETH-80 三分类 CNN 训练/测试。
- `improved_delayed_phase_encoding.py`：改进版多通道脉冲编码 + 数据集评估。
- `scripts/run_4090_experiments.sh`：面向 4090 的一键实验（基线 + 调参 + 消融）。
- `scripts/summarize_metrics.py`：将 `artifacts/metrics/*.pkl` 汇总为 CSV 表格。

---

## 1) 4090 环境准备

先验证 PyTorch 和 CUDA：

```bash
python -c "import torch;print('torch=',torch.__version__);print('cuda=',torch.cuda.is_available());print('gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

如果 `cuda=False`，请安装支持 CUDA 的 PyTorch。

---

## 2) 核心脚本能力

`improved_delayed_phase_encoding.py` 已支持：

1. **多通道编码**：`cnn_feature + intensity + edge`。
2. **自适应延迟相位编码**：局部强度越高，编码时间窗越短。
3. **事件驱动门控**：按感受野标准差阈值抑制低变化输入。
4. **STDP 离线更新函数**：可用于后续突触可塑性实验。
5. **数据集评估**：输出 `accuracy / macro-f1 / active_ratio / encode_ms`。
6. **设备选择**：`--device auto|cuda|cpu`（推荐 4090 使用 `--device cuda` 或 `auto`）。

---

## 3) 如何运行（4090 推荐）

> 依赖至少：`numpy torch torchvision pillow`。

### 3.1 单图模式

```bash
python improved_delayed_phase_encoding.py \
  --device cuda \
  --model ./modelpara.pth \
  --image ./dataset/ETH3x100/apple/apple_65.png \
  --event-threshold 0.08 \
  --save-spikes ./artifacts/spikes/apple_65_spikes.pkl
```

可选：加 `--plot` 看脉冲图。

### 3.2 数据集模式（推荐主实验入口）

```bash
python improved_delayed_phase_encoding.py \
  --device cuda \
  --model ./modelpara.pth \
  --train-csv ./Species_train_annotation.csv \
  --test-csv ./Species_test_annotation.csv \
  --event-threshold 0.08 \
  --save-metrics ./artifacts/metrics/exp_baseline.pkl
```

输出指标：

- `test_accuracy`
- `test_macro_f1`
- `train/test_mean_active_ratio`
- `train/test_mean_encode_ms`

---

## 4) 一键跑“调参 + 消融”

直接执行：

```bash
bash scripts/run_4090_experiments.sh
```

这个脚本会自动完成：

1. baseline（`event-threshold=0.08`）
2. `event-threshold` 扫描：`0.00, 0.02, 0.05, 0.08, 0.12, 0.16`
3. RF 消融：`4x4` vs `5x5`

结果位置：

- 指标：`artifacts/metrics/*.pkl`
- 日志：`artifacts/logs/*.log`

---

## 5) 结果汇总与分析

把所有指标汇总成 CSV：

```bash
python scripts/summarize_metrics.py
```

将生成：

- `artifacts/tables/metrics_summary.csv`

建议先看这几类结论（论文最常用）：

1. 在精度接近时，`mean_active_ratio` 是否更低（更稀疏）。
2. 在精度接近时，`mean_encode_ms` 是否更低（更高效）。
3. `macro-f1` 是否和 `accuracy` 同步（避免类别偏置假提升）。

---

## 6) 推荐实验问题（可直接写进实验设计）

1. 多通道编码相对单通道是否提升 `accuracy/macro-f1`？
2. 提高 `event-threshold` 是否实现“稀疏-精度”平衡？
3. RF 粒度变化（4x4 vs 5x5）对速度与精度影响如何？
4. STDP 更新前后是否提升跨批次稳定性？
5. ETH-80 最优参数迁移到其他数据集时是否稳健？
