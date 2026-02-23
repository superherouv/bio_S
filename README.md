# Biomimetic Visual Information Spatiotemporal Encoding Method for In Vitro Biological Neural Networks

该仓库用于 BNN/SNN 视觉编码与解码实验，当前包含：

- `cnn_ETH__training.py`：ETH-80 三分类 CNN 训练/测试。
- `improved_delayed_phase_encoding.py`：改进版多通道脉冲编码 + 基线解码评估。

---

## 1) 当前已实现的改进

`improved_delayed_phase_encoding.py` 支持以下能力：

1. **多通道编码**：`cnn_feature + intensity + edge`。
2. **自适应延迟相位编码**：局部强度越高，编码时间窗越短。
3. **事件驱动门控**：按感受野标准差阈值抑制低变化输入。
4. **STDP 离线更新函数**：可用于后续突触可塑性实验。
5. **数据集级评估**：编码后聚合特征，使用原型解码器输出 `accuracy / macro-f1 / 编码时延 / 激活比例`。

---

## 2) 如何运行

> 先确保环境安装依赖（至少 `numpy torch torchvision pillow`）。

### 2.1 单图编码模式

```bash
python improved_delayed_phase_encoding.py \
  --model ./modelpara.pth \
  --image ./dataset/ETH3x100/apple/apple_65.png \
  --event-threshold 0.08 \
  --save-spikes ./artifacts/apple_65_spikes.pkl
```

可选：加 `--plot` 看脉冲 eventplot。

### 2.2 数据集评估模式（推荐下一步）

```bash
python improved_delayed_phase_encoding.py \
  --model ./modelpara.pth \
  --train-csv ./Species_train_annotation.csv \
  --test-csv ./Species_test_annotation.csv \
  --event-threshold 0.08 \
  --save-metrics ./artifacts/eth80_metrics.pkl
```

运行后会输出：

- `test_accuracy`
- `test_macro_f1`
- `train/test_mean_active_ratio`（事件驱动门控后平均激活比例）
- `train/test_mean_encode_ms`（编码耗时）

---

## 3) 你现在可以重点改进的“提问/实验问题”

建议把下一轮实验问题收敛成以下 5 个“可验证问题”：

1. **编码收益问题**：多通道编码相对单通道（仅 CNN 特征）是否提升 `accuracy / macro-f1`？
2. **稀疏-精度权衡问题**：提高 `event-threshold` 是否能降低激活比例与时延，同时保持可接受精度？
3. **时窗自适应问题**：关闭/开启自适应时窗后，延迟统计（first spike latency）与分类性能差异是多少？
4. **可塑性问题**：引入 STDP 更新前后，跨批次测试性能是否更稳定？
5. **可迁移性问题**：ETH-80 调好的编码参数迁移到 CIFAR-10 子任务时是否仍有效？

---

## 4) 期望什么样的结果

以“可发表的实验节奏”为例，建议你先追求以下趋势（不是绝对阈值）：

- 在相近准确率下，`mean_active_ratio` 明显下降（说明更稀疏、更事件驱动）。
- 在相近准确率下，`mean_encode_ms` 下降（说明编码效率提高）。
- `macro-f1` 与 `accuracy` 一致提升或至少不退化（说明不是仅靠类别偏置提升）。
- 对不同类别都有效，而不是只提升某一类（可进一步补充 per-class 指标）。

---

## 5) 后续可直接加到代码的方向

- 增加 Logistic Regression / SVM 解码器做对照。
- 增加 t-SNE/UMAP 可视化（编码特征聚类可解释性）。
- 增加 STTC、Node Strength、Participation Coefficient 的生物学连接性分析。
- 增加消融实验脚本（开关：multi-channel / adaptive / event-driven / STDP）。
