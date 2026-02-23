# Biomimetic Visual Information Spatiotemporal Encoding Method for In Vitro Biological Neural Networks

该仓库目前包含 ETH-80 三分类视觉实验的基础 CNN 和改进版延迟相位编码实现。

## 新增：改进版 BNN/SNN 编码脚本

`improved_delayed_phase_encoding.py` 已升级为可运行的实验管线，支持：

- **多通道编码**：CNN 特征 + 原图强度通道 + 边缘通道。
- **自适应延迟相位编码**：根据局部输入强度动态缩放编码时间窗。
- **事件驱动门控**：低变化感受野可跳过，减少无效脉冲。
- **STDP 更新函数**：用于后续 BNN/SNN 联合学习实验。

### 快速运行

```bash
python improved_delayed_phase_encoding.py \
  --image ./dataset/ETH3x100/apple/apple_65.png \
  --model ./modelpara.pth \
  --event-threshold 0.08 \
  --save-spikes ./artifacts/apple_65_spikes.pkl
```

如需可视化编码结果，加 `--plot`。

## 现有训练脚本

- `cnn_ETH__training.py`: ETH-80 三分类 CNN 训练 / 测试脚本。

> 注：本仓库后续将继续补充更完整的 BNN-SNN 解码和评估流程（准确率、收敛、能效、连接性指标）。
