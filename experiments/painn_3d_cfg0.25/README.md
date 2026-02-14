# PaiNN 3D (z=0.3) — p_cfg=0.25 (消融实验)

> CFG dropout 消融: 较高的 classifier-free guidance dropout 比率

## 模型配置

| 参数 | 值 |
|------|-----|
| Backbone | PaiNN |
| 协议 | 3D (allow_z=True, z_scale=0.3) |
| p_cfg | 0.25 |
| 学习率 | 1.5e-4 |
| 配置文件 | `configs/flow/painn_conditional_flow.yml` |

## 检查点

- **最佳**: `checkpoints/run_2025-12-30/epoch0400_unweightedvalloss1.1710_posmae0.5330.pt`
- 训练约 550 epochs
- 训练时间戳: 2025-12-30

### 关键 epoch 指标

| Epoch | Val Loss | PosMae |
|-------|----------|--------|
| 100 | 1.0816 | 0.9837 |
| 200 | 1.0498 | 0.7385 |
| 300 | 1.1554 | 0.6462 |
| 400 | 1.1710 | **0.5330** |
| 500 | 1.2600 | 0.6093 |

## Grid Search 结果

已对多个 epoch 做了初步测试（grid_search/ 下）：
- `epoch0100/` — 早期阶段
- `epoch0200/` — Val-loss 最优阶段
- `epoch0300/` — PosMae 较优阶段
- `epoch0400/` — PosMae 最低点 ★
- `epoch0500/` — 性能下滑

## 待办

- [ ] E6: 运行完整 grid search (w∈{0,1,3,5} × K∈{5,10,30})
