# PaiNN 3D (z=0.3) — p_cfg=0.10 (消融实验)

> CFG dropout 消融: 较低的 classifier-free guidance dropout 比率

## 模型配置

| 参数 | 值 |
|------|-----|
| Backbone | PaiNN |
| 协议 | 3D (allow_z=True, z_scale=0.3) |
| p_cfg | 0.10 |
| 学习率 | 1.5e-4 |
| 配置文件 | `configs/flow/painn_conditional_flow.yml` |

## 检查点

- **最佳**: `checkpoints/run_2025-12-27/epoch0850_unweightedvalloss1.5183_posmae0.5607.pt`
- 训练约 1050+ epochs
- 训练时间戳: 2025-12-27

### 关键 epoch 指标

| Epoch | Val Loss | PosMae |
|-------|----------|--------|
| 300 | 1.2028 | 0.6739 |
| 400 | 1.1904 | 0.6132 |
| 600 | 1.1770 | 0.6262 |
| 850 | 1.5183 | **0.5607** |
| 900 | 1.9157 | 0.6376 |

## Grid Search 结果

已对多个 epoch 做了初步测试（grid_search/ 下）：
- `epoch0300/` — Val-loss 最优阶段
- `epoch0600/` — PosMae 较优阶段
- `epoch0850/` — PosMae 最低点
- `epoch1050/` — 过拟合阶段
- `epoch1250/` — 过拟合阶段

## 待办

- [ ] E6: 运行完整 grid search (w∈{0,1,3,5} × K∈{5,10,30})
