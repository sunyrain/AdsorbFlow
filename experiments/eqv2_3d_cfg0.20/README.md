# EqV2 3D (z=0.3) — p_cfg=0.20 (最佳模型)

> 对应论文 Table 1 "AdsorbFlow (EqV2)" 行 — 当前最佳结果 SR=72.7%

## 模型配置

| 参数 | 值 |
|------|-----|
| Backbone | EquiformerV2 |
| 协议 | 3D (allow_z=True, z_scale=0.3) |
| p_cfg | 0.20 |
| 学习率 | 2.0e-4 |
| 配置文件 | `configs/flow/eqv2_conditional_flow.yml` |

## 检查点

- **最佳**: `checkpoints/run_2026-01-13/best_checkpoint.pt`
- 训练 215 epochs（每 epoch 保存，总计 29 GB）
- 训练时间戳: 2026-01-13

### 关键 epoch 指标

| Epoch | Val Loss | PosMae |
|-------|----------|--------|
| 50 | — | ~1.6 |
| 100 | — | ~1.1 |
| 187 | 1.2734 | 0.8954 |
| 203 | 1.2825 | **0.9141** |
| 208 | 1.1788 | 1.1035 |
| 214 | 1.2660 | 0.9449 |

## Grid Search 结果

### epoch187 (w∈{0,1,3,5,7,10} × K∈{5,10,30})
- **最佳: SR=70.5% (w=7, K=5)**
- 路径: `grid_search/epoch187_w0-10_K5-10-30/`

### epoch208 (w∈{0,1,3,5,7,10,15} × K∈{5,10,30}) ★ 当前全局最佳
- **最佳: SR=72.7%, ΔE=0.320 (w=7, K=30)**
- 路径: `grid_search/epoch208_w0-15_K5-10-30/`

### 选定结果详表 (epoch208)

| w | K=5 | K=10 | K=30 |
|---|-----|------|------|
| 0 | — | — | — |
| 3 | — | — | — |
| 7 | 68.2% | 65.9% | **72.7%** |
| 10 | — | — | — |

## 注意事项

- EqV2 模型每个 epoch 都保存了检查点 → 29 GB；可考虑清理只保留关键 epoch
- 有约 15 个 EqV2 调试 run 在 `experiments/archive/eqv2_debug_runs/`

## 待办

- [ ] 论文制图数据已基本齐全
- [ ] 可选: 在更多 epoch 上做 grid search 确认最佳 epoch
