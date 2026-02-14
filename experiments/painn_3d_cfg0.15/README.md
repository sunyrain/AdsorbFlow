# PaiNN 3D (z=0.3) — p_cfg=0.15 (主力模型)

> 对应论文 Table 1 "AdsorbFlow (PaiNN)" 行

## 模型配置

| 参数 | 值 |
|------|-----|
| Backbone | PaiNN |
| 协议 | 3D (allow_z=True, z_scale=0.3) |
| p_cfg | 0.15 |
| 学习率 | 1.5e-4 |
| 训练集 | train_split (1.2 GB) |
| 配置文件 | `configs/flow/painn_conditional_flow.yml` |

## 检查点

- **最佳**: `checkpoints/run_2025-12-30/best_checkpoint.pt` (epoch~180, best posmae)
- 训练约 550 epochs，保存了每 50 epoch 的权重
- 训练时间戳: 2025-12-30

### 关键 epoch 指标

| Epoch | Val Loss | PosMae |
|-------|----------|--------|
| 150 | 1.0381 | 0.8726 |
| 200 | 1.1132 | 0.7308 |
| 400 | 1.1241 | 0.5708 |
| 550 | 1.3918 | 0.5906 |

## Grid Search 结果

### epoch155 (w∈{0,1,3} × K=10)
- 最佳: SR=56.8%, ΔE=0.441 (w=3, K=10)
- 路径: `grid_search/epoch155_w0-1-3_K10/`

### epoch180 (w∈{0,1,1.5,3} × K=10) ★ 当前最佳
- **最佳: SR=65.9%, ΔE=0.387 (w=3, K=10)**
- 路径: `grid_search/epoch180_w0-1-1.5-3_K10/`

## 待办

- [ ] E5: 补充更多推理步数 K∈{1,3,5,20,30,50} (w=3 固定)
- [ ] E6: 与 cfg0.10/cfg0.25 做消融对比
