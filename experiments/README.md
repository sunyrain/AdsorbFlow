# AdsorbFlow 实验总览

> 最后更新: 2026-02-11

本目录按 **模型/配置** 组织所有实验，每个子文件夹通过符号链接指向原始数据（不移动数据文件）。

---

## 目录结构

```
experiments/
├── painn_3d_cfg0.15/     ★ PaiNN 主力模型 (SR=65.9%)
│   ├── checkpoints/      → checkpoints/2025-12-30-00-38-24-...
│   ├── grid_search/      → grid_search_runs/... (epoch155, epoch180)
│   ├── training_logs/    → results/...
│   └── README.md
│
├── painn_3d_cfg0.10/       PaiNN 消融 (p_cfg=0.10)
│   ├── checkpoints/      → checkpoints/2025-12-27-09-55-12-...
│   ├── grid_search/      → grid_search_runs/test_ck_* (5个epoch)
│   ├── training_logs/    → results/...
│   └── README.md
│
├── painn_3d_cfg0.25/       PaiNN 消融 (p_cfg=0.25)
│   ├── checkpoints/      → checkpoints/2025-12-30-19-35-28-...
│   ├── grid_search/      → grid_search_runs/test_0.25_* (5个epoch)
│   ├── training_logs/    → results/...
│   └── README.md
│
├── eqv2_3d_cfg0.20/      ★ EqV2 最佳模型 (SR=72.7%)
│   ├── checkpoints/      → checkpoints/2026-01-13-23-21-36-...
│   ├── grid_search/      → grid_search_runs/... (epoch187, epoch208)
│   ├── training_logs/    → results/...
│   └── README.md
│
├── painn_2d_cfg0.15/       PaiNN 2D 对比实验 (❌ 待训练)
│   ├── checkpoints/      (空，训练后填充)
│   ├── grid_search/      → 早期测试 epoch060/070 (参考)
│   ├── training_logs/    (空)
│   └── README.md
│
└── archive/                归档: 调试/失败/早期实验
    ├── eqv2_debug_runs/  → ~16 个 EqV2 调参失败 run
    ├── painn_2d_early/     早期 2D 模型 (epoch70)
    └── early_test_runs/  → test_z_*, test_ck, test_0.25 等
```

---

## 模型性能对比

| 实验目录 | Backbone | p_cfg | Best PosMae | Best SR@10 | 状态 |
|----------|----------|-------|-------------|------------|------|
| `eqv2_3d_cfg0.20/` | EqV2 | 0.20 | 0.9141 | **72.7%** (w=7,K=30) | ✅ 完成 |
| `painn_3d_cfg0.15/` | PaiNN | 0.15 | 0.5708 | **65.9%** (w=3,K=10) | ✅ 完成 |
| `painn_3d_cfg0.10/` | PaiNN | 0.10 | 0.5607 | 待完整测试 | 🔄 部分 |
| `painn_3d_cfg0.25/` | PaiNN | 0.25 | 0.5330 | 待完整测试 | 🔄 部分 |
| `painn_2d_cfg0.15/` | PaiNN | 0.15 | — | — | ❌ 待训练 |

---

## 共享资源位置

| 资源 | 路径 | 大小 |
|------|------|------|
| GemNet-OC MLFF 权重 | `gemnet_oc_base_s2ef_2M.pt` | 149 MB |
| PaiNN Scaling Factors | `configs/scaling_factors/painn_nb6_scaling_factors.pt` | — |
| 训练数据 | `train_split/` | 1.2 GB |
| 全能量训练数据 | `train_allE/` | 1.3 GB |
| 验证集 | `val_split/` | 133 MB |
| 评估参考集 | `val_nonrelaxed_update/` | 5 MB |
| OOD 验证集 | `valood50_R1I0.1/` | 5.5 MB |
| OC20-Dense Mappings | `oc20_dense_mappings/` | 17 MB |
| SO3 预计算 | `so3_precompute/` | — |

---

## 磁盘占用摘要

| 目录 | 大小 | 说明 |
|------|------|------|
| `checkpoints/` | 41 GB | 模型权重（含 29GB EqV2 每epoch存储） |
| `grid_search_runs/` | 104 GB | 推理/relaxation traj 文件 |
| `logs/` | 2.6 GB | wandb 日志 |
| `results/` | 13 MB | 训练 val 指标 JSON |
| `test_*/` | 4.7 GB | 早期零散测试 |
| **总计** | **~155 GB** | |

### 清理建议
1. EqV2 检查点只保留 best + epoch187/208 → 节省 ~25 GB
2. 删除 archive 中的空 EqV2 调试目录 → 释放 inode
3. grid_search 中非最佳组合的 traj 可按需删除 → 节省 ~50 GB
