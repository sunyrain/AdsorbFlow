# AdsorbFlow 实验 TODO 清单

> 最后更新: 2026-02-13
> 本文档列出所有待完成的实验，按优先级排序。
> 在 AdsorbFlow 工作区中调用 Copilot 进行具体部署。

---

## 现有资产盘点

> **实验文件已按模型/配置组织到 `experiments/` 目录，详见 [`experiments/README.md`](experiments/README.md)**

### 已训练的模型检查点

| ID | Backbone | p_cfg | 最优epoch | 实验目录 | 原始路径 |
|----|----------|-------|-----------|----------|----------|
| **PaiNN-3D-0.15** | PaiNN | 0.15 | ep180 (best_posmae) | `experiments/painn_3d_cfg0.15/` | `checkpoints/2025-12-30-00-38-24-...` |
| **PaiNN-3D-0.20** | PaiNN | 0.20 | ep350 (posmae=0.6921) | `experiments/painn_3d_cfg0.20/` | `checkpoints/2026-02-12-09-14-40-...` |
| **PaiNN-3D-0.25** | PaiNN | 0.25 | ep400 (posmae=0.5330) | `experiments/painn_3d_cfg0.25/` | `checkpoints/2025-12-30-19-35-28-...` |
| **PaiNN-3D-0.10** | PaiNN | 0.10 | ep850 (posmae=0.5607) | `experiments/painn_3d_cfg0.10/` | `checkpoints/2025-12-27-09-55-12-...` |
| **EqV2-3D-0.20** | EqV2 | 0.20 | ep208 (SR=72.7%) | `experiments/eqv2_3d_cfg0.20/` | `checkpoints/2026-01-13-23-21-36-...` |
| **PaiNN-2D-0.20** | PaiNN | 0.20 | ep281 (posmae=0.5626) | `experiments/painn_2d_cfg0.20/` | `checkpoints/2026-02-11-23-00-16-...` |

### 已有 Grid Search 结果

| 模型 | Epoch | Grid | 最佳结果 | 状态 |
|------|-------|------|---------|------|
| PaiNN-3D-0.15 | 155 | w∈{0,1,3} × K=10 | SR=56.8%, ΔE=0.441 | ✅ |
| PaiNN-3D-0.15 | 180 | w∈{0,1,1.5,3} × K=10 | SR=65.9%, ΔE=0.387 | ✅ |
| **PaiNN-3D-0.20** | best | w∈{0,1,2,3,5,7,10} × K∈{5,10,30} | **SR=63.6% (w=1,K=30), ΔE=0.515** | ✅ 21/21 |
| **PaiNN-2D-0.20** | best | w∈{0,1,2,3,5,10} × K∈{5,10,30} | **SR=63.6% (w=5,K=5), ΔE=0.398** | ✅ 18/18 |
| EqV2-0.20 | 187 | w∈{0,1,2,3,5,7,10} × K∈{5,10,30} | SR=70.5% (w=7,K=5), ΔE=0.342 | ✅ 21/21 |
| EqV2-0.20 | 208 | w∈{0,1,3,5,7,10,15} × K∈{5,10,30} | **SR=72.7% (w=7,K=30), ΔE=0.320** | ✅ 21/21 |

---

## 📋 实验清单

### ═══════════════════════════════════════
### P0 — 必须完成（论文核心表格缺失数据）
### ═══════════════════════════════════════

---

### E1: PaiNN 2D 模型训练 ✅ 已完成
- **目标**: 训练 allow_z=False 的 PaiNN 模型，用于 2D vs 3D 对比 (论文 Table 2)
- **配置**: p_cfg=0.20, allow_z=False, tr_sigma_z_scale=0, 2×RTX4090
- **结果**: 370 epochs 训练，best posmae=0.5626 (epoch 281)，val loss 约 ep150 后开始上升
- **检查点**: `checkpoints/2026-02-11-23-00-16-z_0_2D_cfg_0.20_tr_3_lr1.5-4/best_checkpoint.pt`
- **状态**: ✅ 完成 (2026-02-12)

---

### E2: PaiNN 2D 模型 Grid Search 评估 ✅ 已完成
- **目标**: 对 E1 训练好的 2D 模型做 cfg × steps 网格搜索
- **网格**: w ∈ {0, 1, 2, 3, 5, 10} × K ∈ {5, 10, 30} = 18 组合
- **结果**: Best SR@10=**63.6%** (w=5, K=5, ΔE=0.398)
- **结果文件**: `grid_search_runs/2026-02-11-23-00-16-z_0_2D_cfg_0.20_tr_3_lr1.5-4_best_checkpoint/val_nonrelaxed_update/nsites_10/grid_search_results_nsites10.jsonl`
- **状态**: ✅ 完成 (2026-02-13)

---

### E3: Anomaly Rate 统计
- **目标**: 从现有 traj 文件中提取 anomaly rate
- **对应论文**: Table 2 两行 Anomaly rate 列 (`\emph{pending}`)
- **依赖**: E2 完成 (2D)；3D 数据已有
- **方法**: 使用 `scripts/eval.py` 中的 `DetectTrajAnomaly` pipeline
  ```bash
  cd /root/autodl-tmp/AdsorbFlow
  python scripts/eval.py \
    --traj-dir <grid_search_run_path>/relaxations \
    --ref-data val_nonrelaxed_update \
    --report-anomaly
  ```
- **需要提取**:
  - 3D (PaiNN-0.15, epoch180, w=3, K=10) — anomaly rate
  - 2D (E1 最佳, 最佳 w, K=10) — anomaly rate
- **预估**: ~30min (纯计算，无需 GPU)
- **状态**: ❌ 未开始

---

### E4: 文献查找 — AdsorbML / AdsorbDiff 基线数值
- **目标**: 从已发表论文中查找 AdsorbML 和 AdsorbDiff 的 SR 数值
- **对应论文**: Table 1 第1-2行 `\emph{from ref.}` 列
- **需要查找**:
  - AdsorbML (Lan et al., 2023): Success Rate @ nsites=10 (0.1 eV tolerance)
  - AdsorbDiff (Adsorbate Diffusion, 2024): Success Rate @ nsites=10 (0.1 eV tolerance)  
  - 注意匹配评估条件: OC20-Dense，GemNet-OC MLFF relaxation
- **来源**:
  - AdsorbML: https://arxiv.org/abs/2211.16486
  - AdsorbDiff: https://arxiv.org/abs/2305.03582 (or newer version)
- **预估**: 1h 查阅论文
- **状态**: ❌ 未开始

---

### ═══════════════════════════════════════
### P1 — 论文图表完善（支撑分析）
### ═══════════════════════════════════════

---

### E5: PaiNN 3D 补充 Grid Search (更多 steps 点)
- **目标**: 当前 PaiNN 只有 K=10 的结果，需补充更多推理步数的数据点
- **对应论文**: Table 1 PaiNN 行；Fig 2 accuracy-efficiency frontier
- **依赖**: 无（使用现有 PaiNN-0.15 epoch180 检查点）
- **网格**: w=3 (固定最佳), K ∈ {1, 3, 5, 20, 30, 50}
- **命令**:
  ```bash
  CKPT=checkpoints/2025-12-30-00-38-24-z_0.3_geo_lift0_cfg_0.15_tr_3_t_opt_I_500_lr1.5-4_para_no_lesson/best_checkpoint.pt
  
  for K in 1 3 5 20 30 50; do
    python -u -m torch.distributed.launch \
      --nproc_per_node=1 --master_port=1234 \
      main.py --mode run-relaxations \
      --config-yml configs/flow/painn_conditional_flow.yml \
      --checkpoint $CKPT \
      --task.relax_opt.traj_dir=grid_search_runs/painn_0.15_ep180_w3_K${K} \
      --task.relax_opt.cfg_scale=3 \
      --task.relax_opt.num_steps=$K \
      --distributed --model.sampling=True --debug
  done
  ```
- **注意**: 需确认 best_checkpoint.pt 对应的 epoch 以及使用哪个 epoch 的检查点 (epoch180 对应的在 grid_search_runs 里已有结果)
- **预估**: 6 × ~15min = ~1.5h
- **状态**: ❌ 未开始

---

### E6: PaiNN p_cfg 消融实验 Grid Search
- **目标**: 对比 p_cfg=0.10 / 0.15 / 0.25 三组 CFG dropout 的效果
- **对应论文**: Discussion / Ablation 分析 (CFG guidance strength)
- **依赖**: 无（使用现有检查点）
- **方法**: 对 p_cfg=0.10 和 p_cfg=0.25 的检查点运行同样的 grid search
  ```bash
  # p_cfg=0.10 (epoch850 - 最低 posmae)
  CKPT_010=checkpoints/2025-12-27-09-55-12-z_0.3_geo_lift0_cfg_0.10_tr_3_t_opt_pbc_I_lr1.5-4_para_no_lesson/epoch0850_unweightedvalloss1.5183_posmae0.5607.pt
  
  for w in 0 1 3 5; do
    for K in 5 10 30; do
      python -u -m torch.distributed.launch \
        --nproc_per_node=1 --master_port=1234 \
        main.py --mode run-relaxations \
        --config-yml configs/flow/painn_conditional_flow.yml \
        --checkpoint $CKPT_010 \
        --task.relax_opt.traj_dir=grid_search_runs/painn_0.10_ep850_w${w}_K${K} \
        --task.relax_opt.cfg_scale=$w \
        --task.relax_opt.num_steps=$K \
        --distributed --model.sampling=True --debug
      done
  done
  
  # p_cfg=0.25 (epoch400 - 最低 posmae)
  CKPT_025=checkpoints/2025-12-30-19-35-28-z_0.3_geo_lift0_cfg_0.25_tr_3_t_opt_I_500_lr1.5-4_para_no_lesson/epoch0400_unweightedvalloss1.1710_posmae0.5330.pt
  
  for w in 0 1 3 5; do
    for K in 5 10 30; do
      python -u -m torch.distributed.launch \
        --nproc_per_node=1 --master_port=1234 \
        main.py --mode run-relaxations \
        --config-yml configs/flow/painn_conditional_flow.yml \
        --checkpoint $CKPT_025 \
        --task.relax_opt.traj_dir=grid_search_runs/painn_0.25_ep400_w${w}_K${K} \
        --task.relax_opt.cfg_scale=$w \
        --task.relax_opt.num_steps=$K \
        --distributed --model.sampling=True --debug
      done
  done
  ```
- **预估**: 2 × 12 组合 × ~15min = ~6h
- **注意**: p_cfg=0.25 已有部分 test_0.25_* 结果在 grid_search_runs，需先检查覆盖情况
- **状态**: ❌ 未开始

---

### ═══════════════════════════════════════
### P2 — 图表制作与论文细节
### ═══════════════════════════════════════

---

### E7: 制作 Fig 1 — AdsorbFlow 方法概览图
- **目标**: 方法流程图 (input → flow matching → relaxation → evaluation)
- **对应论文**: `\includegraphics{fig1_adsorbflow_overview.pdf}` (当前被注释)
- **工具**: matplotlib / tikz / draw.io
- **内容**: 
  - 左: 催化剂表面 + 吸附物 (初始随机位姿 → 目标位姿)
  - 中: Flow matching ODE 积分示意 (t=0 → t=1)
  - 右: CFG guidance 条件能量信号
  - 下: PaiNN vs EqV2 backbone 对比
- **状态**: ❌ 未开始

---

### E8: 制作 Fig 2 — Accuracy-Efficiency Frontier
- **目标**: SR@10 vs 推理步数 K 的曲线图
- **对应论文**: `\includegraphics{fig2_steps_frontier.pdf}` (当前被注释)
- **依赖**: E5 完成 (PaiNN更多K点)
- **数据**:
  - EqV2 (w=7): K=5 → 70.5%, K=10 → 65.9%, K=30 → 63.6% (epoch187)
  - EqV2 (w=7): K=5 → 68.2%, K=10 → 65.9%, K=30 → 72.7% (epoch208)
  - PaiNN (w=3): K=10 → 65.9% (epoch180), 其余待 E5 补充
- **状态**: ❌ 未开始

---

### E9: 制作 Fig 3 — 3D Case Studies
- **目标**: 可视化若干成功放置案例 (分子在表面上的位姿)
- **对应论文**: `\includegraphics{fig3_3d_case_studies.pdf}` (当前被注释)
- **工具**: ASE + matplotlib 可视化 traj 文件
- **数据**: 从 grid_search_runs 中选取典型成功/失败案例
- **状态**: ❌ 未开始

---

### E10: 制作 Fig 4 — CFG Guidance 效果对比
- **目标**: 展示不同 guidance scale w 对生成质量的影响
- **对应论文**: `\includegraphics{fig4_closed_loop.pdf}` (当前被注释)
- **数据**: EqV2 grid search 的 w=0/3/7/10 对比热力图或柱状图
- **状态**: ❌ 未开始

---

### ═══════════════════════════════════════
### P3 — 论文元数据（非实验）
### ═══════════════════════════════════════

---

### M1: 填写作者信息
- **位置**: sn-article.tex L44-56
- **内容**: 真实作者姓名、邮箱、机构地址
- **状态**: ❌ 待填写

### M2: 填写致谢和作者贡献
- **位置**: sn-article.tex L335, L338
- **内容**: 资助来源、计算资源、CRediT 作者贡献
- **状态**: ❌ 待填写

### M3: 完善参考文献
- **位置**: sn-article.tex L349 (PaiNN 条目)
- **内容**: 补充完整引用格式
- **状态**: ❌ 待填写

---

## 执行顺序建议

```
阶段 1 (最紧迫):
  E1 (训练 2D PaiNN) ──────────────────── ~12h GPU
  E4 (查文献填基线)  ──────────────────── ~1h 人工
     ↓
阶段 2 (E1完成后):
  E2 (2D Grid Search) ─────────────────── ~3h GPU
  E5 (PaiNN补充步数) ──── 可与E2并行 ──── ~1.5h GPU
     ↓
阶段 3:
  E3 (Anomaly Rate) ───────────────────── ~0.5h CPU
  E6 (p_cfg消融) ──────────────────────── ~6h GPU (可选)
     ↓
阶段 4 (数据齐全后):
  E7-E10 (制图) ───────────────────────── ~1天
  M1-M3 (元数据) ──────────────────────── ~0.5h
```

## 总 GPU 时间预估

| 实验 | GPU 时间 | 优先级 |
|------|---------|--------|
| E1 训练 | ~12h | P0 |
| E2 2D Grid Search | ~3h | P0 |
| E3 Anomaly | ~0.5h (CPU) | P0 |
| E5 PaiNN 补充 K | ~1.5h | P1 |
| E6 p_cfg 消融 | ~6h | P1 (可选) |
| **总计** | **~23h GPU** | |

---

## 快速检查命令

```bash
# 进入工作目录
cd /root/autodl-tmp/AdsorbFlow

# 检查 GPU
nvidia-smi

# 检查 PyTorch
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"

# 检查数据
ls train_split/ val_split/ val_nonrelaxed_update/

# 检查现有检查点
ls checkpoints/

# 检查已有结果
ls grid_search_runs/

# 查看某个 grid search 的结果
cat grid_search_runs/<run_dir>/results.jsonl | python3 -m json.tool
```

---

*本文档由 Copilot 自动生成，请在实验文件夹中调用 Copilot 进行具体实验部署。*
