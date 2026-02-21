# AdsorbFlow 实验命令全记录

> 生成日期：2026-02-17
> 硬件：4× RTX 4090 (24GB), 64 CPU cores, 480GB RAM
> 环境：conda env `adsorbdiff`, VASP 6.3.0

---

## 一、环境配置

```bash
# 激活 conda 环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate adsorbdiff

# 安装项目
cd /root/autodl-tmp/AdsorbFlow
pip install -e .

# VASP 相关环境变量
export VASP_PP_PATH="/root/autodl-tmp/potpaw_PBE_54"
export HDF5_USE_FILE_LOCKING=FALSE   # 避免多进程 HDF5 锁冲突
```

---

## 二、模型训练

### 2.1 PaiNN-3D (p_cfg=0.15, lr=1.5e-4)

```bash
python -u -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=1236 \
    main.py --mode train \
    --config-yml configs/flow/painn_conditional_flow.yml \
    --distributed \
    --identifier z_0.3_geo_lift0_cfg_0.15_tr_3_t_opt_pbc_I_lr1.5-4_para_no_lesson
```

> Checkpoint: `2025-12-30-00-38-24-...`, Best: ep180, pos_mae=0.6214

### 2.2 PaiNN-3D (p_cfg=0.10, lr=1.5e-4)

```bash
python -u -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=1236 \
    main.py --mode train \
    --config-yml configs/flow/painn_conditional_flow.yml \
    --distributed \
    --identifier z_0.3_geo_lift0_cfg_0.10_tr_3_t_opt_pbc_I_lr1.5-4_para_no_lesson \
    --optim.p_cfg=0.10
```

> Checkpoint: `2025-12-27-09-55-12-...`, Best: ep850, pos_mae=0.5607

### 2.3 PaiNN-3D (p_cfg=0.25, lr=1.5e-4)

```bash
python -u -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=1236 \
    main.py --mode train \
    --config-yml configs/flow/painn_conditional_flow.yml \
    --distributed \
    --identifier z_0.3_geo_lift0_cfg_0.25_tr_3_t_opt_I_500_lr1.5-4_para_no_lesson \
    --optim.p_cfg=0.25
```

> Checkpoint: `2025-12-30-19-35-28-...`, Best: ep400, pos_mae=0.5330

### 2.4 PaiNN-2D (p_cfg=0.20, lr=1.5e-4, z=0)

```bash
python -u -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=1234 \
    main.py --mode train \
    --config-yml configs/flow/painn_conditional_flow.yml \
    --distributed \
    --identifier z_0_2D_cfg_0.20_tr_3_lr1.5-4 \
    --optim.p_cfg=0.20 \
    --optim.flow.allow_z=False \
    --optim.flow.tr_sigma_z_scale=0
```

> Checkpoint: `2026-02-11-23-00-16-...`, Best: ep281, pos_mae=0.5626

### 2.5 PaiNN-3D (p_cfg=0.20, lr=1.5e-4)

```bash
python -u -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=1234 \
    main.py --mode train \
    --config-yml configs/flow/painn_conditional_flow.yml \
    --distributed \
    --identifier z_0.3_geo_lift0_cfg_0.20_tr_3_t_opt_pbc_I_500_lr1.5-4_para_painn \
    --optim.p_cfg=0.20
```

> Checkpoint: `2026-02-12-09-14-40-...`, Best: ep350, pos_mae=0.6921

### 2.6 EqV2-3D (p_cfg=0.20, lr=2.0e-4)

```bash
python -u -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=1234 \
    main.py --mode train \
    --config-yml configs/flow/eqv2_conditional_flow.yml \
    --distributed \
    --identifier z_0.3_geo_lift0_cfg_0.20_tr_3_t_opt_pbc_I_500_lr2.0-4_para_eqv2
```

> Checkpoint: `2026-01-13-23-21-36-...`, Best: ep187 (pos_mae=0.8954), ep208 (pos_mae=1.1035)

### 2.7 EqV2-2D (p_cfg=0.20, lr=2.0e-4, z=0)

```bash
nohup python -u -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=1234 \
    main.py --mode train \
    --config-yml configs/flow/eqv2_conditional_flow.yml \
    --distributed \
    --identifier z_0_2D_cfg_0.20_tr_3_lr2.0-4_eqv2 \
    --optim.flow.allow_z=False \
    --optim.flow.tr_sigma_z_scale=0 \
    > logs/train_eqv2_2d.log 2>&1 &
```

> Checkpoint: `2026-02-14-11-05-36-...`, Best: ep180 (pos_mae=0.9085)
> ⚠️ 训练在 ep202 崩溃 (ChildFailedError)

---

## 三、Grid Search（MLFF 评估）

### 3.1 通用命令格式

```bash
python -u scripts/grid_search_cfg_flow.py \
    --cfg-scales <w1 w2 ...> \
    --num-steps <K1 K2 ...> \
    --flow-checkpoint <flow_ckpt_path> \
    --relax-checkpoint gemnet_oc_base_s2ef_2M.pt \
    --model-type <painn|eqv2> \
    --nsites 10 --gpus <N> --master-port <port> --skip-existing
```

### 3.2 EqV2-3D ep187 Grid Search

```bash
python -u scripts/grid_search_cfg_flow.py \
    --cfg-scales 0 1 2 3 5 7 10 \
    --num-steps 5 10 30 \
    --flow-checkpoint checkpoints/2026-01-13-23-21-36-z_0.3_geo_lift0_cfg_0.20_tr_3_t_opt_pbc_I_500_lr2.0-4_para_eqv2/epoch0187_unweightedvalloss1.2734_posmae0.8954.pt \
    --relax-checkpoint gemnet_oc_base_s2ef_2M.pt \
    --model-type eqv2 --nsites 10 --gpus 4 --master-port 1237 --skip-existing
```

> 最佳: w=7 K=5, MLFF SR@10=70.5%

### 3.3 EqV2-3D ep208 Grid Search

```bash
python -u scripts/grid_search_cfg_flow.py \
    --cfg-scales 0 1 3 5 7 10 15 \
    --num-steps 5 10 30 \
    --flow-checkpoint checkpoints/2026-01-13-23-21-36-z_0.3_geo_lift0_cfg_0.20_tr_3_t_opt_pbc_I_500_lr2.0-4_para_eqv2/epoch0208_unweightedvalloss1.2685_posmae1.1035.pt \
    --relax-checkpoint gemnet_oc_base_s2ef_2M.pt \
    --model-type eqv2 --nsites 10 --gpus 4 --master-port 1237 --skip-existing
```

> 最佳: w=7 K=30, MLFF SR@10=72.7%

### 3.4 EqV2-2D ep180 Grid Search

```bash
python -u scripts/grid_search_cfg_flow.py \
    --cfg-scales 0 1 3 5 7 10 \
    --num-steps 5 10 30 \
    --flow-checkpoint checkpoints/2026-02-14-11-05-36-z_0_2D_cfg_0.20_tr_3_lr2.0-4_eqv2/epoch0180_unweightedvalloss1.0316_posmae0.9085.pt \
    --relax-checkpoint gemnet_oc_base_s2ef_2M.pt \
    --model-type eqv2 --nsites 10 --gpus 4 --master-port 1237 --skip-existing
```

> 最佳: w=7 K=5, MLFF SR@10=72.7%

### 3.5 PaiNN-2D ep_best Grid Search

```bash
python -u scripts/grid_search_cfg_flow.py \
    --cfg-scales 0 1 3 5 7 10 \
    --num-steps 5 10 30 \
    --flow-checkpoint checkpoints/2026-02-11-23-00-16-z_0_2D_cfg_0.20_tr_3_lr1.5-4/ \
    --relax-checkpoint gemnet_oc_base_s2ef_2M.pt \
    --model-type painn --nsites 10 --gpus 2 --master-port 1236 --skip-existing
```

> 最佳: w=5 K=5, MLFF SR@10=63.6%

### 3.6 PaiNN-3D-0.20 ep150 & ep200 Grid Search

```bash
nohup python -u scripts/grid_search_cfg_flow.py \
    --cfg-scales 0 1 3 5 7 10 \
    --num-steps 5 10 30 \
    --flow-checkpoint checkpoints/2026-02-12-09-14-40-z_0.3_geo_lift0_cfg_0.20_tr_3_t_opt_pbc_I_500_lr1.5-4_para_painn/epoch0150_*.pt \
    --relax-checkpoint gemnet_oc_base_s2ef_2M.pt \
    --model-type painn --nsites 10 --gpus 2 --master-port 1241 --skip-existing \
    > logs/grid_3d_ep150.log 2>&1 &
```

> 最佳: w=1 K=30, MLFF SR@10=63.6%

---

## 四、VASP DFT 单点验证

### 4.1 通用流程

VASP 验证脚本分两步：
1. **生成 VASP 输入**: 对 Level 1/2/5/10，每个 SID 选最优非异常结构
2. **并行运行 VASP SP**: 8 核/任务，ProcessPoolExecutor 并行

VASP 参数：`nsw=0, encut=350, gga=RP, ncore=4, isym=0, lreal=Auto`
VASP 路径：`/root/autodl-tmp/vasp-autodl/vasp.6.3.0/bin/vasp_std`
赝势路径：`/root/autodl-tmp/potpaw_PBE_54/`

### 4.2 EqV2-3D ep187 w=7 K=5

```bash
# 一键脚本：生成输入 + 运行 VASP
nohup python -u scripts/run_vasp_dft/run_vasp_ep187_cfg7_steps5.py \
    > logs/vasp_ep187_cfg7_steps5.log 2>&1 &
```

> 结果: VASP SR@10 = 63.6% (28/44)

### 4.3 EqV2-2D ep180 w=7 K=5

```bash
# 一键脚本：生成输入 + 运行 VASP（跳过 MLFF 异常结构，无超时限制）
export HDF5_USE_FILE_LOCKING=FALSE
nohup python -u scripts/run_vasp_dft/run_vasp_eqv2_2d_ep180_cfg7_steps5.py \
    > logs/vasp_eqv2_2d_ep180_cfg7_steps5.log 2>&1 &
```

> 101 个任务，全部收敛（0 失败），耗时 ~11 小时
> 结果: VASP SR@10 = 61.4% (27/44)

### 4.4 VASP 结果分析（通用）

```bash
# 修改 analyze_multisite_results.py 中的 BASE_PATH/NSITES_DIR/CFG_DIR 后运行
python scripts/run_vasp_dft/analyze_multisite_results.py
```

### 4.5 重跑未收敛的 VASP 计算

```bash
# 修改 rerun_unconverged_vasp.py 中的目标列表后运行
export HDF5_USE_FILE_LOCKING=FALSE
nohup python -u scripts/run_vasp_dft/rerun_unconverged_vasp.py \
    > logs/rerun_vasp.log 2>&1 &
```

---

## 五、实验结果汇总

### 5.1 MLFF Grid Search 最佳结果

| 模型 | Backbone | 维度 | Epoch | Best w | Best K | MLFF SR@10 |
|:-----|:---------|:-----|:------|:-------|:-------|:-----------|
| EqV2-3D | EqV2 | 3D | 208 | 7 | 30 | **72.7%** |
| EqV2-2D | EqV2 | 2D | 180 | 7 | 5 | **72.7%** |
| EqV2-3D | EqV2 | 3D | 187 | 7 | 5 | 70.5% |
| PaiNN-3D-0.25 | PaiNN | 3D | 300 | 3 | 10 | 65.9% |
| PaiNN-2D | PaiNN | 2D | best | 5 | 5 | 63.6% |
| PaiNN-3D-0.20 | PaiNN | 3D | best | 1 | 30 | 63.6% |
| PaiNN-3D-0.15 | PaiNN | 3D | 70 | 1 | 10 | 61.4% |

### 5.2 VASP DFT 验证结果

| 模型 | Epoch | w | K | VASP SR@1 | VASP SR@2 | VASP SR@5 | VASP SR@10 | MLFF SR@10 |
|:-----|:------|:--|:--|:----------|:----------|:----------|:-----------|:-----------|
| **EqV2-3D** | 187 | 7 | 5 | 31.8% (14/44) | 38.6% (17/44) | 40.9% (18/44) | **63.6%** (28/44) | 70.5% |
| **EqV2-2D** | 180 | 7 | 5 | 25.0% (11/44) | 38.6% (17/44) | 50.0% (22/44) | **61.4%** (27/44) | 72.7% |
| EqV2-3D | 208 | 7 | 5 | 31.8% (14/44) | 40.9% (18/44) | 47.7% (21/44) | **56.8%** (25/44) | 68.2% |
| PaiNN-3D | 180 | 3 | 10 | 29.5% (13/44) | 43.2% (19/44) | 52.3% (23/44) | **54.5%** (24/44) | 52.3% |

### 5.3 EqV2-2D ep180 全部 Grid Search 结果

| w | K | SR@1 | SR@5 | SR@10 | Mean SR | ΔE_mean |
|:--|:--|:-----|:-----|:------|:--------|:--------|
| **7** | **5** | 31.8% | 61.4% | **72.7%** | 35.2% | 0.295 |
| 10 | 30 | 40.9% | 63.6% | 70.5% | 38.4% | 0.333 |
| 10 | 10 | 29.5% | 63.6% | 70.5% | 38.0% | 0.284 |
| 7 | 10 | 40.9% | 56.8% | 70.5% | 37.3% | 0.293 |
| 3 | 5 | 31.8% | 65.9% | 70.5% | 35.5% | 0.290 |
| 7 | 30 | 31.8% | 63.6% | 68.2% | 37.3% | 0.297 |
| 5 | 10 | 38.6% | 63.6% | 68.2% | 34.3% | 0.300 |
| 0 | 30 | 31.8% | 63.6% | 68.2% | 26.6% | 0.412 |
| 3 | 10 | 31.8% | 56.8% | 68.2% | 32.7% | 0.308 |
| 1 | 30 | 34.1% | 59.1% | 68.2% | 28.6% | 0.364 |
| 5 | 5 | 31.8% | 59.1% | 65.9% | 34.5% | 0.294 |
| 1 | 5 | 27.3% | 61.4% | 65.9% | 28.0% | 0.344 |
| 1 | 10 | 34.1% | 61.4% | 65.9% | 28.9% | 0.356 |
| 5 | 30 | 36.4% | 59.1% | 65.9% | 34.8% | 0.311 |
| 0 | 10 | 31.8% | 63.6% | 65.9% | 25.2% | 0.410 |
| 10 | 5 | 36.4% | 56.8% | 63.6% | 32.3% | 0.374 |
| 0 | 5 | 20.5% | 54.5% | 63.6% | 22.7% | 0.426 |
| 3 | 30 | 36.4% | 56.8% | 61.4% | 32.7% | 0.297 |

---

## 六、关键脚本索引

| 脚本 | 功能 |
|:-----|:-----|
| `main.py` | 训练/推理入口 |
| `scripts/grid_search_cfg_flow.py` | Grid search 自动化（flow → LMDB → relax → eval） |
| `scripts/eval.py` | MLFF 评估: SR 计算、异常检测 |
| `scripts/run_vasp_dft/run_vasp_ep187_cfg7_steps5.py` | VASP SP (EqV2-3D ep187) |
| `scripts/run_vasp_dft/run_vasp_eqv2_2d_ep180_cfg7_steps5.py` | VASP SP (EqV2-2D ep180) |
| `scripts/run_vasp_dft/write_vasp_inputs_multisite.py` | 多层级 VASP 输入生成 |
| `scripts/run_vasp_dft/launch_vasp_multisite.py` | VASP 并行运行器 |
| `scripts/run_vasp_dft/analyze_multisite_results.py` | VASP 结果分析 |
| `scripts/run_vasp_dft/rerun_unconverged_vasp.py` | 重跑未收敛 VASP |
| `configs/flow/painn_conditional_flow.yml` | PaiNN 训练配置 |
| `configs/flow/eqv2_conditional_flow.yml` | EqV2 训练配置 |

---

## 七、数据文件

| 文件 | 内容 |
|:-----|:-----|
| `oc20_dense_mappings/oc20dense_targets.pkl` | DFT target 吸附能 `{sid: E_ads}` |
| `oc20_dense_mappings/oc20dense_ref_energies.pkl` | 参考能量(裸表面+气态分子) `{sid: E_ref}` |
| `oc20_dense_mappings/oc20dense_tags.pkl` | 原子标签 `{sid: tags_array}` |
| `gemnet_oc_base_s2ef_2M.pt` | GemNet-OC 弛豫模型权重 |

---

## 八、评估标准

- **判据**: AdsorbML 单侧标准，`E_ads(pred) - E_ads(target) ≤ 0.1 eV` 即为成功
- **吸附能**: `E_ads = E_total - E_ref(sid)`
- **SR@k**: 使用前 k 个 site 的结果取并集（任一 site 成功即成功）
- **异常检测**: dissociated / desorbed / surface_changed / intercalated
- **分母**: 44 个 SID (val_nonrelaxed_update)

---

## 九、常用监控命令

```bash
# 查看 GPU 使用
nvidia-smi

# 查看 grid search 进度
tail -f logs/grid_search_*.log

# 查看 VASP 进度
tail -f logs/vasp_*.log

# 查看正在运行的进程
pgrep -af "grid_search|vasp_std|main.py"

# 统计 VASP 完成数
for d in cfg7_steps5/vasp_level_*/*/; do
    [[ -f "$d/OUTCAR" ]] && grep -q "General timing" "$d/OUTCAR" && echo OK || echo FAIL
done | sort | uniq -c
```
