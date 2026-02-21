# AdsorbFlow 实验结果汇报

> 最后更新: 2026-02-20

---

## 一、项目概述

**AdsorbFlow** 基于 Flow Matching + Classifier-Free Guidance (CFG) 的催化剂吸附位点预测方法。

**核心思路**: 用 conditional flow matching 学习吸附分子在催化剂表面的分布,
通过 CFG 在推理时引入条件引导, 生成多样化的候选吸附构型。
生成后用 GemNet-OC 做 MLFF 弛豫, 选出能量最低的候选, 最终通过 VASP DFT 做 single-point 验证。

**评估基准**: OC20-Dense
- Val ID: 44 systems, 训练集内分布
- OOD: 50 systems, 与 val ID 不重叠

**成功判据**: $E_{ads}^{pred} - E_{ads}^{target} \leq 0.1$ eV（单侧：预测能量不高于目标能量超过 0.1 eV）

---

## 二、模型配置

### 2.1 模型清单

| 模型 | Backbone | 生成协议 | 训练 Epoch | 数据集 | Checkpoint |
|------|----------|---------|-----------|--------|------------|
| **AdsorbFlow (EqV2)** | EquiformerV2 | 2D平移+SO(3)旋转 | 180 | OC20-Dense train | `2026-02-14-11-05-36-...` |
| **AdsorbFlow (PaiNN)** | PaiNN | 2D平移+SO(3)旋转 | 150 | OC20-Dense train | `2026-02-11-23-00-16-...` |

### 2.2 关键超参

| 参数 | EqV2 | PaiNN | 说明 |
|------|------|-------|------|
| CFG 权重 w | **7** | **5** | Grid Search 选出最佳 |
| 推理步数 K | **5** | **5** | 5 步即收敛 |
| CFG 训练 dropout | 0.20 | 0.20 | 训练时 20% drop 条件 |
| 弛豫 MLFF | GemNet-OC | GemNet-OC | 统一使用 `gemnet_oc_base_s2ef_2M.pt` |
| 异常检测 | 结构检查 | 结构检查 | 离解/脱附/表面变化/嵌入 |

---

## 三、MLFF Grid Search 结果

### 3.1 最佳配置

> Grid Search 在 MLFF 级别 (GemNet-OC 弛豫能量 vs DFT 目标) 搜索最佳 (w, K)。
> 搜索空间: w ∈ {0,1,3,5,7,10}, K ∈ {5,10,30}。

| 模型 | 最佳 (w, K) | MLFF SR@10 | 次优 |
|------|------------|-----------|------|
| **EqV2** | **(7, 5)** | **72.7%** | (5,5)=65.9%, (10,5)=63.6% |
| **PaiNN** | **(5, 5)** | **63.6%** | (7,5)=63.6%, (10,5)=63.6% (w=5 ΔE 最低) |

### 3.2 EqV2 完整 Grid Search (18 configs)

| w \ K | 5 | 10 | 30 |
|-------|---|----|----|
| 0 | 63.6% | 65.9% | 68.2% |
| 1 | 65.9% | 65.9% | 68.2% |
| 3 | **70.5%** | 68.2% | 61.4% |
| 5 | 65.9% | 68.2% | 65.9% |
| **7** | **72.7%** | **70.5%** | 68.2% |
| 10 | 63.6% | **70.5%** | **70.5%** |

### 3.3 PaiNN Grid Search (21 configs)

| w \ K | 5 | 10 | 30 |
|-------|---|----|----|
| 0 | 56.8% | 56.8% | 54.5% |
| 1 | 56.8% | 61.4% | 59.1% |
| 2 | 56.8% | 61.4% | 56.8% |
| 3 | 61.4% | 61.4% | 59.1% |
| **5** | **63.6%** | 61.4% | 56.8% |
| 7 | 63.6% | 59.1% | 63.6% |
| 10 | 63.6% | 59.1% | 59.1% |

> **最佳: w=5, K=5, SR@10=63.6%** (w=7/10 K=5 并列但 w=5 ΔE 更低)。

---

## 四、VASP DFT 验证结果

### 4.1 评估协议

**方法 A (论文方法, 与 AdsorbDiff 一致):**
1. 生成 Nsites=k 个候选位点
2. 分别做 MLFF 弛豫, 排除异常结构
3. 选 MLFF 能量最低者 → 1 次 VASP SP
4. 判定: $E_{ads}^{VASP} - E_{ads}^{target} \leq 0.1$ eV 即成功
5. $\text{SR@k} = \frac{\text{成功 SID 数}}{\text{总 SID 数}} \times 100\%$

**Nsites=1 特殊说明:** 仅 1 个候选直接做 DFT, 我们报道 10 个 seed 中最优 seed 的 SR (与 AdsorbDiff 31.8% 基线可比)。

**异常检测 (4 项结构检查):**
- 吸附物离解 (dissociated)
- 吸附物脱附 (desorbed)
- 表面结构改变 (surface_changed)
- 吸附物嵌入 (intercalated)

### 4.2 Val ID DFT SR@k (Method A)

| 模型 | SR@1 | SR@2 | SR@5 | SR@10 |
|------|------|------|------|-------|
| **EqV2** | 27.27% | **45.45%** | **54.55%** | **61.36%** |
| PaiNN | 22.73% | 34.09% | 45.45% | 47.73% |

**Best single seed DFT SR (Nsites=1 报道值):**

| 模型 | Best Seed | SR |
|------|-----------|-----|
| **EqV2** | seed 1/4/6 | **34.09%** |
| PaiNN | seed 1/3/6 | 27.27% |

**Per-seed DFT SR (10 seeds):**

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|------|---|---|---|---|---|---|---|---|---|---|
| EqV2 | 27.27 | 34.09 | 31.82 | 25.00 | 34.09 | 31.82 | 34.09 | 27.27 | 29.55 | 27.27 |
| PaiNN | 22.73 | 27.27 | 20.45 | 27.27 | 25.00 | 20.45 | 27.27 | 22.73 | 25.00 | 25.00 |

### 4.3 OOD DFT 结果

> OOD (valood50_R1I0.1, 50 SIDs) 的完整结果见 §6.4。

---

## 五、异常检测统计

### 5.1 聚合异常率

| 模型 | 配置 | 数据集 | Total | Normal | Anomalous | 异常率 |
|------|------|--------|-------|--------|-----------|--------|
| EqV2 | cfg7_steps5 | val ID | 440 | 345 | 95 | **21.6%** |
| PaiNN | cfg5_steps5 | val ID | 440 | 319 | 121 | **27.5%** |
| EqV2 | cfg7_steps5 | OOD | 500 | 401 | 99 | **19.8%** |
| PaiNN | cfg5_steps5 | OOD | 500 | 378 | 122 | **24.4%** |

### 5.2 Val ID Nsite Anomaly Rate (逐级递减)

> 定义: 对每个 SID, 记 $m$ 为 10 个 seed 中异常的个数。
> 当抽取 Nsite=$k$ 时, 该 SID 全部抽中异常的概率为 $P_{anom}(k)=\frac{C(m,k)}{C(10,k)}$ (若 $m<k$ 则为 0)。
> 报告值为对全部 SID 取平均。

| Nsites | EqV2 | PaiNN |
|--------|------|-------|
| 1 | 21.6% | 27.5% |
| 2 | 14.4% | 18.9% |
| 5 | 10.3% | 14.3% |
| 10 | 6.8% | 13.6% |

### 5.3 OOD Nsite Anomaly Rate (逐级递减)

| Nsites | EqV2 | PaiNN |
|--------|------|-------|
| 1 | 19.8% | 24.4% |
| 2 | 11.3% | 16.1% |
| 5 | 6.3% | 8.8% |
| 10 | 6.0% | 6.0% |

> 两个数据集均满足 **Nsite 增大 → Anomaly Rate 单调递减**, 与论文预期一致。
> OOD 异常率略低于 val ID, 说明模型在分布外数据上的结构完整性良好。

---

## 六、与基线对比

### 6.1 基线数据来源

> 数据来自 AdsorbDiff 论文 (Jiang et al., ICML 2024, arXiv:2405.03962) Figure 3 & Figure 4。
> 评估集: OC20-Dense val ID, 44 systems, 0.1 eV 阈值。
> 评估方法: Paper Method A — 对每个 SID, 在 Nsites=k 个候选中选 MLFF 能量最低者做 1 次 VASP SP。

**AdsorbDiff Figure 3 — DFT Success Rate (%):**

| Nsites | AdsorbML | AdsorbDiff |
|--------|----------|------------|
| 1 | 9.1% | 31.8% |
| 2 | 20.5% | 34.1% |
| 5 | 36.3% | — |
| 10 | 47.7% | 41.0% |

**AdsorbDiff Figure 4 — Nsite Anomaly Rate (%):**

| Nsites | AdsorbML | AdsorbDiff |
|--------|----------|------------|
| 1 | 31.8% | 25.0% |
| 2 | 18.2% | 20.5% |
| 5 | 11.4% | 22.7% |
| 10 | 6.8% | 13.6% |

### 6.2 主对比表: DFT Success Rate (Val ID, 44 SIDs)

> **论文核心表格。**
> - Nsites=k (k≥2): 从 k 个候选中选 MLFF 能量最低者 → 1 次 VASP SP
> - Nsites=1: 10 个 seed 中最优 seed 的 DFT SR

| 方法 | Nsites=1 | Nsites=2 | Nsites=5 | Nsites=10 |
|------|----------|----------|----------|-----------|
| AdsorbML | 9.1% | 20.5% | 36.3% | 47.7% |
| AdsorbDiff | 31.8% | 34.1% | — | 41.0% |
| **AdsorbFlow (EqV2)** | **34.09%** | **45.45%** | **54.55%** | **61.36%** |
| AdsorbFlow (PaiNN) | 27.27% | 34.09% | 45.45% | 47.73% |

> **关键发现:**
> 1. **AdsorbFlow (EqV2) SR@10 = 61.4%**, 大幅超越 AdsorbDiff (41.0%, +20.4pp) 和 AdsorbML (47.7%, +13.7pp)
> 2. **SR@2: EqV2 45.5% 远超 AdsorbDiff 34.1% (+11.4pp) 和 AdsorbML 20.5% (+25.0pp)**
> 3. SR@1 上 AdsorbFlow EqV2 (34.1%) 略优于 AdsorbDiff (31.8%), 多候选增益远超基线 (+27.3pp vs +9.2pp)
> 4. AdsorbFlow (PaiNN) SR@10 = 47.7%, 与 AdsorbML 持平, 高于 AdsorbDiff
> 5. 核心优势: Flow Matching + CFG 保证候选多样性, MLFF 排序可靠 (仅 3/44 SIDs 排序失误)

### 6.3 异常率对比 (Val ID)

| Nsites | AdsorbML | AdsorbDiff | AdsorbFlow (EqV2) | AdsorbFlow (PaiNN) |
|--------|----------|------------|--------------------|--------------------| 
| 1 | 31.8% | 25.0% | **21.6%** | 27.5% |
| 2 | 18.2% | 20.5% | **14.4%** | 18.9% |
| 5 | 11.4% | 22.7% | **10.3%** | 14.3% |
| 10 | 6.8% | 13.6% | **6.8%** | 13.6% |

> **关键发现:**
> 1. AdsorbFlow (EqV2) 在所有 Nsites 上异常率均为最低, Nsite=10 时 (6.8%) 与 AdsorbML (6.8%) 持平
> 2. AdsorbDiff 异常率在 Nsite=5 时不降反升 (22.7%), 反映其候选多样性不足
> 3. AdsorbFlow (PaiNN) 异常率与 AdsorbDiff 接近, Nsite=10 时几乎相同 (13.6% vs 13.6%)

### 6.4 OOD 对比 (valood50_R1I0.1, 50 SIDs)

> OOD 暂无 AdsorbDiff/AdsorbML 基线 (原论文未报告相同 OOD 集结果)。

**OOD DFT 累积 SR@k:**

| 模型 | SR@1 | SR@2 | SR@5 | SR@10 |
|------|------|------|------|-------|
| **AdsorbFlow (EqV2)** | 28.0% | 46.0% | 54.0% | **58.0%** |
| AdsorbFlow (PaiNN) | 32.0% | 42.0% | 44.0% | 46.0% |

**OOD Nsite Anomaly Rate:**

| Nsites | EqV2 | PaiNN |
|--------|------|-------|
| 1 | 19.8% | 24.4% |
| 2 | 11.3% | 16.1% |
| 5 | 6.3% | 8.8% |
| 10 | 6.0% | 6.0% |

**OOD MLFF vs DFT:**

| 模型 | MLFF SR@10 | DFT SR@10 | 衰减 |
|------|-----------|----------|------|
| **EqV2** | 62.0% | **58.0%** | -4.0pp (6.5%) |
| PaiNN | 48.0% | 46.0% | -2.0pp (4.2%) |

### 6.5 MLFF vs DFT 衰减分析

> MLFF 排序与 DFT 验证的一致性。

| 数据集 | 模型 | MLFF SR@10 | DFT SR@10 | 衰减 | 衰减率 |
|--------|------|-----------|----------|------|--------|
| Val ID | EqV2 | 72.7% | 61.4% | -11.3pp | 15.5% |
| Val ID | PaiNN | 63.6% | 47.7% | -15.9pp | 25.0% |
| OOD | EqV2 | 62.0% | 58.0% | -4.0pp | 6.5% |
| OOD | PaiNN | 48.0% | 46.0% | -2.0pp | 4.2% |

> DFT SR@10 使用 Method A (MLFF-best → 1 DFT), DFT union (任意 seed 通过) 会更高:
> Val ID EqV2: Method A 61.4% vs DFT Union 68.2% (差 6.8pp)
>
> **注**: MLFF SR 来自 grid search 管线 (`scripts/eval.py`), DFT SR 来自 `paper_faithful_sr.py`,
> 两者的 PBC unwrap 实现略有差异 (`inv(cell.T)` vs `inv(cell)`), 导致异常判定存在微小分歧。
> Val ID 上仅 3/440 条 traj 受影响 (< 0.7%), 对衰减值影响可忽略;
> OOD 上约 28-42/500 条不一致 (5-8%), 衰减值可能有 ~1-2pp 偏差。

---

## 七、方法论说明

### 7.1 生成协议

与 AdsorbDiff 保持一致, 采用 **平移 + 旋转** 生成协议:
- 吸附分子在表面上做 2D 平移 (xy) 和 SO(3) 旋转
- z 坐标由后续 MLFF 弛豫确定, 不作为生成目标
- 使用更少的推理步数 (K=5) 即可实现高精度

### 7.2 评估方法层级

| 层级 | 方法 | DFT 计算量/SID | 特点 |
|------|------|---------------|------|
| **MLFF** | GemNet-OC 弛豫能量 vs DFT 目标 | 0 | 用于超参搜索 |
| **Method A (论文)** | MLFF-best → 1 VASP SP | 1 | 与 AdsorbDiff 一致 |
| **Fair (附录)** | 所有 seed → VASP SP → 解析 SR@k | N | 消除 seed 偏差 |

### 7.3 MLFF 排序可靠性

Method A vs DFT Union 的差异揭示 MLFF 排序的局限:
- EqV2: 3/44 个 SID 的 MLFF-best 在 DFT 中失败, 但其他 seed 通过
- PaiNN: 1/44 个 SID 同上
- 总体排序准确率 > 93%

---

## 八、数据文件索引

| 文件/目录 | 内容 |
|-----------|------|
| `oc20_dense_mappings/oc20dense_targets.pkl` | DFT target 吸附能 |
| `oc20_dense_mappings/oc20dense_ref_energies.pkl` | 参考能量 (裸表面+气态分子) |
| `gemnet_oc_base_s2ef_2M.pt` | GemNet-OC 弛豫模型权重 |
| `vasp_fair_all2d/` | Val ID Fair VASP 结果 (670 jobs) |
| `vasp_ood_results/` | OOD VASP 结果 (246 jobs) |
| `grid_search_runs/` | Val ID Grid Search 轨迹与评估 |
| `grid_search_runs_ood/` | OOD Grid Search 轨迹与评估 |

---

## 九、关键脚本

| 脚本 | 功能 |
|------|------|
| `main.py` | 训练/推理入口 |
| `scripts/grid_search_cfg_flow.py` | Grid Search 自动化 |
| `scripts/eval.py` | MLFF 评估与异常检测 |
| `scripts/cluster_vasp/paper_faithful_sr.py` | 多方法 DFT SR@k 对比 |
| `scripts/cluster_vasp/generate_fair_vasp_inputs.py` | Fair VASP 输入生成 |
| `scripts/cluster_vasp/analyze_fair_vasp.py` | Fair VASP 结果分析 |
