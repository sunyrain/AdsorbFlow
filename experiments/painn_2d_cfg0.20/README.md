# PaiNN 2D (z=0, fixed) — p_cfg=0.20

> 对应论文 Table 2 "AdsorbFlow (2D, fixed z)" 行 — **P0 优先级**

## 模型配置

| 参数 | 值 |
|------|-----|
| Backbone | PaiNN |
| 协议 | 2D (allow_z=False, tr_sigma_z_scale=0) |
| p_cfg | 0.20 |
| 学习率 | 1.5e-4 |
| 训练集 | train_split (1.2 GB) |
| 配置文件 | `configs/flow/painn_conditional_flow.yml` (命令行覆盖 allow_z, z_scale, p_cfg) |

## 训练日志

- 启动时间: 2026-02-11
- GPU: 2× RTX 4090 (DDP, 有效 batch=32)
- 预估: ~500 epochs, ~6-8h

## 训练命令

```bash
cd .
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

## 早期测试结果 (p_cfg=0.15, 仅供参考)
- `grid_search/early_epoch060/` — epoch60, posmae=0.9163
- `grid_search/early_epoch070/` — epoch70, posmae=1.0663

## 待办

- [x] 启动训练
- [ ] 监控 loss 收敛
- [ ] Grid Search w∈{0,1,3,5,7} × K∈{5,10,30}
- [ ] 提取 Anomaly Rate
