# Archive — 归档的调试/早期实验

这些是调参失败、早期测试或未完成的实验 run，保留作为历史参考。

## eqv2_debug_runs/
EqV2 模型在 2026-01-13 的多次调参尝试，大部分为空目录或仅训练了几个 epoch。
最终成功的 run 在 `experiments/eqv2_3d_cfg0.20/`。

## painn_2d_early/
早期 2D PaiNN 模型（2025-12-17），仅训练到 epoch70，效果不佳。

## early_test_runs/
各种早期手动测试的 traj 目录：
- `test_z_0.3*` — 3D 模型早期测试
- `test_0.25` — p_cfg=0.25 快速测试
- `test_ck` — p_cfg=0.10 检查点测试
- `test_z_0*` — 2D 模型早期测试

**这些数据可以安全删除以节省磁盘空间。**
