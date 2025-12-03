"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
from collections import deque
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import Batch

from adsorbdiff.utils.typing import assert_is_instance
from adsorbdiff.datasets.lmdb_dataset import data_list_collater

from .optimizers.lbfgs_torch import LBFGS, TorchCalc
from .diffusers.denoising_torch import Denoiser, DiffTorchCalc
from .diffusers.flow_torch import FlowTorch

def ml_relax(
    batch,
    model,
    steps: int,
    fmax: float,
    relax_opt,
    save_full_traj,
    device: str = "cuda:0",
    transform=None,
    early_stop_batch: bool = False,
):
    """
    Runs ML-based relaxations.
    Args:
        batch: object
        model: object
        steps: int
            Max number of steps in the structure relaxation.
        fmax: float
            Structure relaxation terminates when the max force
            of the system is no bigger than fmax.
        relax_opt: str
            Optimizer and corresponding parameters to be used for structure relaxations.
        save_full_traj: bool
            Whether to save out the full ASE trajectory. If False, only save out initial and final frames.
    """
    batches = deque([batch])
    relaxed_batches = []
    while batches:
        batch = batches.popleft()
        oom = False
        ids = batch.sid
        calc = TorchCalc(model, transform)

        # Run ML-based relaxation
        traj_dir = relax_opt.get("traj_dir", None)
        optimizer = LBFGS(
            batch,
            calc,
            maxstep=relax_opt.get("maxstep", 0.04),
            memory=relax_opt["memory"],
            damping=relax_opt.get("damping", 1.0),
            alpha=relax_opt.get("alpha", 70.0),
            device=device,
            save_full_traj=save_full_traj,
            traj_dir=Path(traj_dir) if traj_dir is not None else None,
            traj_names=ids,
            early_stop_batch=early_stop_batch,
            max_adsorbate_surface_dist=relax_opt.get(
                "max_adsorbate_surface_dist"
            ),
            ads_clip_scale=relax_opt.get("ads_clip_scale", 0.5),
            ads_clip_attempts=relax_opt.get("ads_clip_attempts", 3),
            ads_clip_margin=relax_opt.get("ads_clip_margin", 0.5),
        )

        e: Optional[RuntimeError] = None
        try:
            relaxed_batch = optimizer.run(fmax=fmax, steps=steps)
            relaxed_batches.append(relaxed_batch)
        except RuntimeError as err:
            e = err
            oom = True
            torch.cuda.empty_cache()

        if oom:
            # move OOM recovery code outside of except clause to allow tensors to be freed.
            data_list = batch.to_data_list()
            if len(data_list) == 1:
                raise assert_is_instance(e, RuntimeError)
            logging.info(
                f"Failed to relax batch with size: {len(data_list)}, splitting into two..."
            )
            mid = len(data_list) // 2
            batches.appendleft(data_list_collater(data_list[:mid]))
            batches.appendleft(data_list_collater(data_list[mid:]))

    relaxed_batch = Batch.from_data_list(relaxed_batches)
    return relaxed_batch


def ml_diffuse(
    batch,
    model,
    denoising_pos_params: dict,
    traj_dir,
    save_full_traj,
    device: str = "cuda:0",
    transform=None,
    early_stop_batch: bool = False,
    logger=None,
):
    """
    Runs ML-based relaxations.
    Args:
        batch: object
        model: object
        steps: int
            Max number of steps in the structure relaxation.
        fmax: float
            Structure relaxation terminates when the max force
            of the system is no bigger than fmax.
        relax_opt: str
            Optimizer and corresponding parameters to be used for structure relaxations.
        save_full_traj: bool
            Whether to save out the full ASE trajectory. If False, only save out initial and final frames.
    """
    batches = deque([batch])
    relaxed_batches = []
    while batches:
        batch = batches.popleft()
        oom = False
        ids = batch.sid
        calc = DiffTorchCalc(model, transform)

        # Run ML-based relaxation

        optimizer = Denoiser(
            batch,
            calc,
            device=device,
            save_full_traj=save_full_traj,
            traj_dir=Path(traj_dir) if traj_dir is not None else None,
            traj_names=ids,
            early_stop_batch=early_stop_batch,
            denoising_pos_params=denoising_pos_params,
            logger=logger,
        )

        e: Optional[RuntimeError] = None
        try:
            relaxed_batch = optimizer.run()
            relaxed_batches.append(relaxed_batch)
        except RuntimeError as err:
            e = err
            oom = True
            torch.cuda.empty_cache()

        if oom:
            # move OOM recovery code outside of except clause to allow tensors to be freed.
            data_list = batch.to_data_list()
            if len(data_list) == 1:
                raise assert_is_instance(e, RuntimeError)
            logging.info(
                f"Failed to relax batch with size: {len(data_list)}, splitting into two..."
            )
            mid = len(data_list) // 2
            batches.appendleft(data_list_collater(data_list[:mid]))
            batches.appendleft(data_list_collater(data_list[mid:]))

    relaxed_batch = Batch.from_data_list(relaxed_batches)
    return relaxed_batch

def ml_flow(
    batch,
    model,
    flow_opt: dict,
    traj_dir: Optional[str] = None,
    save_full_traj: bool = True,
    device: str = "cuda:0",
    transform=None,
    early_stop_batch: bool = False,
    logger=None,
):
    """
    Runs Flow-Matching based sampling (with optional classifier-free guidance) for adsorbate-only rigid DOFs.
    Args:
        batch: torch_geometric.data.Batch
        model: object
            外层模型/Trainer，需暴露 .model(cond, mode="fm") 接口（与你训练/推理一致）
        flow_opt: dict
            Flow 采样配置，例如：
                {
                    "num_steps": 300,
                    "cfg_scale": 5.0,
                    "allow_z": False,      # 是否允许 z 向平移
                    "write_every": 0,         # >0 则每隔 write_every 步写一帧
                    "traj_dir": "...",        # （可选）也可放在 flow_opt 内
                }
        traj_dir: str | None
            轨迹目录；若也在 flow_opt 里提供，优先使用函数参数（不为 None 则覆盖 flow_opt 内配置）
        save_full_traj: bool
            是否保存完整轨迹（False 则仅写末帧）
        device: str
        transform: callable | None
            保持签名一致；本函数不直接用 transform（图更新依赖模型内部）
        early_stop_batch: bool
            保持签名一致；Flow 采样当前不使用该选项
        logger: object | None
    """
    batches = deque([batch])
    relaxed_batches = []
    while batches:
        batch = batches.popleft()
        oom = False
        ids = batch.sid

        # 统一处理 traj_dir：函数参数优先
        _flow_opt = dict(flow_opt or {})
        if traj_dir is not None:
            _flow_opt["traj_dir"] = traj_dir
        # FlowTorch 的构造函数接收单独的 traj_dir/traj_names 参数
        ft_traj_dir = _flow_opt.pop("traj_dir", None)

        sampler = FlowTorch(
            batch=batch,
            model=model,  # 注意：这里传外层（含 .model 与 .model.sampling）
            flow_opt=_flow_opt,
            device=device,
            save_full_traj=save_full_traj,
            traj_dir=Path(ft_traj_dir) if ft_traj_dir is not None else None,
            traj_names=ids,
            logger=logger,
        )

        e: Optional[RuntimeError] = None
        try:
            relaxed_batch = sampler.run()
            relaxed_batches.append(relaxed_batch)
        except RuntimeError as err:
            e = err
            oom = True
            torch.cuda.empty_cache()

        if oom:
            # 与其它入口保持一致的 OOM 恢复策略
            data_list = batch.to_data_list()
            if len(data_list) == 1:
                raise assert_is_instance(e, RuntimeError)
            logging.info(
                f"Failed to relax batch with size: {len(data_list)}, splitting into two..."
            )
            mid = len(data_list) // 2
            batches.appendleft(data_list_collater(data_list[:mid]))
            batches.appendleft(data_list_collater(data_list[mid:]))

    relaxed_batch = Batch.from_data_list(relaxed_batches)
    return relaxed_batch
