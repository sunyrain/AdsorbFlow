#!/usr/bin/env python3
"""Batch GemNet-OC relaxation: load model ONCE, process multiple LMDB dirs.

This avoids the ~7 min model loading overhead per site in grid_search_cfg_flow.py.
Usage (with DDP):
    python -u -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 \
        scripts/batch_relax.py \
        --relax-config configs/relaxation/gemnet_oc/gemnet_relax.yml \
        --checkpoint configs/relaxation/gemnet_oc/gemnet_oc_base_s2ef_2M.pt \
        --tasks tasks.json

tasks.json format:
    [{"lmdb_src": "/path/to/final_struct_lmdb", "traj_dir": "/path/to/relaxations"}, ...]
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adsorbdiff.datasets import data_list_collater
from adsorbdiff.utils import distutils
from adsorbdiff.utils.flags import flags
from adsorbdiff.utils.registry import registry
from adsorbdiff.utils.utils import (
    build_config,
    setup_imports,
    setup_logging,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch GemNet-OC relaxation")
    parser.add_argument("--relax-config", type=str, required=True,
                        help="GemNet relaxation config YAML")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="GemNet-OC checkpoint file")
    parser.add_argument("--tasks", type=str, required=True,
                        help="JSON file with list of {lmdb_src, traj_dir} tasks")
    # DDP args injected by torch.distributed.launch (both formats)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    tasks_path = Path(args.tasks)
    with open(tasks_path) as f:
        tasks = json.load(f)

    if not tasks:
        logging.info("No tasks to process")
        return

    # Use the first task's lmdb_src as dummy relax dataset for initial loading
    first_lmdb = tasks[0]["lmdb_src"]

    # Build OCP config using the standard machinery
    # We create a fake args namespace for build_config
    config_path = str((REPO_ROOT / args.relax_config).resolve())
    checkpoint_path = str((REPO_ROOT / args.checkpoint).resolve())

    # Use flags parser to build proper args
    ocp_parser = flags.get_parser()
    ocp_args_list = [
        "--mode", "run-relaxations",
        "--config-yml", config_path,
        "--checkpoint", checkpoint_path,
        f"--task.relax_dataset.src={first_lmdb}",
        f"--task.relax_opt.traj_dir={tasks[0]['traj_dir']}",
        "--distributed",
        "--debug",
    ]
    ocp_args, override_args = ocp_parser.parse_known_args(ocp_args_list)

    config = build_config(ocp_args, override_args)

    # Setup distributed
    if ocp_args.distributed:
        distutils.setup(config)

    try:
        setup_imports(config)

        t0 = time.time()

        # Build trainer (loads model, datasets, optimizer etc.)
        trainer_cls = registry.get_trainer_class(config.get("trainer", "ocp"))
        trainer = trainer_cls(
            task=config.get("task", {}),
            model=config["model"],
            outputs=config.get("outputs", {}),
            dataset=config["dataset"],
            optimizer=config["optim"],
            loss_fns=config.get("loss_functions", {}),
            eval_metrics=config.get("evaluation_metrics", {}),
            identifier=config["identifier"],
            timestamp_id=config.get("timestamp_id", None),
            run_dir=config.get("run_dir", "./"),
            is_debug=config.get("is_debug", False),
            print_every=config.get("print_every", 10),
            seed=config.get("seed", 0),
            logger=config.get("logger", "tensorboard"),
            local_rank=config["local_rank"],
            amp=config.get("amp", False),
            cpu=config.get("cpu", False),
            slurm=config.get("slurm", {}),
            noddp=config.get("noddp", False),
            name="s2ef",
        )

        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)

        if distutils.is_master():
            logging.info(f"Model loaded in {time.time() - t0:.1f}s. Processing {len(tasks)} tasks.")

        # Process each task
        for task_idx, task in enumerate(tasks):
            lmdb_src = task["lmdb_src"]
            traj_dir = task["traj_dir"]

            if distutils.is_master():
                logging.info(f"[{task_idx+1}/{len(tasks)}] Relaxing: {lmdb_src} -> {traj_dir}")

            # Update config
            trainer.config["task"]["relax_dataset"] = {"src": lmdb_src}
            trainer.config["task"]["relax_opt"]["traj_dir"] = traj_dir

            # Ensure traj_dir exists
            Path(traj_dir).mkdir(parents=True, exist_ok=True)

            # Close previous relax dataset if any
            if hasattr(trainer, 'relax_dataset') and trainer.relax_dataset is not None:
                trainer.relax_dataset.close_db()

            # Rebuild relax dataset and loader
            relax_dataset = registry.get_dataset_class("lmdb")(
                {"src": lmdb_src}
            )
            eval_batch_size = trainer.config["optim"].get(
                "eval_batch_size", trainer.config["optim"]["batch_size"]
            )
            relax_sampler = trainer.get_sampler(
                relax_dataset, eval_batch_size, shuffle=False
            )
            relax_loader = trainer.get_dataloader(
                relax_dataset, relax_sampler
            )

            trainer.relax_dataset = relax_dataset
            trainer.relax_sampler = relax_sampler
            trainer.relax_loader = relax_loader

            # Run relaxation
            t1 = time.time()
            trainer.run_relaxations()
            if distutils.is_master():
                logging.info(f"  Done in {time.time() - t1:.1f}s")

        distutils.synchronize()
        if distutils.is_master():
            logging.info(f"All {len(tasks)} tasks completed in {time.time() - t0:.1f}s")
    finally:
        if ocp_args.distributed:
            distutils.cleanup()


if __name__ == "__main__":
    main()
