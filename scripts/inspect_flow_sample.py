#!/usr/bin/env python3
"""Inspect a single Flow Matching sample end-to-end.

Loads a trainer from a config, extracts a specific LMDB sample, and
traces each stage: interpolant construction, clipping, model outputs,
and masked loss computation. Useful for debugging exploding gradients
or understanding how clipping affects individual samples.

Example:
    python scripts/inspect_flow_sample.py \
        --config-yml configs/flow/painn_so3_flow.yml \
        --split train --index 42 --resamples 3 --checkpoint path/to/ckpt.pt
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from adsorbdiff.utils import distutils
from adsorbdiff.utils.registry import registry
from adsorbdiff.utils.utils import build_config, setup_imports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace Flow Matching preprocessing and loss for selected samples")
    parser.add_argument("--config-yml", required=True, help="Path to the training configuration YAML")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "relax"], help="Dataset split to sample from")
    parser.add_argument("--index", type=int, default=0, help="Index of the sample within the chosen split (default: 0)")
    parser.add_argument("--count", type=int, default=None, help="Inspect this many samples (default: 1 unless --sid/--fid provided)")
    parser.add_argument("--resamples", type=int, default=1, help="Number of stochastic resamples per sample (default: 1)")
    parser.add_argument("--identifier", default="inspect", help="Identifier used when constructing the trainer (default: inspect)")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path to load model weights")
    parser.add_argument("--device", default="cpu", help="Device for model execution, e.g. cpu or cuda:0 (default: cpu)")
    parser.add_argument("--num-workers", type=int, default=0, help="Override dataloader worker count (default: 0)")
    parser.add_argument("--batch-size", type=int, default=1, help="Override batch size when instantiating the trainer (default: 1)")
    parser.add_argument("--train-mode", action="store_true", help="Leave the model in training mode instead of forcing eval mode")
    parser.add_argument("--dump-json", default=None, help="Optional path to dump the collected report as JSON")
    parser.add_argument("--override", nargs="*", default=[], help="Optional config overrides (key=value) applied after the YAML")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)")
    parser.add_argument("--sid", nargs="*", default=None, help="One or more sample IDs (sid) to inspect instead of using --index")
    parser.add_argument("--fid", nargs="*", default=None, help="Optional facet IDs (fid) to filter samples when --sid is ambiguous")
    parser.add_argument(
        "--sid-fid",
        nargs="*",
        default=None,
        help="Explicit sid:fid pairs to inspect (takes precedence over --sid/--fid). Use formats like sid:fid or sid,fid",
    )
    return parser.parse_args()


def make_runner_args(cli: argparse.Namespace) -> Namespace:
    cpu_flag = not cli.device or cli.device.lower().startswith("cpu")
    return Namespace(
        mode="train",
        config_yml=Path(cli.config_yml),
        identifier=cli.identifier,
        debug=True,
        run_dir="./",
        print_every=10,
        seed=cli.seed,
        amp=False,
        checkpoint=cli.checkpoint,
        timestamp_id=None,
        submit=False,
        summit=False,
        logdir=Path("logs"),
        slurm_partition="debug",
        slurm_mem=8,
        slurm_timeout=1,
        num_gpus=1,
        distributed=False,
        cpu=cpu_flag,
        num_nodes=1,
        distributed_port=13356,
        distributed_backend="nccl",
        local_rank=0,
        no_ddp=True,
        gp_gpus=None,
    )


def to_python(obj: Any) -> Any:
    if torch.is_tensor(obj):
        tensor = obj.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if tensor.numel() == 0:
            return []
        if tensor.dim() == 0:
            return float(tensor.item())
        return tensor.tolist()
    if isinstance(obj, dict):
        return {key: to_python(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(value) for value in obj]
    return obj


def summarize_tags(batch) -> Dict[int, int]:
    tags = batch.tags
    if tags.device.type != "cpu":
        tags = tags.cpu()
    unique, counts = torch.unique(tags, return_counts=True)
    return {int(k.item()): int(v.item()) for k, v in zip(unique, counts)}


def normalize_identifier_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = to_python(value)
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            value = str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts: List[str] = []
        for elem in value:
            norm = normalize_identifier_value(elem)
            if norm is None:
                continue
            parts.append(norm)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return "_".join(parts)
    if isinstance(value, (int, float)):
        if isinstance(value, bool):
            return "1" if value else "0"
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def prepare_identifier_set(values: Optional[List[str]]) -> Set[str]:
    if not values:
        return set()
    result: Set[str] = set()
    for raw in values:
        norm = normalize_identifier_value(raw)
        if norm:
            result.add(norm)
    return result


def parse_sid_fid_pairs(values: Optional[List[str]]) -> List[Tuple[Optional[str], Optional[str]]]:
    pairs: List[Tuple[Optional[str], Optional[str]]] = []
    if not values:
        return pairs

    separators = (":", ",", "=", "/")
    for raw in values:
        if raw is None:
            continue
        token = str(raw).strip()
        if not token:
            continue
        sid_part = token
        fid_part: Optional[str] = None
        for sep in separators:
            if sep in token:
                sid_part, fid_part = token.split(sep, 1)
                break
        sid_norm = normalize_identifier_value(sid_part.strip()) if sid_part else None
        fid_norm = normalize_identifier_value(fid_part.strip()) if fid_part else None
        if sid_norm is None and fid_norm is None:
            continue
        pairs.append((sid_norm, fid_norm))
    return pairs


def sample_matches_identifiers(data, sid_set: Set[str], fid_set: Set[str]) -> bool:
    if not sid_set and not fid_set:
        return True
    sid_val = normalize_identifier_value(getattr(data, "sid", None))
    fid_val = normalize_identifier_value(getattr(data, "fid", None))
    if sid_set and sid_val not in sid_set:
        return False
    if fid_set and fid_val not in fid_set:
        return False
    return True


def find_matching_samples(dataset, sid_set: Set[str], fid_set: Set[str], limit: Optional[int]) -> List[Tuple[int, Any]]:
    matches: List[Tuple[int, Any]] = []
    dataset_len = len(dataset)
    for idx in range(dataset_len):
        data = dataset[idx]
        if sample_matches_identifiers(data, sid_set, fid_set):
            matches.append((idx, data))
            if limit is not None and len(matches) >= limit:
                break
    return matches


def select_dataset(trainer, split: str):
    if split == "train":
        dataset = getattr(trainer, "train_dataset", None)
    elif split == "val":
        dataset = getattr(trainer, "val_dataset", None)
    elif split == "test":
        dataset = getattr(trainer, "test_dataset", None)
    else:
        dataset = getattr(trainer, "relax_dataset", None)
    if dataset is None:
        raise RuntimeError(f"Dataset split '{split}' is not available in the trainer.")
    return dataset


def ensure_cpu(batch):
    if hasattr(batch, "to"):
        return batch.to("cpu")
    return batch


def inspect_pass(trainer, batch, resample_id: int) -> Tuple[Dict[str, Any], Any]:
    record: Dict[str, Any] = {"resample_id": resample_id}
    flow_debug = getattr(batch, "_flow_debug", None)
    if flow_debug is not None:
        record["flow_debug"] = to_python(flow_debug)
    else:
        record["flow_debug"] = None

    for attr in ["t", "tr_sched", "rot_sched", "tr_sched_deriv", "rot_sched_deriv", "v_tr_target", "v_rot_target"]:
        if hasattr(batch, attr):
            record[attr] = to_python(getattr(batch, attr))

    if hasattr(batch, "rot_symmetry"):
        record["rot_symmetry"] = to_python(getattr(batch, "rot_symmetry"))
    else:
        record["rot_symmetry"] = None

    trainer._last_output_was_clipped = False
    trainer._clip_masks = {}
    trainer._combined_clip_mask = None

    with torch.no_grad():
        batch_dev = batch.to(trainer.device)
        raw_out = trainer.model(batch_dev, mode="fm")
        sanitized = trainer._sanitize_outputs(raw_out)
        loss_total = trainer._compute_loss(sanitized, batch_dev)

    record["raw_output"] = to_python(raw_out)
    record["sanitized_output"] = to_python(sanitized)
    record["clip_masks"] = {key: to_python(mask) for key, mask in trainer._clip_masks.items()}
    record["combined_clip_mask"] = to_python(trainer._combined_clip_mask) if trainer._combined_clip_mask is not None else None
    record["last_output_was_clipped"] = bool(trainer._last_output_was_clipped)

    v_tr = sanitized.get("v_tr")
    v_rot_raw = sanitized.get("v_rot")
    target_tr = batch_dev.v_tr_target[:, : v_tr.shape[-1]].to(v_tr)
    target_rot = batch_dev.v_rot_target.to(v_rot_raw)

    v_rot_projected = v_rot_raw
    rot_proj = getattr(batch_dev, "rot_projector", None)
    if rot_proj is not None:
        rot_proj = rot_proj.to(v_rot_raw)
        target_rot = torch.matmul(rot_proj, target_rot.unsqueeze(-1)).squeeze(-1)
        v_rot_projected = torch.matmul(rot_proj, v_rot_raw.unsqueeze(-1)).squeeze(-1)

    diff_tr = v_tr - target_tr
    diff_rot = v_rot_projected - target_rot

    combined_mask = trainer._combined_clip_mask
    if combined_mask is not None:
        combined_mask = combined_mask.to(v_tr.device)
        if combined_mask.shape[0] != v_tr.shape[0]:
            min_len = min(combined_mask.shape[0], v_tr.shape[0])
            combined_mask = combined_mask[:min_len]
            if min_len < v_tr.shape[0]:
                pad = torch.zeros(v_tr.shape[0] - min_len, dtype=torch.bool, device=v_tr.device)
                combined_mask = torch.cat([combined_mask, pad], dim=0)
        valid_mask = ~combined_mask.bool()
    else:
        valid_mask = torch.ones(v_tr.shape[0], dtype=torch.bool, device=v_tr.device)

    record["valid_mask"] = to_python(valid_mask)
    record["valid_samples"] = int(valid_mask.sum().item())
    record["total_samples"] = int(valid_mask.numel())

    diff_tr_valid = diff_tr[valid_mask]
    diff_rot_valid = diff_rot[valid_mask]
    target_tr_valid = target_tr[valid_mask]
    target_rot_valid = target_rot[valid_mask]
    raw_tr_valid = raw_out["v_tr"][valid_mask]
    raw_rot_valid = raw_out["v_rot"][valid_mask]
    sanitized_tr_valid = v_tr[valid_mask]
    sanitized_rot_valid = v_rot_projected[valid_mask]

    record["translation"] = {
        "target": to_python(target_tr_valid),
        "raw": to_python(raw_tr_valid),
        "sanitized": to_python(sanitized_tr_valid),
        "diff": to_python(diff_tr_valid),
        "target_norm": float(target_tr_valid.norm(dim=-1).mean().item()) if target_tr_valid.numel() > 0 else 0.0,
        "sanitized_norm": float(sanitized_tr_valid.norm(dim=-1).mean().item()) if sanitized_tr_valid.numel() > 0 else 0.0,
    }
    record["rotation"] = {
        "target": to_python(target_rot_valid),
        "raw": to_python(raw_rot_valid),
    "sanitized": to_python(sanitized_rot_valid),
    "sanitized_pre_projection": to_python(v_rot_raw[valid_mask]) if rot_proj is not None else None,
        "diff": to_python(diff_rot_valid),
        "target_norm": float(target_rot_valid.norm(dim=-1).mean().item()) if target_rot_valid.numel() > 0 else 0.0,
        "sanitized_norm": float(sanitized_rot_valid.norm(dim=-1).mean().item()) if sanitized_rot_valid.numel() > 0 else 0.0,
    }

    weight = None
    loss_tr = diff_tr_valid.new_tensor(0.0)
    loss_rot = diff_rot_valid.new_tensor(0.0)
    if trainer.fm_endpoint_weight_exponent != 0.0:
        t_vals = batch_dev.t.squeeze(-1).to(v_tr)
        weight = torch.clamp(1.0 - t_vals, min=1.0e-3)
        weight = weight[valid_mask]
        weight = weight.pow(-trainer.fm_endpoint_weight_exponent)
        if weight.numel() > 0:
            loss_tr = torch.mean(diff_tr_valid.pow(2) * weight.unsqueeze(-1))
            loss_rot = torch.mean(diff_rot_valid.pow(2) * weight.unsqueeze(-1))
    else:
        if diff_tr_valid.numel() > 0:
            loss_tr = diff_tr_valid.pow(2).mean()
        if diff_rot_valid.numel() > 0:
            loss_rot = diff_rot_valid.pow(2).mean()

    record["weights"] = to_python(weight) if weight is not None else None
    record["loss"] = {
        "total": float(loss_total.detach().cpu().item()),
        "translation": float(loss_tr.detach().cpu().item()),
        "rotation": float(loss_rot.detach().cpu().item()),
    }

    batch_cpu = ensure_cpu(batch_dev)
    return record, batch_cpu


def print_report(sample: Dict[str, Any]) -> None:
    header = (
        f"Split={sample['split']} idx={sample['index']} | "
        f"natoms={sample['natoms']} ads_atoms={sample['adsorbate_atoms']} | "
        f"tags={sample['tag_summary']}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    if sample["identifiers"]:
        print(f"Identifiers: {sample['identifiers']}")
    if sample.get("requested_filter"):
        print(f"Requested filter: {sample['requested_filter']}")
    for res in sample["resamples"]:
        mask_info = res["valid_samples"], res["total_samples"]
        loss_info = res["loss"]
        clip = res["combined_clip_mask"]
        symmetry = res.get("rot_symmetry")
        print(
            f"  [resample {res['resample_id']}] valid={mask_info[0]}/{mask_info[1]} "
            f"loss_total={loss_info['total']:.3e} (tr={loss_info['translation']:.3e}, rot={loss_info['rotation']:.3e})"
        )
        if clip is not None:
            print(f"    combined_clip_mask: {clip}")
        if symmetry:
            print(f"    rot_symmetry: {symmetry}")
        if res["weights"] is not None:
            w = res["weights"]
            if isinstance(w, list) and w:
                w_min = min(w)
                w_max = max(w)
                print(f"    endpoint weights: min={w_min:.3e} max={w_max:.3e}")
        if res["flow_debug"]:
            clip_flags = res["flow_debug"].get("clip_flags", {}) if isinstance(res["flow_debug"], dict) else {}
            print(f"    clip_flags: {clip_flags}")
        tr = res["translation"]
        rot = res["rotation"]
        if tr["sanitized"]:
            print(f"    v_tr sanitized={tr['sanitized']} target={tr['target']} diff={tr['diff']}")
        if rot["sanitized"]:
            print(f"    v_rot sanitized={rot['sanitized']} target={rot['target']} diff={rot['diff']}")


def main() -> None:
    cli = parse_args()
    runner_args = make_runner_args(cli)
    config = build_config(runner_args, cli.override)
    config["is_debug"] = True
    config["identifier"] = cli.identifier
    config["run_dir"] = "./"
    config["seed"] = cli.seed
    config["cpu"] = runner_args.cpu
    config["local_rank"] = 0
    config["optim"]["num_workers"] = cli.num_workers
    config["optim"]["batch_size"] = max(1, cli.batch_size)
    config["optim"]["eval_batch_size"] = max(1, config["optim"].get("eval_batch_size", cli.batch_size))

    setup_imports(config)
    trainer_name = config.get("trainer", "ocp")
    if trainer_name in ["forces", "equiformerv2_forces"]:
        task_name = "s2ef"
    elif trainer_name in ["energy", "equiformerv2_energy"]:
        task_name = "is2re"
    else:
        task_name = "ocp"
    trainer_cls = registry.get_trainer_class(trainer_name)
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
        local_rank=config.get("local_rank", 0),
        amp=config.get("amp", False),
        cpu=config.get("cpu", False),
        slurm=config.get("slurm", {}),
        noddp=config.get("noddp", False),
        name=task_name,
    )

    if cli.checkpoint:
        ckpt_path = Path(cli.checkpoint).expanduser().resolve()
        trainer.load_checkpoint(checkpoint_path=ckpt_path.as_posix())

    if not cli.train_mode:
        trainer.model.eval()
    else:
        trainer.model.train()

    dataset = select_dataset(trainer, cli.split)
    collater = trainer.ocp_collater

    sid_fid_pairs = parse_sid_fid_pairs(cli.sid_fid)
    sid_set = prepare_identifier_set(cli.sid)
    fid_set = prepare_identifier_set(cli.fid)
    sample_entries: List[Dict[str, Any]] = []

    if sid_fid_pairs:
        for sid_val, fid_val in sid_fid_pairs:
            sid_subset = {sid_val} if sid_val else set()
            fid_subset = {fid_val} if fid_val else set()
            matches = find_matching_samples(dataset, sid_subset, fid_subset, limit=1)
            if not matches:
                label = {"sid": sid_val, "fid": fid_val}
                raise RuntimeError(f"No samples matched explicit pair filter {label}.")
            idx, data = matches[0]
            sample_entries.append({"idx": idx, "data": data, "filter": {"sid": sid_val, "fid": fid_val}})
    elif sid_set or fid_set:
        if cli.count is not None:
            limit = cli.count
        elif sid_set:
            limit = len(sid_set)
        elif fid_set:
            limit = len(fid_set)
        else:
            limit = None
        matches = find_matching_samples(dataset, sid_set, fid_set, limit)
        if not matches:
            raise RuntimeError("No samples matched the requested sid/fid filters.")
        for idx, data in matches:
            sample_entries.append({"idx": idx, "data": data, "filter": None})
    else:
        sequential_count = cli.count if cli.count is not None else 1
        start_idx = cli.index
        dataset_len = len(dataset)
        max_required = start_idx + sequential_count
        if max_required > dataset_len:
            raise IndexError(
                f"Requested indices [{start_idx}, {max_required}) exceed dataset length {dataset_len}."
            )
        for offset in range(sequential_count):
            sample_entries.append({"idx": start_idx + offset, "data": None, "filter": None})

    results: List[Dict[str, Any]] = []
    for sample_offset, entry in enumerate(sample_entries):
        idx = entry["idx"]
        preload_data = entry.get("data")
        requested_filter = entry.get("filter")
        data = preload_data if preload_data is not None else dataset[idx]
        batch = collater([data])
        if hasattr(batch, "pos_relaxed") and batch.pos_relaxed is not None:
            batch.pos = batch.pos_relaxed
        base_pos = trainer._clone_base_positions(batch)
        batch = trainer._build_interpolant(batch)

        sample_record: Dict[str, Any] = {
            "split": cli.split,
            "index": idx,
            "identifiers": {},
            "resamples": [],
            "tag_summary": summarize_tags(batch),
            "natoms": int(batch.natoms.sum().item()) if hasattr(batch, "natoms") else 0,
            "adsorbate_atoms": int((batch.tags == 2).sum().item()) if hasattr(batch, "tags") else 0,
        }
        if requested_filter:
            clean_filter = {k: v for k, v in requested_filter.items() if v is not None}
            if clean_filter:
                sample_record["requested_filter"] = clean_filter
        for key in ("sid", "fid", "eid", "data_id", "id"):
            if hasattr(batch, key):
                sample_record["identifiers"][key] = to_python(getattr(batch, key))

        for resample_id in range(cli.resamples):
            pass_record, batch = inspect_pass(trainer, batch, resample_id)
            sample_record["resamples"].append(pass_record)
            if resample_id + 1 < cli.resamples:
                batch = trainer._resample_interpolant(batch, base_pos)
        results.append(sample_record)
        print_report(sample_record)

    if cli.dump_json:
        dump_path = Path(cli.dump_json).expanduser().resolve()
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w", encoding="utf-8") as f:
            json.dump([to_python(rec) for rec in results], f, indent=2)
        print(f"Saved report to {dump_path}")

    # ensure DDP state cleaned up if it was initialized elsewhere
    if distutils.initialized():
        distutils.cleanup()


if __name__ == "__main__":
    main()
