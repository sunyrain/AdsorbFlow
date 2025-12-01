#!/usr/bin/env python3
"""Inspect energy-conditioning vectors stored in a checkpoint.

The script prints the learned ``energy_null`` vector together with the
energy-embedding output for a zero-energy input (E = 0). It then compares
norms and relative orientation (cosine similarity and angular deviation).

Example:
    python scripts/inspect_energy_vectors.py \
        --checkpoint checkpoints/2025-10-24-00-23-28-pretrainis2rs_sde_std0.1-10_so30.01-1.55_painn_new/checkpoint.pt
"""

import argparse
import math
from typing import Dict, Iterable, Tuple

import torch


def _load_state_dict(path: str, use_ema: bool) -> Dict[str, torch.Tensor]:
    """Load a (EMA) state dict from a checkpoint file."""
    checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        if isinstance(checkpoint, torch.nn.Module):
            return checkpoint.state_dict()
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)!r}")

    candidate_keys = []
    if use_ema:
        candidate_keys.extend([
            "ema_state_dict",
            "ema_state",
            "ema_model_state",
            "ema_model_state_dict",
        ])
    candidate_keys.extend([
        "state_dict",
        "model_state_dict",
        "model_state",
        "model",
    ])

    for key in candidate_keys:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint  # already a state dict

    raise ValueError(
        "Could not locate a state_dict inside the checkpoint. "
        "Consider specifying --ema if you need the EMA weights."
    )


def _find_tensor(state_dict: Dict[str, torch.Tensor], suffixes: Iterable[str]) -> Tuple[str, torch.Tensor]:
    """Return the first tensor whose key ends with one of the suffixes."""
    for suffix in suffixes:
        for key, tensor in state_dict.items():
            if key.endswith(suffix):
                return key, tensor.detach().cpu()
    suffix_str = ", ".join(suffixes)
    raise KeyError(f"Could not find any tensor ending with: {suffix_str}")


def _angle_from_cosine(cosine: float) -> float:
    """Convert cosine similarity to degrees with proper clamping."""
    cosine_clamped = max(min(cosine, 1.0), -1.0)
    return math.degrees(math.acos(cosine_clamped))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect energy-conditioning vectors.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint file.")
    parser.add_argument(
        "--ema",
        action="store_true",
        help="Use EMA weights if available.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Number of decimal places when printing scalars (default: 6).",
    )
    parser.add_argument(
        "--energy",
        type=float,
        nargs="+",
        default=[0.0],
        help="Energy values to probe (default: 0.0). You can pass multiple values.",
    )
    args = parser.parse_args()

    torch.set_printoptions(precision=args.precision, linewidth=160)

    state_dict = _load_state_dict(args.checkpoint, use_ema=args.ema)

    null_key, energy_null = _find_tensor(state_dict, ["energy_null"])
    emb_bias_key, emb_bias = _find_tensor(state_dict, ["energy_embedding.bias"])
    emb_weight_key, emb_weight = _find_tensor(state_dict, ["energy_embedding.weight"])

    if energy_null.ndim != 1:
        raise ValueError(f"Expected energy_null to be 1-D, got shape {tuple(energy_null.shape)}")
    if emb_bias.ndim != 1:
        raise ValueError(f"Expected energy_embedding.bias to be 1-D, got shape {tuple(emb_bias.shape)}")
    if energy_null.shape != emb_bias.shape:
        raise ValueError(
            "Vector shapes do not match: "
            f"energy_null {tuple(energy_null.shape)} vs. energy_embedding.bias {tuple(emb_bias.shape)}"
        )
    if emb_weight.ndim == 2 and emb_weight.shape[1] == 1:
        emb_weight_vec = emb_weight.view(-1)
    elif emb_weight.ndim == 1:
        emb_weight_vec = emb_weight
    else:
        raise ValueError(
            "Unexpected energy_embedding.weight shape: "
            f"{tuple(emb_weight.shape)}"
        )
    if emb_weight_vec.shape != energy_null.shape:
        raise ValueError(
            "Vector shapes do not match: "
            f"energy_null {tuple(energy_null.shape)} vs. energy_embedding.weight {tuple(emb_weight_vec.shape)}"
        )

    null_norm = energy_null.norm().item()

    print(f"Loaded state dict keys from: {args.checkpoint}")
    print(f"Using tensor '{null_key}' as energy_null")
    print(f"Using tensor '{emb_weight_key}' as energy_embedding.weight")
    print(f"Using tensor '{emb_bias_key}' as energy_embedding.bias")
    # print("\nenergy_null vector:")
    # print(energy_null)
    print("null_shape", energy_null.shape)
    print(f"  ||energy_null|| = {null_norm:.{args.precision}f}")

    for energy_value in args.energy:
        energy_embed = emb_weight_vec * energy_value + emb_bias
        embed_norm = energy_embed.norm().item()
        if null_norm > 0.0 and embed_norm > 0.0:
            cosine = torch.dot(energy_null, energy_embed).item() / (null_norm * embed_norm)
            angle_deg = _angle_from_cosine(cosine)
        else:
            cosine = float("nan")
            angle_deg = float("nan")
        diff_norm = (energy_null - energy_embed).norm().item()
        norm_ratio = embed_norm / null_norm if null_norm > 0 else float("inf")

        # print(f"\nenergy_embedding(E={energy_value}) vector:")
        # print(energy_embed)
        print("Metrics:")
        print("embed_shape", energy_embed.shape)
        print(f"  ||embedding(E={energy_value})|| = {embed_norm:.{args.precision}f}")
        print(f"  Norm ratio (embedding / null) = {norm_ratio:.{args.precision}f}")
        print(f"  Difference norm = {diff_norm:.{args.precision}f}")
        print(f"  Cosine similarity = {cosine:.{args.precision}f}")
        print(f"  Angle (degrees) = {angle_deg:.{args.precision}f}")


if __name__ == "__main__":
    main()
