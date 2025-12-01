"""Rotation-only variant of the PaiNN flow model.

This keeps the high-level architecture from ``painn_denoising`` but drops the
translation head from the FM outputs so it can be paired with a rotation-only
trainer.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from adsorbdiff.utils.registry import registry

from .painn_denoising import PaiNN as PaiNNDenoising


@registry.register_model("painn_rotation_only")
class PaiNNRotationOnly(PaiNNDenoising):
    """PaiNN wrapper that exposes only the rotational flow head."""

    def _forward_flow(
        self,
        data,
        forces: torch.Tensor,
        forces2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if forces2 is None:
            raise RuntimeError(
                "PaiNNRotationOnly requires 'so3_denoising=True' so that the rotation head is available."
            )
        base_outputs = super()._forward_flow(data, forces, forces2)
        if not isinstance(base_outputs, dict):
            return base_outputs

        outputs = dict(base_outputs)
        outputs.pop("v_tr", None)

        debug_stats = outputs.get("_debug_force_pre_scatter")
        if isinstance(debug_stats, dict):
            debug_stats = dict(debug_stats)
            debug_stats.pop("translation", None)
            if debug_stats:
                outputs["_debug_force_pre_scatter"] = debug_stats
            else:
                outputs.pop("_debug_force_pre_scatter", None)

        return outputs
