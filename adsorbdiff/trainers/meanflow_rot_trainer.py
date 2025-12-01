"""Rotation-only variant of the MeanFlow trainer."""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, Optional

import torch

from adsorbdiff.utils.registry import registry

from .meanflow_trainer import MeanFlowTrainer


@registry.register_trainer("meanflow_rot")
class MeanFlowRotationTrainer(MeanFlowTrainer):
    """Drops translation supervision while keeping the rest of MeanFlow logic."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        flow_cfg = self.config["optim"].get("flow", {})
        self.rot_loss_weight = float(flow_cfg.get("rot_loss_weight", 10.0))

    def _compute_loss(self, out, batch):
        v_rot = out.get("v_rot")
        if v_rot is None:
            raise KeyError("Rotation-only trainer expects 'v_rot' in the model output.")
        target_rot = getattr(batch, "v_rot_target", None)
        if target_rot is None:
            raise KeyError("Batch is missing 'v_rot_target', cannot compute rotation loss.")
        if target_rot.shape[-1] > v_rot.shape[-1]:
            target_rot = target_rot[:, : v_rot.shape[-1]]
        target_rot = target_rot.to(v_rot).clone()

        rot_debug = out.get("_debug_rot_projection") if isinstance(out, dict) else None
        debug_target = rot_debug.get("target_after") if isinstance(rot_debug, dict) else None
        target_is_aligned = False
        if torch.is_tensor(debug_target):
            target_rot = debug_target.to(v_rot)
            target_is_aligned = True

        if not target_is_aligned:
            rot_proj = getattr(batch, "rot_projector", None)
            if rot_proj is not None:
                if not torch.is_tensor(rot_proj):
                    rot_proj = torch.as_tensor(rot_proj)
                if rot_proj.device != v_rot.device:
                    rot_proj = rot_proj.to(v_rot.device)
                target_rot = torch.matmul(rot_proj, target_rot.unsqueeze(-1)).squeeze(-1)
                target_rot = torch.matmul(rot_proj, target_rot.unsqueeze(-1)).squeeze(-1)
                v_rot = torch.matmul(rot_proj, v_rot.unsqueeze(-1)).squeeze(-1)

            target_rot = self._align_discrete_symmetry_targets(v_rot, target_rot, batch)

        diff_rot = v_rot - target_rot

        combined_clip_mask = getattr(self, "_combined_clip_mask", None)
        if torch.is_tensor(combined_clip_mask):
            combined_mask = combined_clip_mask.to(v_rot.device)
            if combined_mask.shape[0] != v_rot.shape[0]:
                min_len = min(combined_mask.shape[0], v_rot.shape[0])
                combined_mask = combined_mask[:min_len]
                if min_len < v_rot.shape[0]:
                    pad = torch.zeros(v_rot.shape[0] - min_len, dtype=torch.bool, device=v_rot.device)
                    combined_mask = torch.cat([combined_mask, pad], dim=0)
            valid_mask = ~combined_mask.bool()
        else:
            valid_mask = torch.ones(v_rot.shape[0], dtype=torch.bool, device=v_rot.device)

        diff_rot_valid = diff_rot[valid_mask]
        if self.fm_endpoint_weight_exponent != 0.0:
            t_vals = batch.t.squeeze(-1).to(v_rot)
            weight = torch.clamp(1.0 - t_vals, min=1.0e-3)
            weight = weight[valid_mask]
            weight = weight.pow(-self.fm_endpoint_weight_exponent)
            if weight.numel() == 0:
                loss_rot = diff_rot_valid.new_zeros(())
            else:
                loss_rot = torch.mean(diff_rot_valid.pow(2) * weight.unsqueeze(-1))
        else:
            if diff_rot_valid.numel() == 0:
                loss_rot = diff_rot_valid.new_zeros(())
            else:
                loss_rot = torch.mean(diff_rot_valid.pow(2))

        weighted_loss_rot = loss_rot * self.rot_loss_weight

        try:
            loss_rot_val = float(weighted_loss_rot.detach().cpu())
        except Exception:
            loss_rot_val = float("nan")
        valid_samples = int(valid_mask.sum().item()) if torch.is_tensor(valid_mask) else 0
        self._last_component_losses = {
            "translation": float("nan"),
            "rotation": loss_rot_val,
            "valid_samples": valid_samples,
        }
        return weighted_loss_rot

    def _log_flow_sample_stats(self, batch, out, context: str) -> None:
        if batch is None or out is None:
            return
        v_rot = out.get("v_rot")
        if v_rot is None:
            logging.warning("[flow-sample][%s] Missing rotation output; skipping stats.", context)
            return
        tgt_rot_raw = batch.v_rot_target if hasattr(batch, "v_rot_target") else None
        t_vals = batch.t if hasattr(batch, "t") else None

        rot_debug = out.get("_debug_rot_projection") if isinstance(out, dict) else None
        v_rot_raw = rot_debug.get("pred_before") if isinstance(rot_debug, dict) else None
        if not torch.is_tensor(v_rot_raw):
            v_rot_raw = v_rot
        tgt_rot_raw = rot_debug.get("target_before", tgt_rot_raw) if isinstance(rot_debug, dict) else tgt_rot_raw
        if torch.is_tensor(tgt_rot_raw) and tgt_rot_raw.shape[-1] > v_rot_raw.shape[-1]:
            tgt_rot_raw = tgt_rot_raw[:, : v_rot_raw.shape[-1]]
        tgt_rot_proj = rot_debug.get("target_after") if isinstance(rot_debug, dict) else None
        projector = getattr(batch, "rot_projector", None)
        if tgt_rot_proj is None and torch.is_tensor(projector) and torch.is_tensor(tgt_rot_raw):
            try:
                proj = projector.to(tgt_rot_raw.device)
                if proj.ndim == 3 and proj.shape[-2:] == (3, 3) and proj.shape[0] == tgt_rot_raw.shape[0]:
                    tgt_rot_proj = torch.matmul(proj, tgt_rot_raw.unsqueeze(-1)).squeeze(-1)
            except Exception:
                tgt_rot_proj = None
        rot_pred_proj = v_rot

        def _stat_compare(out_tensor, tgt_tensor):
            if not (torch.is_tensor(out_tensor) and torch.is_tensor(tgt_tensor)):
                return "n/a", "n/a"
            out_data = out_tensor.detach().float().reshape(-1)
            tgt_data = tgt_tensor.detach().float().reshape(-1)
            mask_out = torch.isfinite(out_data)
            mask_tgt = torch.isfinite(tgt_data)
            if not (mask_out.any() and mask_tgt.any()):
                return "n/a", "n/a"
            out_vals = out_data[mask_out]
            tgt_vals = tgt_data[mask_tgt]
            if out_vals.numel() == 0 or tgt_vals.numel() == 0:
                return "n/a", "n/a"
            out_std = float(out_vals.std(unbiased=False).item()) if out_vals.numel() > 1 else 0.0
            tgt_std = float(tgt_vals.std(unbiased=False).item()) if tgt_vals.numel() > 1 else 0.0
            if tgt_std > 0.0:
                std_ratio = f"{(out_std / tgt_std):.4e}"
            else:
                std_ratio = "n/a"
            mean_diff = f"{float(out_vals.mean().item() - tgt_vals.mean().item()):.4e}"
            return std_ratio, mean_diff

        rot_std_ratio_raw, rot_mean_diff_raw = _stat_compare(v_rot_raw, tgt_rot_raw)
        rot_std_ratio_proj, rot_mean_diff_proj = _stat_compare(rot_pred_proj, tgt_rot_proj)

        logging.info(
            "[flow-sample][%s] t: %s | rot_raw target=%s | out=%s",
            context,
            self._tensor_summary(t_vals),
            self._tensor_summary(tgt_rot_raw),
            self._tensor_summary(v_rot_raw),
        )
        logging.info(
            "[flow-sample][%s] rot_proj target=%s | out=%s",
            context,
            self._tensor_summary(tgt_rot_proj),
            self._tensor_summary(rot_pred_proj),
        )
        logging.info(
            "[flow-sample][%s] rot_raw std_ratio=%s mean_diff=%s | rot_proj std_ratio=%s mean_diff=%s",
            context,
            rot_std_ratio_raw,
            rot_mean_diff_raw,
            rot_std_ratio_proj,
            rot_mean_diff_proj,
        )

        def _rot_alignment(pred: Optional[torch.Tensor], target: Optional[torch.Tensor]):
            if not (torch.is_tensor(pred) and torch.is_tensor(target)):
                return "n/a", "n/a"
            v_flat = pred.detach().float()
            tgt_flat = target.detach().float()
            tgt_norm = torch.linalg.norm(tgt_flat, dim=-1)
            v_norm = torch.linalg.norm(v_flat, dim=-1)
            valid = tgt_norm > 1.0e-6
            if not valid.any():
                return "n/a", "n/a"
            dot = (v_flat[valid] * tgt_flat[valid]).sum(dim=-1)
            denom = v_norm[valid] * tgt_norm[valid] + 1.0e-8
            cos_mean = f"{float(torch.clamp(dot / denom, -1.0, 1.0).mean().item()):.4f}"
            norm_ratio = v_norm[valid] / (tgt_norm[valid] + 1.0e-8)
            return cos_mean, f"{float(norm_ratio.mean().item()):.4f}"

        rot_cosine_raw, rot_norm_ratio_raw = _rot_alignment(v_rot_raw, tgt_rot_raw)
        rot_cosine_proj, rot_norm_ratio_proj = _rot_alignment(rot_pred_proj, tgt_rot_proj)

        logging.info(
            "[flow-sample][%s] rot_raw cos_mean=%s norm_ratio=%s | rot_proj cos_mean=%s norm_ratio=%s",
            context,
            rot_cosine_raw,
            rot_norm_ratio_raw,
            rot_cosine_proj,
            rot_norm_ratio_proj,
        )

        recent_losses = getattr(self, "_last_component_losses", None)
        if isinstance(recent_losses, dict):
            logging.info(
                "[flow-sample][%s] loss_rot=%.4e | valid=%d",
                context,
                recent_losses.get("rotation", float("nan")),
                int(recent_losses.get("valid_samples", 0)),
            )

        def _preview(name: str, pred: Optional[torch.Tensor], target: Optional[torch.Tensor]) -> None:
            if pred is None or target is None:
                logging.info("[flow-sample][%s] %s preview: n/a", context, name)
                return
            try:
                pred_cpu = pred.detach().float().cpu()
                tgt_cpu = target.detach().float().cpu()
            except Exception:
                logging.info("[flow-sample][%s] %s preview: detach failed", context, name)
                return
            if pred_cpu.ndim == 1:
                pred_cpu = pred_cpu.unsqueeze(-1)
            if tgt_cpu.ndim == 1:
                tgt_cpu = tgt_cpu.unsqueeze(-1)
            rows = min(4, pred_cpu.shape[0])
            pred_snippet = pred_cpu[:rows]
            tgt_snippet = tgt_cpu[:rows]
            pred_fmt = ["[" + ", ".join(f"{val:.3e}" for val in row.tolist()) + "]" for row in pred_snippet]
            tgt_fmt = ["[" + ", ".join(f"{val:.3e}" for val in row.tolist()) + "]" for row in tgt_snippet]
            logging.info(
                "[flow-sample][%s] %s pred: %s",
                context,
                name,
                " | ".join(pred_fmt),
            )
            logging.info(
                "[flow-sample][%s] %s label: %s",
                context,
                name,
                " | ".join(tgt_fmt),
            )

        _preview("rotation-raw", v_rot_raw, tgt_rot_raw)
        _preview("rotation-proj", rot_pred_proj, tgt_rot_proj)

        sym_labels = getattr(batch, "rot_symmetry", None)
        if isinstance(sym_labels, list) and sym_labels:
            counts = Counter(sym_labels)
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            logging.info("[flow-sample][%s] rot_symmetry_counts: %s", context, summary)
            eigvals = getattr(batch, "rot_symmetry_eigvals", None)
            if eigvals is not None:
                if torch.is_tensor(eigvals):
                    eig_source = eigvals.detach().cpu()
                elif isinstance(eigvals, (list, tuple)) and eigvals and torch.is_tensor(eigvals[0]):
                    eig_source = torch.stack([e.detach().cpu() for e in eigvals], dim=0)
                else:
                    eig_source = torch.as_tensor(eigvals)
                    eig_source = eig_source.detach().cpu() if torch.is_tensor(eig_source) else None
                if eig_source is not None and eig_source.shape[0] == len(sym_labels):
                    grouped = defaultdict(list)
                    for label, vec in zip(sym_labels, eig_source):
                        grouped[label].append(vec)
                    formatted = []
                    for label in sorted(grouped):
                        vec_strs = [
                            "[" + ", ".join(f"{float(val):.4e}" for val in vec.tolist()) + "]"
                            for vec in grouped[label]
                        ]
                        formatted.append(f"{label}: {', '.join(vec_strs)}")
                    if formatted:
                        logging.info(
                            "[flow-sample][%s] rot_symmetry_eigvals(norm): %s",
                            context,
                            " | ".join(formatted),
                        )

        base_model = getattr(self, "_unwrapped_model", None)
        if base_model is None:
            base_model = getattr(self.model, "model", self.model)
        energy_null = getattr(base_model, "energy_null", None)
        if torch.is_tensor(energy_null):
            with torch.no_grad():
                vals = energy_null.detach().float().reshape(-1)
                finite = torch.isfinite(vals)
                if finite.any():
                    eff = vals[finite]
                    norm = float(eff.norm().item())
                    mean = float(eff.mean().item())
                    std = float(eff.std(unbiased=False).item()) if eff.numel() > 1 else 0.0
                    emin = float(eff.min().item())
                    emax = float(eff.max().item())
                    logging.info(
                        "[flow-sample][%s] energy_null norm=%.4e mean=%.4e std=%.4e min=%.4e max=%.4e",
                        context,
                        norm,
                        mean,
                        std,
                        emin,
                        emax,
                    )
                else:
                    logging.info("[flow-sample][%s] energy_null has no finite entries", context)

    def _compute_metrics(self, out, batch, evaluator, metrics: Dict = {}):
        v_rot = out.get("v_rot")
        if v_rot is None:
            raise KeyError("Rotation-only trainer expects 'v_rot' in the model output.")
        target_rot = getattr(batch, "v_rot_target", None)
        if target_rot is None:
            raise KeyError("Batch is missing 'v_rot_target', cannot compute metrics.")
        if target_rot.shape[-1] > v_rot.shape[-1]:
            target_rot = target_rot[:, : v_rot.shape[-1]]
        target_rot = target_rot.to(v_rot)

        rot_proj = getattr(batch, "rot_projector", None)
        if rot_proj is not None:
            if not torch.is_tensor(rot_proj):
                rot_proj = torch.as_tensor(rot_proj)
            if rot_proj.device != v_rot.device:
                rot_proj = rot_proj.to(v_rot.device)
            target_rot = torch.matmul(rot_proj, target_rot.unsqueeze(-1)).squeeze(-1)
            target_rot = torch.matmul(rot_proj, target_rot.unsqueeze(-1)).squeeze(-1)
            v_rot = torch.matmul(rot_proj, v_rot.unsqueeze(-1)).squeeze(-1)

        target_rot = self._align_discrete_symmetry_targets(v_rot, target_rot, batch)
        diff = v_rot - target_rot
        mse = torch.mean(diff.pow(2), dim=-1)
        stats = metrics.get("rot_mse")
        if stats is None:
            stats = {"metric": 0.0, "total": 0.0, "numel": 0.0}
            metrics["rot_mse"] = stats
        stats["total"] += float(mse.sum().item())
        stats["numel"] += float(mse.numel())
        if stats["numel"] > 0:
            stats["metric"] = stats["total"] / stats["numel"]
        else:
            stats["metric"] = 0.0
        return metrics
