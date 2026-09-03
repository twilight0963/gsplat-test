from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class VoxelGuidedConfig:
    n_along_shortest: int = 80
    tau: float = 3.5
    gamma1: float = 1e-4
    gamma2: int = 2
    gamma3: float = 0.075
    decay_rate: float = 4.0
    warmup_iters: int = 500
    densify_interval: int = 100
    prune_interval: int = 100




    _base: int = 100_000


def _voxel_ijk(positions: torch.Tensor, origin: torch.Tensor, voxel_size: float) -> torch.Tensor:

    return torch.floor((positions - origin) / voxel_size).to(torch.int64)


def _flatten_ijk(ijk: torch.Tensor, base: int) -> torch.Tensor:
    off = base // 2
    i, j, k = (ijk[:, 0] + off), (ijk[:, 1] + off), (ijk[:, 2] + off)
    return i + j * base + k * base * base


class VoxelGuidedOptimizer:
    PARAM_NAMES = ("means", "colors", "scales", "quats", "opacities")

    def __init__(self, init_means: torch.Tensor, config: VoxelGuidedConfig, device: str):
        self.cfg = config
        self.device = device

        bbox_min = init_means.min(dim=0).values
        bbox_max = init_means.max(dim=0).values
        extent = (bbox_max - bbox_min).clamp_min(1e-6)
        self.voxel_size = float(extent.min().item()) / max(config.n_along_shortest, 1)
        self.origin = bbox_min.detach().clone()

        self.voxel_id = self._assign(init_means)
        n = init_means.shape[0]
        self.grad_accum = torch.zeros((n, 3), device=device)
        self.grad_count = torch.zeros(n, device=device)



    def _assign(self, positions: torch.Tensor) -> torch.Tensor:
        ijk = _voxel_ijk(positions, self.origin, self.voxel_size)
        return _flatten_ijk(ijk, self.cfg._base)

    def _voxel_center(self, ijk: torch.Tensor) -> torch.Tensor:
        return self.origin + (ijk.to(self.origin.dtype) + 0.5) * self.voxel_size



    def accumulate_step(self, means: torch.Tensor) -> None:
        if means.grad is not None:
            self.grad_accum += means.grad.detach()
            self.grad_count += 1

    def dampen_gradients(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        colors: torch.Tensor,
        quats: torch.Tensor,
        opacities: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            voxel_center = self._voxel_center(self._ijk_from_flat(self.voxel_id))
            drift = (means.detach() - voxel_center).norm(dim=-1)
            max_linear_scale = scales.detach().exp().max(dim=-1).values
            bound = self.cfg.tau * self.voxel_size
            excess = torch.clamp(torch.maximum(drift, max_linear_scale) - bound, min=0.0)
            decay = torch.exp(-self.cfg.decay_rate * excess / self.voxel_size)
            unconstrained = excess > 0

        for tensor in (means, scales, colors, quats, opacities):
            if tensor.grad is not None:
                view_shape = (-1,) + (1,) * (tensor.grad.dim() - 1)
                tensor.grad.mul_(decay.view(*view_shape))

        return unconstrained

    def _ijk_from_flat(self, flat_id: torch.Tensor) -> torch.Tensor:
        base = self.cfg._base
        off = base // 2
        k = torch.div(flat_id, base * base, rounding_mode="floor")
        rem = flat_id - k * base * base
        j = torch.div(rem, base, rounding_mode="floor")
        i = rem - j * base
        return torch.stack([i - off, j - off, k - off], dim=-1)



    def maybe_densify_and_prune(
        self,
        step: int,
        optimizer: torch.optim.Optimizer,
        means: torch.Tensor,
        colors: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        opacities: torch.Tensor,
    ):
        params = {"means": means, "colors": colors, "scales": scales, "quats": quats, "opacities": opacities}

        if step >= self.cfg.warmup_iters and step % self.cfg.densify_interval == 0:
            params = self._densify(optimizer, params)

        if step >= self.cfg.warmup_iters and step % self.cfg.prune_interval == 0:
            params = self._prune(optimizer, params)

        return tuple(params[name] for name in self.PARAM_NAMES)

    def _densify(self, optimizer, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        means = params["means"]
        with torch.no_grad():
            avg_grad = self.grad_accum / self.grad_count.clamp_min(1).unsqueeze(-1)
            avg_grad_norm = avg_grad.norm(dim=-1)

            ijk = self._ijk_from_flat(self.voxel_id)
            voxel_center = self._voxel_center(ijk)
            drift = (means.detach() - voxel_center).norm(dim=-1)
            max_linear_scale = params["scales"].detach().exp().max(dim=-1).values
            unconstrained = torch.maximum(drift, max_linear_scale) > self.cfg.tau * self.voxel_size

            candidates = unconstrained & (avg_grad_norm > self.cfg.gamma1)
            if candidates.any():
                direction = torch.sign(avg_grad[candidates])
                target_ijk = ijk[candidates] + direction.to(torch.int64)
                target_flat = _flatten_ijk(target_ijk, self.cfg._base)

                occupied = set(self.voxel_id.tolist())
                empty_mask = torch.tensor(
                    [int(v.item()) not in occupied for v in target_flat],
                    device=means.device, dtype=torch.bool,
                )
                grow_idx = candidates.nonzero(as_tuple=True)[0][empty_mask]
                grow_target_flat = target_flat[empty_mask]

            else:
                grow_idx = means.new_zeros((0,), dtype=torch.long)
                grow_target_flat = means.new_zeros((0,), dtype=torch.long)

        if grow_idx.numel() > 0:
            new_means = self._voxel_center(self._ijk_from_flat(grow_target_flat))
            clones = {
                "means": new_means,
                "colors": params["colors"][grow_idx].clone(),
                "scales": params["scales"][grow_idx].clone(),
                "quats": params["quats"][grow_idx].clone(),
                "opacities": params["opacities"][grow_idx].clone(),
            }
            params = _extend_optimizer(optimizer, params, clones, self.PARAM_NAMES)

            self.voxel_id = torch.cat([self.voxel_id, grow_target_flat])
            self.grad_accum = torch.cat([self.grad_accum, torch.zeros_like(self.grad_accum[grow_idx])])
            self.grad_count = torch.cat([self.grad_count, torch.zeros_like(self.grad_count[grow_idx])])




        self.grad_accum.zero_()
        self.grad_count.zero_()
        return params

    def _prune(self, optimizer, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            unique_ids, inverse = torch.unique(self.voxel_id, return_inverse=True)
            counts = torch.zeros_like(unique_ids, dtype=torch.float32).index_add_(
                0, inverse, torch.ones_like(inverse, dtype=torch.float32)
            )
            opac = params["opacities"].detach().sigmoid()
            opac_sum = torch.zeros_like(unique_ids, dtype=torch.float32).index_add_(0, inverse, opac)
            avg_opacity = opac_sum / counts.clamp_min(1)

            voxel_ok = (counts >= self.cfg.gamma2) & (avg_opacity >= self.cfg.gamma3)
            keep = voxel_ok[inverse]

        if keep.all():
            return params

        params = _mask_optimizer(optimizer, params, keep, self.PARAM_NAMES)
        self.voxel_id = self.voxel_id[keep]
        self.grad_accum = self.grad_accum[keep]
        self.grad_count = self.grad_count[keep]
        return params








def _find_group(optimizer, name):
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return group
    raise KeyError(
        f"optimizer has no param group named {name!r} - build it with "
        f"`{{'params': [...], 'lr': ..., 'name': {name!r}}}`"
    )


def _extend_optimizer(optimizer, params, additions, names):
    updated = {}
    for name in names:
        group = _find_group(optimizer, name)
        old_tensor = group["params"][0]
        addition = additions[name]

        stored = optimizer.state.get(old_tensor, None)
        new_tensor = torch.cat([old_tensor.detach(), addition], dim=0).requires_grad_(True)
        if stored is not None:
            stored["exp_avg"] = torch.cat([stored["exp_avg"], torch.zeros_like(addition)], dim=0)
            stored["exp_avg_sq"] = torch.cat([stored["exp_avg_sq"], torch.zeros_like(addition)], dim=0)
            del optimizer.state[old_tensor]
            optimizer.state[new_tensor] = stored

        group["params"][0] = new_tensor
        updated[name] = new_tensor
    return updated


def _mask_optimizer(optimizer, params, keep_mask, names):
    updated = {}
    for name in names:
        group = _find_group(optimizer, name)
        old_tensor = group["params"][0]

        stored = optimizer.state.get(old_tensor, None)
        new_tensor = old_tensor.detach()[keep_mask].requires_grad_(True)
        if stored is not None:
            stored["exp_avg"] = stored["exp_avg"][keep_mask]
            stored["exp_avg_sq"] = stored["exp_avg_sq"][keep_mask]
            del optimizer.state[old_tensor]
            optimizer.state[new_tensor] = stored

        group["params"][0] = new_tensor
        updated[name] = new_tensor
    return updated
