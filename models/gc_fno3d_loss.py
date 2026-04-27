"""
GC-FNO3D Loss Functions — Physics-Informed + Data-Driven
=========================================================

Composite loss for training the :class:`GC_FNO3D` dual-head model.

Components
----------
1. **Data loss** (field + scalar):
   - MSE between predicted and ground-truth ``[u, v, w, p]`` fields
   - MSE between predicted and ground-truth ``C_T``

2. **Physics loss** (PINN residuals, no ground-truth needed):
   - Continuity:  ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
   - Momentum:    NS x/y/z residuals
   - All evaluated on the interior grid via 2nd-order central FD

3. **Boundary loss** (soft penalty, supplements the hard BCs):
   - No-slip:  ||vel||² inside hull  → 0
   - Inlet:    (u − U∞)² + v² + w²  → 0
   - Outlet:   zero-gradient penalty

Grid convention (matches models/gc_fno3d.py):
    x = dim-4 (W=64)   streamwise
    y = dim-3 (H=128)  transverse
    z = dim-2 (D=64)   vertical

Usage
-----
    from models.gc_fno3d_loss import GC_FNO3DLoss, LossConfig
    criterion = GC_FNO3DLoss(LossConfig())
    losses = criterion(pred_dict, gt_field, gt_ct, sdf)
    losses["total"].backward()

License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LossConfig:
    """Weights and physics parameters for the composite loss."""

    # --- Data loss weights ---
    lambda_field:  float = 1.0     # MSE on [u, v, w, p] fields
    lambda_ct:     float = 10.0    # MSE on scalar C_T (higher weight for importance)

    # --- Physics loss weights ---
    lambda_continuity: float = 1.0
    lambda_momentum:   float = 1.0

    # --- Boundary loss weights (soft penalty, supplements hard BCs) ---
    lambda_noslip:  float = 10.0
    lambda_inlet:   float = 5.0
    lambda_outlet:  float = 1.0

    # --- Physics parameters ---
    reynolds:       float = 1e6
    inlet_velocity: float = 1.0    # non-dimensional U∞

    # --- Grid spacing (non-dim by hull length L) ---
    #     Default domain: x ∈ [-0.5, 2.0], y ∈ [-0.5, 0.5], z ∈ [-0.5, 0.3]
    #     Grid: (D, H, W) = (64, 128, 64)
    dx: float = (2.0 - (-0.5)) / 63    # ≈ 0.03968
    dy: float = (0.5 - (-0.5)) / 127   # ≈ 0.00787
    dz: float = (0.3 - (-0.5)) / 63    # ≈ 0.01270


# ═══════════════════════════════════════════════════════════════════════════════
# Finite Difference Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ddx(f: torch.Tensor, dx: float) -> torch.Tensor:
    """∂f/∂x  — 2nd-order central difference along W (dim=-1).

    Returns shape ``(..., W-2)``; input must have W ≥ 3.
    """
    return (f[..., 2:] - f[..., :-2]) / (2.0 * dx)


def _ddy(f: torch.Tensor, dy: float) -> torch.Tensor:
    """∂f/∂y  — central diff along H (dim=-2).  Returns ``(..., H-2, :)``."""
    return (f[..., 2:, :] - f[..., :-2, :]) / (2.0 * dy)


def _ddz(f: torch.Tensor, dz: float) -> torch.Tensor:
    """∂f/∂z  — central diff along D (dim=-3).  Returns ``(..., D-2, :, :)``."""
    return (f[..., 2:, :, :] - f[..., :-2, :, :]) / (2.0 * dz)


def _lap_x(f: torch.Tensor, dx: float) -> torch.Tensor:
    """∂²f/∂x²  along W."""
    return (f[..., 2:] - 2.0 * f[..., 1:-1] + f[..., :-2]) / (dx ** 2)


def _lap_y(f: torch.Tensor, dy: float) -> torch.Tensor:
    """∂²f/∂y²  along H."""
    return (f[..., 2:, :] - 2.0 * f[..., 1:-1, :] + f[..., :-2, :]) / (dy ** 2)


def _lap_z(f: torch.Tensor, dz: float) -> torch.Tensor:
    """∂²f/∂z²  along D."""
    return (f[..., 2:, :, :] - 2.0 * f[..., 1:-1, :, :] + f[..., :-2, :, :]) / (dz ** 2)


def _trim_interior(f: torch.Tensor) -> torch.Tensor:
    """Trim 1 cell from each side in D, H, W → interior sub-volume."""
    return f[:, :, 1:-1, 1:-1, 1:-1]


# ═══════════════════════════════════════════════════════════════════════════════
# Composite Loss
# ═══════════════════════════════════════════════════════════════════════════════

class GC_FNO3DLoss(nn.Module):
    """Composite loss for training :class:`models.gc_fno3d.GC_FNO3D`.

    Supports three modes of operation:

    * **Supervised only** — provide ``gt_field`` and/or ``gt_ct``
    * **Physics only** (PINN) — provide ``sdf`` only, no ground-truth
    * **Hybrid** — both data and physics terms active

    Parameters
    ----------
    config : LossConfig
        Weights and physics parameters.

    Examples
    --------
    >>> criterion = GC_FNO3DLoss(LossConfig())
    >>> pred = model(x)   # {"field": (B,4,D,H,W), "C_T": (B,1)}
    >>> losses = criterion(pred, gt_field=gt, gt_ct=ct, sdf=sdf)
    >>> losses["total"].backward()
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.cfg = config

    # ── Main entry point ─────────────────────────────────────────────────

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt_field: Optional[torch.Tensor] = None,
        gt_ct: Optional[torch.Tensor] = None,
        sdf: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all active loss terms.

        Parameters
        ----------
        pred : dict
            Model output: ``{"field": (B,4,D,H,W), "C_T": (B,1)}``.
        gt_field : Tensor | None
            Ground-truth flow field ``(B, 4, D, H, W)``.
        gt_ct : Tensor | None
            Ground-truth ``C_T``  ``(B, 1)``.
        sdf : Tensor | None
            SDF ``(B, 1, D, H, W)`` for physics / BC losses.

        Returns
        -------
        dict
            ``"total"`` (differentiable) and itemised scalars for logging.
        """
        cfg = self.cfg
        field = pred["field"]       # (B, 4, D, H, W)
        ct    = pred["C_T"]         # (B, 1)

        total = torch.tensor(0.0, device=field.device, requires_grad=True)
        log: Dict[str, float] = {}

        # ── 1. Data losses ──────────────────────────────────────────────
        if gt_field is not None:
            loss_field = F.mse_loss(field, gt_field)
            total = total + cfg.lambda_field * loss_field
            log["data_field"] = loss_field.item()

        if gt_ct is not None:
            loss_ct = F.mse_loss(ct, gt_ct)
            total = total + cfg.lambda_ct * loss_ct
            log["data_ct"] = loss_ct.item()

        # ── 2. Physics losses (require SDF) ─────────────────────────────
        if sdf is not None:
            physics = self._physics_loss(field, sdf)
            total = total + cfg.lambda_continuity * physics["continuity"]
            total = total + cfg.lambda_momentum   * physics["momentum"]
            log["continuity"] = physics["continuity"].item()
            log["momentum"]   = physics["momentum"].item()

            # Boundary penalties
            bc = self._boundary_loss(field, sdf)
            total = total + cfg.lambda_noslip  * bc["noslip"]
            total = total + cfg.lambda_inlet   * bc["inlet"]
            total = total + cfg.lambda_outlet  * bc["outlet"]
            log["noslip"] = bc["noslip"].item()
            log["inlet"]  = bc["inlet"].item()
            log["outlet"] = bc["outlet"].item()

        return {"total": total, **log}

    # ── Physics: Continuity + Momentum ───────────────────────────────────

    def _physics_loss(
        self, field: torch.Tensor, sdf: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Steady incompressible Navier-Stokes residuals via central FD.

        Only evaluated in the **fluid interior** (SDF > 0, 1 cell inward).
        """
        dx, dy, dz = self.cfg.dx, self.cfg.dy, self.cfg.dz
        inv_Re = 1.0 / self.cfg.reynolds

        u = field[:, 0:1]
        v = field[:, 1:2]
        w = field[:, 2:3]
        p = field[:, 3:4]

        # ── Continuity: ∇·u = 0 ─────────────────────────────────────────
        du_dx = _ddx(u, dx)[:, :, 1:-1, 1:-1, :]
        dv_dy = _ddy(v, dy)[:, :, 1:-1, :, 1:-1]
        dw_dz = _ddz(w, dz)[:, :, :, 1:-1, 1:-1]

        D2 = min(du_dx.shape[2], dv_dy.shape[2], dw_dz.shape[2])
        H2 = min(du_dx.shape[3], dv_dy.shape[3], dw_dz.shape[3])
        W2 = min(du_dx.shape[4], dv_dy.shape[4], dw_dz.shape[4])

        div = (du_dx[:, :, :D2, :H2, :W2]
               + dv_dy[:, :, :D2, :H2, :W2]
               + dw_dz[:, :, :D2, :H2, :W2])

        sdf_int = _trim_interior(sdf)[:, :, :D2, :H2, :W2]
        fluid = (sdf_int > 0).float()

        loss_cont = torch.mean((div * fluid) ** 2)

        # ── Momentum: NS residuals ───────────────────────────────────────
        u_i = _trim_interior(u)[:, :, :D2, :H2, :W2]
        v_i = _trim_interior(v)[:, :, :D2, :H2, :W2]
        w_i = _trim_interior(w)[:, :, :D2, :H2, :W2]

        # First derivatives (trimmed to common interior)
        du_dx_i = du_dx[:, :, :D2, :H2, :W2]
        du_dy   = _ddy(u, dy)[:, :, 1:-1, :, 1:-1][:, :, :D2, :H2, :W2]
        du_dz   = _ddz(u, dz)[:, :, :, 1:-1, 1:-1][:, :, :D2, :H2, :W2]

        dv_dx   = _ddx(v, dx)[:, :, 1:-1, 1:-1, :][:, :, :D2, :H2, :W2]
        dv_dy_i = dv_dy[:, :, :D2, :H2, :W2]
        dv_dz   = _ddz(v, dz)[:, :, :, 1:-1, 1:-1][:, :, :D2, :H2, :W2]

        dw_dx   = _ddx(w, dx)[:, :, 1:-1, 1:-1, :][:, :, :D2, :H2, :W2]
        dw_dy   = _ddy(w, dy)[:, :, 1:-1, :, 1:-1][:, :, :D2, :H2, :W2]
        dw_dz_i = dw_dz[:, :, :D2, :H2, :W2]

        dp_dx = _ddx(p, dx)[:, :, 1:-1, 1:-1, :][:, :, :D2, :H2, :W2]
        dp_dy = _ddy(p, dy)[:, :, 1:-1, :, 1:-1][:, :, :D2, :H2, :W2]
        dp_dz = _ddz(p, dz)[:, :, :, 1:-1, 1:-1][:, :, :D2, :H2, :W2]

        # Laplacians
        lu_x = _lap_x(u, dx)[:, :, 1:-1, 1:-1, :][:, :, :D2, :H2, :W2]
        lu_y = _lap_y(u, dy)[:, :, 1:-1, :, 1:-1][:, :, :D2, :H2, :W2]
        lu_z = _lap_z(u, dz)[:, :, :, 1:-1, 1:-1][:, :, :D2, :H2, :W2]

        lv_x = _lap_x(v, dx)[:, :, 1:-1, 1:-1, :][:, :, :D2, :H2, :W2]
        lv_y = _lap_y(v, dy)[:, :, 1:-1, :, 1:-1][:, :, :D2, :H2, :W2]
        lv_z = _lap_z(v, dz)[:, :, :, 1:-1, 1:-1][:, :, :D2, :H2, :W2]

        lw_x = _lap_x(w, dx)[:, :, 1:-1, 1:-1, :][:, :, :D2, :H2, :W2]
        lw_y = _lap_y(w, dy)[:, :, 1:-1, :, 1:-1][:, :, :D2, :H2, :W2]
        lw_z = _lap_z(w, dz)[:, :, :, 1:-1, 1:-1][:, :, :D2, :H2, :W2]

        # u·∂u/∂x + v·∂u/∂y + w·∂u/∂z + ∂p/∂x − (1/Re)∇²u = 0
        res_x = (u_i * du_dx_i + v_i * du_dy + w_i * du_dz
                 + dp_dx - inv_Re * (lu_x + lu_y + lu_z))
        res_y = (u_i * dv_dx + v_i * dv_dy_i + w_i * dv_dz
                 + dp_dy - inv_Re * (lv_x + lv_y + lv_z))
        res_z = (u_i * dw_dx + v_i * dw_dy + w_i * dw_dz_i
                 + dp_dz - inv_Re * (lw_x + lw_y + lw_z))

        loss_mom = torch.mean(
            (res_x * fluid) ** 2
            + (res_y * fluid) ** 2
            + (res_z * fluid) ** 2
        )

        return {"continuity": loss_cont, "momentum": loss_mom}

    # ── Boundary penalties (soft) ────────────────────────────────────────

    def _boundary_loss(
        self, field: torch.Tensor, sdf: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Soft BC penalties that supplement the hard enforcement in the model."""
        u = field[:, 0:1]
        v = field[:, 1:2]
        w = field[:, 2:3]
        U_inf = self.cfg.inlet_velocity

        # No-slip: velocity must be zero inside the solid
        solid = (sdf <= 0).float()
        loss_noslip = torch.mean((u ** 2 + v ** 2 + w ** 2) * solid)

        # Inlet: u = U∞, v = w = 0  at first W-slice
        loss_inlet = (
            torch.mean((u[:, :, :, :, 0] - U_inf) ** 2)
            + torch.mean(v[:, :, :, :, 0] ** 2)
            + torch.mean(w[:, :, :, :, 0] ** 2)
        )

        # Outlet: zero-gradient ∂vel/∂x ≈ 0  at last W-slice
        loss_outlet = (
            torch.mean((u[:, :, :, :, -1] - u[:, :, :, :, -2]) ** 2)
            + torch.mean((v[:, :, :, :, -1] - v[:, :, :, :, -2]) ** 2)
            + torch.mean((w[:, :, :, :, -1] - w[:, :, :, :, -2]) ** 2)
        )

        return {"noslip": loss_noslip, "inlet": loss_inlet, "outlet": loss_outlet}


# ═══════════════════════════════════════════════════════════════════════════════
# Drag Coefficient Estimator (post-processing)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def estimate_drag_from_field(
    field: torch.Tensor,
    sdf: torch.Tensor,
    dx: float = LossConfig.dx,
    dy: float = LossConfig.dy,
    dz: float = LossConfig.dz,
) -> torch.Tensor:
    """Estimate drag coefficient ``C_D`` from the predicted pressure field.

    Integrates pressure force on the hull surface (SDF ≈ 0) in the
    streamwise direction using a simple surface-pressure approach:

        F_x = ∬_S  p · n_x  dA

    The hull surface is identified as cells adjacent to the SDF
    zero-crossing (|SDF| < threshold).

    Parameters
    ----------
    field : (B, 4, D, H, W)  — predicted [u, v, w, p]
    sdf   : (B, 1, D, H, W)  — signed distance field
    dx, dy, dz : float        — grid spacings

    Returns
    -------
    Tensor (B,)  — estimated drag coefficient per sample.
    """
    p = field[:, 3:4]    # pressure

    # Hull surface: cells where SDF crosses zero
    threshold = max(dx, dy, dz) * 1.5
    surface = (sdf.abs() < threshold).float()      # (B, 1, D, H, W)

    # Pressure gradient in x (streamwise) → proxy for n_x · p
    dp_dx = (p[..., 2:] - p[..., :-2]) / (2.0 * dx)
    surf_trimmed = surface[..., 1:-1]

    # Integrate
    cell_area = dy * dz
    F_x = (dp_dx * surf_trimmed).sum(dim=(-3, -2, -1)).squeeze(-1) * cell_area

    return F_x


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke Test
# ═══════════════════════════════════════════════════════════════════════════════

def _smoke_test():
    """Verify loss computation and backward pass."""
    from models.gc_fno3d import GC_FNO3D, FNOConfig

    cfg = FNOConfig()
    D, H, W = cfg.grid_dims
    B = 2

    model = GC_FNO3D.from_config(cfg)
    x   = torch.randn(B, 6, D, H, W)
    sdf = x[:, 0:1]

    pred = model(x, sdf)

    # Supervised + physics hybrid
    gt_field = torch.randn(B, 4, D, H, W)
    gt_ct    = torch.randn(B, 1)

    criterion = GC_FNO3DLoss(LossConfig())
    losses = criterion(pred, gt_field=gt_field, gt_ct=gt_ct, sdf=sdf)

    print(f"[GC_FNO3DLoss] Loss terms:")
    for k, v in losses.items():
        if k == "total":
            continue
        print(f"  {k:20s} = {v:.4e}" if isinstance(v, float) else f"  {k:20s} = {v.item():.4e}")
    total_val = losses['total'].item() if hasattr(losses['total'], 'item') else losses['total']
    print(f"  {'total':20s} = {total_val:.4e}")

    losses["total"].backward()
    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    print(f"[GC_FNO3DLoss] Grad norm: {grad_norm:.4f}")

    # Physics-only mode (no GT)
    model.zero_grad()
    pred2 = model(x, sdf)
    losses2 = criterion(pred2, sdf=sdf)
    losses2["total"].backward()
    print(f"[GC_FNO3DLoss] Physics-only total: {losses2['total'].item():.4e}")
    print("[GC_FNO3DLoss] OK — Smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
