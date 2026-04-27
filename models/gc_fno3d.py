"""
Geometry-Conditioned 3D Fourier Neural Operator (GC-FNO3D)
==========================================================

Production model for steady incompressible Navier-Stokes prediction
conditioned on hull geometry via SDF.

Architecture (Li et al. 2021 + geometry conditioning):
  [SDF, x, y, z, Re, Fr]  →  Lift(1×1×1)  →  N × FNOBlock3d  →  Field Head  →  [u, v, w, p]
                                                                →  Scalar Head →  C_T

Boundary condition strategy (hard enforcement in forward pass):
  1. No-slip wall:  velocity *= (SDF > 0).float()   — exact boolean mask
  2. Inlet  (x_min): u = U∞, v = w = 0              — Dirichlet
  3. Outlet (x_max): ∂u/∂x = 0, p = 0               — Neumann + gauge

Grid convention:
  dim-2 (D=64)  →  z vertical
  dim-3 (H=128) →  y transverse
  dim-4 (W=64)  →  x streamwise

References:
  Li et al. (2021) — Fourier Neural Operator for Parametric PDEs, ICLR
  Guo et al. (2022) — Geometry-Conditioned Operator Learning

License: Apache 2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FNOConfig:
    """All hyperparameters for GC_FNO3D in one place.

    Separates model architecture from training / physics config
    (which lives in agents/sdf_utils.SolverConfig).
    """

    # --- Grid shape ---
    grid_d: int = 64     # z (vertical)
    grid_h: int = 128    # y (transverse)
    grid_w: int = 64     # x (streamwise)

    # --- FNO architecture ---
    in_channels:  int = 6    # [SDF, x, y, z, Re, Fr]
    out_channels: int = 4    # [u, v, w, p]  (field head)
    fno_width:    int = 32   # hidden channel width
    fno_blocks:   int = 4    # number of spectral residual blocks
    modes:        int = 8    # truncated Fourier modes per dimension

    # --- Scalar head ---
    scalar_outputs: int = 1  # C_T  (total resistance coefficient)

    # --- Hard BC ---
    inlet_velocity: float = 1.0   # non-dimensional U∞

    @property
    def grid_dims(self) -> Tuple[int, int, int]:
        return (self.grid_d, self.grid_h, self.grid_w)


# ═══════════════════════════════════════════════════════════════════════════════
# SpectralConv3d — Truncated Fourier multiplication
# ═══════════════════════════════════════════════════════════════════════════════

class SpectralConv3d(nn.Module):
    """3D spectral convolution via truncated Fourier modes.

    Implements:  y = iFFT³( R_θ · FFT³(x) )

    For a real-valued input of shape ``(B, C_in, D, H, W)``, the real FFT
    (``rfftn``) produces a half-spectrum of size ``W//2+1`` along the last
    axis.  The full-spectrum axes (D, H) have both positive and negative
    frequency components, giving **4 quadrants** of learnable weights:

        (+D, +H)   (+D, −H)   (−D, +H)   (−D, −H)

    Each quadrant stores ``(C_in, C_out, modes, modes, modes_w)`` complex
    weights, where ``modes_w = modes // 2 + 1`` to stay within the rFFT
    half-spectrum bound.

    Parameters
    ----------
    in_channels, out_channels : int
        Number of input / output feature channels.
    modes : int
        Number of low-frequency modes retained per **full-spectrum**
        dimension (D, H).  Along W the kept modes are
        ``modes_w = min(modes, W//2 + 1)``.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_ch  = in_channels
        self.out_ch = out_channels
        self.modes  = modes
        # W uses rFFT → half-spectrum; clamp to at most modes
        self.modes_w = modes // 2 + 1

        scale = 1.0 / (in_channels * out_channels)

        def _weight() -> nn.Parameter:
            """Complex weight stored as real tensor [..., 2]."""
            return nn.Parameter(
                scale * torch.randn(
                    in_channels, out_channels,
                    modes, modes, self.modes_w, 2,
                )
            )

        # 4 quadrants: (±D, ±H) × (+W half-spectrum)
        self.w_pp = _weight()   # +D, +H
        self.w_pn = _weight()   # +D, −H
        self.w_np = _weight()   # −D, +H
        self.w_nn = _weight()   # −D, −H

    # ── Complex einsum helper ────────────────────────────────────────────

    @staticmethod
    def _compmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Batched complex multiply:  [B,Ci,d,h,w] × [Ci,Co,d,h,w] → [B,Co,d,h,w]."""
        return torch.einsum("bidhw,iodhw->bodhw", a, b)

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  (B, C_in, D, H, W)  — real-valued spatial features.

        Returns
        -------
        Tensor  (B, C_out, D, H, W) — spectral-convolved output.
        """
        B, C, D, H, W = x.shape
        md = self.modes
        mw = self.modes_w

        # Forward FFT (real → complex half-spectrum along W)
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")

        # Allocate output spectrum (zeros → un-touched modes stay zero)
        out_ft = torch.zeros(
            B, self.out_ch, D, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        # Convert stored real-pairs → complex view
        w_pp = torch.view_as_complex(self.w_pp)
        w_pn = torch.view_as_complex(self.w_pn)
        w_np = torch.view_as_complex(self.w_np)
        w_nn = torch.view_as_complex(self.w_nn)

        # Quadrant 1: +D, +H
        out_ft[:, :, :md, :md, :mw] = self._compmul(
            x_ft[:, :, :md, :md, :mw], w_pp)
        # Quadrant 2: +D, −H
        out_ft[:, :, :md, -md:, :mw] = self._compmul(
            x_ft[:, :, :md, -md:, :mw], w_pn)
        # Quadrant 3: −D, +H
        out_ft[:, :, -md:, :md, :mw] = self._compmul(
            x_ft[:, :, -md:, :md, :mw], w_np)
        # Quadrant 4: −D, −H
        out_ft[:, :, -md:, -md:, :mw] = self._compmul(
            x_ft[:, :, -md:, -md:, :mw], w_nn)

        # Inverse FFT back to spatial domain
        return torch.fft.irfftn(out_ft, s=(D, H, W), norm="ortho")


# ═══════════════════════════════════════════════════════════════════════════════
# FNOBlock3d — Spectral residual block
# ═══════════════════════════════════════════════════════════════════════════════

class FNOBlock3d(nn.Module):
    """Single FNO residual block:  SpectralConv3d + 1×1×1 bypass + norm + GELU.

    The bypass (pointwise convolution) captures purely local features that
    the global Fourier path may miss, ensuring the block is at least as
    expressive as a standard 1×1×1 convolutional layer.
    """

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv3d(width, width, modes)
        self.bypass   = nn.Conv3d(width, width, kernel_size=1)
        self.norm     = nn.InstanceNorm3d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


# ═══════════════════════════════════════════════════════════════════════════════
# GC_FNO3D — Main Model
# ═══════════════════════════════════════════════════════════════════════════════

class GC_FNO3D(nn.Module):
    """Geometry-Conditioned 3D Fourier Neural Operator.

    Dual-head architecture:

    * **Field head** — predicts the full 3D flow field ``[u, v, w, p]``
      on the ``(D, H, W)`` grid with hard boundary conditions.
    * **Scalar head** — predicts an auxiliary drag coefficient ``C_T``
      by global-average-pooling the last hidden state and passing
      through an MLP.

    Input
    -----
    ``x``  : ``(B, 6, D, H, W)``  — ``[SDF, x, y, z, Re, Fr]``

    Output
    ------
    ``dict`` with keys:

    * ``"field"``  : ``(B, 4, D, H, W)``  — ``[u, v, w, p]`` with hard BCs
    * ``"C_T"``    : ``(B, 1)``           — total resistance coefficient
    """

    def __init__(
        self,
        modes: int = 8,
        width: int = 32,
        in_channels: int = 6,
        out_channels: int = 4,
        n_blocks: int = 4,
        scalar_outputs: int = 1,
        inlet_velocity: float = 1.0,
    ):
        super().__init__()
        self.width = width
        self.inlet_velocity = inlet_velocity

        # ── Lifting: 6 channels → hidden width ──────────────────────────
        self.lift = nn.Conv3d(in_channels, width, kernel_size=1)

        # ── Spectral backbone ───────────────────────────────────────────
        self.blocks = nn.ModuleList([
            FNOBlock3d(width, modes) for _ in range(n_blocks)
        ])

        # ── Field head: width → 4 output channels [u, v, w, p] ─────────
        self.field_head = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width, out_channels, kernel_size=1),
        )

        # ── Scalar head: GAP → MLP → C_T ───────────────────────────────
        # Global-Average-Pooling over (D, H, W) → (B, width)
        # then 2-layer MLP → scalar_outputs
        self.scalar_head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, scalar_outputs),
        )

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        sdf: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Parameters
        ----------
        x   : (B, 6, D, H, W)  — full FNO input
        sdf : (B, 1, D, H, W)  — SDF for hard BC mask.
               If ``None``, channel 0 of ``x`` is used.

        Returns
        -------
        dict
            ``{"field": (B,4,D,H,W), "C_T": (B,1)}``
        """
        # ── Extract SDF ─────────────────────────────────────────────────
        if sdf is None:
            sdf = x[:, 0:1, :, :, :]

        # ── Backbone ────────────────────────────────────────────────────
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)

        # ── Scalar head (before BC masking) ─────────────────────────────
        # GAP over spatial dims → (B, width)
        h_pooled = h.mean(dim=(-3, -2, -1))
        C_T = self.scalar_head(h_pooled)       # (B, 1)

        # ── Field head ──────────────────────────────────────────────────
        out = self.field_head(h)               # (B, 4, D, H, W)

        # ── Hard BC 1: No-slip wall — exact boolean SDF mask ────────────
        fluid_mask = (sdf > 0).float()         # (B, 1, D, H, W)
        out_vel = out[:, :3] * fluid_mask      # u, v, w → 0 inside hull
        out_p   = out[:, 3:]                   # pressure is NOT masked

        # ── Hard BC 2: Inlet (first W-slice = x_min) ───────────────────
        U_inf = self.inlet_velocity
        out_vel[:, 0:1, :, :, 0:1] = U_inf    # u = U∞
        out_vel[:, 1:3, :, :, 0:1] = 0.0      # v = w = 0

        # ── Hard BC 3: Outlet (last W-slice = x_max) ───────────────────
        # Zero-gradient Neumann: copy from second-to-last slice.
        out_vel[:, :, :, :, -1:] = out_vel[:, :, :, :, -2:-1].detach()
        # Gauge pressure: p = 0 at outlet.
        out_p[:, :, :, :, -1:] = 0.0

        field = torch.cat([out_vel, out_p], dim=1)   # (B, 4, D, H, W)

        return {"field": field, "C_T": C_T}

    # ── Convenience ──────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: FNOConfig) -> "GC_FNO3D":
        """Construct from an :class:`FNOConfig` dataclass."""
        return cls(
            modes=cfg.modes,
            width=cfg.fno_width,
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            n_blocks=cfg.fno_blocks,
            scalar_outputs=cfg.scalar_outputs,
            inlet_velocity=cfg.inlet_velocity,
        )

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════════
# Shape Smoke Test
# ═══════════════════════════════════════════════════════════════════════════════

def _smoke_test():
    """Quick forward-pass shape verification."""
    cfg = FNOConfig()
    D, H, W = cfg.grid_dims
    B = 2

    model = GC_FNO3D.from_config(cfg)
    print(f"[GC_FNO3D] Parameters: {model.count_parameters():,}")

    x = torch.randn(B, 6, D, H, W)
    out = model(x)

    assert out["field"].shape == (B, 4, D, H, W), \
        f"field shape mismatch: {out['field'].shape}"
    assert out["C_T"].shape == (B, 1), \
        f"C_T shape mismatch: {out['C_T'].shape}"

    # Verify hard BCs
    u_inlet = out["field"][:, 0, :, :, 0]
    assert torch.allclose(u_inlet, torch.full_like(u_inlet, cfg.inlet_velocity)), \
        "Inlet u ≠ U∞"
    assert torch.allclose(out["field"][:, 3, :, :, -1],
                          torch.zeros_like(out["field"][:, 3, :, :, -1])), \
        "Outlet p ≠ 0"

    # Backward pass
    loss = out["field"].sum() + out["C_T"].sum()
    loss.backward()
    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    print(f"[GC_FNO3D] Grad norm: {grad_norm:.4f}")
    print(f"[GC_FNO3D] field: {list(out['field'].shape)}  C_T: {list(out['C_T'].shape)}")
    print("[GC_FNO3D] OK — Smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
