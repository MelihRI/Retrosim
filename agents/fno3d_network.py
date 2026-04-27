"""
3D Fourier Neural Operator — Navier-Stokes PINN Solver
=======================================================
Geometry-conditioned via SDF.  Pure PyTorch, no third-party solvers.

Architecture:
  [SDF, x, y, z, Re, Fr]  →  Lift  →  N × SpectralConv3d blocks  →  Project  →  [u, v, w, p]
  Hard BC: output_vel *= (SDF > 0)  — exact boolean mask enforcing zero-velocity inside hull.

References:
  Li et al. (2021) — Fourier Neural Operator for Parametric PDEs, ICLR.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.sdf_utils import SolverConfig


# ─── Device ──────────────────────────────────────────────────────────────────
def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        if torch.backends.mps.is_available():
            return torch.device('mps')
    except AttributeError:
        pass
    return torch.device('cpu')

DEVICE = _get_device()


# ═══════════════════════════════════════════════════════════════════════════════
# SpectralConv3d
# ═══════════════════════════════════════════════════════════════════════════════

class SpectralConv3d(nn.Module):
    """3D Spectral Convolution via truncated Fourier modes.

    Performs:  iFFT³( R_θ · FFT³(x) )
    where R_θ is a learnable complex weight tensor for each of the
    4 quadrants in the (D,H) frequency plane (W uses rFFT half-spectrum).
    """

    def __init__(self, in_ch: int, out_ch: int,
                 modes_d: int, modes_h: int, modes_w: int):
        super().__init__()
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.modes_d = modes_d
        self.modes_h = modes_h
        self.modes_w = modes_w

        scale = 1.0 / (in_ch * out_ch)
        # 4 quadrants: (±D, ±H) × (+W half-spectrum)
        def _w():
            return nn.Parameter(scale * torch.rand(
                in_ch, out_ch, modes_d, modes_h, modes_w, 2))

        self.w1 = _w()  # +D +H
        self.w2 = _w()  # +D -H
        self.w3 = _w()  # -D +H
        self.w4 = _w()  # -D -H

    @staticmethod
    def _compmul3d(inp, weights):
        """Batched complex multiply: [B,Ci,d,h,w] × [Ci,Co,d,h,w] → [B,Co,d,h,w]"""
        return torch.einsum("bidhw,iodhw->bodhw", inp, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        md, mh, mw = self.modes_d, self.modes_h, self.modes_w

        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm='ortho')

        out_ft = torch.zeros(B, self.out_ch, D, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        w1 = torch.view_as_complex(self.w1)
        w2 = torch.view_as_complex(self.w2)
        w3 = torch.view_as_complex(self.w3)
        w4 = torch.view_as_complex(self.w4)

        # Quadrant 1: +D +H
        out_ft[:, :, :md, :mh, :mw] = self._compmul3d(
            x_ft[:, :, :md, :mh, :mw], w1)
        # Quadrant 2: +D -H
        out_ft[:, :, :md, -mh:, :mw] = self._compmul3d(
            x_ft[:, :, :md, -mh:, :mw], w2)
        # Quadrant 3: -D +H
        out_ft[:, :, -md:, :mh, :mw] = self._compmul3d(
            x_ft[:, :, -md:, :mh, :mw], w3)
        # Quadrant 4: -D -H
        out_ft[:, :, -md:, -mh:, :mw] = self._compmul3d(
            x_ft[:, :, -md:, -mh:, :mw], w4)

        return torch.fft.irfftn(out_ft, s=(D, H, W), norm='ortho')


# ═══════════════════════════════════════════════════════════════════════════════
# FNO Block (Residual)
# ═══════════════════════════════════════════════════════════════════════════════

class FNOBlock3d(nn.Module):
    """SpectralConv3d + 1×1×1 bypass + InstanceNorm + GELU."""

    def __init__(self, width: int, modes_d: int, modes_h: int, modes_w: int):
        super().__init__()
        self.spectral = SpectralConv3d(width, width, modes_d, modes_h, modes_w)
        self.bypass   = nn.Conv3d(width, width, 1)
        self.norm     = nn.InstanceNorm3d(width)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


# ═══════════════════════════════════════════════════════════════════════════════
# Main Network: FNO3d_NS_Solver
# ═══════════════════════════════════════════════════════════════════════════════

class FNO3d_NS_Solver(nn.Module):
    """Geometry-Conditioned 3D FNO for steady incompressible Navier-Stokes.

    Input:  [B, 6, D, H, W]  =  [SDF, x, y, z, Re, Fr]
    Output: [B, 4, D, H, W]  =  [u, v, w, p]

    Boundary Condition Strategy (hard enforcement):
      1. Hull wall:  velocity *= (SDF > 0).float()  — exact no-slip
         Boolean mask ensures EXACTLY zero velocity inside the hull,
         unlike the previous sigmoid(α·SDF) which only approximated it.
      2. Inlet (x_min): u = U_∞, v = w = 0
      3. Outlet (x_max): zero-gradient Neumann for velocity; p = 0 (gauge)
      4. Pressure is NOT masked — it may be nonzero inside the hull.
    """

    def __init__(self, config: SolverConfig):
        super().__init__()
        self.config = config
        w = config.fno_width

        # Lifting: 6 input channels → hidden width
        self.lift = nn.Conv3d(config.in_channels, w, 1)

        # Spectral blocks
        self.blocks = nn.ModuleList([
            FNOBlock3d(w, config.modes_d, config.modes_h, config.modes_w)
            for _ in range(config.fno_blocks)
        ])

        # Projection: hidden → 4 output channels [u, v, w, p]
        self.proj = nn.Sequential(
            nn.Conv3d(w, w, 1),
            nn.GELU(),
            nn.Conv3d(w, config.out_channels, 1),
        )

    def forward(self, x: torch.Tensor,
                sdf: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:   [B, 6, D, H, W]  full FNO input
            sdf: [B, 1, D, H, W]  SDF for hard BC mask (channel 0 of x)
        Returns:
            out: [B, 4, D, H, W]  = [u, v, w, p]
        """
        cfg = self.config

        # Extract SDF from input if not provided separately
        if sdf is None:
            sdf = x[:, 0:1, :, :, :]

        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        out = self.proj(h)

        # ── Hard BC 1: No-slip wall via exact boolean SDF mask ──────────
        # fluid_mask = 1 where SDF > 0 (fluid domain), 0 inside hull.
        # This replaces the previous sigmoid(α·SDF) soft mask.
        fluid_mask = (sdf > 0).float()         # [B, 1, D, H, W]
        out_vel = out[:, :3] * fluid_mask       # u, v, w → EXACTLY 0 inside hull
        out_p   = out[:, 3:]                    # p is NOT masked

        # ── Hard BC 2: Inlet (x_min = first W-slice) ───────────────────
        # Enforce u = U_∞, v = w = 0 at the domain inlet.
        inlet_velocity = cfg.inlet_velocity     # default 1.0 (non-dim)
        out_vel[:, 0:1, :, :, 0:1] = inlet_velocity   # u = U∞
        out_vel[:, 1:3, :, :, 0:1] = 0.0               # v = w = 0

        # ── Hard BC 3: Outlet (x_max = last W-slice) ───────────────────
        # Zero-gradient Neumann: copy from second-to-last slice.
        out_vel[:, :, :, :, -1:] = out_vel[:, :, :, :, -2:-1].detach()
        # Gauge pressure: p = 0 at outlet.
        out_p[:, :, :, :, -1:] = 0.0

        out = torch.cat([out_vel, out_p], dim=1)

        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Navier-Stokes PINN Loss (Finite Differences)
# ═══════════════════════════════════════════════════════════════════════════════

class NavierStokesPINNLoss(nn.Module):
    """Physics-informed loss enforcing steady incompressible NS on a 3D grid.

    Uses 2nd-order central finite differences (10× faster than autograd).

    Loss = λ_c·L_cont + λ_m·L_mom + λ_ns·L_noslip + λ_in·L_inlet + λ_out·L_outlet

    Grid convention: x=dim4(W), y=dim3(H), z=dim2(D)
    """

    def __init__(self, config: SolverConfig, dx: float, dy: float, dz: float):
        super().__init__()
        self.cfg = config
        self.dx = dx
        self.dy = dy
        self.dz = dz

    # ── Finite Difference Helpers ────────────────────────────────────────

    @staticmethod
    def _ddx(f, dx):
        """∂f/∂x  central diff along W (dim=-1). Returns [..., W-2]."""
        return (f[..., 2:] - f[..., :-2]) / (2.0 * dx)

    @staticmethod
    def _ddy(f, dy):
        """∂f/∂y  central diff along H (dim=-2). Returns [..., H-2, :]."""
        return (f[..., 2:, :] - f[..., :-2, :]) / (2.0 * dy)

    @staticmethod
    def _ddz(f, dz):
        """∂f/∂z  central diff along D (dim=-3). Returns [..., D-2, :, :]."""
        return (f[..., 2:, :, :] - f[..., :-2, :, :]) / (2.0 * dz)

    @staticmethod
    def _laplacian_x(f, dx):
        """∂²f/∂x² along W."""
        return (f[..., 2:] - 2*f[..., 1:-1] + f[..., :-2]) / (dx**2)

    @staticmethod
    def _laplacian_y(f, dy):
        """∂²f/∂y² along H."""
        return (f[..., 2:, :] - 2*f[..., 1:-1, :] + f[..., :-2, :]) / (dy**2)

    @staticmethod
    def _laplacian_z(f, dz):
        """∂²f/∂z² along D."""
        return (f[..., 2:, :, :] - 2*f[..., 1:-1, :, :] + f[..., :-2, :, :]) / (dz**2)

    def _trim_interior(self, f):
        """Trim 1 cell from each side in D, H, W to match derivative shapes."""
        return f[:, :, 1:-1, 1:-1, 1:-1]

    # ── Loss Components ──────────────────────────────────────────────────

    def forward(self, pred: torch.Tensor, sdf: torch.Tensor) -> dict:
        """
        Args:
            pred: [B, 4, D, H, W]  network output [u, v, w, p]
            sdf:  [B, 1, D, H, W]  signed distance field
        Returns:
            dict with 'total' and individual loss terms.
        """
        cfg = self.cfg
        dx, dy, dz = self.dx, self.dy, self.dz

        u = pred[:, 0:1]; v = pred[:, 1:2]; w = pred[:, 2:3]; p = pred[:, 3:4]

        # ── 1. Continuity: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0 ──────────────────
        # Compute on interior (trim 1 from each side)
        du_dx = self._ddx(u, dx)[:, :, 1:-1, 1:-1, :]
        dv_dy = self._ddy(v, dy)[:, :, 1:-1, :, 1:-1]
        dw_dz = self._ddz(w, dz)[:, :, :, 1:-1, 1:-1]

        # Align to common interior [B,1,D-2,H-2,W-2]
        D2 = min(du_dx.shape[2], dv_dy.shape[2], dw_dz.shape[2])
        H2 = min(du_dx.shape[3], dv_dy.shape[3], dw_dz.shape[3])
        W2 = min(du_dx.shape[4], dv_dy.shape[4], dw_dz.shape[4])

        div = (du_dx[:,:,:D2,:H2,:W2] + dv_dy[:,:,:D2,:H2,:W2]
               + dw_dz[:,:,:D2,:H2,:W2])

        # Weight by fluid mask (only penalise in fluid region)
        sdf_int = self._trim_interior(sdf)[:,:,:D2,:H2,:W2]
        fluid_mask = (sdf_int > 0).float()

        loss_cont = torch.mean((div * fluid_mask) ** 2)

        # ── 2. Momentum (NS residuals) ──────────────────────────────────
        # Interior versions of all fields
        u_i = self._trim_interior(u)[:,:,:D2,:H2,:W2]
        v_i = self._trim_interior(v)[:,:,:D2,:H2,:W2]
        w_i = self._trim_interior(w)[:,:,:D2,:H2,:W2]

        # First derivatives on interior
        du_dx_i = du_dx[:,:,:D2,:H2,:W2]
        du_dy = self._ddy(u, dy)[:, :, 1:-1, :, 1:-1][:,:,:D2,:H2,:W2]
        du_dz = self._ddz(u, dz)[:, :, :, 1:-1, 1:-1][:,:,:D2,:H2,:W2]

        dv_dx = self._ddx(v, dx)[:, :, 1:-1, 1:-1, :][:,:,:D2,:H2,:W2]
        dv_dy_i = dv_dy[:,:,:D2,:H2,:W2]
        dv_dz = self._ddz(v, dz)[:, :, :, 1:-1, 1:-1][:,:,:D2,:H2,:W2]

        dw_dx = self._ddx(w, dx)[:, :, 1:-1, 1:-1, :][:,:,:D2,:H2,:W2]
        dw_dy = self._ddy(w, dy)[:, :, 1:-1, :, 1:-1][:,:,:D2,:H2,:W2]
        dw_dz_i = dw_dz[:,:,:D2,:H2,:W2]

        dp_dx = self._ddx(p, dx)[:, :, 1:-1, 1:-1, :][:,:,:D2,:H2,:W2]
        dp_dy = self._ddy(p, dy)[:, :, 1:-1, :, 1:-1][:,:,:D2,:H2,:W2]
        dp_dz = self._ddz(p, dz)[:, :, :, 1:-1, 1:-1][:,:,:D2,:H2,:W2]

        # Laplacians
        lap_u_x = self._laplacian_x(u, dx)[:,:,1:-1,1:-1,:][:,:,:D2,:H2,:W2]
        lap_u_y = self._laplacian_y(u, dy)[:,:,1:-1,:,1:-1][:,:,:D2,:H2,:W2]
        lap_u_z = self._laplacian_z(u, dz)[:,:,:,1:-1,1:-1][:,:,:D2,:H2,:W2]

        lap_v_x = self._laplacian_x(v, dx)[:,:,1:-1,1:-1,:][:,:,:D2,:H2,:W2]
        lap_v_y = self._laplacian_y(v, dy)[:,:,1:-1,:,1:-1][:,:,:D2,:H2,:W2]
        lap_v_z = self._laplacian_z(v, dz)[:,:,:,1:-1,1:-1][:,:,:D2,:H2,:W2]

        lap_w_x = self._laplacian_x(w, dx)[:,:,1:-1,1:-1,:][:,:,:D2,:H2,:W2]
        lap_w_y = self._laplacian_y(w, dy)[:,:,1:-1,:,1:-1][:,:,:D2,:H2,:W2]
        lap_w_z = self._laplacian_z(w, dz)[:,:,:,1:-1,1:-1][:,:,:D2,:H2,:W2]

        inv_Re = 1.0 / cfg.reynolds

        # x-momentum: u·∂u/∂x + v·∂u/∂y + w·∂u/∂z + ∂p/∂x - (1/Re)∇²u = 0
        res_x = (u_i*du_dx_i + v_i*du_dy + w_i*du_dz
                 + dp_dx - inv_Re*(lap_u_x + lap_u_y + lap_u_z))
        res_y = (u_i*dv_dx + v_i*dv_dy_i + w_i*dv_dz
                 + dp_dy - inv_Re*(lap_v_x + lap_v_y + lap_v_z))
        res_z = (u_i*dw_dx + v_i*dw_dy + w_i*dw_dz_i
                 + dp_dz - inv_Re*(lap_w_x + lap_w_y + lap_w_z))

        loss_mom = torch.mean(
            (res_x * fluid_mask)**2 +
            (res_y * fluid_mask)**2 +
            (res_z * fluid_mask)**2)

        # ── 3. No-slip BC (soft — always computed for logging) ──────────
        solid_mask = (sdf <= 0).float()
        vel_sq = u**2 + v**2 + w**2
        loss_noslip = torch.mean(vel_sq * solid_mask)

        # ── 4. Inlet BC: u=U∞, v=w=0 at x_min (first W-slice) ─────────
        U_inf = cfg.inlet_velocity
        loss_inlet = (torch.mean((u[:,:,:,:, 0] - U_inf)**2)
                      + torch.mean(v[:,:,:,:, 0]**2)
                      + torch.mean(w[:,:,:,:, 0]**2))

        # ── 5. Outlet BC: zero-gradient ∂u/∂x=0 at x_max ──────────────
        loss_outlet = (torch.mean((u[:,:,:,:,-1] - u[:,:,:,:,-2])**2)
                       + torch.mean((v[:,:,:,:,-1] - v[:,:,:,:,-2])**2)
                       + torch.mean((w[:,:,:,:,-1] - w[:,:,:,:,-2])**2))

        # ── Total ───────────────────────────────────────────────────────
        total = (cfg.lambda_continuity * loss_cont
                 + cfg.lambda_momentum  * loss_mom
                 + cfg.lambda_bc_noslip * loss_noslip
                 + cfg.lambda_bc_inlet  * loss_inlet
                 + cfg.lambda_bc_outlet * loss_outlet)

        return {
            'total':      total,
            'continuity': loss_cont.item(),
            'momentum':   loss_mom.item(),
            'noslip':     loss_noslip.item(),
            'inlet':      loss_inlet.item(),
            'outlet':     loss_outlet.item(),
        }
