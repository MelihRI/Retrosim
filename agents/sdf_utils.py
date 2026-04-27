"""
SDF Utilities & Solver Configuration
=====================================
Signed Distance Field generation from STL meshes and analytical hulls,
plus the SolverConfig dataclass for the 3D-FNO NS solver.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SolverConfig:
    """All configurable parameters for the 3D-FNO NS solver.

    Grid axes convention:
        D (dim=2) → z vertical   |  H (dim=3) → y transverse
        W (dim=4) → x streamwise (longest)
    """

    # --- Grid (low-res default for CPU debugging) ---
    grid_depth:  int = 32     # z
    grid_height: int = 16     # y
    grid_width:  int = 64     # x

    # --- Domain bounds (non-dim by hull length L) ---
    x_min: float = -0.5;  x_max: float = 2.0
    y_min: float = -0.5;  y_max: float = 0.5
    z_min: float = -0.5;  z_max: float = 0.3

    # --- FNO architecture ---
    fno_width:  int = 32
    fno_blocks: int = 4
    modes_d: int = 8;   modes_h: int = 6;   modes_w: int = 16
    in_channels: int = 6   # [SDF, x, y, z, Re, Fr]
    out_channels: int = 4  # [u, v, w, p]

    # --- Physics ---
    reynolds: float = 1e6
    froude:   float = 0.26
    inlet_velocity: float = 1.0

    # --- Boundary enforcement ---
    hard_bc: bool = True
    bc_sharpness: float = 50.0  # DEPRECATED: was α in sigmoid(α·SDF). Now using exact boolean mask (SDF > 0).

    # --- Loss weights ---
    lambda_continuity: float = 1.0
    lambda_momentum:   float = 1.0
    lambda_bc_noslip:  float = 10.0
    lambda_bc_inlet:   float = 5.0
    lambda_bc_outlet:  float = 1.0

    # --- Training ---
    epochs:        int = 500
    learning_rate: float = 1e-3
    weight_decay:  float = 1e-5
    batch_size:    int = 1
    use_amp:       bool = False

    @property
    def grid_dims(self) -> Tuple[int, int, int]:
        return (self.grid_depth, self.grid_height, self.grid_width)

    @property
    def domain_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {'x': (self.x_min, self.x_max),
                'y': (self.y_min, self.y_max),
                'z': (self.z_min, self.z_max)}


# ═══════════════════════════════════════════════════════════════════════════════
# SDF Generator
# ═══════════════════════════════════════════════════════════════════════════════

class SDFGenerator:
    """Generates Signed Distance Fields and assembles FNO input tensors.

    φ(x) > 0 → fluid  |  φ(x) = 0 → surface  |  φ(x) < 0 → solid
    """

    def __init__(self, config: SolverConfig):
        self.config = config
        self._build_grids()

    # ── Coordinate Grids ─────────────────────────────────────────────────

    def _build_grids(self):
        cfg = self.config
        D, H, W = cfg.grid_dims

        x = torch.linspace(cfg.x_min, cfg.x_max, W)
        y = torch.linspace(cfg.y_min, cfg.y_max, H)
        z = torch.linspace(cfg.z_min, cfg.z_max, D)

        # grid_z/y/x each have shape [D, H, W]
        self.grid_z, self.grid_y, self.grid_x = torch.meshgrid(
            z, y, x, indexing='ij')

        self.dx = (cfg.x_max - cfg.x_min) / max(W - 1, 1)
        self.dy = (cfg.y_max - cfg.y_min) / max(H - 1, 1)
        self.dz = (cfg.z_max - cfg.z_min) / max(D - 1, 1)

    # ── SDF from STL ─────────────────────────────────────────────────────

    def compute_sdf_from_stl(self, stl_path: str) -> torch.Tensor:
        """Compute SDF from a watertight STL mesh via trimesh.

        Mesh normalisation strategy (matches CFD convention):
          - Bow at x_domain * 0.2, stern at x_domain * 0.8
          - Centred in Y (beam axis) and Z (depth axis)
          - Cross-section scaled to fit inner 30% of the Y-Z domain
          - This leaves room for inlet (x < 0.2) and wake (x > 0.8)

        Returns: [1, 1, D, H, W] float32 tensor.
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError("pip install trimesh")

        print(f"[SDF] Loading: {os.path.basename(stl_path)}")
        mesh = trimesh.load(stl_path, force='mesh')

        if not mesh.is_watertight:
            print("[SDF] Mesh not watertight -- attempting repair")
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)

        cfg = self.config

        # ---- Step 1: Compute domain extents ----
        dx = cfg.x_max - cfg.x_min  # streamwise domain length
        dy = cfg.y_max - cfg.y_min  # transverse domain width
        dz = cfg.z_max - cfg.z_min  # vertical domain height

        # ---- Step 2: Compute target placement in DOMAIN coords ----
        # Hull occupies x=[0.2, 0.8] of domain (60% for hull, 40% for wake)
        hull_x_start = cfg.x_min + 0.2 * dx   # bow
        hull_x_end   = cfg.x_min + 0.8 * dx   # stern
        hull_x_len   = hull_x_end - hull_x_start

        # Centre of Y and Z domain
        hull_y_center = (cfg.y_min + cfg.y_max) * 0.5
        hull_z_center = (cfg.z_min + cfg.z_max) * 0.5

        # ---- Step 3: Scale mesh to fit ----
        mesh_ext = mesh.bounding_box.extents  # [Lx, Ly, Lz] of original mesh
        mesh_center = mesh.bounding_box.centroid

        # Primary scale: fit hull length into the 60% x-domain slot
        scale_x = hull_x_len / max(mesh_ext[0], 1e-6)
        # Cross-section: fit into inner 30% of Y and Z domains
        scale_y = (0.30 * dy) / max(mesh_ext[1], 1e-6)
        scale_z = (0.30 * dz) / max(mesh_ext[2], 1e-6)
        # Use UNIFORM scale to preserve aspect ratio (smallest of all 3)
        scale = min(scale_x, scale_y, scale_z)

        # Centre mesh at origin, then scale uniformly
        mesh.apply_translation(-mesh_center)
        mesh.apply_scale(scale)

        # Translate to target position in domain
        scaled_ext = mesh.bounding_box.extents
        target_center = np.array([
            hull_x_start + scaled_ext[0] * 0.5,  # bow flush with x=0.2
            hull_y_center,
            hull_z_center,
        ])
        mesh.apply_translation(target_center - mesh.bounding_box.centroid)

        print(f"[SDF] Hull placed: x=[{mesh.bounds[0][0]:.3f}, {mesh.bounds[1][0]:.3f}] "
              f"y=[{mesh.bounds[0][1]:.3f}, {mesh.bounds[1][1]:.3f}] "
              f"z=[{mesh.bounds[0][2]:.3f}, {mesh.bounds[1][2]:.3f}]")
        print(f"[SDF] Domain:      x=[{cfg.x_min}, {cfg.x_max}] "
              f"y=[{cfg.y_min}, {cfg.y_max}] z=[{cfg.z_min}, {cfg.z_max}]")

        # ---- Step 4: Query SDF at every grid point ----
        D, H, W = cfg.grid_dims
        pts = torch.stack([self.grid_x.flatten(),
                           self.grid_y.flatten(),
                           self.grid_z.flatten()], dim=-1).numpy()

        print(f"[SDF] Computing distances for {len(pts)} points ...")
        _, dists, _ = trimesh.proximity.closest_point(mesh, pts)
        inside = mesh.contains(pts)
        sdf_np = dists.copy()
        sdf_np[inside] *= -1.0

        sdf = torch.tensor(sdf_np, dtype=torch.float32).reshape(1, 1, D, H, W)
        n_solid = int((sdf < 0).sum().item())
        print(f"[SDF] Range [{sdf.min():.4f}, {sdf.max():.4f}] | "
              f"solid voxels: {n_solid} / {D*H*W} ({100*n_solid/(D*H*W):.1f}%)")
        return sdf

    # ── Analytical KCS-like Hull ─────────────────────────────────────────

    def generate_analytical_hull_sdf(self) -> torch.Tensor:
        """Approximate KCS hull as a tapered ellipsoid. Returns [1,1,D,H,W]."""
        L, half_B, draft = 1.0, 0.12, 0.06

        # Parametric position along hull [0,1]
        xi = (self.grid_x - 0.0) / L
        valid = (xi >= 0) & (xi <= 1)

        # Taper: sin(π·ξ) → full beam at midship, zero at bow/stern
        taper = torch.zeros_like(xi)
        taper[valid] = torch.sin(np.pi * xi[valid])

        a = half_B * taper                    # y semi-axis
        b = draft                              # z semi-axis (constant)
        z_off = self.grid_z + draft * 0.5      # shift so keel at z≈-draft/2

        ell = (self.grid_y / (a + 1e-8)) ** 2 + (z_off / (b + 1e-8)) ** 2
        min_ax = torch.minimum(a, torch.full_like(a, b))
        sdf = (torch.sqrt(ell + 1e-8) - 1.0) * (min_ax + 1e-8)

        # Outside hull length → positive (fluid)
        sdf[~valid] = torch.sqrt(
            self.grid_y[~valid]**2 + z_off[~valid]**2).clamp(min=0.01)

        sdf = sdf.unsqueeze(0).unsqueeze(0)
        print(f"[SDF] Analytical hull | solid voxels: {(sdf < 0).sum().item()}")
        return sdf

    # ── Assemble FNO Input ───────────────────────────────────────────────

    def build_fno_input(self, sdf: torch.Tensor,
                        reynolds: Optional[float] = None,
                        froude: Optional[float] = None) -> torch.Tensor:
        """Concatenate [SDF, x, y, z, Re, Fr] → [B, 6, D, H, W]."""
        cfg = self.config
        B = sdf.shape[0]
        D, H, W = cfg.grid_dims
        Re = reynolds or cfg.reynolds
        Fr = froude or cfg.froude

        # Normalise coords to [-1, 1]
        x_n = 2.0 * (self.grid_x - cfg.x_min) / (cfg.x_max - cfg.x_min) - 1.0
        y_n = 2.0 * (self.grid_y - cfg.y_min) / (cfg.y_max - cfg.y_min) - 1.0
        z_n = 2.0 * (self.grid_z - cfg.z_min) / (cfg.z_max - cfg.z_min) - 1.0

        def _expand(t):
            return t.unsqueeze(0).unsqueeze(0).expand(B, 1, D, H, W)

        re_ch = torch.full((B, 1, D, H, W), np.log10(Re) / 10.0)
        fr_ch = torch.full((B, 1, D, H, W), Fr)

        return torch.cat([sdf, _expand(x_n), _expand(y_n), _expand(z_n),
                          re_ch, fr_ch], dim=1)
