"""
Geometry Assembler — Hull + Appendage → SDF → FNO Input Pipeline
=================================================================

Production geometry pipeline that takes a parametric design vector,
optionally merges an appendage STL (rudder, propeller bracket, etc.),
computes a watertight SDF on a fixed 3-D grid, and assembles the
6-channel FNO input tensor in a single deterministic call.

Architecture position:
  UI / Optimizer  →  GeometryAssembler.build()  →  FNO3d_NS_Solver.forward()

Grid convention  (matches agents/sdf_utils.py / fno3d_network.py):
    dim-2  D = 64   z  (vertical)
    dim-3  H = 128  y  (transverse)
    dim-4  W = 64   x  (streamwise, longest domain axis)

SDF sign convention:
    φ(x) < 0  →  inside solid (hull + appendage)
    φ(x) = 0  →  surface
    φ(x) > 0  →  fluid domain

References:
    - Li et al. (2021) — FNO for Parametric PDEs, ICLR
    - MIT DeCoDE Lab — Ship-D parametric hull generator
"""

from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Fixed production grid resolution
# ---------------------------------------------------------------------------
GRID_D: int = 64    # z — vertical
GRID_H: int = 128   # y — transverse
GRID_W: int = 64    # x — streamwise

# ---------------------------------------------------------------------------
# Domain bounds (non-dimensionalised by hull length L)
#   Matches agents/sdf_utils.SolverConfig defaults used by the FNO solver.
# ---------------------------------------------------------------------------
X_MIN, X_MAX = -0.5, 2.0   # streamwise: enough room for inlet + wake
Y_MIN, Y_MAX = -0.5, 0.5   # transverse: beam centred at 0
Z_MIN, Z_MAX = -0.5, 0.3   # vertical:   keel-to-freeboard range


# ═══════════════════════════════════════════════════════════════════════════════
# Abstract Base Class
# ═══════════════════════════════════════════════════════════════════════════════

class GeometryAssembler(ABC):
    """Abstract interface for the hull geometry → FNO input pipeline.

    Every concrete assembler must implement :meth:`build`, which converts
    raw design parameters into the three artefacts consumed by the
    GC-FNO solver:

    * **combined_stl** — a ``numpy-stl`` Mesh object (hull ± appendage).
    * **sdf**          — ``float32`` array ``(1, D, H, W)``, negative
      inside the solid, positive in the fluid domain.
    * **input_tensor** — ``float32`` array ``(6, D, H, W)`` with channels
      ``[SDF, x, y, z, Re, Fr]``.
    """

    @abstractmethod
    def build(
        self,
        design_vector: np.ndarray,
        appendage_stl: Optional[Path],
        appendage_transform: Optional[np.ndarray],
        Re: float,
        Fr: float,
        operating_param: float,
    ) -> dict:
        """Assemble the complete geometry pipeline output.

        Parameters
        ----------
        design_vector : np.ndarray
            45-element Ship-D design vector (see ``FFDHullMorpher.DESIGN_VECTOR_KEYS``).
        appendage_stl : Path | None
            Optional path to an appendage STL (rudder, propeller bracket, …).
            If ``None`` the hull is used alone.
        appendage_transform : np.ndarray | None
            4×4 homogeneous transform applied to the appendage mesh before
            merging.  Ignored when *appendage_stl* is ``None``.
        Re : float
            Reynolds number (dimensional).  Stored as ``log10(Re)/10`` in
            the tensor to keep magnitudes ≈ O(1).
        Fr : float
            Froude number (dimensionless).
        operating_param : float
            Scalar operating condition (e.g. draft ratio, loading factor)
            available for subclass-specific logic.

        Returns
        -------
        dict
            ``{"combined_stl": stl.mesh.Mesh,
               "sdf": np.ndarray (1, D, H, W) float32,
               "input_tensor": np.ndarray (6, D, H, W) float32}``
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# Concrete Implementation: STL-Based Assembler
# ═══════════════════════════════════════════════════════════════════════════════

class STLGeometryAssembler(GeometryAssembler):
    """Production assembler that uses ``HullParameterization`` + ``trimesh``.

    Workflow inside :meth:`build`:
      1. Generate hull surface from the 45-D design vector.
      2. Convert ``(vertices, faces)`` → ``numpy-stl`` Mesh.
      3. Optionally load and transform an appendage STL, then merge.
      4. Compute SDF on the fixed ``(64, 128, 64)`` grid via ``trimesh``.
      5. Assemble the 6-channel FNO input tensor.
    """

    def __init__(
        self,
        n_stations: int = 100,
        n_waterlines: int = 60,
        hull_x_frac: Tuple[float, float] = (0.2, 0.8),
        yz_fill_frac: float = 0.30,
    ):
        """
        Parameters
        ----------
        n_stations : int
            Longitudinal resolution of the parametric hull mesh.
        n_waterlines : int
            Vertical resolution of the parametric hull mesh.
        hull_x_frac : (float, float)
            Fraction of the streamwise domain occupied by the hull.
            ``(0.2, 0.8)`` places the bow at 20 % and stern at 80 %,
            leaving 20 % for inlet development and 20 % for the wake.
        yz_fill_frac : float
            Fraction of the Y-Z domain cross-section the hull may fill
            (used to compute the uniform scaling factor).
        """
        self.n_stations = n_stations
        self.n_waterlines = n_waterlines
        self.hull_x_frac = hull_x_frac
        self.yz_fill_frac = yz_fill_frac

        # Pre-compute the fixed coordinate grids (NumPy, float32).
        self._x, self._y, self._z, self._grid_pts = self._build_grids()

    # ── Grid Construction ────────────────────────────────────────────────

    @staticmethod
    def _build_grids() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build 3-D coordinate grids and the flattened query-point array.

        Returns (grid_x, grid_y, grid_z, grid_pts) where each ``grid_*``
        has shape ``(D, H, W)`` and ``grid_pts`` is ``(D*H*W, 3)``.
        """
        xs = np.linspace(X_MIN, X_MAX, GRID_W, dtype=np.float32)
        ys = np.linspace(Y_MIN, Y_MAX, GRID_H, dtype=np.float32)
        zs = np.linspace(Z_MIN, Z_MAX, GRID_D, dtype=np.float32)

        # indexing='ij' → shape (D, H, W) matching z, y, x order
        grid_z, grid_y, grid_x = np.meshgrid(zs, ys, xs, indexing='ij')

        # Flattened query points for trimesh: columns are (x, y, z)
        grid_pts = np.stack(
            [grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1
        ).astype(np.float64)

        return grid_x, grid_y, grid_z, grid_pts

    # ── Core Pipeline ────────────────────────────────────────────────────

    def build(
        self,
        design_vector: np.ndarray,
        appendage_stl: Optional[Path] = None,
        appendage_transform: Optional[np.ndarray] = None,
        Re: float = 1e6,
        Fr: float = 0.26,
        operating_param: float = 1.0,
    ) -> dict:
        """Run the full geometry → SDF → tensor pipeline.

        See :class:`GeometryAssembler` for the interface contract.
        """
        # --- Step 1: Generate hull mesh from the design vector -----------
        stl_mesh_obj = self._generate_hull_stl(design_vector)

        # --- Step 2: Optionally merge appendage --------------------------
        if appendage_stl is not None:
            stl_mesh_obj = self._merge_appendage(
                stl_mesh_obj, appendage_stl, appendage_transform,
            )

        # --- Step 3: Compute SDF on the fixed grid -----------------------
        sdf = self._compute_sdf(stl_mesh_obj)

        # --- Step 4: Assemble [SDF, x, y, z, Re, Fr] tensor -------------
        input_tensor = self._assemble_input_tensor(sdf, Re, Fr)

        return {
            "combined_stl": stl_mesh_obj,
            "sdf": sdf,
            "input_tensor": input_tensor,
        }

    # ── Step 1: Hull STL Generation ──────────────────────────────────────

    def _generate_hull_stl(self, design_vector: np.ndarray):
        """Convert the 45-D design vector into a ``numpy-stl`` Mesh.

        Uses ``HullParameterization`` from the existing geometry engine
        and wraps the ``(vertices, faces)`` output into a proper STL mesh.

        Returns
        -------
        stl.mesh.Mesh
            Closed, watertight hull mesh.
        """
        try:
            from stl import mesh as stl_mesh_module
        except ImportError:
            raise ImportError(
                "numpy-stl is required for mesh generation. "
                "Install with: pip install numpy-stl"
            )

        from core.geometry.FFDHullMorpher import HullParameterization

        # HullParameterization expects a dict keyed by DV name.
        # If the caller passed a raw ndarray, convert via the canonical key list.
        if isinstance(design_vector, np.ndarray):
            from core.geometry.FFDHullMorpher import DESIGN_VECTOR_KEYS
            if design_vector.shape[0] != len(DESIGN_VECTOR_KEYS):
                raise ValueError(
                    f"design_vector has {design_vector.shape[0]} elements, "
                    f"expected {len(DESIGN_VECTOR_KEYS)}."
                )
            dv_dict = dict(zip(DESIGN_VECTOR_KEYS, design_vector.tolist()))
        else:
            dv_dict = design_vector  # already a dict

        hull = HullParameterization(dv_dict)
        vertices, faces = hull.generate_mesh(
            n_stations=self.n_stations,
            n_waterlines=self.n_waterlines,
            include_bulb=True,
        )

        # Build numpy-stl Mesh from (vertices, faces)
        stl_obj = stl_mesh_module.Mesh(
            np.zeros(len(faces), dtype=stl_mesh_module.Mesh.dtype)
        )
        for i, face in enumerate(faces):
            for j in range(3):
                stl_obj.vectors[i][j] = vertices[face[j]]

        return stl_obj

    # ── Step 2: Appendage Merge ──────────────────────────────────────────

    @staticmethod
    def _merge_appendage(
        hull_stl,
        appendage_path: Path,
        transform: Optional[np.ndarray],
    ):
        """Load an appendage STL, optionally transform it, and merge with
        the hull into a single ``stl.mesh.Mesh`` object.

        Parameters
        ----------
        hull_stl : stl.mesh.Mesh
            Primary hull mesh.
        appendage_path : Path
            Path to the appendage STL file.
        transform : np.ndarray | None
            4×4 homogeneous transformation matrix.  Applied to the
            appendage *before* merging.

        Returns
        -------
        stl.mesh.Mesh
            Combined hull + appendage mesh.
        """
        from stl import mesh as stl_mesh_module

        appendage_path = Path(appendage_path)
        if not appendage_path.exists():
            warnings.warn(
                f"Appendage STL not found: {appendage_path}. "
                "Proceeding with hull only."
            )
            return hull_stl

        app_mesh = stl_mesh_module.Mesh.from_file(str(appendage_path))

        # Apply rigid-body transform if provided
        if transform is not None:
            if transform.shape != (4, 4):
                raise ValueError(
                    f"appendage_transform must be (4, 4), got {transform.shape}."
                )
            R = transform[:3, :3]
            t = transform[:3, 3]
            for i in range(len(app_mesh.vectors)):
                for j in range(3):
                    app_mesh.vectors[i][j] = R @ app_mesh.vectors[i][j] + t

        # Concatenate both meshes
        combined = stl_mesh_module.Mesh(
            np.concatenate([hull_stl.data, app_mesh.data])
        )
        return combined

    # ── Step 3: SDF Computation ──────────────────────────────────────────

    def _compute_sdf(self, stl_mesh_obj) -> np.ndarray:
        """Compute the Signed Distance Field on the fixed grid.

        Strategy (matches ``agents/sdf_utils.SDFGenerator``):
          1. Convert ``numpy-stl`` mesh → ``trimesh`` for SDF queries.
          2. Normalise the mesh into the domain using the hull placement
             fraction (bow at 20 %, stern at 80 % of x-domain).
          3. Query closest-point distances + inside/outside at every
             grid node.

        Returns
        -------
        np.ndarray
            Shape ``(1, D, H, W)``, dtype ``float32``.
            Negative inside the solid, positive in the fluid.
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError(
                "trimesh is required for SDF computation. "
                "Install with: pip install trimesh"
            )

        # Convert numpy-stl vectors → trimesh.Trimesh
        vectors = stl_mesh_obj.vectors  # (N_faces, 3, 3)
        n_faces = len(vectors)
        tri_verts = vectors.reshape(-1, 3)
        tri_faces = np.arange(n_faces * 3, dtype=np.int32).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=tri_verts, faces=tri_faces,
                               process=True)

        # Attempt watertight repair
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)

        # ---- Normalise mesh into the domain ----
        dx = X_MAX - X_MIN
        dy = Y_MAX - Y_MIN
        dz = Z_MAX - Z_MIN

        hull_x_start = X_MIN + self.hull_x_frac[0] * dx
        hull_x_end   = X_MIN + self.hull_x_frac[1] * dx
        hull_x_len   = hull_x_end - hull_x_start

        hull_y_center = (Y_MIN + Y_MAX) * 0.5
        hull_z_center = (Z_MIN + Z_MAX) * 0.5

        mesh_ext = mesh.bounding_box.extents
        mesh_center = mesh.bounding_box.centroid

        # Uniform scale: fit hull into the x-slot AND keep it inside
        # the inner yz_fill_frac of the cross-section.
        scale_x = hull_x_len / max(mesh_ext[0], 1e-6)
        scale_y = (self.yz_fill_frac * dy) / max(mesh_ext[1], 1e-6)
        scale_z = (self.yz_fill_frac * dz) / max(mesh_ext[2], 1e-6)
        scale = min(scale_x, scale_y, scale_z)

        mesh.apply_translation(-mesh_center)
        mesh.apply_scale(scale)

        # Place bow flush with hull_x_start
        scaled_ext = mesh.bounding_box.extents
        target_center = np.array([
            hull_x_start + scaled_ext[0] * 0.5,
            hull_y_center,
            hull_z_center,
        ])
        mesh.apply_translation(target_center - mesh.bounding_box.centroid)

        # ---- Query SDF ----
        _, dists, _ = trimesh.proximity.closest_point(mesh, self._grid_pts)
        inside = mesh.contains(self._grid_pts)

        sdf_flat = dists.copy().astype(np.float32)
        sdf_flat[inside] *= -1.0

        sdf = sdf_flat.reshape(1, GRID_D, GRID_H, GRID_W)

        n_solid = int((sdf < 0).sum())
        n_total = GRID_D * GRID_H * GRID_W
        print(
            f"[GeometryAssembler] SDF range [{sdf.min():.4f}, {sdf.max():.4f}] | "
            f"solid voxels: {n_solid}/{n_total} ({100 * n_solid / n_total:.1f}%)"
        )

        return sdf

    # ── Step 4: FNO Input Tensor Assembly ────────────────────────────────

    def _assemble_input_tensor(
        self, sdf: np.ndarray, Re: float, Fr: float,
    ) -> np.ndarray:
        """Build the 6-channel FNO input ``[SDF, x, y, z, Re, Fr]``.

        Coordinate channels are normalised to ``[-1, 1]``.
        Reynolds number is stored as ``log10(Re) / 10`` to keep all
        channels at comparable magnitudes.

        Parameters
        ----------
        sdf : np.ndarray  (1, D, H, W)
        Re  : float
        Fr  : float

        Returns
        -------
        np.ndarray  (6, D, H, W), float32
        """
        D, H, W = GRID_D, GRID_H, GRID_W

        # Normalised coordinates [-1, 1]
        x_norm = (2.0 * (self._x - X_MIN) / (X_MAX - X_MIN) - 1.0)
        y_norm = (2.0 * (self._y - Y_MIN) / (Y_MAX - Y_MIN) - 1.0)
        z_norm = (2.0 * (self._z - Z_MIN) / (Z_MAX - Z_MIN) - 1.0)

        # Scalar physics channels (broadcast to full grid)
        re_ch = np.full((D, H, W), np.log10(max(Re, 1.0)) / 10.0,
                        dtype=np.float32)
        fr_ch = np.full((D, H, W), Fr, dtype=np.float32)

        # Stack: [SDF(1,D,H,W) → (D,H,W), x, y, z, Re, Fr]
        tensor = np.stack(
            [sdf[0], x_norm, y_norm, z_norm, re_ch, fr_ch], axis=0,
        ).astype(np.float32)

        assert tensor.shape == (6, D, H, W), (
            f"Expected (6, {D}, {H}, {W}), got {tensor.shape}"
        )
        return tensor
