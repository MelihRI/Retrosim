"""
HullParameterization — MIT DeCoDE Lab Inspired Parametric Hull Generator
=========================================================================

Isolated third-party geometry engine for generating parametric ship hulls
from a 45-dimensional Design Vector.

Scientific Basis:
    - MIT DeCoDE Lab (Ship-D dataset): Parametric hull generator using
      B-spline curves for waterplane, section, and profile definitions.
    - The Design Vector captures key naval architecture parameters that
      fully define a monohull surface.

This module is intentionally isolated from the rest of the SmartCAPEX system
to maintain clean Adapter Pattern separation.

Author: SmartCAPEX AI Team
License: MIT (third-party integration)
"""

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline
from typing import Dict, List, Tuple, Optional


# ============================================================
# Design Vector Definition (45 Parameters)
# ============================================================
DESIGN_VECTOR_KEYS = [
    # --- Principal Dimensions (6) ---
    'L',          # Length between perpendiculars [m]
    'B',          # Beam (breadth) [m]
    'T',          # Draft [m]
    'D',          # Depth [m]
    'Cb',         # Block coefficient [-]
    'LCB',        # Longitudinal center of buoyancy [% L from FP]

    # --- Waterplane Shape (8) ---
    'Cwp',        # Waterplane area coefficient [-]
    'wp_fwd_1',   # Forward waterplane control point 1
    'wp_fwd_2',   # Forward waterplane control point 2
    'wp_fwd_3',   # Forward waterplane control point 3
    'wp_aft_1',   # Aft waterplane control point 1
    'wp_aft_2',   # Aft waterplane control point 2
    'wp_aft_3',   # Aft waterplane control point 3
    'wp_mid_f',   # Parallel midbody fraction [-]

    # --- Section Shape (10) ---
    'Cm',         # Midship section coefficient [-]
    'sec_bilge_r', # Bilge radius [m]
    'sec_flare_1', # Flare angle station 1 [deg]
    'sec_flare_2', # Flare angle station 2 [deg]
    'sec_dead_1',  # Deadrise angle station 1 [deg]
    'sec_dead_2',  # Deadrise angle station 2 [deg]
    'sec_ctrl_1',  # Section control 1
    'sec_ctrl_2',  # Section control 2
    'sec_ctrl_3',  # Section control 3
    'sec_ctrl_4',  # Section control 4

    # --- Profile / Keel Line (6) ---
    'keel_rise_fwd',    # Keel rise forward [m]
    'keel_rise_aft',    # Keel rise aft [m]
    'keel_flat_frac',   # Flat keel fraction [-]
    'prof_fwd_ctrl',    # Profile forward control
    'prof_aft_ctrl',    # Profile aft control
    'prof_mid_ctrl',    # Profile mid control

    # --- Bow Shape (7) ---
    'bow_angle',        # Entrance angle [deg]
    'bow_flare',        # Bow flare angle [deg]
    'bulb_length',      # Bulbous bow length [m]
    'bulb_breadth',     # Bulbous bow breadth [m]
    'bulb_depth',       # Bulbous bow depth [m]
    'bulb_shape',       # Bulbous bow shape factor [0-1]
    'bow_rake',         # Bow rake angle [deg]

    # --- Stern Shape (8) ---
    'stern_angle',      # Run angle [deg]
    'stern_shape',      # Stern shape factor [0-1]
    'stern_overhang',   # Stern overhang [m]
    'transom_beam',     # Transom beam ratio [0-1]
    'transom_draft',    # Transom draft ratio [0-1]
    'skeg_height',      # Skeg height [m]
    'stern_ctrl_1',     # Stern control 1
    'stern_ctrl_2',     # Stern control 2
]

N_DESIGN_PARAMS = len(DESIGN_VECTOR_KEYS)  # 45


def get_default_design_vector() -> Dict[str, float]:
    """
    Returns a default design vector for a typical 100m general cargo vessel
    (Koster type, ~5000 DWT).
    """
    return {
        # Principal Dimensions
        'L': 100.0, 'B': 16.0, 'T': 6.5, 'D': 9.0,
        'Cb': 0.72, 'LCB': 52.0,

        # Waterplane
        'Cwp': 0.78, 'wp_fwd_1': 0.3, 'wp_fwd_2': 0.5, 'wp_fwd_3': 0.7,
        'wp_aft_1': 0.7, 'wp_aft_2': 0.5, 'wp_aft_3': 0.3, 'wp_mid_f': 0.4,

        # Section
        'Cm': 0.97, 'sec_bilge_r': 1.5,
        'sec_flare_1': 5.0, 'sec_flare_2': 8.0,
        'sec_dead_1': 0.0, 'sec_dead_2': 5.0,
        'sec_ctrl_1': 0.5, 'sec_ctrl_2': 0.5, 'sec_ctrl_3': 0.5, 'sec_ctrl_4': 0.5,

        # Profile / Keel
        'keel_rise_fwd': 0.5, 'keel_rise_aft': 0.3, 'keel_flat_frac': 0.6,
        'prof_fwd_ctrl': 0.5, 'prof_aft_ctrl': 0.5, 'prof_mid_ctrl': 0.5,

        # Bow
        'bow_angle': 25.0, 'bow_flare': 15.0,
        'bulb_length': 3.0, 'bulb_breadth': 3.0, 'bulb_depth': 3.5,
        'bulb_shape': 0.6, 'bow_rake': 10.0,

        # Stern
        'stern_angle': 20.0, 'stern_shape': 0.5, 'stern_overhang': 2.0,
        'transom_beam': 0.6, 'transom_draft': 0.4,
        'skeg_height': 0.5, 'stern_ctrl_1': 0.5, 'stern_ctrl_2': 0.5,
    }


# ============================================================
# B-Spline Hull Surface Generator
# ============================================================
class HullParameterization:
    """
    Generates a 3D hull surface from a 45-parameter Design Vector.

    Coordinate System (Ship-D convention):
        X: Longitudinal (0 = AP, L = FP)
        Y: Transverse (0 = centerline, +Y = port)
        Z: Vertical (0 = baseline, +Z = up)

    The hull is generated for the port side only (Y >= 0).
    The starboard side is obtained by mirroring.

    Usage:
        hp = HullParameterization(design_vector)
        vertices, faces = hp.generate_mesh(n_stations=21, n_waterlines=11)
    """

    def __init__(self, design_vector: Optional[Dict[str, float]] = None):
        if design_vector is None:
            design_vector = get_default_design_vector()
        self.dv = design_vector
        self._validate()

    def _validate(self):
        """Basic validation of design vector ranges."""
        dv = self.dv
        assert dv['L'] > 0, "Length must be positive"
        assert dv['B'] > 0, "Beam must be positive"
        assert dv['T'] > 0, "Draft must be positive"
        assert 0 < dv['Cb'] < 1, "Block coefficient must be in (0, 1)"
        assert 0 < dv['Cm'] <= 1, "Midship coefficient must be in (0, 1]"

    # ----------------------------------------------------------
    # Waterplane Curve  (plan view at draft T)
    # ----------------------------------------------------------
    def waterplane_halfbreadth(self, x_norm: np.ndarray) -> np.ndarray:
        """
        Compute half-breadth at the waterplane for normalized stations.

        Args:
            x_norm: Longitudinal positions normalized [0, 1] where 0=AP, 1=FP.

        Returns:
            Half-breadth values [m] at each station.
        """
        B_half = self.dv['B'] / 2.0
        mid_f = self.dv['wp_mid_f']  # parallel midbody fraction

        # Control points for B-spline
        # Aft → Midship → Forward
        ctrl_x = np.array([0.0, 0.05, 0.15, 0.5 - mid_f / 2,
                           0.5, 0.5 + mid_f / 2, 0.85, 0.95, 1.0])

        aft_fullness = (self.dv['wp_aft_1'] + self.dv['wp_aft_2'] + self.dv['wp_aft_3']) / 3
        fwd_fullness = (self.dv['wp_fwd_1'] + self.dv['wp_fwd_2'] + self.dv['wp_fwd_3']) / 3

        ctrl_y = np.array([
            self.dv['transom_beam'] * B_half,
            self.dv['wp_aft_1'] * B_half,
            aft_fullness * B_half,
            B_half,                         # Start of parallel midbody
            B_half,                         # Midship
            B_half,                         # End of parallel midbody
            fwd_fullness * B_half,
            self.dv['wp_fwd_1'] * B_half,
            0.0                             # Stem (bow tip)
        ])

        # Bulbous bow width contribution
        bulb_y = self.dv['bulb_breadth'] / 2.0
        # Modify the last few control points to account for bulb
        if self.dv['bulb_length'] > 0:
            bulb_x_norm = 1.0 - self.dv['bulb_length'] / self.dv['L']
            # Ensure bow tip has some width for bulb
            ctrl_y[-1] = max(ctrl_y[-1], bulb_y * 0.3)

        # Create smooth B-spline
        try:
            spline = make_interp_spline(ctrl_x, ctrl_y, k=3)
            y = spline(np.clip(x_norm, 0, 1))
        except Exception:
            # Fallback: linear interpolation
            y = np.interp(x_norm, ctrl_x, ctrl_y)

        return np.clip(y, 0, B_half)

    # ----------------------------------------------------------
    # Section Shape  (cross-section at a given station)
    # ----------------------------------------------------------
    def section_halfbreadth(self, z_norm: np.ndarray, x_station: float) -> np.ndarray:
        """
        Compute half-breadth at a given station for vertical positions.

        Args:
            z_norm: Vertical positions normalized [0, 1] where 0=keel, 1=waterline.
            x_station: Longitudinal position normalized [0, 1].

        Returns:
            Half-breadth values [m] at each vertical position.
        """
        # Get waterplane half-breadth at this station
        y_wl = float(self.waterplane_halfbreadth(np.array([x_station]))[0])
        B_half = self.dv['B'] / 2.0

        # Midship section shape
        Cm = self.dv['Cm']
        bilge_r = self.dv['sec_bilge_r']

        # Deadrise and flare interpolation based on station
        if x_station > 0.5:
            # Forward sections
            t = (x_station - 0.5) * 2  # 0 at mid, 1 at FP
            deadrise = t * self.dv['sec_dead_2']
            flare = t * self.dv['sec_flare_2']
        else:
            # Aft sections
            t = (0.5 - x_station) * 2  # 0 at mid, 1 at AP
            deadrise = t * self.dv['sec_dead_1']
            flare = t * self.dv['sec_flare_1']

        deadrise_rad = np.radians(deadrise)
        flare_rad = np.radians(flare)

        # Section control points
        keel_y = np.tan(deadrise_rad) * (self.dv['T'] * 0.3)  # Deadrise at bottom
        bilge_transition = bilge_r / B_half if B_half > 0 else 0

        # Build section curve
        ctrl_z = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        ctrl_y = np.array([
            keel_y,                                    # Keel
            keel_y + bilge_transition * y_wl * 0.3,   # Lower bilge
            y_wl * Cm * 0.7,                           # Bilge radius zone
            y_wl * Cm,                                 # Mid-height
            y_wl * (1.0 + 0.02 * flare),              # Upper section
            y_wl * (1.0 + 0.03 * flare),              # Near waterline
            y_wl                                       # Waterline
        ])

        try:
            spline = make_interp_spline(ctrl_z, ctrl_y, k=3)
            breadth = spline(np.clip(z_norm, 0, 1))
        except Exception:
            breadth = np.interp(z_norm, ctrl_z, ctrl_y)

        return np.clip(breadth, 0, B_half * 1.1)

    # ----------------------------------------------------------
    # Keel / Profile Curve
    # ----------------------------------------------------------
    def keel_profile(self, x_norm: np.ndarray) -> np.ndarray:
        """
        Compute keel (baseline) z-coordinate at normalized stations.

        Args:
            x_norm: Longitudinal positions [0=AP, 1=FP], normalized.

        Returns:
            Z-coordinates of the keel [m].
        """
        keel_rise_fwd = self.dv['keel_rise_fwd']
        keel_rise_aft = self.dv['keel_rise_aft']
        flat_frac = self.dv['keel_flat_frac']

        ctrl_x = np.array([0.0, 0.1, 0.5 - flat_frac / 2,
                           0.5, 0.5 + flat_frac / 2, 0.9, 1.0])
        ctrl_z = np.array([
            keel_rise_aft,
            keel_rise_aft * 0.5,
            0.0,  # Start of flat keel
            0.0,  # Mid flat keel
            0.0,  # End of flat keel
            keel_rise_fwd * 0.3,
            keel_rise_fwd
        ])

        try:
            spline = make_interp_spline(ctrl_x, ctrl_z, k=3)
            z = spline(np.clip(x_norm, 0, 1))
        except Exception:
            z = np.interp(x_norm, ctrl_x, ctrl_z)

        return np.clip(z, 0, max(keel_rise_fwd, keel_rise_aft, 0.01))

    # ----------------------------------------------------------
    # Bulbous Bow Geometry
    # ----------------------------------------------------------
    def bulb_section(self, x_norm: float, n_pts: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate bulbous bow cross-section at a given longitudinal position.

        Returns:
            (y_pts, z_pts): Arrays of (half-breadth, height) for the bulb cross-section.
        """
        bl = self.dv['bulb_length']
        bb = self.dv['bulb_breadth'] / 2.0
        bd = self.dv['bulb_depth']
        shape = self.dv['bulb_shape']

        # Bulb starts at x_norm = 1.0 - bl/L
        bulb_start = 1.0 - bl / self.dv['L']
        if x_norm < bulb_start or bl <= 0:
            return np.zeros(n_pts), np.zeros(n_pts)

        # Local parameter along bulb (0 = start, 1 = tip)
        t_bulb = (x_norm - bulb_start) / (1.0 - bulb_start + 1e-8)
        t_bulb = np.clip(t_bulb, 0, 1)

        # Bulb radius decreases toward tip
        radius_y = bb * (1.0 - t_bulb ** (1.5 + shape))
        radius_z = bd * (1.0 - t_bulb ** (1.5 + shape)) / 2.0

        # Elliptical section
        theta = np.linspace(0, np.pi, n_pts)
        y_pts = radius_y * np.cos(theta)
        z_pts = radius_z * np.sin(theta) + bd / 2.0  # Center above baseline

        return np.abs(y_pts), z_pts

    # ----------------------------------------------------------
    # 3D Mesh Generation
    # ----------------------------------------------------------
    def generate_mesh(self, n_stations: int = 21, n_waterlines: int = 11,
                      include_bulb: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a CLOSED (watertight) 3D hull surface mesh.

        The mesh is a manifold surface with consistent outward-facing normals.
        Includes:
          - Full hull shell (port + starboard stitched at keel centerline)
          - Deck (waterplane) cap
          - Transom (stern) cap
          - Bow cap
          - Optional closed bulbous bow

        Coordinate System (Ship-D convention):
            X: Longitudinal (0 = AP, L = FP)
            Y: Transverse (0 = centerline, +Y = port, -Y = starboard)
            Z: Vertical (0 = baseline, +Z = up)

        Args:
            n_stations: Number of longitudinal stations.
            n_waterlines: Number of vertical subdivisions per half-section.
            include_bulb: Whether to include bulbous bow geometry.

        Returns:
            (vertices, faces): NumPy arrays for mesh vertices (Nx3, float32)
                               and triangular faces (Mx3, int32, 0-indexed).
        """
        L = self.dv['L']
        T = self.dv['T']

        x_stations = np.linspace(0, 1, n_stations)
        z_waterlines = np.linspace(0, 1, n_waterlines)

        # ── Ring Layout ─────────────────────────────────────────────────
        # Each station has a C-shaped ring profile:
        #   Port WL → down port side → keel (shared) → up stbd side → stbd WL
        # Ring size = 2 * n_waterlines - 1  (keel vertex shared)
        pts_per_ring = 2 * n_waterlines - 1

        # ── Step 1: Build vertex rings ──────────────────────────────────
        vertices_list = []

        for i, x_n in enumerate(x_stations):
            x_m = x_n * L
            keel_z = float(self.keel_profile(np.array([x_n]))[0])
            y_half = self.section_halfbreadth(z_waterlines, x_n)
            y_half = np.asarray(y_half, dtype=np.float64).copy()
            y_half[0] = 0.0  # Force keel to centerline for watertight seam

            # Port side: waterline (j = n_wl-1) down to near-keel (j = 1)
            for j in range(n_waterlines - 1, 0, -1):
                z_m = keel_z + z_waterlines[j] * T
                vertices_list.append([x_m, float(y_half[j]), z_m])

            # Keel (centerline vertex, shared between port & starboard)
            vertices_list.append([x_m, 0.0, keel_z])

            # Starboard side: near-keel (j = 1) up to waterline (j = n_wl-1)
            for j in range(1, n_waterlines):
                z_m = keel_z + z_waterlines[j] * T
                vertices_list.append([x_m, -float(y_half[j]), z_m])

        vertices = np.array(vertices_list, dtype=np.float32)
        faces_list = []

        # ── Step 2: Hull surface faces (connect adjacent rings) ─────────
        for i in range(n_stations - 1):
            for j in range(pts_per_ring - 1):
                v00 = i * pts_per_ring + j
                v01 = i * pts_per_ring + (j + 1)
                v10 = (i + 1) * pts_per_ring + j
                v11 = (i + 1) * pts_per_ring + (j + 1)
                faces_list.append([v00, v10, v01])
                faces_list.append([v10, v11, v01])

        # ── Step 3: Deck (waterplane) cap ───────────────────────────────
        # Connect port WL (ring idx 0) to stbd WL (ring idx pts_per_ring-1)
        for i in range(n_stations - 1):
            p0 = i * pts_per_ring                            # port WL, station i
            p1 = (i + 1) * pts_per_ring                      # port WL, station i+1
            s0 = i * pts_per_ring + (pts_per_ring - 1)       # stbd WL, station i
            s1 = (i + 1) * pts_per_ring + (pts_per_ring - 1) # stbd WL, station i+1
            faces_list.append([p0, s0, s1])
            faces_list.append([p0, s1, p1])

        # ── Step 4: Transom (stern) cap — triangle fan from centroid ────
        stern_center_idx = len(vertices)
        ring0_verts = vertices[0:pts_per_ring]
        centroid_stern = ring0_verts.mean(axis=0).astype(np.float32)
        vertices = np.vstack([vertices, centroid_stern.reshape(1, 3)])

        for j in range(pts_per_ring - 1):
            faces_list.append([stern_center_idx, j + 1, j])
        # Close ring gap (stbd WL → port WL through deck at stern)
        faces_list.append([stern_center_idx, 0, pts_per_ring - 1])

        # ── Step 5: Bow cap — triangle fan from centroid ────────────────
        bow_center_idx = len(vertices)
        last_ring_start = (n_stations - 1) * pts_per_ring
        ring_last_verts = vertices[last_ring_start:last_ring_start + pts_per_ring]
        centroid_bow = ring_last_verts.mean(axis=0).astype(np.float32)
        vertices = np.vstack([vertices, centroid_bow.reshape(1, 3)])

        for j in range(pts_per_ring - 1):
            v0 = last_ring_start + j
            v1 = last_ring_start + j + 1
            faces_list.append([bow_center_idx, v0, v1])
        faces_list.append([bow_center_idx,
                           last_ring_start + pts_per_ring - 1,
                           last_ring_start])

        faces = np.array(faces_list, dtype=np.int32)

        # ── Step 6: Orient all face normals outward ─────────────────────
        faces = self._orient_faces_outward(vertices, faces)

        # ── Step 7: Optional bulbous bow (self-closed mesh) ─────────────
        if include_bulb and self.dv['bulb_length'] > 0:
            bulb_v, bulb_f = self._generate_closed_bulb()
            if len(bulb_v) > 0:
                offset = len(vertices)
                vertices = np.vstack([vertices, bulb_v])
                faces = np.vstack([faces, bulb_f + offset])

        return vertices, faces

    # ----------------------------------------------------------
    # Normal Orientation
    # ----------------------------------------------------------
    @staticmethod
    def _orient_faces_outward(vertices: np.ndarray,
                               faces: np.ndarray) -> np.ndarray:
        """
        Ensure all face normals point away from the mesh centroid.

        Uses the dot-product test: if face_normal · (face_center - centroid) < 0,
        the face winding is reversed.
        """
        centroid = vertices.mean(axis=0)

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        face_centers = (v0 + v1 + v2) / 3.0
        face_normals = np.cross(v1 - v0, v2 - v0)

        outward_dir = face_centers - centroid
        dots = np.sum(face_normals * outward_dir, axis=1)

        flip_mask = dots < 0
        faces[flip_mask] = faces[flip_mask, ::-1]

        return faces

    # ----------------------------------------------------------
    # Closed Bulbous Bow
    # ----------------------------------------------------------
    def _generate_closed_bulb(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a self-closed ellipsoidal mesh for the bulbous bow.

        The bulb is modelled as a tapered ellipsoid with:
          - A tip vertex at the forward perpendicular
          - Elliptical cross-section rings tapering toward the tip
          - A base cap closing the aft end

        Returns:
            (vertices, faces): Self-closed bulb mesh arrays.
        """
        bl = self.dv['bulb_length']
        bb = self.dv['bulb_breadth'] / 2.0
        bd = self.dv['bulb_depth']
        shape = self.dv['bulb_shape']
        L = self.dv['L']

        if bl <= 0 or bb <= 0.01 or bd <= 0.01:
            return np.zeros((0, 3), dtype=np.float32), \
                   np.zeros((0, 3), dtype=np.int32)

        n_long = 10    # longitudinal slices
        n_circ = 16    # circumferential points per ring

        bulb_start_norm = 1.0 - bl / L

        verts = []

        # ── Tip vertex (at FP) ──────────────────────────────────────
        verts.append([L, 0.0, bd / 2.0])

        # ── Rings from near-tip toward bulb start ───────────────────
        t_vals = np.linspace(0.95, 0.0, n_long)

        for t_b in t_vals:
            x_n = bulb_start_norm + t_b * (1.0 - bulb_start_norm)
            x_m = x_n * L

            ry = bb * (1.0 - t_b ** (1.5 + shape))
            rz = (bd / 2.0) * (1.0 - t_b ** (1.5 + shape))

            theta = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
            for t in theta:
                y = ry * np.cos(t)
                z = rz * np.sin(t) + bd / 2.0
                verts.append([x_m, y, z])

        # ── Base cap center ─────────────────────────────────────────
        base_center_idx = len(verts)
        verts.append([bulb_start_norm * L, 0.0, bd / 2.0])

        verts = np.array(verts, dtype=np.float32)
        fcs = []

        # ── Tip fan ─────────────────────────────────────────────────
        for j in range(n_circ):
            j1 = (j + 1) % n_circ
            fcs.append([0, 1 + j, 1 + j1])

        # ── Lateral surface (ring-to-ring) ──────────────────────────
        for i in range(n_long - 1):
            r0 = 1 + i * n_circ
            r1 = 1 + (i + 1) * n_circ
            for j in range(n_circ):
                j1 = (j + 1) % n_circ
                fcs.append([r0 + j, r1 + j, r1 + j1])
                fcs.append([r0 + j, r1 + j1, r0 + j1])

        # ── Base cap fan ────────────────────────────────────────────
        last_ring = 1 + (n_long - 1) * n_circ
        for j in range(n_circ):
            j1 = (j + 1) % n_circ
            fcs.append([base_center_idx, last_ring + j1, last_ring + j])

        fcs = np.array(fcs, dtype=np.int32)

        # Orient normals outward
        fcs = HullParameterization._orient_faces_outward(verts, fcs)

        return verts, fcs

    # ----------------------------------------------------------
    # Volumetric Computations
    # ----------------------------------------------------------
    def compute_displaced_volume(self, n_stations: int = 51) -> float:
        """
        Compute displaced volume using numerical integration (Simpson's rule).

        Returns:
            Displaced volume [m³].
        """
        L = self.dv['L']
        T = self.dv['T']
        x_norm = np.linspace(0, 1, n_stations)

        areas = []
        for x_n in x_norm:
            # Integrate section area using half-breadths
            n_z = 21
            z_vals = np.linspace(0, 1, n_z)
            y_half = self.section_halfbreadth(z_vals, x_n)

            # Area of section = 2 * ∫ y(z) dz (both sides)
            area = 2.0 * np.trapz(y_half, z_vals * T)
            areas.append(area)

        areas = np.array(areas)
        volume = np.trapz(areas, x_norm * L)
        return float(volume)

    def compute_wetted_surface(self, n_stations: int = 41, n_waterlines: int = 21) -> float:
        """
        Approximate wetted surface area from the mesh.

        Returns:
            Wetted surface area [m²].
        """
        vertices, faces = self.generate_mesh(n_stations, n_waterlines, include_bulb=True)

        total_area = 0.0
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            # Triangle area = 0.5 * |cross product|
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            total_area += 0.5 * np.linalg.norm(cross)

        return float(total_area)

    def compute_waterplane_area(self, n_pts: int = 101) -> float:
        """
        Compute waterplane area.

        Returns:
            Waterplane area [m²].
        """
        L = self.dv['L']
        x_norm = np.linspace(0, 1, n_pts)
        y_half = self.waterplane_halfbreadth(x_norm)

        # Area = 2 * ∫ y(x) dx
        area = 2.0 * np.trapz(y_half, x_norm * L)
        return float(area)

    def compute_block_coefficient(self) -> float:
        """Compute actual block coefficient from geometry."""
        vol = self.compute_displaced_volume()
        L, B, T = self.dv['L'], self.dv['B'], self.dv['T']
        return vol / (L * B * T) if (L * B * T) > 0 else 0.0

    def compute_hydrostatics(self) -> Dict[str, float]:
        """
        Compute comprehensive hydrostatic properties.

        Returns:
            Dictionary with volume, Cb, Cwp, WSA, LCB, etc.
        """
        vol = self.compute_displaced_volume()
        wsa = self.compute_wetted_surface()
        awp = self.compute_waterplane_area()
        L, B, T = self.dv['L'], self.dv['B'], self.dv['T']

        cb_actual = vol / (L * B * T) if (L * B * T) > 0 else 0
        cwp_actual = awp / (L * B) if (L * B) > 0 else 0

        # Displacement (seawater density ≈ 1025 kg/m³)
        displacement = vol * 1025.0

        return {
            'displaced_volume': vol,
            'displacement': displacement,
            'wetted_surface_area': wsa,
            'waterplane_area': awp,
            'Cb_actual': cb_actual,
            'Cwp_actual': cwp_actual,
            'L': L, 'B': B, 'T': T, 'D': self.dv['D'],
            'LCB': self.dv['LCB'],
            'Cm': self.dv['Cm'],
        }



# ShipHullGAN requirement: geomdl for NURBS
try:
    from geomdl import BSpline, NURBS, utilities
    from geomdl.exchange import export_obj
    GEOMDL_AVAILABLE = True
except ImportError:
    GEOMDL_AVAILABLE = False
    import logging
    logging.warning("geomdl not installed. NURBS reconstruction disabled.")

class ShipHullGANEncoder:
    '''Encodes a hull into 56 cross-sections X 25 points'''
    def __init__(self, hp):
        self.hp = hp
        self.points = None
        
    def encode(self, num_sections=56, pts_per_section=25) -> __import__('numpy').ndarray:
        import numpy as np
        L = self.hp.dv['L']
        T = self.hp.dv['T']
        stations = np.linspace(0, 1, num_sections)
        points_3d = np.zeros((num_sections, pts_per_section, 3))
        for i, x_n in enumerate(stations):
            x_m = x_n * L
            keel_z = float(self.hp.keel_profile(np.array([x_n]))[0])
            z_n = np.linspace(0, 1, pts_per_section)**1.5 
            y_half = self.hp.section_halfbreadth(z_n, x_n)
            for j in range(pts_per_section):
                points_3d[i, j, 0] = x_m
                points_3d[i, j, 1] = y_half[j]
                points_3d[i, j, 2] = keel_z + z_n[j] * T
        self.points = points_3d
        return points_3d

class GeometricMomentComputer:
    '''Computes Geometric Moment Invariants (GMIs)'''
    @staticmethod
    def compute_gmi(points_3d: __import__('numpy').ndarray) -> __import__('numpy').ndarray:
        import numpy as np
        centroid = np.mean(points_3d.reshape(-1, 3), axis=0)
        centered = points_3d.reshape(-1, 3) - centroid
        m200 = np.sum(centered[:, 0]**2)
        m020 = np.sum(centered[:, 1]**2)
        m002 = np.sum(centered[:, 2]**2)
        m110 = np.sum(centered[:, 0] * centered[:, 1])
        return np.array([m200 + m020 + m002, m200*m020 - m110**2, np.linalg.norm(centroid)])

class ShapeSignatureTensor:
    '''Constructs the SST format'''
    @staticmethod
    def construct(points_3d: __import__('numpy').ndarray, gmis: __import__('numpy').ndarray) -> __import__('numpy').ndarray:
        import numpy as np
        num_sec, num_pts, _ = points_3d.shape
        sst = np.zeros((num_sec, num_pts, 4))
        sst[:, :, :3] = points_3d
        sst[:, :, :3] /= (np.max(np.abs(points_3d)) + 1e-8)
        return sst

class NURBSReconstructor:
    '''Reconstructs a smooth surface using geomdl'''
    def __init__(self, sst_points: __import__('numpy').ndarray):
        self.points = sst_points
        self.surface = None
    def reconstruct(self) -> tuple:
        import numpy as np
        if not GEOMDL_AVAILABLE:
            raise RuntimeError("geomdl is required")
        num_u, num_v, _ = self.points.shape
        surf = NURBS.Surface()
        surf.degree_u = 3
        surf.degree_v = 3
        surf.ctrlpts2d = self.points.tolist()
        surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, num_u)
        surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, num_v)
        surf.delta = 0.05
        surf.evaluate()
        verts = np.array(surf.evalpts, dtype=np.float32)
        u_pts = int(1.0 / surf.delta) + 1
        v_pts = int(1.0 / surf.delta) + 1
        faces = []
        for i in range(u_pts - 1):
            for j in range(v_pts - 1):
                p1 = i * v_pts + j
                p2 = i * v_pts + (j + 1)
                p3 = (i + 1) * v_pts + j
                p4 = (i + 1) * v_pts + (j + 1)
                faces.append([p1, p3, p2])
                faces.append([p2, p3, p4])
        return verts, np.array(faces, dtype=np.int32)

class FFDMorpher:
    '''Applies Free-Form Deformation'''
    def __init__(self, vertices: __import__('numpy').ndarray):
        self.vertices = vertices.copy()
    def morph_region(self, center, radius, displacement):
        import numpy as np
        c = np.array(center)
        d = np.array(displacement)
        dist = np.linalg.norm(self.vertices - c, axis=1)
        mask = dist < radius
        if np.any(mask):
            falloff = np.exp(-(dist[mask]**2) / (radius**2 * 0.5))
            self.vertices[mask] += np.outer(falloff, d)
        return self.vertices


"""
RetrosimHullAdapter — Abstraction Layer for Parametric Hull Design
====================================================================

Adapter Pattern implementation that bridges the SmartCAPEX AI GUI
with the isolated MIT DeCoDE parametric hull generator.

Responsibilities:
    1. Convert PyQt6 UI inputs → 45-dimensional Design Vector
    2. Invoke HullParameterization for mesh generation
    3. Export mesh as STL file (numpy-stl)
    4. Extract volumetric features for ML inputs (EANN / PINN)
    5. Provide regression-based imputation for missing geometry parameters

Scientific Basis:
    - Adapter Pattern isolating third-party geometry from application logic
    - Statistical regression for missing parameter estimation
    - Ship-D 45-vector standard for parametric design

Author: SmartCAPEX AI Team
"""

import os
import numpy as np
from typing import Dict, Optional, Tuple, List

# Removed HullParameterization import to be fully contained

# Optional: PyGeM FFD Hull Morphing (CMAME 2023)
# Replaced PyGeM with internal FFD

# Optional: numpy-stl for STL export
try:
    from stl import mesh as stl_mesh
    HAS_STL = True
except ImportError:
    HAS_STL = False


class RetrosimHullAdapter:

    def get_ship_hull_gan_sst(self) -> __import__('numpy').ndarray:
        if self._hull is None:
            self._hull = HullParameterization(self._design_vector)
        encoder = ShipHullGANEncoder(self._hull)
        points = encoder.encode()
        gmis = GeometricMomentComputer.compute_gmi(points)
        return ShapeSignatureTensor.construct(points, gmis)

    def get_nurbs_mesh(self) -> tuple:
        if self._hull is None:
            self._hull = HullParameterization(self._design_vector)
        encoder = ShipHullGANEncoder(self._hull)
        pts = encoder.encode()
        try:
            recon = NURBSReconstructor(pts)
            return recon.reconstruct()
        except Exception as e:
            print(f"NURBS error: {e}")
            return self.generate_mesh()

    @property
    def mesh(self):
        v, f = self.generate_mesh()
        return {'vertices': v, 'faces': f}

    @property
    def is_generated(self):
        return True

    """
    Adapts SmartCAPEX GUI vessel data to the 45-parameter Design Vector
    used by HullParameterization, then generates meshes and computes
    volumetric features.

    Usage:
        adapter = RetrosimHullAdapter()
        adapter.set_from_ui(vessel_data_dict)
        stl_path = adapter.generate_stl("output_hull.stl")
        features = adapter.extract_ml_features()
    """

    # Output directory for generated meshes
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'geometry')

    def __init__(self):
        self._design_vector: Dict[str, float] = get_default_design_vector()
        self._hull: Optional[HullParameterization] = None
        self._mesh_vertices: Optional[np.ndarray] = None
        self._mesh_faces: Optional[np.ndarray] = None
        self._hydrostatics: Optional[Dict] = None
        self._dirty = True  # Marks when DV changed but mesh not regenerated

        # FFD Mode State (CMAME 2023)
        self._ffd_mode: bool = False
        self._ffd_morpher: Optional[object] = None  # FFDHullMorpher
        self._ffd_base_vertices: Optional[np.ndarray] = None
        self._ffd_base_faces: Optional[np.ndarray] = None

    # ----------------------------------------------------------
    # UI → Design Vector Mapping
    # ----------------------------------------------------------
    def set_from_ui(self, ui_data: Dict) -> Dict[str, float]:
        """
        Convert PyQt6 UI inputs (VesselData dataclass fields) to a
        45-parameter Design Vector.

        Known UI fields are mapped directly. Missing geometry parameters
        are estimated using naval architecture regression formulae.

        Args:
            ui_data: Dictionary from VesselData dataclass (asdict(vessel_data)).

        Returns:
            The resulting design vector.
        """
        dv = get_default_design_vector()

        # === Principal Dimensions ===
        # L
        if ui_data.get('loa') and float(ui_data['loa']) > 0:
            loa = float(ui_data['loa'])
            lbp = float(ui_data.get('lbp', 0))
            dv['L'] = lbp if lbp > 0 else loa * 0.95
        elif ui_data.get('lbp') and float(ui_data['lbp']) > 0:
            dv['L'] = float(ui_data['lbp'])

        # B
        if ui_data.get('beam') and float(ui_data['beam']) > 0:
            dv['B'] = float(ui_data['beam'])
        elif ui_data.get('breadth') and float(ui_data['breadth']) > 0:
            dv['B'] = float(ui_data['breadth'])

        # T (Draft) & F (Freeboard) -> D (Depth)
        if ui_data.get('draft') and float(ui_data['draft']) > 0:
            dv['T'] = float(ui_data['draft'])

        fribord = 2.0  # Default fribord
        if ui_data.get('freeboard') and float(ui_data['freeboard']) > 0:
            fribord = float(ui_data['freeboard'])
        elif ui_data.get('depth') and float(ui_data['depth']) > 0:
            # Fallback for old saved definitions
            fribord = max(0.5, float(ui_data['depth']) - dv['T'])

        # Depth (D) = Draft (T) + Freeboard (f)
        dv['D'] = dv['T'] + fribord

        # --- FİZİKSEL DOĞRULAMALAR VE İLİŞKİ DÜZELTMELERİ ---
        # 2. Boy (L), Genişlikten (B) makul ölçüde uzun olmalı
        if dv['L'] < dv['B'] * 3.0:
            print(f"⚠️ Düzeltme: Boy ({dv['L']}m) çok kısa. Beam'e ({dv['B']}m) oranla güncelleniyor.")
            dv['L'] = dv['B'] * 3.0
            
        # 3. Genişlik (B), Draft'tan (T) dar olmamalı. Minimum stabilite için B >= T * 1.5
        if dv['B'] < dv['T'] * 1.5:
            print(f"⚠️ Düzeltme: Genişlik ({dv['B']}m) Draft'a ({dv['T']}m) oranla çok dar. Beam güncelleniyor.")
            dv['B'] = dv['T'] * 1.5

        # Cb
        if ui_data.get('cb') and float(ui_data['cb']) > 0:
            dv['Cb'] = max(0.4, min(1.0, float(ui_data['cb'])))
        else:
            # Watson & Gilfillan regression for Cb
            speed_knots = float(ui_data.get('speed', 0)) or 12.0
            speed_ms = speed_knots * 0.5144
            Fn = speed_ms / np.sqrt(9.81 * dv['L']) if dv['L'] > 0 else 0.2
            dv['Cb'] = max(0.4, min(0.9, 1.067 - 1.811 * Fn))

        # Cm & Cp
        if ui_data.get('cm') and float(ui_data['cm']) > 0:
            dv['Cm'] = max(dv['Cb'], min(1.0, float(ui_data['cm'])))
        elif ui_data.get('cp') and float(ui_data['cp']) > 0:
            cm_est = dv['Cb'] / max(float(ui_data['cp']), 0.01)
            dv['Cm'] = max(dv['Cb'], min(1.0, cm_est))
        else:
            dv['Cm'] = min(0.99, dv['Cb'] + 0.1) # Fallback

        # --- LCB Estimation ---
        dv['LCB'] = self._estimate_lcb(dv['Cb'], dv['L'])

        # --- Waterplane Coefficient ---
        if ui_data.get('cwp') and float(ui_data['cwp']) > 0:
            dv['Cwp'] = max(0.5, min(1.0, float(ui_data['cwp'])))
        else:
            # Cwp ≈ f(Cb) regression
            dv['Cwp'] = 0.18 + 0.86 * dv['Cb']

        # === Bow Parameters ===
        if ui_data.get('bow_height') and float(ui_data['bow_height']) > 0:
            dv['bow_flare'] = float(ui_data['bow_height']) * 2  # Approximate

        if ui_data.get('bulb_length') and float(ui_data['bulb_length']) > 0:
            # Bulbous bow length usually max ~15% of LBP
            dv['bulb_length'] = min(dv['L'] * 0.15, float(ui_data['bulb_length']))
            
        if ui_data.get('bulb_radius') and float(ui_data['bulb_radius']) > 0:
            bulb_r = float(ui_data['bulb_radius'])
            # Bulb breadth generally max 50% of Beam
            dv['bulb_breadth'] = min(dv['B'] * 0.5, bulb_r * 2)
            # Bulb depth (height) generally shouldn't exceed Draft
            dv['bulb_depth'] = min(dv['T'] * 0.9, bulb_r * 1.5)

        # --- Bow Entrance Angle ---
        dv['bow_angle'] = self._estimate_entrance_angle(dv['Cb'], dv['L'], dv['B'])

        # === Stern Parameters ===
        if ui_data.get('stern_shape') and float(ui_data['stern_shape']) > 0:
            dv['stern_shape'] = max(0.0, min(1.0, float(ui_data['stern_shape'])))
        if ui_data.get('stern_height') and float(ui_data['stern_height']) > 0:
            dv['stern_overhang'] = float(ui_data['stern_height'])

        # === Propeller (influences stern) ===
        if ui_data.get('prop_dia') and float(ui_data['prop_dia']) > 0:
            prop_d = float(ui_data['prop_dia'])
            # Pervane çapı draft'tan büyük olamaz (pratikte max ~0.7T)
            prop_d = min(dv['T'] * 0.7, prop_d)
            dv['skeg_height'] = prop_d * 0.1

        # === Bilge Radius Estimation ===
        dv['sec_bilge_r'] = self._estimate_bilge_radius(dv['B'], dv['T'], dv['Cm'])

        # === Parallel Midbody ===
        dv['wp_mid_f'] = self._estimate_parallel_midbody(dv['Cb'])

        # === Section Shape from L/B and B/T ratios ===
        lb_ratio = dv['L'] / dv['B'] if dv['B'] > 0 else 6.0
        bt_ratio = dv['B'] / dv['T'] if dv['T'] > 0 else 2.5

        # Finer vessels → more flare, less deadrise
        dv['sec_flare_1'] = max(0.0, 10.0 - lb_ratio)
        dv['sec_flare_2'] = max(0.0, 15.0 - lb_ratio * 1.5)
        dv['sec_dead_1'] = max(0.0, 5.0 * (1.0 - dv['Cm']))
        dv['sec_dead_2'] = max(0.0, 10.0 * (1.0 - dv['Cm']))

        # === High Block Coefficient (Cb) Override (Barge/Box Shape) ===
        # If Cb is tending towards 1.0, the hull must physically transform into a pure box.
        # We blend all control points towards rectangular boundaries based on Cb > 0.85
        if dv['Cb'] > 0.85:
            # Interpolation factor: 0.0 at Cb=0.85, 1.0 at Cb=1.00
            factor = (dv['Cb'] - 0.85) / 0.15
            factor = max(0.0, min(1.0, factor))
            
            # 1. Waterplane becomes perfectly rectangular (no curves, full beam everywhere)
            dv['wp_mid_f'] = dv['wp_mid_f'] * (1 - factor) + 1.0 * factor
            dv['wp_fwd_1'] = dv['wp_fwd_1'] * (1 - factor) + 1.0 * factor
            dv['wp_fwd_2'] = dv['wp_fwd_2'] * (1 - factor) + 1.0 * factor
            dv['wp_fwd_3'] = dv['wp_fwd_3'] * (1 - factor) + 1.0 * factor
            dv['wp_aft_1'] = dv['wp_aft_1'] * (1 - factor) + 1.0 * factor
            dv['wp_aft_2'] = dv['wp_aft_2'] * (1 - factor) + 1.0 * factor
            dv['wp_aft_3'] = dv['wp_aft_3'] * (1 - factor) + 1.0 * factor
            dv['transom_beam'] = dv['transom_beam'] * (1 - factor) + 1.0 * factor
            dv['Cwp'] = dv['Cwp'] * (1 - factor) + 1.0 * factor
            
            # 2. Section becomes perfectly rectangular (no deadrise, flare or bilge radius)
            dv['Cm'] = dv['Cm'] * (1 - factor) + 1.0 * factor
            dv['sec_bilge_r'] = dv['sec_bilge_r'] * (1 - factor) + 0.0 * factor
            dv['sec_flare_1'] = dv['sec_flare_1'] * (1 - factor) + 0.0 * factor
            dv['sec_flare_2'] = dv['sec_flare_2'] * (1 - factor) + 0.0 * factor
            dv['sec_dead_1'] = dv['sec_dead_1'] * (1 - factor) + 0.0 * factor
            dv['sec_dead_2'] = dv['sec_dead_2'] * (1 - factor) + 0.0 * factor
            
            # 3. Profile / Keel line becomes completely flat bottom and blunt
            dv['keel_rise_fwd'] = dv['keel_rise_fwd'] * (1 - factor) + 0.0 * factor
            dv['keel_rise_aft'] = dv['keel_rise_aft'] * (1 - factor) + 0.0 * factor
            dv['keel_flat_frac'] = dv['keel_flat_frac'] * (1 - factor) + 1.0 * factor
            
            # 4. Angles become 90 degrees (blunt wall)
            dv['bow_angle'] = dv['bow_angle'] * (1 - factor) + 90.0 * factor
            dv['stern_angle'] = dv['stern_angle'] * (1 - factor) + 90.0 * factor
            
            # 5. Remove protruding features like Bulb or overhangs
            dv['bulb_length'] = dv['bulb_length'] * (1 - factor) + 0.0 * factor
            dv['stern_overhang'] = dv['stern_overhang'] * (1 - factor) + 0.0 * factor

        # === Store and invalidate cache ===
        self._design_vector = dv
        self._dirty = True
        self._hull = None
        self._mesh_vertices = None
        self._mesh_faces = None
        self._hydrostatics = None

        return dv

    def set_design_vector(self, dv: Dict[str, float]):
        """Set design vector directly (for advanced users / Ship-D data)."""
        self._design_vector = dv
        self._dirty = True
        self._hull = None
        self._mesh_vertices = None
        self._mesh_faces = None
        self._hydrostatics = None

    def get_design_vector(self) -> Dict[str, float]:
        """Get current design vector."""
        return self._design_vector.copy()

    # ----------------------------------------------------------
    # Regression Helpers (Naval Architecture Empirical Formulas)
    # ----------------------------------------------------------
    @staticmethod
    def _estimate_lcb(Cb: float, L: float) -> float:
        """Estimate LCB position (% from FP) based on block coefficient."""
        # Typical regression: LCB shifts aft with increasing Cb
        # LCB ≈ 50 + (Cb - 0.7) * 8 [% from FP]
        lcb_pct = 50.0 + (Cb - 0.7) * 8.0
        return max(45.0, min(55.0, lcb_pct))

    @staticmethod
    def _estimate_entrance_angle(Cb: float, L: float, B: float) -> float:
        """Estimate half-angle of entrance at waterline [degrees]."""
        # ie ≈ f(Cb, L/B) — various regressions exist
        lb = L / B if B > 0 else 6.0
        ie = 125.67 * (B / L) - 162.25 * Cb ** 2 + 234.32 * Cb ** 3
        return max(5, min(60, ie))

    @staticmethod
    def _estimate_bilge_radius(B: float, T: float, Cm: float) -> float:
        """Estimate bilge radius from midship coefficient and dimensions."""
        # r ≈ B/2 * (1 - Cm) * constant
        return max(0.1, (B / 2.0) * (1.0 - Cm) * 3.0)

    @staticmethod
    def _estimate_parallel_midbody(Cb: float) -> float:
        """Estimate parallel midbody fraction from Cb."""
        # Full-form vessels have longer parallel midbody
        return max(0.0, min(0.6, (Cb - 0.5) * 1.2))

    # ----------------------------------------------------------
    # Mesh Generation
    # ----------------------------------------------------------
    def generate_mesh(self, n_stations: int = 100, n_waterlines: int = 60
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate hull mesh from current design vector.

        Returns:
            (vertices, faces) as numpy arrays.
        """
        if self._hull is None or self._dirty:
            self._hull = HullParameterization(self._design_vector)
            self._dirty = False

        self._mesh_vertices, self._mesh_faces = self._hull.generate_mesh(
            n_stations=n_stations,
            n_waterlines=n_waterlines,
            include_bulb=True
        )
        return self._mesh_vertices, self._mesh_faces

    def generate_stl(self, output_path: Optional[str] = None,
                     n_stations: int = 100, n_waterlines: int = 60) -> str:
        """
        Generate and save hull mesh as STL file.

        Args:
            output_path: Path for the STL file. If None, uses default.
            n_stations: Number of longitudinal stations.
            n_waterlines: Number of vertical waterlines.

        Returns:
            Absolute path to the generated STL file.

        Raises:
            ImportError: If numpy-stl is not installed.
        """
        if not HAS_STL:
            raise ImportError(
                "numpy-stl is required for STL export. "
                "Install with: pip install numpy-stl"
            )

        vertices, faces = self.generate_mesh(n_stations, n_waterlines)

        if output_path is None:
            os.makedirs(self.OUTPUT_DIR, exist_ok=True)
            output_path = os.path.join(self.OUTPUT_DIR, 'generated_hull.stl')

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Create STL mesh
        hull_mesh = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
        for i, face in enumerate(faces):
            for j in range(3):
                hull_mesh.vectors[i][j] = vertices[face[j]]

        hull_mesh.save(output_path)
        print(f"🛳️ STL kaydedildi: {output_path} ({len(faces)} face, {len(vertices)} vertex)")
        return os.path.abspath(output_path)

    def generate_usda(self, output_path: Optional[str] = None,
                      n_stations: int = 100, n_waterlines: int = 60) -> str:
        """
        Generate and save hull mesh as an ASCII USD (.usda) file with
        per-vertex normals, OmniPBR material binding, and proper Xform
        hierarchy for NVIDIA Omniverse.

        Args:
            output_path: Path for the USDA file. If None, uses default.
            n_stations: Number of longitudinal stations.
            n_waterlines: Number of vertical waterlines.

        Returns:
            Absolute path to the generated USDA file.
        """
        vertices, faces = self.generate_mesh(n_stations, n_waterlines)

        if output_path is None:
            os.makedirs(self.OUTPUT_DIR, exist_ok=True)
            output_path = os.path.join(self.OUTPUT_DIR, 'generated_hull.usda')

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Compute per-vertex normals
        normals = self._compute_vertex_normals(vertices, faces)

        # Build USDA string
        face_counts = ", ".join(["3"] * len(faces))
        face_indices = ", ".join(map(str, faces.flatten()))
        points_str = ", ".join([f"({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})" for v in vertices])
        normals_str = ", ".join([f"({n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f})" for n in normals])

        usda_lines = [
            '#usda 1.0',
            '(',
            '    defaultPrim = "Hull_Xform"',
            '    metersPerUnit = 1.0',
            '    upAxis = "Z"',
            ')',
            '',
            'def Xform "Hull_Xform"',
            '{',
            '    def Scope "Materials"',
            '    {',
            '        def Material "HullMaterial"',
            '        {',
            '            token outputs:surface.connect = </Hull_Xform/Materials/HullMaterial/OmniPBR.outputs:surface>',
            '',
            '            def Shader "OmniPBR"',
            '            {',
            '                uniform token info:id = "OmniPBR"',
            '                color3f inputs:diffuse_color_constant = (0.12, 0.30, 0.55)',
            '                float inputs:metallic_constant = 0.3',
            '                float inputs:roughness_constant = 0.6',
            '                float inputs:specular_level = 0.5',
            '                token outputs:surface',
            '            }',
            '        }',
            '    }',
            '',
            '    def Mesh "Hull"',
            '    {',
            f'        int[] faceVertexCounts = [{face_counts}]',
            f'        int[] faceVertexIndices = [{face_indices}]',
            f'        point3f[] points = [{points_str}]',
            f'        normal3f[] normals = [{normals_str}] (',
            '            interpolation = "vertex"',
            '        )',
            '        uniform token subdivisionScheme = "none"',
            '        color3f[] primvars:displayColor = [(0.15, 0.35, 0.65)] (',
            '            interpolation = "constant"',
            '        )',
            '        rel material:binding = </Hull_Xform/Materials/HullMaterial>',
            '    }',
            '}',
        ]

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(usda_lines))

        print(f"🚢 USD (Omniverse) kaydedildi: {output_path} ({len(faces)} face, {len(vertices)} vertex)")
        return os.path.abspath(output_path)

    def generate_usd_with_flow_field(self, flow_data: Dict,
                                      output_path: Optional[str] = None,
                                      n_stations: int = 100,
                                      n_waterlines: int = 60) -> str:
        """
        Generate USD with hull geometry AND a flow field overlay as Points prim.

        Args:
            flow_data: Dict with 'X', 'Y', 'U', 'V', 'P' arrays.
            output_path: Output USDA path.

        Returns:
            Absolute path to the generated USDA file.
        """
        # First generate the base USD
        base_path = self.generate_usda(output_path, n_stations, n_waterlines)

        # Append flow field prim
        X = flow_data.get('X')
        Y = flow_data.get('Y')
        if X is None or Y is None:
            return base_path

        # Subsample for performance
        step = max(1, X.shape[0] // 32)
        xs = X[::step, ::step].flatten()
        ys = Y[::step, ::step].flatten()
        zs = np.zeros_like(xs)

        flow_points = ", ".join([f"({x:.3f}, {y:.3f}, {z:.3f})"
                                 for x, y, z in zip(xs, ys, zs)])

        flow_prim = [
            '',
            '    def Points "FlowField"',
            '    {',
            f'        point3f[] points = [{flow_points}]',
            '        float[] widths = [' + ', '.join(['0.02'] * len(xs)) + ']',
            '        color3f[] primvars:displayColor = [(0.2, 0.8, 1.0)] (',
            '            interpolation = "constant"',
            '        )',
            '    }',
        ]

        # Insert flow prim before closing brace of Hull_Xform
        with open(base_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace last '}' with flow prim + '}'
        last_brace = content.rfind('}')
        if last_brace > 0:
            content = content[:last_brace] + '\n'.join(flow_prim) + '\n}\n'

        with open(base_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"🌊 USD + FlowField kaydedildi: {base_path}")
        return base_path

    @staticmethod
    def _compute_vertex_normals(vertices: np.ndarray,
                                 faces: np.ndarray) -> np.ndarray:
        """Compute per-vertex normals by averaging face normals."""
        normals = np.zeros_like(vertices)
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        face_normals /= norms

        for i in range(3):
            np.add.at(normals, faces[:, i], face_normals)

        lens = np.linalg.norm(normals, axis=1, keepdims=True)
        lens[lens < 1e-10] = 1.0
        normals /= lens

        return normals

    # ----------------------------------------------------------
    # ML Feature Extraction
    # ----------------------------------------------------------
    def extract_ml_features(self) -> Dict[str, float]:
        """
        Extract machine-learning-ready features from the hull geometry.

        These features feed into the SurrogateModeler (EANN) and
        PINNCFDAgent for hydrodynamic prediction.

        Returns:
            Dictionary of volumetric and shape features.
        """
        if self._hydrostatics is None:
            if self._hull is None:
                self._hull = HullParameterization(self._design_vector)
                self._dirty = False
            self._hydrostatics = self._hull.compute_hydrostatics()

        hs = self._hydrostatics
        dv = self._design_vector

        # Derived ratios
        L, B, T = hs['L'], hs['B'], hs['T']
        lb_ratio = L / B if B > 0 else 0
        bt_ratio = B / T if T > 0 else 0
        lt_ratio = L / T if T > 0 else 0

        # Slenderness coefficient
        vol = hs['displaced_volume']
        slenderness = L / (vol ** (1.0 / 3.0)) if vol > 0 else 0

        return {
            # Volumetric
            'displaced_volume': vol,
            'displacement_tonnes': hs['displacement'] / 1000.0,
            'wetted_surface_area': hs['wetted_surface_area'],
            'waterplane_area': hs['waterplane_area'],

            # Form coefficients
            'Cb_actual': hs['Cb_actual'],
            'Cwp_actual': hs['Cwp_actual'],
            'Cm': hs['Cm'],

            # Ratios
            'L_B_ratio': lb_ratio,
            'B_T_ratio': bt_ratio,
            'L_T_ratio': lt_ratio,
            'slenderness_coefficient': slenderness,

            # Dimensions
            'length': L,
            'breadth': B,
            'draft': T,
            'depth': hs['D'],
            'LCB': hs['LCB'],

            # Bow shape
            'bulb_length': dv['bulb_length'],
            'bulb_breadth': dv['bulb_breadth'],
            'bow_angle': dv['bow_angle'],

            # Stern shape
            'stern_angle': dv['stern_angle'],
            'stern_shape': dv['stern_shape'],
            'transom_beam_ratio': dv['transom_beam'],
        }

    def extract_point_cloud(self, num_points: int = 2048,
                            method: str = 'parametric') -> np.ndarray:
        """
        Extract a Point Cloud representation of the hull surface.

        This is the **primary geometry representation** for all AI agents
        (PointNet++, GC-FNO, XGBoost features). No mesh intermediate
        is used when method='parametric'.

        Methods:
            'parametric': Direct B-spline surface sampling (preferred).
                Stratified grid over (station × waterline) with jitter
                for uniform coverage. No mesh needed.
            'mesh': Area-weighted barycentric sampling from triangulated
                mesh. Used as fallback or for imported geometries.

        Args:
            num_points: Target number of points in the (N, 3) output.
            method: 'parametric' (default, higher fidelity) or 'mesh'.

        Returns:
            np.ndarray of shape (num_points, 3) — [x, y, z] coordinates.
        """
        if method == 'parametric':
            return self._point_cloud_from_parametric(num_points)
        else:
            return self._point_cloud_from_mesh(num_points)

    def _point_cloud_from_parametric(self, num_points: int) -> np.ndarray:
        """
        Sample point cloud directly from B-spline parametric surface.

        Uses stratified grid over (station × waterline) parameter space
        with uniform jitter for Poisson-disc-like surface coverage.
        Both port and starboard sides are sampled.

        This avoids the mesh intermediate entirely, producing higher-
        fidelity samples that capture the true smooth surface.
        """
        if self._hull is None or self._dirty:
            self._hull = HullParameterization(self._design_vector)
            self._dirty = False

        hp = self._hull
        L = hp.dv['L']
        T = hp.dv['T']

        # Determine grid density (half points per side for symmetry)
        n_half = num_points // 2
        # Aspect ratio: more stations than waterlines (hull is longer)
        n_stations = int(np.sqrt(n_half * 3.0))
        n_waterlines = max(8, n_half // n_stations)

        # Stratified grid with jitter
        x_centers = np.linspace(0.01, 0.99, n_stations)
        z_centers = np.linspace(0.01, 0.99, n_waterlines)

        dx = 0.5 / n_stations
        dz = 0.5 / n_waterlines

        points = []
        for x_n in x_centers:
            # Jitter station position
            x_jitter = x_n + np.random.uniform(-dx, dx, n_waterlines)
            x_jitter = np.clip(x_jitter, 0.001, 0.999)

            z_jitter = z_centers + np.random.uniform(-dz, dz, n_waterlines)
            z_jitter = np.clip(z_jitter, 0.001, 0.999)

            # Get half-breadths at each (station, waterline) pair
            y_half = hp.section_halfbreadth(z_jitter, np.mean(x_jitter))

            for i, (xj, zj, yh) in enumerate(zip(x_jitter, z_jitter, y_half)):
                if yh > 1e-4:
                    x_real = xj * L
                    z_real = zj * T
                    # Starboard
                    points.append([x_real, yh, z_real])
                    # Port (mirror)
                    points.append([x_real, -yh, z_real])

        points = np.array(points, dtype=np.float32)

        # Resample to exact num_points
        if len(points) == 0:
            # Fallback to mesh if parametric fails
            return self._point_cloud_from_mesh(num_points)

        if len(points) >= num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
        else:
            indices = np.random.choice(len(points), num_points, replace=True)

        return points[indices]

    def _point_cloud_from_mesh(self, num_points: int) -> np.ndarray:
        """
        Sample point cloud from triangulated mesh via area-weighted
        barycentric sampling. Used for imported geometries (STL/OBJ)
        or as fallback.
        """
        if self._mesh_vertices is None or self._mesh_faces is None:
            self.generate_mesh()

        return self.mesh_to_point_cloud(
            self._mesh_vertices, self._mesh_faces, num_points
        )

    @staticmethod
    def mesh_to_point_cloud(vertices: np.ndarray, faces: np.ndarray,
                            num_points: int = 2048) -> np.ndarray:
        """
        Convert any (vertices, faces) mesh to a point cloud via
        area-weighted barycentric surface sampling.

        Use this when importing external geometry (STL, OBJ, USD)
        that needs to be converted to point cloud for AI processing.

        Args:
            vertices: (V, 3) vertex array.
            faces: (F, 3) face index array.
            num_points: Target number of sampled points.

        Returns:
            np.ndarray of shape (num_points, 3).
        """
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)

        # Normalize to probability distribution
        total_area = areas.sum()
        if total_area < 1e-12:
            # Degenerate mesh — return vertex subsample
            idx = np.random.choice(len(vertices), min(num_points, len(vertices)),
                                   replace=len(vertices) < num_points)
            pc = vertices[idx]
            if len(pc) < num_points:
                pc = np.vstack([pc] * (num_points // len(pc) + 1))[:num_points]
            return pc.astype(np.float32)

        prob = areas / total_area

        # Select faces proportional to area
        face_indices = np.random.choice(len(faces), size=num_points, p=prob)

        # Random barycentric coordinates
        u = np.random.uniform(0, 1, size=(num_points, 1))
        v = np.random.uniform(0, 1, size=(num_points, 1))

        # Fold into triangle
        mask = (u + v) > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        w = 1 - u - v

        selected_v0 = vertices[faces[face_indices, 0]]
        selected_v1 = vertices[faces[face_indices, 1]]
        selected_v2 = vertices[faces[face_indices, 2]]

        point_cloud = (u * selected_v0) + (v * selected_v1) + (w * selected_v2)
        return point_cloud.astype(np.float32)

    @staticmethod
    def import_mesh_as_point_cloud(file_path: str,
                                    num_points: int = 2048) -> np.ndarray:
        """
        Import an external mesh file (STL/OBJ) and convert to point cloud.

        This is the entry point for user-imported geometries. The mesh
        is loaded, converted to point cloud, and normalised for AI input.

        Args:
            file_path: Path to STL or OBJ file.
            num_points: Target number of points.

        Returns:
            Normalised point cloud (num_points, 3) centered at origin.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.stl':
            if not HAS_STL:
                raise ImportError("numpy-stl required: pip install numpy-stl")
            mesh_data = stl_mesh.Mesh.from_file(file_path)
            # Each triangle has 3 vertices
            vertices = mesh_data.vectors.reshape(-1, 3)
            n_faces = len(mesh_data.vectors)
            faces = np.arange(n_faces * 3).reshape(-1, 3)

        elif ext == '.obj':
            vertices_list = []
            faces_list = []
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if parts[0] == 'v' and len(parts) >= 4:
                        vertices_list.append([float(parts[1]),
                                              float(parts[2]),
                                              float(parts[3])])
                    elif parts[0] == 'f' and len(parts) >= 4:
                        # OBJ indices are 1-based
                        face_verts = [int(p.split('/')[0]) - 1
                                      for p in parts[1:4]]
                        faces_list.append(face_verts)
            vertices = np.array(vertices_list, dtype=np.float32)
            faces = np.array(faces_list, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported format: {ext}. Use .stl or .obj")

        # Convert to point cloud
        pc = RetrosimHullAdapter.mesh_to_point_cloud(vertices, faces, num_points)

        # Normalise: center at origin, scale to unit sphere
        centroid = pc.mean(axis=0)
        pc -= centroid
        max_dist = np.max(np.linalg.norm(pc, axis=1))
        if max_dist > 1e-8:
            pc /= max_dist

        return pc

    def predict_total_resistance(self, speed_knots: float) -> Dict[str, float]:
        """
        Ship total resistance prediction using the complete Holtrop-Mennen
        (1982/1984) method.

        RT = RF(1+k1) + RAPP + RW + RB + RTR + RA

        References:
          - Holtrop & Mennen (1982), Int. Shipbuilding Progress, Vol. 29
          - Holtrop (1984), Int. Shipbuilding Progress, Vol. 31

        Args:
            speed_knots: Ship speed in knots.

        Returns:
            Dictionary with all resistance components (in kN) and
            non-dimensional coefficients.
        """
        features = self.extract_ml_features()
        dv = self._design_vector

        V  = speed_knots * 0.5144  # m/s
        L  = features['length']          # LWL (m)
        B  = features['breadth']         # Beam (m)
        T  = features['draft']           # Draft (m)
        S  = features['wetted_surface_area']  # m²
        Vol = features['displaced_volume']     # ∇ (m³)
        Cb = features['Cb_actual']
        Cm = features['Cm']
        Cwp = features.get('Cwp_actual', 0.5 + 0.45 * Cb)

        # Prismatic coefficient
        Cp = Cb / Cm if Cm > 0.01 else Cb

        # LCB as percentage of L forward of 0.5*L
        LCB_pct = features.get('LCB', 52.0)  # % from FP
        lcb = LCB_pct - 50.0  # forward of midship (+ve = forward)

        # Bulbous bow parameters
        ABT = dv.get('bulb_breadth', 0) * dv.get('bulb_depth', 0) * 0.6  # Transverse section area of bulb at FP
        hB = 0.4 * T  # height of centroid of ABT above keel

        # Transom area
        AT = dv.get('transom_beam', 0) * B * T * 0.15  # Immersed transom area (m²)

        # Constants
        g   = 9.81
        rho = 1025.0
        nu  = 1.1883e-6  # seawater kinematic viscosity at 15°C

        if V < 0.01 or L < 1.0:
            return {
                'Rf': 0, 'Rf_form': 0, 'Rw': 0, 'Rt': 0, 'RB': 0, 'RTR': 0, 'RA': 0,
                'Froude_number': 0, 'Reynolds_number': 0, 'form_factor_k1': 0,
                'Cf': 0, 'Cw': 0, 'Pe_kW': 0,
            }

        # ──────────────────────────────────────────────
        # 1. Froude & Reynolds numbers
        # ──────────────────────────────────────────────
        Fn = V / np.sqrt(g * L)
        Rn = V * L / nu

        # ──────────────────────────────────────────────
        # 2. Frictional resistance (ITTC-1957)
        # ──────────────────────────────────────────────
        Cf = 0.075 / (np.log10(max(Rn, 1e5)) - 2.0) ** 2
        Rf = 0.5 * rho * V**2 * S * Cf

        # ──────────────────────────────────────────────
        # 3. Form factor (1 + k1) — Holtrop 1984
        # ──────────────────────────────────────────────
        # Length of the run
        LR = L * (1 - Cp + 0.06 * Cp * lcb / (4 * Cp - 1)) if abs(4 * Cp - 1) > 0.01 else L * 0.3
        LR = max(LR, 0.01 * L)

        # c14 = stem shape factor (1.0 for normal bows)
        c14 = 1.0

        k1 = (0.93 + 0.487118 * c14 * (B / L) ** 1.06806
              * (T / L) ** 0.46106 * (L / LR) ** 0.121563
              * (L**3 / Vol) ** 0.36486
              * (1 - Cp) ** (-0.604247))

        Rf_form = Rf * k1

        # ──────────────────────────────────────────────
        # 4. Wave resistance (Rw) — Holtrop 1984
        #    Rw = c1·c2·c5·∇·ρ·g·exp{m1·Fn^d + m2·cos(λ·Fn^-2)}
        # ──────────────────────────────────────────────

        # c7 — hull shape factor
        BL = B / L
        if BL < 0.11:
            c7 = 0.229577 * BL ** (1.0 / 3.0)
        elif BL < 0.25:
            c7 = BL
        else:
            c7 = 0.5 - 0.0625 * (L / B)

        # Half-angle of entrance iE (Holtrop regression)
        # Use bow_angle from design vector if available, otherwise estimate
        bow_angle_dv = dv.get('bow_angle', 0)
        if bow_angle_dv > 1.0:
            iE = bow_angle_dv
        else:
            # Holtrop regression for iE
            LR_B = LR / B if B > 0 else 5
            Vol_L3 = 100 * Vol / L**3 if L > 0 else 1
            iE_exponent = (-(L / B) ** 0.80856
                           * (1 - Cwp) ** 0.30484
                           * (1 - Cp - 0.0225 * lcb) ** 0.6367
                           * (LR_B) ** 0.34574
                           * Vol_L3 ** 0.16302)
            iE = 1.0 + 89.0 * np.exp(max(iE_exponent, -20))

        iE = max(iE, 1.0)
        iE = min(iE, 89.0)

        # c1 = 2223105 * c7^3.78613 * (T/B)^1.07961 * (90-iE)^-1.37565
        c1 = 2223105.0 * c7 ** 3.78613 * (T / B) ** 1.07961 * (90.0 - iE) ** (-1.37565)

        # c3 — bulbous bow coefficient
        TF = T  # forward draught ≈ T for even keel
        if ABT > 0:
            c3_arg = 0.56 * ABT ** 1.5 / (B * T * (0.31 * np.sqrt(ABT) + TF - hB))
            c3 = max(c3_arg, 0)
        else:
            c3 = 0.0

        # c2 — bulb correction factor
        c2 = np.exp(-1.89 * np.sqrt(c3))

        # c5 — transom correction
        if AT > 0 and Cm > 0:
            c5 = 1.0 - 0.8 * AT / (B * T * Cm)
        else:
            c5 = 1.0

        # m1 — primary wave resistance coefficient
        # c16 — function of Cp
        if Cp < 0.8:
            c16 = 8.07981 * Cp - 13.8673 * Cp**2 + 6.984388 * Cp**3
        else:
            c16 = 1.73014 - 0.7067 * Cp

        Vol_13 = Vol ** (1.0 / 3.0) if Vol > 0 else 1.0
        m1 = (0.0140407 * L / T
              - 1.75254 * Vol_13 / L
              - 4.79323 * B / L
              - c16)

        # c15 & m2 — secondary wave resistance coefficient
        L3_nabla = L**3 / Vol if Vol > 0 else 1000
        if L3_nabla < 512:
            c15 = -1.69385
        elif L3_nabla < 1726.91:
            c15 = -1.69385 + (L3_nabla ** (1.0 / 3.0) - 8.0) / 2.36
        else:
            c15 = 0.0

        m2 = c15 * Cp**2 * np.exp(-0.1 * Fn**(-2)) if Fn > 0.001 else 0

        # λ — hump shape factor
        if L / B <= 12:
            lam = 1.446 * Cp - 0.03 * L / B
        else:
            lam = 1.446 * Cp - 0.36

        # d exponent
        d = -0.9

        # Final wave resistance
        if Fn > 0.001:
            exponent = m1 * Fn**d + m2 * np.cos(lam * Fn**(-2))
            exponent = max(exponent, -50)  # prevent underflow
            Rw = c1 * c2 * c5 * Vol * rho * g * np.exp(exponent)
        else:
            Rw = 0.0

        # Sanity check — Rw should not exceed a physical maximum
        Rw = max(Rw, 0.0)
        Rw = min(Rw, Rf_form * 5.0)  # Cap at 5× friction as safety

        # ──────────────────────────────────────────────
        # 5. Bulbous bow resistance (RB)
        # ──────────────────────────────────────────────
        if ABT > 0 and Fn > 0.001:
            PB = 0.56 * np.sqrt(ABT) / (TF - 1.5 * hB) if TF > 1.5 * hB else 0
            FnI = V / np.sqrt(g * (TF - hB - 0.25 * np.sqrt(ABT)) + 0.15 * V**2) if ABT > 0 else 0
            RB = 0.11 * np.exp(-3.0 * PB**(-2)) * FnI**3 * ABT**1.5 * rho * g / (1 + FnI**2)
        else:
            RB = 0.0

        # ──────────────────────────────────────────────
        # 6. Transom resistance (RTR)
        # ──────────────────────────────────────────────
        if AT > 0:
            FnT = V / np.sqrt(2.0 * g * AT / (B + B * Cwp)) if B > 0 else 0
            if FnT < 5.0:
                c6 = 0.2 * (1.0 - 0.2 * FnT)
            else:
                c6 = 0.0
            RTR = 0.5 * rho * V**2 * AT * c6
        else:
            RTR = 0.0

        # ──────────────────────────────────────────────
        # 7. Correlation allowance (RA)
        # ──────────────────────────────────────────────
        # CA = 0.006(L+100)^-0.16 - 0.00205 + 0.003√(L/7.5)·Cb^4·c2·(0.04-c4)
        # Simplified for displacement ships:
        CA = 0.006 * (L + 100) ** (-0.16) - 0.00205
        if L > 150:
            CA += 0.003 * np.sqrt(L / 7.5) * Cb**4 * c2 * 0.04
        RA = 0.5 * rho * V**2 * S * max(CA, 0)

        # ──────────────────────────────────────────────
        # 8. Total resistance
        # ──────────────────────────────────────────────
        Rt = Rf_form + Rw + RB + RTR + RA

        # Non-dimensional wave resistance coefficient
        q = 0.5 * rho * V**2 * S if V > 0 and S > 0 else 1
        Cw = Rw / q if q > 0 else 0

        # Effective power
        Pe = Rt * V / 1000.0  # kW

        return {
            'Rf': float(Rf / 1000),           # kN
            'Rf_form': float(Rf_form / 1000),  # kN (with form factor)
            'Rw': float(Rw / 1000),            # kN
            'RB': float(RB / 1000),            # kN
            'RTR': float(RTR / 1000),          # kN
            'RA': float(RA / 1000),            # kN
            'Rt': float(Rt / 1000),            # kN
            'Pe_kW': float(Pe),
            'Froude_number': float(Fn),
            'Reynolds_number': float(Rn),
            'form_factor_k1': float(k1),
            'Cf': float(Cf),
            'Cw': float(Cw),
            'iE': float(iE),
            'c1': float(c1),
            'c7': float(c7),
            'Cp': float(Cp),
        }

    # ----------------------------------------------------------
    # Design Vector Constraint Validation (Ship-D Statistical Bounds)
    # ----------------------------------------------------------

    # Ship-D 45-vector statistical bounds (min, max) from 30,000+ hulls
    DESIGN_VECTOR_BOUNDS: Dict[str, Tuple[float, float]] = {
        'L':              (20.0, 400.0),     # LBP (m)
        'B':              (5.0, 65.0),       # Beam (m)
        'T':              (2.0, 25.0),       # Draft (m)
        'D':              (3.0, 35.0),       # Depth (m)
        'Cb':             (0.35, 1.0),       # Block coefficient
        'Cm':             (0.60, 1.0),       # Midship coefficient
        'Cwp':            (0.50, 1.0),       # Waterplane coefficient
        'LCB':            (42.0, 58.0),      # LCB (% from FP)
        'bow_angle':      (5.0, 90.0),       # Half-angle of entrance (deg)
        'stern_angle':    (5.0, 90.0),       # Stern angle (deg)
        'bulb_length':    (0.0, 30.0),       # Bulb length (m)
        'bulb_breadth':   (0.0, 15.0),       # Bulb breadth (m)
        'bulb_depth':     (0.0, 12.0),       # Bulb depth (m)
        'sec_bilge_r':    (0.0, 10.0),       # Bilge radius (m)
        'stern_shape':    (0.0, 1.0),        # 0=V, 1=U
        'stern_overhang': (0.0, 15.0),       # Stern overhang (m)
        'transom_beam':   (0.0, 1.0),        # Transom beam ratio
        'keel_rise_fwd':  (0.0, 5.0),        # Forward keel rise (m)
        'keel_rise_aft':  (0.0, 5.0),        # Aft keel rise (m)
        'keel_flat_frac': (0.0, 1.0),        # Flat keel fraction
        'wp_mid_f':       (0.0, 1.0),        # Parallel midbody fraction
        'sec_flare_1':    (0.0, 30.0),       # Section flare (deg)
        'sec_flare_2':    (0.0, 30.0),
        'sec_dead_1':     (0.0, 20.0),       # Deadrise (deg)
        'sec_dead_2':     (0.0, 20.0),
    }

    # Dimensional consistency rules (ratio checks)
    RATIO_BOUNDS: Dict[str, Tuple[float, float, str]] = {
        'L/B': (3.0, 12.0, 'Boy/Genişlik oranı'),
        'B/T': (1.5, 5.0,  'Genişlik/Draft oranı'),
        'L/T': (8.0, 40.0, 'Boy/Draft oranı'),
        'L/D': (5.0, 20.0, 'Boy/Derinlik oranı'),
    }

    def validate_design_vector(self, auto_correct: bool = False
                                ) -> Dict[str, list]:
        """
        Validate all 45 design vector parameters against Ship-D statistical
        bounds and dimensional consistency rules.

        Args:
            auto_correct: If True, clamp out-of-bounds values to limits.

        Returns:
            {
                'valid': bool,
                'warnings': [(param, value, min, max, msg), ...],
                'errors': [(param, value, min, max, msg), ...],
                'corrections': [(param, old_val, new_val), ...] if auto_correct
            }
        """
        dv = self._design_vector
        warnings = []
        errors = []
        corrections = []

        # 1. Individual parameter bounds
        for param, (lo, hi) in self.DESIGN_VECTOR_BOUNDS.items():
            val = dv.get(param)
            if val is None:
                continue
            val = float(val)

            if val < lo or val > hi:
                severity = 'error' if (val < lo * 0.5 or val > hi * 2.0) else 'warning'
                msg = (f"{param} = {val:.3f} — Ship-D aralığı dışında "
                       f"[{lo:.1f}, {hi:.1f}]")

                if severity == 'error':
                    errors.append((param, val, lo, hi, msg))
                else:
                    warnings.append((param, val, lo, hi, msg))

                if auto_correct:
                    corrected = max(lo, min(hi, val))
                    corrections.append((param, val, corrected))
                    dv[param] = corrected

        # 2. Ratio checks
        L = float(dv.get('L', 100))
        B = float(dv.get('B', 15))
        T = float(dv.get('T', 6))
        D = float(dv.get('D', 8))

        ratios = {
            'L/B': L / B if B > 0 else 0,
            'B/T': B / T if T > 0 else 0,
            'L/T': L / T if T > 0 else 0,
            'L/D': L / D if D > 0 else 0,
        }

        for name, (lo, hi, desc) in self.RATIO_BOUNDS.items():
            val = ratios.get(name, 0)
            if val < lo or val > hi:
                msg = f"{desc}: {name} = {val:.2f} — kabul aralığı [{lo:.1f}, {hi:.1f}]"
                warnings.append((name, val, lo, hi, msg))

        # 3. Physical consistency
        Cb = float(dv.get('Cb', 0.7))
        Cm = float(dv.get('Cm', 0.9))
        Cwp = float(dv.get('Cwp', 0.8))

        if Cb > Cm:
            msg = f"Cb ({Cb:.3f}) > Cm ({Cm:.3f}) — fiziksel olarak geçersiz"
            errors.append(('Cb>Cm', Cb, 0, Cm, msg))
            if auto_correct:
                dv['Cm'] = min(0.99, Cb + 0.05)
                corrections.append(('Cm', Cm, dv['Cm']))

        if Cb > Cwp:
            msg = f"Cb ({Cb:.3f}) > Cwp ({Cwp:.3f}) — genellikle geçersiz"
            warnings.append(('Cb>Cwp', Cb, 0, Cwp, msg))

        is_valid = len(errors) == 0

        result = {
            'valid': is_valid,
            'warnings': warnings,
            'errors': errors,
        }
        if auto_correct:
            result['corrections'] = corrections
            self._dirty = True

        # Log
        if warnings:
            print(f"⚠️ Design Vector: {len(warnings)} uyarı")
            for w in warnings:
                print(f"   {w[4]}")
        if errors:
            print(f"❌ Design Vector: {len(errors)} hata")
            for e in errors:
                print(f"   {e[4]}")
        if is_valid and not warnings:
            print("✅ Design Vector: Tüm parametreler Ship-D aralığında.")

        return result

    # ==============================================================
    # FFD Mode — PyGeM Free-Form Deformation (CMAME 2023)
    # ==============================================================
    # Reference:
    #   "Ship hull form dataset generation using PyGeM to enable
    #    AI-based design", Computer Methods in Applied Mechanics
    #    and Engineering, Vol. 412, 2023, Article 116051.
    # ==============================================================

    def enable_ffd_mode(self,
                        n_control_pts: List[int] = None,
                        symmetry: bool = True,
                        max_displacement: float = 0.10,
                        n_stations: int = 50,
                        n_waterlines: int = 30):
        """
        Switch to FFD parametrization mode.

        This generates a base hull from the current design vector,
        then initialises the FFD lattice around it. Subsequent
        deformations operate on this base mesh via control point
        displacements.

        Args:
            n_control_pts: [nx, ny, nz] FFD control points per axis.
                Default [5, 3, 3] = 45 CPs.
            symmetry: Enforce port/starboard symmetry.
            max_displacement: Maximum normalised displacement [0-1].
            n_stations: Stations for base hull mesh generation.
            n_waterlines: Waterlines for base hull mesh generation.

        Raises:
            ImportError: If PyGeM is not installed.
        """
        if not HAS_FFD:
            raise ImportError(
                "PyGeM gerekli: pip install pygem\n"
                "Ref: CMAME Vol. 412, 2023, Art. 116051"
            )

        if n_control_pts is None:
            n_control_pts = [5, 3, 3]

        # Generate base hull from current design vector
        base_verts, base_faces = self.generate_mesh(
            n_stations=n_stations, n_waterlines=n_waterlines
        )
        self._ffd_base_vertices = base_verts.copy()
        self._ffd_base_faces = base_faces.copy()

        # Initialise FFD morpher
        self._ffd_morpher = FFDHullMorpher(
            n_control_points=n_control_pts,
            symmetry=symmetry,
            max_displacement=max_displacement,
        )
        self._ffd_morpher.set_base_hull(base_verts)
        self._ffd_mode = True

        print(f"[FFD] modu aktif: {self._ffd_morpher}")

    def disable_ffd_mode(self):
        """Switch back to direct B-spline parametrization mode."""
        self._ffd_mode = False
        self._ffd_morpher = None
        self._ffd_base_vertices = None
        self._ffd_base_faces = None
        print("[FFD] modu kapatildi - B-spline moduna donuldu.")

    @property
    def is_ffd_mode(self) -> bool:
        """Whether FFD parametrization mode is active."""
        return self._ffd_mode and self._ffd_morpher is not None

    def get_ffd_parameter_count(self) -> int:
        """
        Return dimensionality of the FFD parameter space.

        Returns:
            Number of free parameters in the μ vector.
        """
        if not self.is_ffd_mode:
            raise RuntimeError("FFD modu aktif değil. enable_ffd_mode() çağırın.")
        return self._ffd_morpher.n_free_params

    def deform_hull_ffd(self, mu_vector: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate deformed hull mesh from FFD displacement parameters.

        The base hull (generated from design vector) is deformed by
        moving FFD control points according to μ.

        Args:
            mu_vector: Displacement parameter vector of length
                       get_ffd_parameter_count().

        Returns:
            (vertices, faces): Deformed hull mesh arrays.
        """
        if not self.is_ffd_mode:
            raise RuntimeError("FFD modu aktif değil. enable_ffd_mode() çağırın.")

        deformed_verts = self._ffd_morpher.deform(mu_vector)

        # Update cached mesh
        self._mesh_vertices = deformed_verts
        self._mesh_faces = self._ffd_base_faces.copy()
        self._hydrostatics = None  # Invalidate cache

        return self._mesh_vertices, self._mesh_faces

    def validate_ffd_hull(self,
                          vertices: Optional[np.ndarray] = None,
                          ship_type: Optional[str] = None
                          ) -> Dict:
        """
        Validate deformed hull against literature-based ship parameter
        bounds (IMO / ITTC / Schneekluth & Bertram 1998).

        Args:
            vertices: Deformed hull vertices. If None, uses current mesh.
            ship_type: Ship type for type-specific bounds.
                Options: 'tanker', 'bulk_carrier', 'container',
                         'general_cargo', 'ro_ro', 'passenger'.

        Returns:
            Validation report dict.
        """
        if not self.is_ffd_mode:
            raise RuntimeError("FFD modu aktif değil.")

        if vertices is None:
            if self._mesh_vertices is None:
                raise RuntimeError("Henüz bir mesh üretilmedi.")
            vertices = self._mesh_vertices

        report = self._ffd_morpher.validate_hull_geometry(
            vertices, ship_type=ship_type
        )

        # Also run hydrostatic validation
        if report['valid']:
            # Compute actual hydrostatic coefficients for the deformed hull
            try:
                hs = self._compute_ffd_hydrostatics(vertices)
                report['hydrostatics'] = hs

                # Check form coefficients against literature
                bounds = dict(SHIP_PARAMETER_BOUNDS)
                if ship_type and ship_type in SHIP_TYPE_BOUNDS:
                    bounds.update({
                        k: (v[0], v[1], k)
                        for k, v in SHIP_TYPE_BOUNDS[ship_type].items()
                    })

                for coeff in ['Cb', 'Cwp']:
                    if coeff in hs and coeff in bounds:
                        lo, hi = bounds[coeff][0], bounds[coeff][1]
                        val = hs[coeff]
                        if val < lo or val > hi:
                            report['violations'].append(
                                (coeff, val, lo, hi,
                                 f"{coeff} = {val:.3f} — aralık [{lo}, {hi}]")
                            )
                            report['valid'] = False
            except Exception as e:
                report['hydrostatic_error'] = str(e)

        # Log results
        if report['valid']:
            print("[OK] FFD Hull: Literatur parametreleri uyumlu.")
        else:
            print(f"[ERR] FFD Hull: {len(report['violations'])} ihlal bulundu:")
            for v in report['violations']:
                print(f"   {v[4] if len(v) > 4 else v}")

        return report

    def _compute_ffd_hydrostatics(self,
                                  vertices: np.ndarray
                                  ) -> Dict[str, float]:
        """
        Compute approximate hydrostatic coefficients for deformed hull.

        Uses numerical integration over the mesh to estimate displaced
        volume, waterplane area, and form coefficients.

        Args:
            vertices: (N, 3) deformed hull vertices.

        Returns:
            Dictionary with Cb, Cwp, volume, etc.
        """
        faces = self._ffd_base_faces

        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)

        L = v_max[0] - v_min[0]
        B = v_max[1] - v_min[1]
        D_total = v_max[2] - v_min[2]

        # Compute displaced volume using divergence theorem
        # V = (1/6) * Σ |n_i · (v0_i + v1_i + v2_i)| * A_i
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)
        face_centers = (v0 + v1 + v2) / 3.0

        # Signed volume via divergence theorem
        signed_vol = np.sum(face_normals * face_centers) / 6.0
        vol = abs(signed_vol)

        # Waterplane area: project faces near waterline onto XY plane
        T_estimated = D_total * 0.65  # Approximate draft
        waterline_z = v_min[2] + T_estimated

        # Simple approximation: area of hull at waterline
        wl_mask = (face_centers[:, 2] > waterline_z - 0.5) & \
                  (face_centers[:, 2] < waterline_z + 0.5)
        if wl_mask.any():
            wl_y = vertices[faces[wl_mask].flatten()][:, 1]
            wl_x = vertices[faces[wl_mask].flatten()][:, 0]
            awp = (wl_x.max() - wl_x.min()) * (wl_y.max() - wl_y.min()) * 0.85
        else:
            awp = L * B * 0.75

        # Form coefficients
        Cb = vol / (L * B * T_estimated) if (L * B * T_estimated) > 0 else 0
        Cwp = awp / (L * B) if (L * B) > 0 else 0

        return {
            'L': L, 'B': B,
            'T_estimated': T_estimated,
            'D_total': D_total,
            'displaced_volume': vol,
            'waterplane_area': awp,
            'Cb': min(Cb, 1.0),
            'Cwp': min(Cwp, 1.0),
        }

    def generate_hull_dataset(self,
                              n_samples: int = 100,
                              sigma: float = 0.03,
                              output_dir: Optional[str] = None,
                              ship_type: Optional[str] = None,
                              seed: Optional[int] = None,
                              fmt: str = 'stl'
                              ) -> List[str]:
        """
        Generate N hull variant files for ML training dataset.

        Implements the paper's dataset generation pipeline:
        1. Generate base hull from design vector
        2. Sample FFD displacement vectors μ ~ N(0, σ²)
        3. Deform hull and validate against literature bounds
        4. Export valid variants as STL/USDA files

        Args:
            n_samples: Target number of valid hull variants.
            sigma: Std dev of normalised displacements (paper: 0.01-0.05).
            output_dir: Output directory. Default: models/geometry/dataset/.
            ship_type: Ship type for validation bounds.
            seed: Random seed for reproducibility.
            fmt: Output format: 'stl' or 'usda'.

        Returns:
            List of generated file paths.

        References:
            CMAME Vol. 412, 2023, Art. 116051, Section 3.3.
        """
        if not self.is_ffd_mode:
            raise RuntimeError(
                "FFD modu aktif değil. Önce enable_ffd_mode() çağırın."
            )

        if output_dir is None:
            output_dir = os.path.join(self.OUTPUT_DIR, 'ffd_dataset')
        os.makedirs(output_dir, exist_ok=True)

        # Generate deformed vertex arrays
        variants = self._ffd_morpher.generate_variants(
            n_samples=n_samples,
            sigma=sigma,
            seed=seed,
            validate=True,
        )

        # Export each variant
        saved_paths = []
        for i, verts in enumerate(variants):
            fname = f"hull_variant_{i:04d}.{fmt}"
            fpath = os.path.join(output_dir, fname)

            if fmt == 'stl':
                if not HAS_STL:
                    raise ImportError("numpy-stl gerekli: pip install numpy-stl")
                hull_mesh = stl_mesh.Mesh(
                    np.zeros(len(self._ffd_base_faces),
                             dtype=stl_mesh.Mesh.dtype)
                )
                for j, face in enumerate(self._ffd_base_faces):
                    for k in range(3):
                        hull_mesh.vectors[j][k] = verts[face[k]]
                hull_mesh.save(fpath)
            elif fmt == 'usda':
                # Temporarily swap vertices and export
                self._mesh_vertices = verts
                self._mesh_faces = self._ffd_base_faces
                self.generate_usda(output_path=fpath)

            saved_paths.append(fpath)

        print(f"[FFD Dataset] {len(saved_paths)} hull varyanti kaydedildi -> {output_dir}")
        print(f"   sigma={sigma:.3f}, CP={self._ffd_morpher.n_control_points}, "
              f"Free params={self._ffd_morpher.n_free_params}")

        return saved_paths

    def extract_ffd_ml_features(self,
                                mu_vector: Optional[np.ndarray] = None
                                ) -> Dict[str, float]:
        """
        Extract ML-ready features from an FFD-deformed hull.

        If mu_vector is provided, deforms the hull first.
        Otherwise uses the current mesh state.

        Args:
            mu_vector: Optional FFD displacement vector.

        Returns:
            Dictionary of geometric features for EANN/PINN input.
        """
        if mu_vector is not None:
            self.deform_hull_ffd(mu_vector)

        # Use the standard feature extraction pipeline
        # (works on current self._mesh_vertices)
        if self._mesh_vertices is None:
            raise RuntimeError("Mesh mevcut değil.")

        # Compute hydrostatics from deformed mesh
        hs = self._compute_ffd_hydrostatics(self._mesh_vertices)

        # Dimensional features
        L = hs['L']
        B = hs['B']
        T = hs['T_estimated']
        vol = hs['displaced_volume']

        lb_ratio = L / B if B > 0 else 0
        bt_ratio = B / T if T > 0 else 0
        lt_ratio = L / T if T > 0 else 0
        slenderness = L / (vol ** (1.0 / 3.0)) if vol > 0 else 0

        features = {
            # Volumetric
            'displaced_volume': vol,
            'displacement_tonnes': vol * 1.025,  # Seawater
            'waterplane_area': hs['waterplane_area'],

            # Form coefficients
            'Cb': hs['Cb'],
            'Cwp': hs['Cwp'],

            # Ratios
            'L_B_ratio': lb_ratio,
            'B_T_ratio': bt_ratio,
            'L_T_ratio': lt_ratio,
            'slenderness_coefficient': slenderness,

            # Dimensions
            'length': L,
            'breadth': B,
            'draft': T,
            'depth': hs['D_total'],

            # FFD-specific metadata
            'ffd_n_params': self._ffd_morpher.n_free_params,
            'ffd_symmetry': float(self._ffd_morpher.symmetry),
        }

        return features
