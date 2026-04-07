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
        Generate a 3D hull surface mesh.

        Args:
            n_stations: Number of longitudinal stations.
            n_waterlines: Number of vertical waterlines per section.
            include_bulb: Whether to include bulbous bow geometry.

        Returns:
            (vertices, faces): NumPy arrays for mesh vertices (Nx3) and
                               triangular faces (Mx3, 0-indexed).
        """
        L = self.dv['L']
        T = self.dv['T']

        x_stations = np.linspace(0, 1, n_stations)
        z_waterlines = np.linspace(0, 1, n_waterlines)

        # Port side vertices
        vertices = []
        for i, x_n in enumerate(x_stations):
            x_m = x_n * L  # Physical x coordinate

            # Keel elevation at this station
            keel_z = float(self.keel_profile(np.array([x_n]))[0])

            # Section half-breadths
            y_half = self.section_halfbreadth(z_waterlines, x_n)

            for j, z_n in enumerate(z_waterlines):
                z_m = keel_z + z_n * T  # Physical z
                y_m = float(y_half[j])  # Physical y (port side)
                vertices.append([x_m, y_m, z_m])

        vertices = np.array(vertices, dtype=np.float32)

        # Generate triangular faces
        faces = []
        for i in range(n_stations - 1):
            for j in range(n_waterlines - 1):
                # Vertex indices
                v00 = i * n_waterlines + j
                v10 = (i + 1) * n_waterlines + j
                v01 = i * n_waterlines + (j + 1)
                v11 = (i + 1) * n_waterlines + (j + 1)

                # Two triangles per quad
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])

        faces = np.array(faces, dtype=np.int32)

        # Mirror to starboard side
        port_verts = vertices.copy()
        starboard_verts = vertices.copy()
        starboard_verts[:, 1] *= -1  # Mirror Y

        n_port = len(port_verts)
        starboard_faces = faces.copy() + n_port
        # Reverse winding for starboard
        starboard_faces = starboard_faces[:, ::-1]

        all_vertices = np.vstack([port_verts, starboard_verts])
        all_faces = np.vstack([faces, starboard_faces])

        # Add bulbous bow if requested
        if include_bulb and self.dv['bulb_length'] > 0:
            bulb_verts, bulb_faces = self._generate_bulb_mesh(n_stations)
            if len(bulb_verts) > 0:
                offset = len(all_vertices)
                all_vertices = np.vstack([all_vertices, bulb_verts])
                all_faces = np.vstack([all_faces, bulb_faces + offset])

        return all_vertices, all_faces

    def _generate_bulb_mesh(self, n_stations: int = 21) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mesh for bulbous bow portion."""
        bl = self.dv['bulb_length']
        L = self.dv['L']

        if bl <= 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)

        bulb_start_x = 1.0 - bl / L
        n_bulb = max(5, n_stations // 3)
        n_circ = 12

        x_range = np.linspace(bulb_start_x, 1.0, n_bulb)
        vertices = []

        for x_n in x_range:
            y_pts, z_pts = self.bulb_section(x_n, n_circ)
            x_m = x_n * L
            for k in range(n_circ):
                vertices.append([x_m, float(y_pts[k]), float(z_pts[k])])

        if len(vertices) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)

        vertices = np.array(vertices, dtype=np.float32)

        # Faces
        faces = []
        for i in range(n_bulb - 1):
            for j in range(n_circ - 1):
                v00 = i * n_circ + j
                v10 = (i + 1) * n_circ + j
                v01 = i * n_circ + (j + 1)
                v11 = (i + 1) * n_circ + (j + 1)
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])

        return vertices, np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)

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
