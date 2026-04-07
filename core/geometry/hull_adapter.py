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

from core.geometry.third_party.HullParameterization import (
    HullParameterization,
    get_default_design_vector,
    DESIGN_VECTOR_KEYS,
    N_DESIGN_PARAMS,
)

# Optional: numpy-stl for STL export
try:
    from stl import mesh as stl_mesh
    HAS_STL = True
except ImportError:
    HAS_STL = False


class RetrosimHullAdapter:
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

    def extract_point_cloud(self, num_points: int = 2048) -> np.ndarray:
        """
        Extract a Point Cloud representation of the hull surface for PointNet++ / GNNs.
        Samples dynamically from the generated mesh.
        
        Args:
            num_points: Target number of nodes in the N x 3 output tensor.
        
        Returns:
            np.ndarray of shape (num_points, 3) containing 3D coordinates.
        """
        if self._mesh_vertices is None or self._mesh_faces is None:
            self.generate_mesh()
            
        vertices = self._mesh_vertices
        faces = self._mesh_faces
        
        # Calculate triangle areas for uniform surface sampling
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        
        # Normalize probabilities
        prob = areas / areas.sum()
        
        # Select faces based on area
        face_indices = np.random.choice(len(faces), size=num_points, p=prob)
        
        # Generate random barycentric coordinates
        u = np.random.uniform(0, 1, size=(num_points, 1))
        v = np.random.uniform(0, 1, size=(num_points, 1))
        
        # Ensure points are inside the triangle
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        w = 1 - u - v
        
        # Compute final 3D point coordinates
        selected_v0 = vertices[faces[face_indices, 0]]
        selected_v1 = vertices[faces[face_indices, 1]]
        selected_v2 = vertices[faces[face_indices, 2]]
        
        point_cloud = (u * selected_v0) + (v * selected_v1) + (w * selected_v2)
        return point_cloud.astype(np.float32)

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
