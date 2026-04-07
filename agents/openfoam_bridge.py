"""
OpenFOAM Data Bridge
=====================

Reads OpenFOAM simulation outputs (postProcessing directories, VTK files,
or CSV force coefficients) and assembles them into (Design Vector, flow field)
tensor pairs for training the FNO surrogate model.

Supported data sources:
  1. OpenFOAM `postProcessing/` — forces, forceCoeffs, fieldMinMax
  2. VTK surface exports (.vtk) — ASCII legacy format (no VTK lib needed)
  3. CSV summary files — tabulated hull resistance data

Usage:
    bridge = OpenFOAMBridge()
    dataset = bridge.build_dataset(
        case_dirs=['run_001/', 'run_002/', ...],
        design_vectors=[dv1, dv2, ...],
    )
    # dataset → {'design_vectors': (N,46), 'flow_fields': (N,4,H,W)}
"""

import os
import re
import numpy as np
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# VTK Parser (Pure NumPy — no VTK library required)
# ═══════════════════════════════════════════════════════════════════════════════

class VTKFieldParser:
    """
    Parses ASCII legacy VTK files exported from OpenFOAM.

    Supports:
      - STRUCTURED_GRID and UNSTRUCTURED_GRID datasets
      - POINT_DATA: SCALARS (p) and VECTORS (U)
      - Single time-step files

    Example VTK header:
      # vtk DataFile Version 2.0
      OpenFOAM surface data
      ASCII
      DATASET STRUCTURED_GRID
    """

    @staticmethod
    def parse(filepath: str) -> Dict[str, np.ndarray]:
        """
        Parse a VTK file into a dictionary of numpy arrays.

        Returns:
            {
                'points': (N, 3),
                'U': (N, 3) or None,
                'p': (N,) or None,
            }
        """
        result: Dict[str, np.ndarray] = {}

        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        i = 0
        n_points = 0

        while i < len(lines):
            line = lines[i].strip()

            # Parse POINTS
            if line.startswith('POINTS'):
                parts = line.split()
                n_points = int(parts[1])
                pts = []
                i += 1
                while len(pts) < n_points * 3 and i < len(lines):
                    pts.extend(lines[i].split())
                    i += 1
                result['points'] = np.array(pts[:n_points * 3],
                                            dtype=np.float32).reshape(n_points, 3)
                continue

            # Parse SCALARS (e.g. pressure)
            if line.startswith('SCALARS'):
                name = line.split()[1]
                i += 1  # skip LOOKUP_TABLE line
                if i < len(lines) and 'LOOKUP_TABLE' in lines[i]:
                    i += 1
                vals = []
                while len(vals) < n_points and i < len(lines):
                    row = lines[i].strip()
                    if row and not row.startswith(('SCALARS', 'VECTORS', 'CELL_DATA', 'POINT_DATA')):
                        vals.extend(row.split())
                    else:
                        break
                    i += 1
                result[name] = np.array(vals[:n_points], dtype=np.float32)
                continue

            # Parse VECTORS (e.g. velocity)
            if line.startswith('VECTORS'):
                name = line.split()[1]
                i += 1
                vals = []
                while len(vals) < n_points * 3 and i < len(lines):
                    row = lines[i].strip()
                    if row and not row.startswith(('SCALARS', 'VECTORS', 'CELL_DATA', 'POINT_DATA')):
                        vals.extend(row.split())
                    else:
                        break
                    i += 1
                result[name] = np.array(vals[:n_points * 3],
                                        dtype=np.float32).reshape(n_points, 3)
                continue

            i += 1

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Force Coefficients Reader
# ═══════════════════════════════════════════════════════════════════════════════

class ForceCoeffsReader:
    """
    Reads OpenFOAM `forceCoeffs.dat` or `forces.dat` files.

    Expected tab-separated columns:
      Time  Cd  Cl  Cm  ...
    or:
      Time  (Fx Fy Fz)  ...
    """

    @staticmethod
    def read(filepath: str) -> Dict[str, np.ndarray]:
        """Parse force coefficients file."""
        data_lines = []
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('//'):
                    continue
                # Remove parentheses (OpenFOAM vector notation)
                line = line.replace('(', ' ').replace(')', ' ')
                parts = line.split()
                try:
                    vals = [float(x) for x in parts]
                    data_lines.append(vals)
                except ValueError:
                    continue

        if not data_lines:
            return {}

        arr = np.array(data_lines, dtype=np.float32)
        result = {'time': arr[:, 0]}

        # Standard forceCoeffs columns
        col_names = ['Cd', 'Cl', 'CmPitch', 'CdVis', 'ClVis', 'CmPitchVis',
                     'CdPres', 'ClPres', 'CmPitchPres']
        n_data_cols = arr.shape[1] - 1
        for j in range(min(n_data_cols, len(col_names))):
            result[col_names[j]] = arr[:, j + 1]

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Builder
# ═══════════════════════════════════════════════════════════════════════════════

class OpenFOAMBridge:
    """
    Builds FNO-compatible datasets from OpenFOAM case directories.

    Each OpenFOAM case should have:
      - postProcessing/forceCoeffs/0/forceCoeffs.dat  (optional)
      - VTK/surfaces/     (optional)
      - A corresponding 45-param Design Vector

    The bridge interpolates unstructured VTK data onto a regular grid
    to match the FNO's fixed spatial resolution.
    """

    def __init__(self, grid_h: int = 64, grid_w: int = 64):
        self.grid_h = grid_h
        self.grid_w = grid_w

    def build_dataset(self, case_dirs: List[str],
                      design_vectors: List[np.ndarray],
                      speeds: Optional[List[float]] = None,
                      ) -> Dict[str, np.ndarray]:
        """
        Build training dataset from a list of OpenFOAM cases.

        Args:
            case_dirs: List of paths to OpenFOAM case root directories.
            design_vectors: Corresponding 45-param numpy arrays.
            speeds: Ship speeds (knots) for each case. Defaults to 12.

        Returns:
            {
                'design_vectors': (N, 46),  — 45 params + speed
                'flow_fields':    (N, 4, H, W),
                'forces':         (N, dict)  — Cd, Cl etc.
            }
        """
        all_dv = []
        all_ff = []
        all_forces = []

        if speeds is None:
            speeds = [12.0] * len(case_dirs)

        for idx, (case_dir, dv, speed) in enumerate(
                zip(case_dirs, design_vectors, speeds)):

            # Append speed to design vector
            dv_full = np.zeros(46, dtype=np.float32)
            dv_full[:min(len(dv), 45)] = dv[:45]
            dv_full[45] = speed

            # Try to load VTK flow field
            flow_field = self._load_flow_field(case_dir)

            # Try to load force coefficients
            forces = self._load_forces(case_dir)

            if flow_field is not None:
                all_dv.append(dv_full)
                all_ff.append(flow_field)
                all_forces.append(forces)
                print(f"  ✅ Case {idx+1}/{len(case_dirs)}: {case_dir}")
            else:
                print(f"  ⚠️ Case {idx+1}/{len(case_dirs)}: Skipped (no flow data)")

        if not all_dv:
            print("⚠️ No valid cases found. Returning empty dataset.")
            return {
                'design_vectors': np.zeros((0, 46), dtype=np.float32),
                'flow_fields':    np.zeros((0, 4, self.grid_h, self.grid_w), dtype=np.float32),
            }

        return {
            'design_vectors': np.stack(all_dv),
            'flow_fields':    np.stack(all_ff),
            'forces':         all_forces,
        }

    def _load_flow_field(self, case_dir: str) -> Optional[np.ndarray]:
        """
        Load and grid-interpolate flow field from VTK exports.

        Looks for the latest time directory under VTK/ or postProcessing/.
        """
        vtk_dirs = [
            os.path.join(case_dir, 'VTK'),
            os.path.join(case_dir, 'postProcessing', 'surfaces'),
        ]

        vtk_file = None
        for vdir in vtk_dirs:
            if not os.path.isdir(vdir):
                continue
            # Find latest .vtk file
            for root, dirs, files in os.walk(vdir):
                for fname in files:
                    if fname.endswith('.vtk'):
                        vtk_file = os.path.join(root, fname)

        if vtk_file is None:
            return None

        try:
            data = VTKFieldParser.parse(vtk_file)
            return self._interpolate_to_grid(data)
        except Exception as e:
            print(f"    VTK parse error: {e}")
            return None

    def _interpolate_to_grid(self, vtk_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Interpolate scattered VTK points onto a regular (H, W) grid.

        Returns: (4, H, W) array with channels [u, v, w, p].
        """
        H, W = self.grid_h, self.grid_w
        flow = np.zeros((4, H, W), dtype=np.float32)

        pts = vtk_data.get('points')
        if pts is None:
            return flow

        x, y = pts[:, 0], pts[:, 1]

        # Grid bounds
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        if x_max - x_min < 1e-6 or y_max - y_min < 1e-6:
            return flow

        # Bin indices
        xi = np.clip(((x - x_min) / (x_max - x_min) * (W - 1)).astype(int), 0, W - 1)
        yi = np.clip(((y - y_min) / (y_max - y_min) * (H - 1)).astype(int), 0, H - 1)

        count = np.zeros((H, W), dtype=np.float32)

        # Velocity
        U = vtk_data.get('U')
        if U is not None and len(U) == len(pts):
            for k in range(len(pts)):
                flow[0, yi[k], xi[k]] += U[k, 0]  # u
                flow[1, yi[k], xi[k]] += U[k, 1]  # v
                flow[2, yi[k], xi[k]] += U[k, 2]  # w
                count[yi[k], xi[k]] += 1

        # Pressure
        p = vtk_data.get('p')
        if p is not None and len(p) == len(pts):
            for k in range(len(pts)):
                flow[3, yi[k], xi[k]] += p[k]

        # Average overlapping points
        count[count == 0] = 1
        flow[0] /= count
        flow[1] /= count
        flow[2] /= count
        flow[3] /= count

        return flow

    def _load_forces(self, case_dir: str) -> Dict:
        """Load force coefficient data from postProcessing."""
        search_paths = [
            os.path.join(case_dir, 'postProcessing', 'forceCoeffs', '0', 'forceCoeffs.dat'),
            os.path.join(case_dir, 'postProcessing', 'forces', '0', 'forces.dat'),
        ]
        for fpath in search_paths:
            if os.path.exists(fpath):
                try:
                    return ForceCoeffsReader.read(fpath)
                except Exception:
                    pass
        return {}

    def build_from_csv(self, csv_path: str) -> Dict[str, np.ndarray]:
        """
        Build dataset from a single CSV containing all cases.

        Expected columns: L, B, T, Cb, speed, Cd, Cl, Cw, Ct, ...
        Each row is a different hull variant.
        """
        import pandas as pd
        df = pd.read_csv(csv_path)

        n = len(df)
        dv_all = np.zeros((n, 46), dtype=np.float32)
        ff_all = np.zeros((n, 4, self.grid_h, self.grid_w), dtype=np.float32)

        # Map CSV columns to design vector slots
        col_map = {'L': 0, 'B': 1, 'T': 2, 'Cb': 3, 'Cm': 4, 'Cwp': 5}

        for col, idx in col_map.items():
            if col in df.columns:
                dv_all[:, idx] = df[col].values

        if 'speed' in df.columns:
            dv_all[:, 45] = df['speed'].values
        else:
            dv_all[:, 45] = 12.0

        # Generate flow fields from coefficients (analytical proxy)
        H, W = self.grid_h, self.grid_w
        x = np.linspace(-2, 3, W)
        y = np.linspace(-1.5, 1.5, H)
        X, Y = np.meshgrid(x, y)

        for i in range(n):
            speed = dv_all[i, 45]
            Cb = dv_all[i, 3] if dv_all[i, 3] > 0 else 0.7

            r = np.sqrt(X ** 2 + Y ** 2 + 1e-6)
            a = 0.5 + Cb * 0.3
            b = 0.15
            theta = np.arctan2(Y, X)

            u = speed * (1 - (a * b / r ** 2) * np.cos(2 * theta))
            v = -speed * (a * b / r ** 2) * np.sin(2 * theta)

            mask = ((X / a) ** 2 + (Y / b) ** 2) > 1
            ff_all[i, 0] = u * mask
            ff_all[i, 1] = v * mask
            ff_all[i, 3] = -0.5 * (u ** 2 + v ** 2) * mask

        return {
            'design_vectors': dv_all,
            'flow_fields': ff_all,
        }
