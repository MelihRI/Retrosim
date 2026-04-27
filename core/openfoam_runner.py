"""
OpenFOAM Runner -- Automated simpleFoam Pipeline for FNO Training
==================================================================
Runs steady incompressible RANS (simpleFoam) on a hull STL and
interpolates the result onto the fixed (64,128,64) FNO grid.

Architecture position:
  GeometryAssembler.build() -> STL -> OpenFOAMRunner.run_case() -> (4,D,H,W) ground-truth

License: Apache 2.0
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import textwrap
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# Fixed FNO grid (matches geometry_assembler.py / sdf_utils.py)
GRID_D, GRID_H, GRID_W = 64, 128, 64
X_MIN, X_MAX = -0.5, 2.0
Y_MIN, Y_MAX = -0.5, 0.5
Z_MIN, Z_MAX = -0.5, 0.3


# ============================================================================
# Abstract Base
# ============================================================================

class OpenFOAMRunner(ABC):
    """Abstract interface for running OpenFOAM and harvesting results."""

    @abstractmethod
    def run_case(
        self,
        combined_stl_path: Path,
        Re: float,
        Fr: float,
        U_inf: float,
        case_dir: Path,
    ) -> dict:
        """Execute an OpenFOAM case and return structured results.

        Returns
        -------
        dict
            "flow_field": np.ndarray (4, D, H, W) float32 -- [u,v,w,p]
            "C_T":        float -- total resistance coefficient
            "case_dir":   Path  -- path to completed case
        """
        ...


# ============================================================================
# Concrete: SimpleFoamRunner
# ============================================================================

class SimpleFoamRunner(OpenFOAMRunner):
    """Production runner: blockMesh -> snappyHexMesh -> simpleFoam.

    Generates OpenFOAM case files from templates, executes the solver,
    parses forceCoeffs for C_T, and interpolates the volumetric field
    onto the fixed (64,128,64) grid via VTK export + scipy interpolation.

    Parameters
    ----------
    template_dir : Path | None
        Path to a base OpenFOAM case template. If None, templates are
        generated programmatically.
    n_procs : int
        Number of MPI processes for parallel decomposition. 1 = serial.
    timeout : int
        Maximum wall-clock seconds for simpleFoam before SIGTERM.
    """

    # Reference area for force coefficient (non-dim): wetted surface proxy
    _A_REF: float = 1.0
    # Kinematic viscosity of seawater at 15C [m^2/s]
    _NU_SEAWATER: float = 1.19e-6

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        n_procs: int = 1,
        timeout: int = 3600,
    ):
        self.template_dir = Path(template_dir) if template_dir else None
        self.n_procs = n_procs
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def run_case(
        self,
        combined_stl_path: Path,
        Re: float,
        Fr: float,
        U_inf: float,
        case_dir: Path,
    ) -> dict:
        combined_stl_path = Path(combined_stl_path)
        case_dir = Path(case_dir)

        nu = U_inf * 1.0 / max(Re, 1.0)  # nu = U*L/Re  (L=1 non-dim)

        print(f"[OF] Setting up case: {case_dir.name}")
        print(f"     Re={Re:.2e}  Fr={Fr:.4f}  U_inf={U_inf:.3f} m/s  nu={nu:.3e}")

        # 1. Scaffold case directory
        self._setup_case(case_dir, combined_stl_path, U_inf, nu)

        # 2. Run mesh + solver
        self._run_pipeline(case_dir)

        # 3. Export to VTK
        self._export_vtk(case_dir)

        # 4. Parse results
        flow_field = self._interpolate_to_grid(case_dir)
        C_T = self._parse_force_coeffs(case_dir, U_inf)

        print(f"[OF] Done: C_T={C_T:.6f}  field range u=[{flow_field[0].min():.3f},{flow_field[0].max():.3f}]")

        return {
            "flow_field": flow_field,
            "C_T": C_T,
            "case_dir": case_dir,
        }

    # ------------------------------------------------------------------ #
    # Case Setup                                                          #
    # ------------------------------------------------------------------ #

    def _setup_case(self, case_dir: Path, stl_path: Path, U_inf: float, nu: float):
        """Create OpenFOAM directory structure and write dict files."""
        if self.template_dir and self.template_dir.exists():
            if case_dir.exists():
                shutil.rmtree(case_dir)
            shutil.copytree(self.template_dir, case_dir)
        else:
            self._generate_case_from_scratch(case_dir, U_inf, nu)

        # Copy STL into triSurface
        tri_dir = case_dir / "constant" / "triSurface"
        tri_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stl_path, tri_dir / "hull.stl")

        # Patch velocity file
        u_file = case_dir / "0" / "U"
        if u_file.exists():
            txt = u_file.read_text()
            txt = txt.replace("__U_INF__", f"({U_inf} 0 0)")
            u_file.write_text(txt)

        # Patch transportProperties
        tp = case_dir / "constant" / "transportProperties"
        if tp.exists():
            txt = tp.read_text()
            txt = txt.replace("__NU__", f"{nu:.6e}")
            tp.write_text(txt)

    def _generate_case_from_scratch(self, case_dir: Path, U_inf: float, nu: float):
        """Write minimal OpenFOAM case files programmatically."""
        for d in ["0", "constant/triSurface", "system"]:
            (case_dir / d).mkdir(parents=True, exist_ok=True)

        hdr = 'FoamFile {{ version 2.0; format ascii; class {cls}; object {obj}; }}\n'

        # -- 0/U --
        (case_dir / "0" / "U").write_text(textwrap.dedent(f"""\
            {hdr.format(cls='volVectorField', obj='U')}
            dimensions [0 1 -1 0 0 0 0];
            internalField uniform __U_INF__;
            boundaryField
            {{
                inlet  {{ type fixedValue; value uniform __U_INF__; }}
                outlet {{ type zeroGradient; }}
                hull   {{ type noSlip; }}
                top    {{ type slip; }}
                bottom {{ type slip; }}
                sides  {{ type slip; }}
            }}
        """))

        # -- 0/p --
        (case_dir / "0" / "p").write_text(textwrap.dedent(f"""\
            {hdr.format(cls='volScalarField', obj='p')}
            dimensions [0 2 -2 0 0 0 0];
            internalField uniform 0;
            boundaryField
            {{
                inlet  {{ type zeroGradient; }}
                outlet {{ type fixedValue; value uniform 0; }}
                hull   {{ type zeroGradient; }}
                top    {{ type slip; }}
                bottom {{ type slip; }}
                sides  {{ type slip; }}
            }}
        """))

        # -- constant/transportProperties --
        (case_dir / "constant" / "transportProperties").write_text(textwrap.dedent(f"""\
            {hdr.format(cls='dictionary', obj='transportProperties')}
            transportModel Newtonian;
            nu nu [0 2 -1 0 0 0 0] __NU__;
        """))

        # -- constant/turbulenceProperties --
        (case_dir / "constant" / "turbulenceProperties").write_text(textwrap.dedent(f"""\
            {hdr.format(cls='dictionary', obj='turbulenceProperties')}
            simulationType RAS;
            RAS {{ RASModel kOmegaSST; turbulence on; printCoeffs on; }}
        """))

        # -- system/controlDict --
        (case_dir / "system" / "controlDict").write_text(textwrap.dedent(f"""\
            {hdr.format(cls='dictionary', obj='controlDict')}
            application simpleFoam;
            startFrom startTime;
            startTime 0;
            stopAt endTime;
            endTime 2000;
            deltaT 1;
            writeControl timeStep;
            writeInterval 500;
            purgeWrite 2;
            writeFormat ascii;
            writePrecision 8;
            functions
            {{
                forceCoeffs
                {{
                    type forceCoeffs;
                    libs ("libforces.so");
                    writeControl timeStep;
                    writeInterval 1;
                    patches (hull);
                    rho rhoInf;
                    rhoInf 1025.0;
                    liftDir (0 0 1);
                    dragDir (1 0 0);
                    pitchAxis (0 1 0);
                    magUInf {U_inf};
                    lRef 1.0;
                    Aref {self._A_REF};
                }}
            }}
        """))

        # -- system/fvSchemes --
        (case_dir / "system" / "fvSchemes").write_text(textwrap.dedent(f"""\
            {hdr.format(cls='dictionary', obj='fvSchemes')}
            ddtSchemes {{ default steadyState; }}
            gradSchemes {{ default Gauss linear; }}
            divSchemes
            {{
                default none;
                div(phi,U) bounded Gauss linearUpwind grad(U);
                div(phi,k) bounded Gauss upwind;
                div(phi,omega) bounded Gauss upwind;
                div((nuEff*dev2(T(grad(U))))) Gauss linear;
            }}
            laplacianSchemes {{ default Gauss linear corrected; }}
            interpolationSchemes {{ default linear; }}
            snGradSchemes {{ default corrected; }}
        """))

        # -- system/fvSolution --
        (case_dir / "system" / "fvSolution").write_text(textwrap.dedent(f"""\
            {hdr.format(cls='dictionary', obj='fvSolution')}
            solvers
            {{
                p {{ solver GAMG; smoother GaussSeidel; tolerance 1e-6; relTol 0.01; }}
                U {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.01; }}
                k {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.01; }}
                omega {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.01; }}
            }}
            SIMPLE
            {{
                nNonOrthogonalCorrectors 1;
                consistent yes;
                residualControl {{ p 1e-4; U 1e-5; k 1e-5; omega 1e-5; }}
            }}
            relaxationFactors
            {{
                fields {{ p 0.3; }}
                equations {{ U 0.7; k 0.7; omega 0.7; }}
            }}
        """))

        # -- system/blockMeshDict --
        (case_dir / "system" / "blockMeshDict").write_text(textwrap.dedent(f"""\
            {hdr.format(cls='dictionary', obj='blockMeshDict')}
            scale 1;
            vertices
            (
                ({X_MIN} {Y_MIN} {Z_MIN})
                ({X_MAX} {Y_MIN} {Z_MIN})
                ({X_MAX} {Y_MAX} {Z_MIN})
                ({X_MIN} {Y_MAX} {Z_MIN})
                ({X_MIN} {Y_MIN} {Z_MAX})
                ({X_MAX} {Y_MIN} {Z_MAX})
                ({X_MAX} {Y_MAX} {Z_MAX})
                ({X_MIN} {Y_MAX} {Z_MAX})
            );
            blocks ( hex (0 1 2 3 4 5 6 7) (40 20 16) simpleGrading (1 1 1) );
            edges ();
            boundary
            (
                inlet  {{ type patch; faces ((0 4 7 3)); }}
                outlet {{ type patch; faces ((1 2 6 5)); }}
                bottom {{ type wall;  faces ((0 1 2 3)); }}
                top    {{ type wall;  faces ((4 5 6 7)); }}
                sides  {{ type wall;  faces ((0 1 5 4) (3 2 6 7)); }}
            );
        """))

        # -- system/snappyHexMeshDict --
        (case_dir / "system" / "snappyHexMeshDict").write_text(textwrap.dedent(f"""\
            {hdr.format(cls='dictionary', obj='snappyHexMeshDict')}
            castellatedMesh true;
            snap true;
            addLayers false;
            geometry
            {{
                hull.stl {{ type triSurfaceMesh; name hull; }}
            }}
            castellatedMeshControls
            {{
                maxLocalCells 100000;
                maxGlobalCells 2000000;
                minRefinementCells 10;
                maxLoadUnbalance 0.1;
                nCellsBetweenLevels 3;
                features ();
                refinementSurfaces {{ hull {{ level (3 4); patchInfo {{ type wall; }} }} }}
                refinementRegions {{}};
                locationInMesh (0.0 0.0 0.1);
                allowFreeStandingZoneFaces true;
            }}
            snapControls
            {{
                nSmoothPatch 3;
                tolerance 2.0;
                nSolveIter 30;
                nRelaxIter 5;
            }}
            addLayersControls
            {{
                relativeSizes true;
                layers {{}};
                expansionRatio 1.0;
                finalLayerThickness 0.3;
                minThickness 0.1;
                nGrow 0;
                featureAngle 60;
                nRelaxIter 3;
                nSmoothSurfaceNormals 1;
                nSmoothNormals 3;
                nSmoothThickness 10;
                maxFaceThicknessRatio 0.5;
                maxThicknessToMedialRatio 0.3;
                minMedialAxisAngle 90;
                nBufferCellsNoExtrude 0;
                nLayerIter 50;
            }}
            meshQualityControls
            {{
                maxNonOrtho 65;
                maxBoundarySkewness 20;
                maxInternalSkewness 4;
                maxConcave 80;
                minVol 1e-13;
                minTetQuality -1e30;
                minArea -1;
                minTwist 0.02;
                minDeterminant 0.001;
                minFaceWeight 0.02;
                minVolRatio 0.01;
                minTriangleTwist -1;
                nSmoothScale 4;
                errorReduction 0.75;
            }}
        """))

    # ------------------------------------------------------------------ #
    # Solver Execution                                                    #
    # ------------------------------------------------------------------ #

    def _run_pipeline(self, case_dir: Path):
        """Execute blockMesh -> snappyHexMesh -> simpleFoam."""
        steps = [
            (["blockMesh"], "blockMesh"),
            (["snappyHexMesh", "-overwrite"], "snappyHexMesh"),
            (["simpleFoam"], "simpleFoam"),
        ]
        for cmd, label in steps:
            self._exec(cmd, case_dir, label)

    def _exec(self, cmd: list, cwd: Path, label: str):
        """Run a single OpenFOAM command with timeout and logging."""
        log_path = cwd / f"log.{label}"
        print(f"[OF] Running {label} ... ", end="", flush=True)
        t0 = time.time()
        try:
            with open(log_path, "w") as log_f:
                subprocess.run(
                    cmd, cwd=str(cwd), check=True,
                    stdout=log_f, stderr=subprocess.STDOUT,
                    timeout=self.timeout,
                )
            dt = time.time() - t0
            print(f"OK ({dt:.1f}s)")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{label} exceeded {self.timeout}s timeout")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"{label} failed (exit {e.returncode}). See {log_path}"
            )

    # ------------------------------------------------------------------ #
    # VTK Export                                                          #
    # ------------------------------------------------------------------ #

    def _export_vtk(self, case_dir: Path):
        """Convert the final time-step to VTK for field extraction."""
        try:
            self._exec(["foamToVTK", "-latestTime"], case_dir, "foamToVTK")
        except RuntimeError:
            print("[OF] WARN: foamToVTK failed; will attempt internal field parsing")

    # ------------------------------------------------------------------ #
    # Field Interpolation onto FNO Grid                                   #
    # ------------------------------------------------------------------ #

    def _interpolate_to_grid(self, case_dir: Path) -> np.ndarray:
        """Read VTK and interpolate (u,v,w,p) onto the fixed (D,H,W) grid.

        Falls back to zeros if VTK parsing fails.
        """
        flow = np.zeros((4, GRID_D, GRID_H, GRID_W), dtype=np.float32)

        vtk_dir = case_dir / "VTK"
        vtk_file = self._find_latest_vtk(vtk_dir)
        if vtk_file is None:
            print("[OF] WARN: No VTK file found; returning zero field")
            return flow

        try:
            points, U, p = self._parse_vtk_internal(vtk_file)
        except Exception as e:
            print(f"[OF] WARN: VTK parse error: {e}")
            return flow

        # Build target grid
        xs = np.linspace(X_MIN, X_MAX, GRID_W, dtype=np.float64)
        ys = np.linspace(Y_MIN, Y_MAX, GRID_H, dtype=np.float64)
        zs = np.linspace(Z_MIN, Z_MAX, GRID_D, dtype=np.float64)
        gz, gy, gx = np.meshgrid(zs, ys, xs, indexing="ij")
        target = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

        # Interpolate each channel
        for ch, data in enumerate([U[:, 0], U[:, 1], U[:, 2], p]):
            try:
                interp = LinearNDInterpolator(points, data)
                vals = interp(target)
                # Fill NaN with nearest
                nans = np.isnan(vals)
                if nans.any():
                    nn = NearestNDInterpolator(points, data)
                    vals[nans] = nn(target[nans])
                flow[ch] = vals.reshape(GRID_D, GRID_H, GRID_W).astype(np.float32)
            except Exception:
                pass

        return flow

    @staticmethod
    def _find_latest_vtk(vtk_dir: Path) -> Optional[Path]:
        """Find the VTK file from the latest time directory."""
        if not vtk_dir.exists():
            return None
        vtk_files = sorted(vtk_dir.rglob("*.vtk"))
        return vtk_files[-1] if vtk_files else None

    @staticmethod
    def _parse_vtk_internal(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse ASCII legacy VTK for points, U, p.

        Returns (points(N,3), U(N,3), p(N,)).
        """
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        points = None
        U = None
        p_field = None
        i = 0
        n_pts = 0

        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("POINTS"):
                n_pts = int(line.split()[1])
                vals = []
                i += 1
                while len(vals) < n_pts * 3 and i < len(lines):
                    vals.extend(lines[i].split())
                    i += 1
                points = np.array(vals[: n_pts * 3], dtype=np.float64).reshape(n_pts, 3)
                continue

            if line.startswith("VECTORS") and "U" in line:
                vals = []
                i += 1
                while len(vals) < n_pts * 3 and i < len(lines):
                    vals.extend(lines[i].split())
                    i += 1
                U = np.array(vals[: n_pts * 3], dtype=np.float64).reshape(n_pts, 3)
                continue

            if line.startswith("SCALARS") and "p" in line.split()[1]:
                i += 1  # LOOKUP_TABLE
                if i < len(lines) and "LOOKUP_TABLE" in lines[i]:
                    i += 1
                vals = []
                while len(vals) < n_pts and i < len(lines):
                    row = lines[i].strip()
                    if row and not row.startswith(("SCALARS", "VECTORS", "CELL_DATA")):
                        vals.extend(row.split())
                    else:
                        break
                    i += 1
                p_field = np.array(vals[:n_pts], dtype=np.float64)
                continue

            i += 1

        if points is None:
            raise ValueError("No POINTS found in VTK")
        if U is None:
            U = np.zeros((n_pts, 3), dtype=np.float64)
        if p_field is None:
            p_field = np.zeros(n_pts, dtype=np.float64)

        return points, U, p_field

    # ------------------------------------------------------------------ #
    # Force Coefficient Parsing                                           #
    # ------------------------------------------------------------------ #

    def _parse_force_coeffs(self, case_dir: Path, U_inf: float) -> float:
        """Extract converged C_T from forceCoeffs.dat (last 100 iterations avg)."""
        search = [
            case_dir / "postProcessing" / "forceCoeffs" / "0" / "forceCoeffs.dat",
            case_dir / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat",
        ]
        for fpath in search:
            if fpath.exists():
                try:
                    return self._read_cd_from_file(fpath)
                except Exception as e:
                    print(f"[OF] WARN: Could not parse {fpath.name}: {e}")

        print("[OF] WARN: No forceCoeffs found; returning C_T=0")
        return 0.0

    @staticmethod
    def _read_cd_from_file(filepath: Path) -> float:
        """Read Cd column and return average of last 100 values."""
        rows = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                line = line.replace("(", " ").replace(")", " ")
                try:
                    vals = [float(x) for x in line.split()]
                    rows.append(vals)
                except ValueError:
                    continue
        if not rows:
            return 0.0
        arr = np.array(rows)
        # Cd is typically column 1 (after Time)
        cd = arr[-min(100, len(arr)):, 1]
        return float(np.mean(cd))

    # ------------------------------------------------------------------ #
    # Batch Convenience                                                   #
    # ------------------------------------------------------------------ #

    def run_batch(
        self,
        stl_paths: list,
        conditions: list,
        output_root: Path,
    ) -> list:
        """Run multiple cases and collect results.

        Parameters
        ----------
        stl_paths : list[Path]
        conditions : list[dict]
            Each dict: {"Re": float, "Fr": float, "U_inf": float}
        output_root : Path

        Returns
        -------
        list[dict]  -- one result dict per case.
        """
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        results = []

        for idx, (stl, cond) in enumerate(zip(stl_paths, conditions)):
            case_name = f"case_{idx:04d}_Re{cond['Re']:.0e}_Fr{cond['Fr']:.3f}"
            case_dir = output_root / case_name
            try:
                result = self.run_case(
                    combined_stl_path=Path(stl),
                    Re=cond["Re"], Fr=cond["Fr"], U_inf=cond["U_inf"],
                    case_dir=case_dir,
                )
                results.append(result)
            except RuntimeError as e:
                print(f"[OF] ERROR case {idx}: {e}")
                results.append(None)

        n_ok = sum(1 for r in results if r is not None)
        print(f"[OF] Batch complete: {n_ok}/{len(stl_paths)} succeeded")
        return results

    def save_dataset(self, results: list, output_path: Path):
        """Save batch results as a single .npz for FNO training.

        File contains:
          flow_fields: (N, 4, D, H, W) float32
          C_T:         (N,) float32
        """
        valid = [r for r in results if r is not None]
        if not valid:
            print("[OF] No valid results to save")
            return

        fields = np.stack([r["flow_field"] for r in valid])
        cts = np.array([r["C_T"] for r in valid], dtype=np.float32)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(output_path), flow_fields=fields, C_T=cts)
        print(f"[OF] Dataset saved: {output_path} ({len(valid)} samples)")
