"""
PhysicsNeMo Agent -- PyQt6 Integration Layer for the 3D-FNO PINN Solver
=========================================================================
Bridges the Retrosim GUI with the SDF-Conditioned 3D Fourier Neural
Operator (sdf_fno3d_solver).  All heavy PyTorch work runs on a QThread
to keep the GUI responsive.

Pipeline:
    GUI.run_analysis(vessel_params)
        -> Worker thread:
            1. RetrosimHullAdapter  -> mesh (vertices, faces)
            2. SDFGenerator         -> 3D SDF tensor from mesh
            3. FNO3d_NS_Solver      -> [u, v, w, p] volume
            4. Slice extraction     -> 2D numpy arrays for plotting
        <- finished_signal(result_dict)

Fallback:
    If the pre-trained FNO checkpoint is missing, the agent transparently
    falls back to Holtrop-Mennen empirical resistance via the hull adapter.

Author: Retrosim Team
"""

import os
import traceback
import tempfile
import numpy as np
from typing import Dict, Optional

import torch
from PyQt6.QtCore import QObject, QThread, pyqtSignal

# ---- Internal Imports (lazy to tolerate missing deps at import time) ----
# Hull geometry
try:
    from core.geometry.FFDHullMorpher import RetrosimHullAdapter
    HAS_GEOMETRY = True
except ImportError:
    HAS_GEOMETRY = False

# SDF + FNO modules (our new solver)
try:
    from agents.sdf_utils import SolverConfig, SDFGenerator
    from agents.fno3d_network import FNO3d_NS_Solver, DEVICE
    HAS_FNO3D = True
except ImportError:
    HAS_FNO3D = False

# Physical constants
RHO_WATER = 1025.0   # kg/m^3
NU_WATER  = 1.188e-6 # m^2/s  (15 C seawater)
G_GRAV    = 9.81     # m/s^2

# Model checkpoint path
MODEL_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'fno3d_pinn.pt')

# Hull assets directory
HULL_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'hulls')

# Vessel type -> STL filename mapping
VESSEL_STL_MAP = {
    'KCS':       'kcs_hull.stl',
    'Tanker':    'tanker_hull.stl',
    'FastBoat':  'fastboat_hull.stl',
    'Container': 'container_hull.stl',
    'Bulk':      'bulk_hull.stl',
}


# =====================================================================
# Worker Thread
# =====================================================================

class PhysicsNeMoWorker(QThread):
    """Runs the full SDF -> FNO -> slice pipeline off the GUI thread.

    Signals:
        progress_signal(int, str)  -- progress percentage + status text
        finished_signal(dict)      -- final result dictionary
        error_signal(str)          -- traceback string on failure
    """

    progress_signal  = pyqtSignal(int, str)
    finished_signal  = pyqtSignal(dict)
    error_signal     = pyqtSignal(str)

    def __init__(self, vessel_params: Dict, config: SolverConfig,
                 vessel_type: str = 'KCS', parent=None):
        super().__init__(parent)
        self.vessel_params = vessel_params
        self.config = config
        self.vessel_type = vessel_type

    # ---- Main entry (runs on worker thread) -------------------------

    def run(self):
        """Execute the full inference pipeline."""
        try:
            self._run_pipeline()
        except Exception:
            tb = traceback.format_exc()
            print(f"[PhysicsNeMo] ERROR:\n{tb}")
            self.error_signal.emit(tb)

    def _run_pipeline(self):
        cfg = self.config
        vp  = self.vessel_params

        # ── Step 1: Geometry ─────────────────────────────────────────
        self.progress_signal.emit(5, "Gemi geometrisi olusturuluyor...")
        print("[PhysicsNeMo] Step 1/5 -- Geometry extraction")

        adapter, features, stl_path = self._extract_geometry(vp)

        # Check for vessel-type STL in assets/hulls/
        stl_path = self._resolve_hull_stl(stl_path)

        L  = features['length']
        B  = features['breadth']
        T  = features['draft']
        Cb = features['Cb_actual']

        speed_knots = float(vp.get('speed', vp.get('inlet_velocity', 12.0)))
        V  = speed_knots * 0.5144
        Fn = V / np.sqrt(G_GRAV * L) if L > 0 else 0.2
        Re = V * L / NU_WATER if L > 0 else 1e7

        print(f"[PhysicsNeMo]   L={L:.1f}m  B={B:.1f}m  T={T:.1f}m  "
              f"Cb={Cb:.3f}  V={speed_knots:.1f}kn  Fn={Fn:.3f}")

        # ── Step 2: SDF Generation ───────────────────────────────────
        self.progress_signal.emit(20, "SDF alani hesaplaniyor...")
        print("[PhysicsNeMo] Step 2/5 -- SDF generation")

        sdf_gen = SDFGenerator(cfg)

        if stl_path and os.path.exists(stl_path):
            try:
                sdf = sdf_gen.compute_sdf_from_stl(stl_path)
                print(f"[PhysicsNeMo]   SDF from STL: {list(sdf.shape)}")
            except Exception as e:
                print(f"[PhysicsNeMo]   STL SDF failed ({e}), using analytical")
                sdf = sdf_gen.generate_analytical_hull_sdf()
        else:
            sdf = sdf_gen.generate_analytical_hull_sdf()
            print(f"[PhysicsNeMo]   SDF analytical: {list(sdf.shape)}")

        # ── Step 3: Assemble FNO input ───────────────────────────────
        self.progress_signal.emit(35, "FNO girdi tensoru hazirlaniyor...")
        print("[PhysicsNeMo] Step 3/5 -- Assemble FNO input")

        fno_input = sdf_gen.build_fno_input(sdf, reynolds=Re, froude=Fn)
        print(f"[PhysicsNeMo]   FNO input: {list(fno_input.shape)}")

        # ── Step 4: Model Inference ──────────────────────────────────
        self.progress_signal.emit(50, "3D-FNO cikarimlari yapiliyor...")
        print("[PhysicsNeMo] Step 4/5 -- FNO inference")

        device = DEVICE
        fno_result = self._run_fno_inference(fno_input, sdf, device, cfg)

        if fno_result is None:
            # Fallback to Holtrop-Mennen
            self.progress_signal.emit(70, "FNO mevcut degil -- Holtrop-Mennen...")
            print("[PhysicsNeMo]   FALLBACK -> Holtrop-Mennen")
            result = self._holtrop_fallback(adapter, speed_knots, features, sdf_gen)
            self.progress_signal.emit(100, "Holtrop-Mennen tamamlandi (fallback)")
            self.finished_signal.emit(result)
            return

        pred_vol = fno_result  # [4, D, H, W] numpy

        # ── Step 5: Post-processing & Slice Extraction ───────────────
        self.progress_signal.emit(80, "Akis alani dilim cikartiliyor...")
        print("[PhysicsNeMo] Step 5/5 -- Post-processing")

        result = self._postprocess(
            pred_vol, sdf, sdf_gen, features, speed_knots, Fn, Re)

        self.progress_signal.emit(100, "3D-FNO analizi tamamlandi!")
        print("[PhysicsNeMo] DONE -- emitting finished_signal")
        self.finished_signal.emit(result)

    # ---- Sub-steps --------------------------------------------------

    def _resolve_hull_stl(self, parametric_stl_path):
        """Priority STL lookup: assets/hulls/{type}.stl > parametric > None."""
        # 1. Check vessel-type STL in assets/hulls/
        stl_name = VESSEL_STL_MAP.get(self.vessel_type)
        if stl_name:
            asset_stl = os.path.join(HULL_DIR, stl_name)
            if os.path.exists(asset_stl):
                print(f"[PhysicsNeMo]   Using hull asset: {asset_stl}")
                return asset_stl
            else:
                print(f"[PhysicsNeMo]   Hull asset not found: {asset_stl}")

        # 2. Try generic name match
        if os.path.isdir(HULL_DIR):
            for fname in os.listdir(HULL_DIR):
                if fname.lower().endswith('.stl'):
                    # If any STL exists, use the first one
                    found = os.path.join(HULL_DIR, fname)
                    print(f"[PhysicsNeMo]   Using first available hull: {found}")
                    return found

        # 3. Fall back to parametric STL
        if parametric_stl_path and os.path.exists(parametric_stl_path):
            return parametric_stl_path
        return None

    def _extract_geometry(self, vp: Dict):
        """Use RetrosimHullAdapter to produce mesh + features + STL."""
        if not HAS_GEOMETRY:
            print("[PhysicsNeMo]   WARNING: Geometry engine unavailable")
            return None, self._default_features(vp), None

        try:
            adapter = RetrosimHullAdapter()
            adapter.set_from_ui(vp)
            features = adapter.extract_ml_features()

            # Generate STL into temp dir for SDF computation
            tmp_dir = os.path.join(MODEL_DIR, 'geometry')
            os.makedirs(tmp_dir, exist_ok=True)
            stl_path = os.path.join(tmp_dir, '_physics_nemo_hull.stl')

            try:
                stl_path = adapter.generate_stl(stl_path)
                print(f"[PhysicsNeMo]   STL generated: {stl_path}")
            except ImportError:
                print("[PhysicsNeMo]   numpy-stl not installed, no STL")
                stl_path = None

            return adapter, features, stl_path
        except Exception as e:
            print(f"[PhysicsNeMo]   Geometry extraction failed: {e}")
            return None, self._default_features(vp), None

    @staticmethod
    def _default_features(vp: Dict) -> Dict:
        """Provide sensible defaults when hull adapter is unavailable."""
        return {
            'length':  float(vp.get('loa', vp.get('lbp', 100))),
            'breadth': float(vp.get('beam', vp.get('breadth', 15))),
            'draft':   float(vp.get('draft', 6)),
            'Cb_actual': float(vp.get('cb', 0.65)),
            'Cm':      float(vp.get('cm', 0.98)),
            'wetted_surface_area': 0,
            'displaced_volume': 0,
        }

    def _run_fno_inference(self, fno_input, sdf, device, cfg):
        """Load checkpoint, run forward pass, return [4,D,H,W] numpy or None."""
        if not HAS_FNO3D:
            print("[PhysicsNeMo]   fno3d_network not importable")
            return None

        if not os.path.exists(MODEL_PATH):
            print(f"[PhysicsNeMo]   Checkpoint not found: {MODEL_PATH}")
            return None

        try:
            model = FNO3d_NS_Solver(cfg).to(device)
            ckpt = torch.load(MODEL_PATH, map_location=device,
                              weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            print(f"[PhysicsNeMo]   Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"[PhysicsNeMo]   Model load failed: {e}")
            return None

        fno_input_d = fno_input.to(device)
        sdf_d       = sdf.to(device)

        with torch.no_grad():
            pred = model(fno_input_d, sdf_d)  # [B, 4, D, H, W]

        return pred.cpu().numpy()[0]  # [4, D, H, W]

    def _postprocess(self, pred_vol, sdf_tensor, sdf_gen,
                     features, speed_knots, Fn, Re):
        """Extract 2D slices and scalar estimates from the 3D volume."""
        u_vol = pred_vol[0]  # [D, H, W]
        v_vol = pred_vol[1]
        w_vol = pred_vol[2]
        p_vol = pred_vol[3]
        sdf_np = sdf_tensor.numpy()[0, 0]  # [D, H, W]

        D, H, W = u_vol.shape

        # --- Waterline slice (mid-depth, z ~ waterline) ---
        z_mid = D // 2
        u_wl = u_vol[z_mid, :, :]   # [H, W]
        v_wl = v_vol[z_mid, :, :]
        p_wl = p_vol[z_mid, :, :]
        sdf_wl = sdf_np[z_mid, :, :]
        vel_mag_wl = np.sqrt(u_wl**2 + v_wl**2)

        # --- Centerline slice (mid-height, y=0 plane) ---
        y_mid = H // 2
        u_cl = u_vol[:, y_mid, :]   # [D, W]
        w_cl = w_vol[:, y_mid, :]
        p_cl = p_vol[:, y_mid, :]
        sdf_cl = sdf_np[:, y_mid, :]
        vel_mag_cl = np.sqrt(u_cl**2 + w_cl**2)

        # --- Grid coordinates for the slices ---
        gx = sdf_gen.grid_x.numpy()  # [D, H, W]
        gy = sdf_gen.grid_y.numpy()
        gz = sdf_gen.grid_z.numpy()

        X_wl = gx[z_mid, :, :]  # [H, W]
        Y_wl = gy[z_mid, :, :]
        X_cl = gx[:, y_mid, :]  # [D, W]
        Z_cl = gz[:, y_mid, :]

        # --- Scalar resistance estimates from pressure integration ---
        # (crude estimate: integrate p * n_x over hull surface voxels)
        L = features['length']
        B = features['breadth']
        T = features['draft']
        S = features.get('wetted_surface_area', 1.7 * L * T + L * B * 0.85)
        V = speed_knots * 0.5144
        q = 0.5 * RHO_WATER * max(V, 0.1) ** 2

        # Pressure drag ~ integral of p over surface voxels
        surface_mask = np.abs(sdf_np) < (sdf_gen.dx * 1.5)
        if surface_mask.sum() > 0:
            p_surface = p_vol[surface_mask].mean()
        else:
            p_surface = p_vol.mean()

        # Approximate resistance coefficients
        Cf = 0.075 / (np.log10(max(Re, 1e5)) - 2.0) ** 2
        Cw_est = max(0.0, float(p_surface) * 0.001)
        Ct_est = Cf + Cw_est
        Rf = q * S * Cf / 1000.0  # kN
        Rw = q * S * Cw_est / 1000.0
        Rt = Rf + Rw
        Pe = Rt * V  # kW

        # Velocity inside hull (quality check)
        u_inside = u_vol[sdf_np < 0]
        max_u_inside = float(np.abs(u_inside).max()) if len(u_inside) > 0 else 0.0

        result = {
            # ---- 2D Slices (ready for matplotlib/pyqtgraph) ----
            # Waterline plane
            'X_wl': X_wl, 'Y_wl': Y_wl,
            'U_wl': u_wl, 'V_wl': v_wl, 'P_wl': p_wl,
            'vel_mag_wl': vel_mag_wl,
            'sdf_wl': sdf_wl,

            # Centerline plane
            'X_cl': X_cl, 'Z_cl': Z_cl,
            'U_cl': u_cl, 'W_cl': w_cl, 'P_cl': p_cl,
            'vel_mag_cl': vel_mag_cl,
            'sdf_cl': sdf_cl,

            # ---- Scalar Resistance ----
            'Cf': float(Cf),
            'Cw': float(Cw_est),
            'Ct': float(Ct_est),
            'Rf_kN': float(Rf),
            'Rw_kN': float(Rw),
            'Rt_kN': float(Rt),
            'Pe_kW': float(Pe),
            'Froude_number': float(Fn),
            'Reynolds_number': float(Re),
            'speed_knots': float(speed_knots),

            # ---- Metadata ----
            'backend': '3D-FNO-PINN',
            'is_fno_active': True,
            'is_fallback': False,
            'grid_dims': list(u_vol.shape),
            'max_u_inside_hull': max_u_inside,
        }

        print(f"[PhysicsNeMo]   Rt={Rt:.2f} kN  Pe={Pe:.1f} kW  "
              f"Fn={Fn:.3f}  |u|_hull={max_u_inside:.5f}")
        return result

    def _holtrop_fallback(self, adapter, speed_knots, features, sdf_gen):
        """Produce result dict using Holtrop-Mennen when FNO is unavailable.

        Even without the FNO model, generates a synthetic uniform flow field
        masked by the SDF so the GUI contour shows the hull shape.
        """
        hm = {}
        if adapter is not None:
            try:
                hm = adapter.predict_total_resistance(speed_knots)
                print(f"[PhysicsNeMo]   Holtrop Rt={hm.get('Rt',0):.1f} N")
            except Exception as e:
                print(f"[PhysicsNeMo]   Holtrop failed: {e}")

        D, H, W = self.config.grid_dims
        gx = sdf_gen.grid_x.numpy()
        gy = sdf_gen.grid_y.numpy()
        gz = sdf_gen.grid_z.numpy()
        z_mid = D // 2
        y_mid = H // 2

        # Build a synthetic flow field masked by SDF (shows hull shape)
        # This makes the hull visible in the contour even without FNO
        sdf_wl = sdf_gen.grid_x.numpy()[z_mid] * 0  # same shape [H, W]
        try:
            # If we have a computed SDF tensor, extract its slices
            sdf_full = sdf_gen.generate_analytical_hull_sdf()
            sdf_np = sdf_full.numpy()[0, 0]  # [D, H, W]
            sdf_wl = sdf_np[z_mid]
            sdf_cl = sdf_np[:, y_mid]

            # Synthetic U: uniform flow * sigmoid mask
            mask_wl = 1.0 / (1.0 + np.exp(-50.0 * sdf_wl))
            mask_cl = 1.0 / (1.0 + np.exp(-50.0 * sdf_cl))
            u_wl = mask_wl * 1.0   # free-stream = 1.0
            u_cl = mask_cl * 1.0
            vel_mag_wl = np.abs(u_wl)
            vel_mag_cl = np.abs(u_cl)
        except Exception:
            sdf_wl = np.zeros((H, W))
            sdf_cl = np.zeros((D, W))
            u_wl = np.ones((H, W))
            u_cl = np.ones((D, W))
            vel_mag_wl = u_wl
            vel_mag_cl = u_cl

        result = {
            'X_wl': gx[z_mid], 'Y_wl': gy[z_mid],
            'U_wl': u_wl.astype(np.float32),
            'V_wl': np.zeros((H, W), dtype=np.float32),
            'P_wl': np.zeros((H, W), dtype=np.float32),
            'vel_mag_wl': vel_mag_wl.astype(np.float32),
            'sdf_wl': sdf_wl.astype(np.float32),

            'X_cl': gx[:, y_mid], 'Z_cl': gz[:, y_mid],
            'U_cl': u_cl.astype(np.float32),
            'W_cl': np.zeros((D, W), dtype=np.float32),
            'P_cl': np.zeros((D, W), dtype=np.float32),
            'vel_mag_cl': vel_mag_cl.astype(np.float32),
            'sdf_cl': sdf_cl.astype(np.float32),

            'Cf': float(hm.get('Cf', 0)),
            'Cw': float(hm.get('Cw', 0)),
            'Ct': float(hm.get('Cf', 0)) + float(hm.get('Cw', 0)),
            'Rf_kN': float(hm.get('Rf_form', 0)) / 1000.0,
            'Rw_kN': float(hm.get('Rw', 0)) / 1000.0,
            'Rt_kN': float(hm.get('Rt', 0)) / 1000.0,
            'Pe_kW': float(hm.get('Pe_kW', 0)),
            'Froude_number': float(hm.get('Froude_number', 0)),
            'Reynolds_number': float(hm.get('Reynolds_number', 0)),
            'speed_knots': float(speed_knots),

            'backend': 'Holtrop-Mennen (fallback)',
            'is_fno_active': False,
            'is_fallback': True,
            'grid_dims': [D, H, W],
            'max_u_inside_hull': 0.0,
        }
        return result


# =====================================================================
# Orchestrator Agent  (GUI calls this)
# =====================================================================

class PhysicsNeMoAgent(QObject):
    """GUI-facing orchestrator for the 3D-FNO PINN solver.

    Usage from the GUI:
        agent = PhysicsNeMoAgent()
        agent.progress_signal.connect(status_bar.update)
        agent.finished_signal.connect(results_panel.display)
        agent.error_signal.connect(error_dialog.show)
        agent.run_analysis(vessel_params_dict)
    """

    # Re-export signals so the GUI can connect directly to the agent
    progress_signal  = pyqtSignal(int, str)
    finished_signal  = pyqtSignal(dict)
    error_signal     = pyqtSignal(str)

    # Default solver configuration (low-res for CPU, switch to hires on GPU)
    DEFAULT_CONFIG = SolverConfig() if HAS_FNO3D else None

    def __init__(self, config: Optional[SolverConfig] = None,
                 vessel_type: str = 'KCS', parent=None):
        super().__init__(parent)
        self.config = config or (self.DEFAULT_CONFIG or SolverConfig())
        self.vessel_type = vessel_type
        self._worker: Optional[PhysicsNeMoWorker] = None
        self._last_result: Optional[Dict] = None

        self._check_model_status()

    # ---- Public API -------------------------------------------------

    def run_analysis(self, vessel_params: Dict,
                     vessel_type: str = None):
        """Start the async FNO inference pipeline.

        If a worker is already running, it is ignored (no double-run).

        Args:
            vessel_params: Dict from GUI with loa, beam, draft, speed, etc.
            vessel_type: Optional hull type ('KCS', 'Tanker', etc.).
                         Falls back to self.vessel_type if not specified.
        """
        if self._worker is not None and self._worker.isRunning():
            print("[PhysicsNeMo] Worker already running -- ignoring request")
            return

        vtype = vessel_type or self.vessel_type

        print(f"\n{'='*60}")
        print(f" PhysicsNeMo Agent -- run_analysis()")
        print(f" Vessel: L={vessel_params.get('loa','?')}  "
              f"B={vessel_params.get('beam','?')}  "
              f"T={vessel_params.get('draft','?')}  "
              f"V={vessel_params.get('speed','?')} kn")
        print(f" Hull type: {vtype}")
        print(f"{'='*60}")

        self._worker = PhysicsNeMoWorker(
            vessel_params, self.config, vessel_type=vtype, parent=self)

        # Wire worker signals to agent signals (relay to GUI)
        self._worker.progress_signal.connect(self.progress_signal.emit)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.error_signal.connect(self.error_signal.emit)

        self._worker.start()

    def is_model_available(self) -> bool:
        """Check whether a trained FNO checkpoint exists."""
        return os.path.exists(MODEL_PATH)

    def get_last_result(self) -> Optional[Dict]:
        """Return the most recent result dictionary (or None)."""
        return self._last_result

    # ---- Configuration Helpers --------------------------------------

    def set_hires(self):
        """Switch to high-resolution grid (GPU recommended)."""
        self.config.grid_depth  = 64
        self.config.grid_height = 32
        self.config.grid_width  = 128
        self.config.modes_d = 16
        self.config.modes_h = 12
        self.config.modes_w = 32
        self.config.fno_width = 48
        print("[PhysicsNeMo] Switched to HIGH-RES grid [64, 32, 128]")

    def set_lowres(self):
        """Switch to low-resolution grid (CPU safe)."""
        self.config.grid_depth  = 32
        self.config.grid_height = 16
        self.config.grid_width  = 64
        self.config.modes_d = 8
        self.config.modes_h = 6
        self.config.modes_w = 16
        self.config.fno_width = 32
        print("[PhysicsNeMo] Switched to LOW-RES grid [32, 16, 64]")

    # ---- Internals --------------------------------------------------

    def _on_finished(self, result: Dict):
        """Cache result and relay to GUI."""
        self._last_result = result
        backend = result.get('backend', '?')
        rt = result.get('Rt_kN', 0)
        print(f"[PhysicsNeMo] Result received: backend={backend}  "
              f"Rt={rt:.2f} kN")
        self.finished_signal.emit(result)

    def _check_model_status(self):
        """Log model availability at startup."""
        if os.path.exists(MODEL_PATH):
            size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            print(f"[PhysicsNeMo] Model checkpoint found: "
                  f"{MODEL_PATH} ({size_mb:.1f} MB)")
        else:
            print(f"[PhysicsNeMo] WARNING: No checkpoint at {MODEL_PATH}")
            print(f"[PhysicsNeMo]   -> Will fallback to Holtrop-Mennen")
            print(f"[PhysicsNeMo]   -> Train with: python agents/sdf_fno3d_solver.py")
