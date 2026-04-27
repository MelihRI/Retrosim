"""
CFDVisualizationWidget -- True CFD Contour Viewer (pyqtgraph.opengl)
=====================================================================
Replaces the legacy fake potential-flow particle system with:
  1. Real STL hull mesh rendering (shaded, anti-fouling paint split)
  2. FNO flow-field contour plane at Z=0 (waterline) from PhysicsNeMoAgent
  3. Scalar HUD overlay (Cf, Cw, Rt, Pe) updated from the solver dict

Public API (unchanged from legacy -- drop-in replacement):
  - update_hull_geometry(stl_path, vessel_data=dict)
  - update_plot(results_dict)
  - ship_speed, ship_L, ship_B, ship_T, ship_Cb  (settable attrs)
"""

import numpy as np
import os
import traceback
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

import pyqtgraph.opengl as gl

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    from stl import mesh as stl_mesh
    HAS_STL = True
except ImportError:
    HAS_STL = False

# Physical constants
RHO_WATER = 1025.0
NU_WATER  = 1.188e-6
G         = 9.81


class CFDVisualizationWidget(QWidget):
    """OpenGL 3D viewer: hull mesh + CFD contour slice."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # Ship state (normalised viewport)
        self.ship_L = 1.5
        self.ship_B = 0.3
        self.ship_T = 0.15
        self.ship_speed = 12.0
        self.ship_Cb = 0.82

        # GL items
        self.hull_mesh_item = None
        self.hull_vertices = None
        self.waterline_item = None
        self.contour_mesh_item = None
        self.domain_box_item = None
        self.light_item = None

        # State
        self.resistance_data = {}
        self._vessel_data = {}
        self._hull_adapter = None
        self.light_elevation = 60.0
        self.light_azimuth = 225.0
        self.light_intensity = 1.0

        self._build_viewport()
        self._setup_hud()
        self._setup_toolbar()
        self._setup_scene()

    # ================================================================
    # UI Construction
    # ================================================================

    def _build_viewport(self):
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setBackgroundColor('#020610')
        self._layout.addWidget(self.gl_widget, stretch=10)
        self.gl_widget.setCameraPosition(distance=5.0, elevation=22, azimuth=230)

    def _setup_hud(self):
        f = QFont("Consolas", 9)

        self.lbl_stats = QLabel("", self.gl_widget)
        self.lbl_stats.setFont(f)
        self.lbl_stats.setStyleSheet(
            "color:#6a9fb5; background:rgba(5,10,20,180);"
            "padding:4px; border:1px solid #1a3a5a; border-radius:3px;")
        self.lbl_stats.move(10, 10)

        self.lbl_resistance = QLabel("", self.gl_widget)
        self.lbl_resistance.setFont(f)
        self.lbl_resistance.setStyleSheet(
            "color:#a0e0ff; background:rgba(5,10,20,160);"
            "padding:6px; border:1px solid #1a3a5a; border-radius:4px;")
        self.lbl_resistance.hide()

    def _setup_toolbar(self):
        bar = QWidget()
        bar.setFixedHeight(36)
        bar.setStyleSheet("background:#0a0e1a; border-top:1px solid #1a2a3a;")
        row = QHBoxLayout(bar)
        row.setContentsMargins(8, 2, 8, 2)
        row.setSpacing(12)
        ls = "color:#7fb3d3; font-size:11px; font-family:Consolas;"
        ss = ("QSlider::groove:horizontal{height:4px;background:#1a3a5a;}"
              "QSlider::handle:horizontal{width:12px;height:12px;"
              "background:#4a8ab0;border-radius:6px;margin:-4px 0;}")

        row.addWidget(QLabel("Light:", styleSheet=ls))
        self.sl_light_elev = QSlider(Qt.Orientation.Horizontal)
        self.sl_light_elev.setRange(0, 90); self.sl_light_elev.setValue(60)
        self.sl_light_elev.setFixedWidth(90); self.sl_light_elev.setStyleSheet(ss)
        self.sl_light_elev.valueChanged.connect(self._on_light_changed)
        row.addWidget(self.sl_light_elev)

        row.addWidget(QLabel("Dir:", styleSheet=ls))
        self.sl_light_az = QSlider(Qt.Orientation.Horizontal)
        self.sl_light_az.setRange(0, 360); self.sl_light_az.setValue(225)
        self.sl_light_az.setFixedWidth(90); self.sl_light_az.setStyleSheet(ss)
        self.sl_light_az.valueChanged.connect(self._on_light_changed)
        row.addWidget(self.sl_light_az)

        row.addWidget(QLabel("Brightness:", styleSheet=ls))
        self.sl_light_int = QSlider(Qt.Orientation.Horizontal)
        self.sl_light_int.setRange(10, 200); self.sl_light_int.setValue(100)
        self.sl_light_int.setFixedWidth(70); self.sl_light_int.setStyleSheet(ss)
        self.sl_light_int.valueChanged.connect(self._on_light_changed)
        row.addWidget(self.sl_light_int)

        btn = QPushButton("Reset")
        btn.setStyleSheet("QPushButton{background:#1a2a3a;color:#7fb3d3;"
                          "border:1px solid #2a4a6a;padding:2px 8px;"
                          "border-radius:3px;font-size:11px;}"
                          "QPushButton:hover{background:#2a3a5a;}")
        btn.clicked.connect(self._reset_light)
        row.addWidget(btn)
        row.addStretch()
        self._layout.addWidget(bar, stretch=0)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        w, h = self.width(), self.height()
        if hasattr(self, 'lbl_stats'):
            self.lbl_stats.adjustSize()
            self.lbl_stats.move(w - self.lbl_stats.width() - 10, 10)
        if hasattr(self, 'lbl_resistance'):
            self.lbl_resistance.adjustSize()
            self.lbl_resistance.move(10, h - self.lbl_resistance.height() - 50)

    # ================================================================
    # Scene Setup
    # ================================================================

    def _setup_scene(self):
        self._build_domain()
        self._build_axes()
        self._build_light()
        self._update_stats_label()

    def _build_domain(self):
        if self.domain_box_item:
            self.gl_widget.removeItem(self.domain_box_item)
        L, B, T = self.ship_L, self.ship_B, self.ship_T
        self.domain_box_item = gl.GLBoxItem()
        self.domain_box_item.setSize(x=4.5*L, y=3*B, z=2.5*T)
        self.domain_box_item.translate(-1.5*L, -1.5*B, -1.5*T)
        self.domain_box_item.setColor((30, 80, 130, 50))
        self.gl_widget.addItem(self.domain_box_item)

        g = gl.GLGridItem()
        g.setSize(x=4.5*L, y=3*B)
        g.setSpacing(x=L*0.5, y=B*0.5)
        g.translate(0, 0, -1.5*T)
        self.gl_widget.addItem(g)

    def _build_axes(self):
        ax = gl.GLAxisItem()
        ax.setSize(x=self.ship_L*0.7, y=self.ship_B, z=self.ship_T*2)
        self.gl_widget.addItem(ax)

    def _build_light(self):
        self._update_light_pos()
        lx, ly, lz = self._light_xyz()
        self.light_item = gl.GLScatterPlotItem(
            pos=np.array([[lx, ly, lz]], dtype=np.float32),
            size=18, pxMode=True, color=(1, .9, .5, .9))
        self.gl_widget.addItem(self.light_item)

    def _light_xyz(self):
        el = np.radians(self.light_elevation)
        az = np.radians(self.light_azimuth)
        d = 8.0
        return (float(d*np.cos(el)*np.sin(az)),
                float(d*np.cos(el)*np.cos(az)),
                float(d*np.sin(el)))

    def _update_light_pos(self):
        self.light_elevation = float(self.sl_light_elev.value())
        self.light_azimuth = float(self.sl_light_az.value())
        self.light_intensity = self.sl_light_int.value() / 100.0
        a = max(0.02, 0.02 + 0.025 * self.light_intensity)
        self.gl_widget.setBackgroundColor(
            (int(a*2*255), int(a*4*255), int(a*10*255), 255))

    def _on_light_changed(self):
        self._update_light_pos()
        if self.light_item:
            lx, ly, lz = self._light_xyz()
            self.light_item.setData(
                pos=np.array([[lx, ly, lz]], dtype=np.float32),
                color=(self.light_intensity, self.light_intensity*.9, .5, .9))

    def _reset_light(self):
        self.sl_light_elev.setValue(60)
        self.sl_light_az.setValue(225)
        self.sl_light_int.setValue(100)

    # ================================================================
    # Hull Geometry Loading (Step 2)
    # ================================================================

    def update_hull_geometry(self, stl_path, vessel_data=None):
        """Load real hull STL and render as shaded mesh."""
        if vessel_data:
            loa = float(vessel_data.get('loa', 190))
            beam = float(vessel_data.get('beam', 32))
            draft = float(vessel_data.get('draft', 12.5))
            self.ship_Cb = float(vessel_data.get('cb', 0.82))
            self.ship_speed = float(vessel_data.get('speed', 12))
            self.ship_L = 1.5
            self.ship_B = 1.5 * beam / max(loa, 1)
            self.ship_T = 1.5 * draft / max(loa, 1)
            self._vessel_data = dict(vessel_data)

        self._build_domain()

        # Remove old hull items
        for item in [self.hull_mesh_item, self.waterline_item]:
            if item:
                try:
                    self.gl_widget.removeItem(item)
                except Exception:
                    pass
        self.hull_mesh_item = None
        self.waterline_item = None

        if not stl_path or not os.path.exists(stl_path):
            print(f"[CFD] No STL path or file missing: {stl_path}")
            return

        # --- Load mesh: prefer trimesh (robust), fallback to numpy-stl ---
        verts, faces, nf = None, None, 0
        try:
            if HAS_TRIMESH:
                tm = trimesh.load(stl_path, force='mesh')
                verts = np.array(tm.vertices, dtype=np.float32)
                faces = np.array(tm.faces, dtype=np.uint32)
                nf = len(faces)
                print(f"[CFD] Loaded via trimesh: {len(verts)} verts, {nf} faces")
            elif HAS_STL:
                hull = stl_mesh.Mesh.from_file(stl_path)
                verts = hull.vectors.reshape(-1, 3).astype(np.float32)
                nf = len(hull.vectors)
                faces = np.arange(nf * 3).reshape(-1, 3).astype(np.uint32)
                print(f"[CFD] Loaded via numpy-stl: {len(verts)} verts, {nf} faces")
            else:
                print("[CFD] ERROR: No mesh loader (install trimesh or numpy-stl)")
                return
        except Exception as e:
            print(f"[CFD] Mesh load error: {e}")
            traceback.print_exc()
            return

        if verts is None or len(verts) < 3:
            return

        # --- Centre at origin, UNIFORM scale preserving aspect ratio ---
        centroid = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
        verts -= centroid
        extent = verts.max(axis=0) - verts.min(axis=0)
        max_dim = max(extent[0], extent[1], extent[2], 1e-6)
        scale = self.ship_L / max_dim  # uniform scale -> preserves L/B/T ratio
        verts *= scale

        # Position: bow forward (+X), waterline at Z=0
        verts[:, 2] -= verts[:, 2].max()  # deck at Z=0
        verts[:, 2] += 0.0               # waterline ~ Z=0

        self.hull_vertices = verts.copy()
        print(f"[CFD] Hull bbox after scale: X=[{verts[:,0].min():.3f}, {verts[:,0].max():.3f}] "
              f"Y=[{verts[:,1].min():.3f}, {verts[:,1].max():.3f}] "
              f"Z=[{verts[:,2].min():.3f}, {verts[:,2].max():.3f}]")

        # --- Compute per-vertex normals for proper shading ---
        normals = np.zeros_like(verts)
        for f in faces:
            v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
            n = np.cross(v1 - v0, v2 - v0)
            normals[f[0]] += n
            normals[f[1]] += n
            normals[f[2]] += n
        norms_len = np.linalg.norm(normals, axis=1, keepdims=True)
        norms_len[norms_len < 1e-10] = 1.0
        normals /= norms_len

        # --- Per-face colour: anti-fouling red below WL, steel grey above ---
        face_z = np.mean(verts[faces], axis=1)[:, 2]  # avg Z per face
        cols = np.zeros((len(faces), 4), dtype=np.float32)
        below = face_z < -0.01
        cols[below]  = [0.55, 0.14, 0.12, 1.0]  # anti-fouling red
        cols[~below] = [0.50, 0.52, 0.58, 1.0]  # steel grey

        try:
            self.hull_mesh_item = gl.GLMeshItem(
                vertexes=verts,
                faces=faces,
                faceColors=cols,
                computeNormals=True,
                smooth=True,
                drawEdges=False,
                shader='normalColor',
                glOptions='opaque',
            )
            self.gl_widget.addItem(self.hull_mesh_item)
        except Exception:
            # Fallback if normalColor shader not available
            self.hull_mesh_item = gl.GLMeshItem(
                vertexes=verts,
                faces=faces,
                faceColors=cols,
                computeNormals=True,
                smooth=True,
                drawEdges=False,
                shader='shaded',
                glOptions='opaque',
            )
            self.gl_widget.addItem(self.hull_mesh_item)

        # Waterline contour
        self._draw_waterline(verts, faces)
        print(f"[CFD] Hull rendered: {nf} faces, shader=normalColor")

    def _draw_waterline(self, verts, faces):
        """Extract Z=0 intersection as white contour line."""
        pts = []
        for tri in faces:
            v = [verts[tri[0]], verts[tri[1]], verts[tri[2]]]
            for i in range(3):
                a, b = v[i], v[(i+1) % 3]
                if (a[2] >= 0) != (b[2] >= 0):
                    t = -a[2] / (b[2] - a[2] + 1e-10)
                    pts.append(a + t * (b - a))
        if len(pts) > 3:
            arr = np.array(pts, dtype=np.float32)
            arr = arr[np.argsort(arr[:, 0])]
            arr[:, 2] = 0.002
            self.waterline_item = gl.GLLinePlotItem(
                pos=arr, color=(1, 1, 1, 0.85), width=2.5, antialias=True)
            self.gl_widget.addItem(self.waterline_item)

    # ================================================================
    # Flow Contour Plane (Step 3) -- from FNO result dict
    # ================================================================

    def update_flow_contours(self, results_dict):
        """Render a 2D colour-mapped plane below the hull from FNO slices.

        Accepts keys: X_wl/Y_wl + vel_mag_wl (preferred) or U_wl or P_wl.
        """
        # --- DEBUG: log what keys arrived ---
        print(f"[CFD] update_flow_contours: keys={list(results_dict.keys())}")

        X = results_dict.get('X_wl')
        Y = results_dict.get('Y_wl')
        # Field priority: vel_mag_wl > U_wl > P_wl > U > P
        field = results_dict.get('vel_mag_wl')
        if field is None:
            field = results_dict.get('U_wl')
        if field is None:
            field = results_dict.get('P_wl')
        if field is None:
            field = results_dict.get('U')
        if field is None:
            field = results_dict.get('P')

        if X is None or Y is None or field is None:
            print(f"[CFD] Contour SKIPPED: X={X is not None}, Y={Y is not None}, field={field is not None}")
            return

        # Ensure numpy
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        field = np.asarray(field, dtype=np.float32)

        # Remove old contour
        if self.contour_mesh_item:
            try:
                self.gl_widget.removeItem(self.contour_mesh_item)
            except Exception:
                pass
            self.contour_mesh_item = None

        H, W = field.shape
        if H < 2 or W < 2:
            print(f"[CFD] Contour SKIPPED: field shape too small {field.shape}")
            return

        # --- Map physical coords to viewport coords ---
        x_min, x_max = float(X.min()), float(X.max())
        y_min, y_max = float(Y.min()), float(Y.max())
        x_range = max(x_max - x_min, 1e-6)
        y_range = max(y_max - y_min, 1e-6)

        L, B, T = self.ship_L, self.ship_B, self.ship_T
        x_norm = (X - x_min) / x_range * 3.0 * L - 1.0 * L
        y_norm = (Y - y_min) / y_range * 2.0 * B - 1.0 * B

        # Place BELOW hull to avoid z-fighting (5% of draft below waterline)
        z_val = -0.05 * max(T, 0.05)
        z_plane = np.full_like(x_norm, z_val, dtype=np.float32)

        # Build vertex array [H*W, 3]
        verts = np.stack([x_norm, y_norm, z_plane], axis=-1).reshape(-1, 3).astype(np.float32)

        # Build faces (two tris per quad) -- vectorised
        i_idx = np.arange(H - 1)[:, None]
        j_idx = np.arange(W - 1)[None, :]
        base = (i_idx * W + j_idx).ravel()
        tri_a = np.column_stack([base, base + 1, base + W])
        tri_b = np.column_stack([base + 1, base + W + 1, base + W])
        faces = np.vstack([tri_a, tri_b]).astype(np.uint32)

        # --- Turbo colormap (vectorised, no Python loop) ---
        colors = self._turbo_colormap_fast(field, faces)

        # Mask hull interior (SDF < 0) as dark
        sdf_wl = results_dict.get('sdf_wl')
        if sdf_wl is not None:
            sdf_flat = np.asarray(sdf_wl, dtype=np.float32).ravel()
            if len(sdf_flat) == H * W:
                face_sdf = (sdf_flat[faces[:, 0]] + sdf_flat[faces[:, 1]] + sdf_flat[faces[:, 2]]) / 3.0
                inside = face_sdf < 0
                colors[inside] = [0.06, 0.06, 0.10, 0.90]

        # Use 'balloon' shader = unlit, shows faceColors exactly as given
        self.contour_mesh_item = gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            faceColors=colors,
            smooth=False,
            drawEdges=False,
            shader='balloon',
            glOptions='translucent',
        )
        self.gl_widget.addItem(self.contour_mesh_item)
        print(f"[CFD] Contour plane ADDED: {H}x{W} -> {len(faces)} tris at Z={z_val:.4f}")

    @staticmethod
    def _turbo_colormap_fast(field, faces):
        """Vectorised turbo-like colormap: field[H,W] + faces -> [nf, 4] RGBA."""
        fmin, fmax = float(field.min()), float(field.max())
        rng = max(fmax - fmin, 1e-8)
        t_flat = ((field.ravel() - fmin) / rng).astype(np.float32)

        # Per-face value = average of 3 vertices
        t = (t_flat[faces[:, 0]] + t_flat[faces[:, 1]] + t_flat[faces[:, 2]]) / 3.0
        nf = len(faces)

        r = np.zeros(nf, dtype=np.float32)
        g = np.zeros(nf, dtype=np.float32)
        b = np.zeros(nf, dtype=np.float32)

        # Segment 0: [0, 0.25) blue -> cyan
        m = t < 0.25
        s = t[m] / 0.25
        r[m] = 0.18;               g[m] = 0.15 + 0.6*s;  b[m] = 0.85 - 0.2*s

        # Segment 1: [0.25, 0.5) cyan -> green
        m = (t >= 0.25) & (t < 0.5)
        s = (t[m] - 0.25) / 0.25
        r[m] = 0.1*s;              g[m] = 0.75 + 0.15*s; b[m] = 0.65 - 0.45*s

        # Segment 2: [0.5, 0.75) green -> yellow
        m = (t >= 0.5) & (t < 0.75)
        s = (t[m] - 0.5) / 0.25
        r[m] = 0.1 + 0.7*s;       g[m] = 0.9 - 0.15*s;  b[m] = 0.2 - 0.1*s

        # Segment 3: [0.75, 1.0] yellow -> red
        m = t >= 0.75
        s = (t[m] - 0.75) / 0.25
        r[m] = 0.8 + 0.2*s;       g[m] = 0.75 - 0.55*s; b[m] = 0.1 - 0.05*s

        cols = np.column_stack([r, g, b, np.full(nf, 0.88, dtype=np.float32)])
        return cols

    # ================================================================
    # Resistance HUD
    # ================================================================

    def _update_hud(self):
        r = self.resistance_data
        if not r:
            return

        backend = r.get('backend', 'N/A')
        txt = (f"Resistance Analysis ({backend})\n"
               f"------------------------------\n"
               f"  Friction     Rf = {r.get('Rf_kN', 0):.2f} kN\n"
               f"  Wave         Rw = {r.get('Rw_kN', 0):.2f} kN\n"
               f"  Total        Rt = {r.get('Rt_kN', 0):.2f} kN\n"
               f"  Eff. Power   Pe = {r.get('Pe_kW', 0):.1f} kW\n"
               f"------------------------------\n"
               f"  Froude  Fn = {r.get('Froude_number', r.get('Froude', 0)):.4f}\n"
               f"  Cf = {r.get('Cf', 0):.6f}   Cw = {r.get('Cw', 0):.6f}")
        self.lbl_resistance.setText(txt)
        self.lbl_resistance.adjustSize()
        self.lbl_resistance.show()
        self.lbl_resistance.move(10, self.height() - self.lbl_resistance.height() - 50)

    def _update_stats_label(self):
        self.lbl_stats.setText("3D-FNO CFD Viewer\nAwaiting analysis...")
        self.lbl_stats.adjustSize()
        w = self.width() if self.width() > 100 else 400
        self.lbl_stats.move(w - self.lbl_stats.width() - 10, 10)

    # ================================================================
    # Holtrop-Mennen Fallback (for when FNO dict lacks fields)
    # ================================================================

    def _estimate_resistance(self):
        """Compute Holtrop-Mennen via hull adapter as fallback."""
        vd = self._vessel_data
        if not self._hull_adapter:
            try:
                from core.geometry.FFDHullMorpher import RetrosimHullAdapter
                self._hull_adapter = RetrosimHullAdapter()
            except ImportError:
                pass
        if self._hull_adapter and vd:
            try:
                self._hull_adapter.set_from_ui(vd)
                r = self._hull_adapter.predict_total_resistance(self.ship_speed)
                return {
                    'Rf_kN': round(r['Rf'] / 1000, 2) if r['Rf'] > 100 else round(r['Rf'], 2),
                    'Rw_kN': round(r['Rw'] / 1000, 2) if r['Rw'] > 100 else round(r['Rw'], 2),
                    'Rt_kN': round(r['Rt'] / 1000, 2) if r['Rt'] > 100 else round(r['Rt'], 2),
                    'Pe_kW': round(r.get('Pe_kW', 0), 1),
                    'Froude_number': round(r.get('Froude_number', 0), 4),
                    'Cf': round(r.get('Cf', 0), 6),
                    'Cw': round(r.get('Cw', 0), 6),
                    'backend': 'Holtrop-Mennen 1984',
                }
            except Exception as e:
                print(f"[CFD] Holtrop adapter error: {e}")
        return {}

    # ================================================================
    # Public API  (drop-in replacement)
    # ================================================================

    def update_plot(self, results=None):
        """Main entry point called by main_window after analysis completes.

        Accepts either:
          - Legacy dict with X, Y, U, V, P keys (ignored for contour)
          - New PhysicsNeMoAgent dict with X_wl, Y_wl, vel_mag_wl, etc.
        """
        if results is None:
            results = {}

        print(f"[CFD] update_plot called: keys={list(results.keys())}")

        # Check if this is a PhysicsNeMo result (has flow field slices)
        has_fno = (results.get('X_wl') is not None and
                   (results.get('vel_mag_wl') is not None or
                    results.get('U_wl') is not None or
                    results.get('P_wl') is not None))

        if has_fno:
            # Full FNO result -- render contour + use its scalars
            self.update_flow_contours(results)
            self.resistance_data = {
                'Rf_kN': results.get('Rf_kN', 0),
                'Rw_kN': results.get('Rw_kN', 0),
                'Rt_kN': results.get('Rt_kN', 0),
                'Pe_kW': results.get('Pe_kW', 0),
                'Froude_number': results.get('Froude_number', 0),
                'Cf': results.get('Cf', 0),
                'Cw': results.get('Cw', 0),
                'backend': results.get('backend', '3D-FNO'),
            }
            # Update stats label
            dims = results.get('grid_dims', [0, 0, 0])
            mu = results.get('max_u_inside_hull', 0)
            self.lbl_stats.setText(
                f"Engine: {results.get('backend', '3D-FNO')}\n"
                f"Grid: {dims}\n"
                f"|u| inside hull: {mu:.5f}")
            self.lbl_stats.adjustSize()
        else:
            # Legacy / fallback path -- use Holtrop-Mennen
            self.resistance_data = self._estimate_resistance()
            self.lbl_stats.setText("Engine: Holtrop-Mennen\n(No FNO contour)")
            self.lbl_stats.adjustSize()

        self._update_hud()
