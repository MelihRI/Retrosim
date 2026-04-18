"""
CFDVisualizationWidget — Physics-Based Ship Hydrodynamics Viewer
=================================================================

Fizik tabanlı gemi-su etkileşim simülasyonu.

Physics Engine:
- Potansiyel akış (Rankine source/sink) ile gövde etrafı hız alanı
- Kelvin dalga sistemi (transverse + diverging + bow wave)
- Su yüzeyi gövde waterplane'inde kesilir (hull pierces water)
- Basınç katsayısı (Cp) tabanlı parçacık renklendirmesi
- Streamline tabanlı parçacık hareketi (gövdeyi dolanan akış)

References:
- Kelvin (1887): Wake half-angle 19.47°
- Michell (1898): Thin-ship wave making theory
- Holtrop & Mennen (1984): Empirical wave resistance
"""

import numpy as np
import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

import pyqtgraph.opengl as gl

# PyJet
try:
    from utils.pyjet_adapter import PyJetFlowField
    HAS_PYJET = True
except ImportError:
    HAS_PYJET = False

try:
    from stl import mesh as stl_mesh
    HAS_STL = True
except ImportError:
    HAS_STL = False

# ─── Physical Constants ─────────────────────────────────────────────────
RHO_WATER = 1025.0
NU_WATER  = 1.188e-6
G         = 9.81
KELVIN_HALF = np.radians(19.47)


# ═════════════════════════════════════════════════════════════════════════
# Physics Engine
# ═════════════════════════════════════════════════════════════════════════

class ShipFlowField:
    """
    Rankine source/sink potential-flow model around a ship hull.

    The hull centreline is populated with sources (bow) and sinks (stern).
    This creates a velocity perturbation field that deflects flow particles
    around the hull geometry and raises/lowers the free surface.
    """

    def __init__(self, L, B, T, Cb, Fr):
        self.L, self.B, self.T = L, B, T
        self.Cb = Cb
        self.Fr = max(Fr, 0.001)
        self.U  = 1.0        # free-stream speed (normalised)
        self._build_sources()

    def _build_sources(self):
        """Distribute Rankine sources along hull centreline."""
        n = 24
        self.sx = np.linspace(-0.46 * self.L, 0.46 * self.L, n)
        xs = self.sx / (0.48 * self.L)   # normalised -1..+1

        # Section area curve (Lackenby)
        area = self.Cb * np.maximum(1.0 - xs**2, 0) ** 0.55
        grad = np.gradient(area, self.sx)

        # Strong enough to produce visible deflection
        self.sq = grad * self.U * self.B * self.T * 2.5
        self.sy = np.zeros(n)
        self.sz = np.full(n, -0.35 * self.T)

    def hull_halfbeam(self, X):
        """Half-beam of hull waterplane at longitudinal position X."""
        xs = np.clip(X / (0.48 * self.L + 1e-6), -1, 1)
        return 0.5 * self.B * self.Cb**0.25 * np.sqrt(np.maximum(1 - xs**2, 0))

    def is_inside_hull(self, X, Y):
        """Boolean mask: True where (X,Y) is inside the hull waterplane."""
        in_x = (X > -0.5 * self.L) & (X < 0.5 * self.L)
        hb = self.hull_halfbeam(X)
        in_y = np.abs(Y) < hb
        return in_x & in_y

    # ─── Velocity field ──────────────────────────────────────────────
    def velocity(self, px, py, pz):
        """Velocity perturbation from source superposition."""
        ux = np.zeros_like(px)
        uy = np.zeros_like(py)
        uz = np.zeros_like(pz)
        c = 1.0 / (4.0 * np.pi)
        for i in range(len(self.sx)):
            dx = px - self.sx[i]
            dy = py - self.sy[i]
            dz = pz - self.sz[i]
            r2 = dx**2 + dy**2 + dz**2 + 0.002
            r3 = r2 ** 1.5
            q = self.sq[i] * c
            ux += q * dx / r3
            uy += q * dy / r3
            uz += q * dz / r3
        return ux, uy, uz

    # ─── Pressure coefficient ────────────────────────────────────────
    def pressure(self, px, py, pz):
        ux, uy, uz = self.velocity(px, py, pz)
        V2 = (self.U + ux)**2 + uy**2 + uz**2
        return np.clip(1.0 - V2 / self.U**2, -2, 1)

    # ─── Free-surface elevation ──────────────────────────────────────
    def elevation(self, X, Y):
        """η(x,y) from potential flow + Kelvin wake."""
        L, B, T, Fr = self.L, self.B, self.T, self.Fr
        eta = np.zeros_like(X)

        # 1. Source-induced near-field displacement
        for i in range(len(self.sx)):
            dx = X - self.sx[i]
            dy = Y - self.sy[i]
            r2 = dx**2 + dy**2 + 0.001
            q = self.sq[i]
            eta += q / (2 * np.pi * self.U) * dx / r2 * np.exp(-np.sqrt(r2) / (2 * L))
        eta *= Fr * 1.5

        # 2. Bow wave stagnation pile-up
        bow_r = np.sqrt((X - 0.46 * L)**2 + Y**2 + 1e-6)
        eta += 0.12 * Fr * T * np.exp(-4 * bow_r / B)

        # 3. Stern trough
        stern_r = np.sqrt((X + 0.46 * L)**2 + Y**2 + 1e-6)
        eta -= 0.05 * Fr * T * np.exp(-3 * stern_r / B)

        # 4. Kelvin wake — transverse waves
        Xrel = 0.5 * L - X   # distance behind bow (+ve = behind)
        k0 = 1.0 / (Fr**2 * L + 1e-6)
        behind = Xrel > 0
        decay_y = np.exp(-0.2 * np.abs(Y) / (B + 1e-6))
        decay_x = np.exp(-0.06 * Xrel / (L + 1e-6))
        eta += behind * 0.05 * Fr * T * np.cos(k0 * Xrel) * decay_y * decay_x

        # 5. Kelvin wake — diverging waves
        theta = np.arctan2(np.abs(Y) + 1e-6, Xrel + 1e-6)
        kelvin = (theta < KELVIN_HALF) & behind
        r = np.sqrt(Xrel**2 + Y**2 + 1e-6)
        env = np.exp(-0.1 * r / L) / np.sqrt(r / L + 0.15)
        eta += kelvin * 0.03 * Fr * T * np.cos(k0 * r * np.cos(theta)) * env

        return eta

    # ─── Hard-body deflection ────────────────────────────────────────
    def deflect(self, px, py, pz):
        """Push particles outside the hull boundary."""
        L, B, T = self.L, self.B, self.T
        hb = self.hull_halfbeam(px)
        in_x = (px > -0.52 * L) & (px < 0.52 * L)

        # Lateral: push away from centreline
        pen = hb - np.abs(py)   # penetration depth (>0 = inside)
        push = np.where(in_x & (pen > -0.3 * B),
                        0.015 * np.exp(3.0 * np.minimum(pen, 0.2 * B) / B), 0)
        dy = np.sign(py + 1e-8) * push

        # Vertical: push down near keel
        keel_pen = -(pz + T)  # >0 = above keel
        dz = np.where(in_x & (np.abs(py) < hb * 1.3) & (keel_pen > -0.3 * T),
                      -0.005 * np.exp(-np.abs(keel_pen) / (T + 1e-6)), 0)
        return dy, dz


# ═════════════════════════════════════════════════════════════════════════
# CFD Visualisation Widget
# ═════════════════════════════════════════════════════════════════════════

class CFDVisualizationWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        # Ship geometry (normalised viewport coords)
        self.ship_L   = 1.5
        self.ship_B   = 0.3
        self.ship_T   = 0.15
        self.ship_speed = 12.0
        self.ship_Cb  = 0.82

        # Physics engine
        self.flow = None          # type: ShipFlowField

        # GL items
        self.hull_mesh_item    = None
        self.hull_vertices     = None
        self.domain_box_item   = None
        self.light_item        = None
        self.waterline_item    = None

        # Particle scatter items
        self.scat_flow  = None
        self.scat_spray = None
        self.scat_wake  = None
        # Particle arrays
        self.p_flow_pos  = None
        self.p_flow_col  = None
        self.p_spray_pos = None
        self.p_spray_col = None
        self.p_wake_pos  = None
        self.p_wake_col  = None

        # State
        self.resistance_data = {}
        self._vessel_data = {}
        self._hull_adapter = None
        self.light_elevation = 60.0
        self.light_azimuth   = 225.0
        self.light_intensity = 1.0
        self._time = 0.0

        # Build
        self._build_viewport()
        self._setup_hud()
        self._setup_toolbar()
        self._setup_scene()
        self._setup_timers()

    # ── UI ────────────────────────────────────────────────────────────
    def _build_viewport(self):
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setBackgroundColor('#020610')
        self._main_layout.addWidget(self.gl_widget, stretch=10)
        self.gl_widget.setCameraPosition(distance=5.0, elevation=22, azimuth=230)

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

        row.addWidget(QLabel("☀ Işık:", styleSheet=ls))
        self.sl_light_elev = QSlider(Qt.Orientation.Horizontal)
        self.sl_light_elev.setRange(0, 90); self.sl_light_elev.setValue(60)
        self.sl_light_elev.setFixedWidth(90); self.sl_light_elev.setStyleSheet(ss)
        self.sl_light_elev.valueChanged.connect(self._on_light_changed)
        row.addWidget(self.sl_light_elev)

        row.addWidget(QLabel("Yön:", styleSheet=ls))
        self.sl_light_az = QSlider(Qt.Orientation.Horizontal)
        self.sl_light_az.setRange(0, 360); self.sl_light_az.setValue(225)
        self.sl_light_az.setFixedWidth(90); self.sl_light_az.setStyleSheet(ss)
        self.sl_light_az.valueChanged.connect(self._on_light_changed)
        row.addWidget(self.sl_light_az)

        row.addWidget(QLabel("Parlaklık:", styleSheet=ls))
        self.sl_light_int = QSlider(Qt.Orientation.Horizontal)
        self.sl_light_int.setRange(10, 200); self.sl_light_int.setValue(100)
        self.sl_light_int.setFixedWidth(70); self.sl_light_int.setStyleSheet(ss)
        self.sl_light_int.valueChanged.connect(self._on_light_changed)
        row.addWidget(self.sl_light_int)

        btn = QPushButton("Sıfırla")
        btn.setStyleSheet("QPushButton{background:#1a2a3a;color:#7fb3d3;"
                          "border:1px solid #2a4a6a;padding:2px 8px;"
                          "border-radius:3px;font-size:11px;}"
                          "QPushButton:hover{background:#2a3a5a;}")
        btn.clicked.connect(self._reset_light)
        row.addWidget(btn)

        if HAS_PYJET:
            self.btn_pyjet = QPushButton("🌊 PyJet (Real-Time)")
            self.btn_pyjet.setCheckable(True)
            self.btn_pyjet.setStyleSheet(btn.styleSheet())
            self.btn_pyjet.clicked.connect(self._toggle_pyjet)
            row.addWidget(self.btn_pyjet)

        row.addStretch()
        self._main_layout.addWidget(bar, stretch=0)

    # ── PyJet Toggle ──────────────────────────────────────────────────
    def _toggle_pyjet(self):
        if self.btn_pyjet.isChecked():
            print("🚀 Switching to PyJet Engine...")
            self.lbl_stats.setText("PyJet solver initializing...")
            # Trigger rebuild using PyJet
            if self.hull_vertices is not None:
                # Need faces to init PyJet
                pass # updated below
        else:
            print("🔙 Reverting to Output Flow...")
            self._rebuild_flow()
            
        self.update_plot()


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

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        w, h = self.width(), self.height()
        if hasattr(self, 'lbl_stats'):
            self.lbl_stats.adjustSize()
            self.lbl_stats.move(w - self.lbl_stats.width() - 10, 10)
        if hasattr(self, 'lbl_resistance'):
            self.lbl_resistance.adjustSize()
            self.lbl_resistance.move(10, h - self.lbl_resistance.height() - 50)

    # ── Scene ─────────────────────────────────────────────────────────
    def _setup_scene(self):
        self._rebuild_flow()
        self._build_domain()
        self._build_axes()
        self._build_light()
        self._create_scatters()

    def _rebuild_flow(self):
        Fr = self._froude()
        self.flow = ShipFlowField(
            self.ship_L, self.ship_B, self.ship_T, self.ship_Cb, Fr)

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
        g.setSize(x=4.5*L, y=3*B); g.setSpacing(x=L*0.5, y=B*0.5)
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
        el, az = np.radians(self.light_elevation), np.radians(self.light_azimuth)
        d = 8.0
        return (float(d*np.cos(el)*np.sin(az)),
                float(d*np.cos(el)*np.cos(az)),
                float(d*np.sin(el)))

    def _update_light_pos(self):
        self.light_elevation = float(self.sl_light_elev.value())
        self.light_azimuth   = float(self.sl_light_az.value())
        self.light_intensity = self.sl_light_int.value() / 100.0
        a = max(0.02, 0.02 + 0.025 * self.light_intensity)
        self.gl_widget.setBackgroundColor(
            (int(a*2*255), int(a*4*255), int(a*10*255), 255))

    def _create_scatters(self):
        self.scat_flow  = gl.GLScatterPlotItem(size=2.8, pxMode=True)
        self.scat_spray = gl.GLScatterPlotItem(size=1.5, pxMode=True)
        self.scat_wake  = gl.GLScatterPlotItem(size=2.0, pxMode=True)
        self.gl_widget.addItem(self.scat_flow)
        self.gl_widget.addItem(self.scat_spray)
        self.gl_widget.addItem(self.scat_wake)

    # ── Timers ────────────────────────────────────────────────────────
    def _setup_timers(self):
        self.frame_count = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self._fps_update)
        self.fps_timer.start(1000)

    # ── Physics helpers ───────────────────────────────────────────────
    def _froude(self):
        V = self.ship_speed * 0.5144
        return V / np.sqrt(G * max(self.ship_L * 190, 1))

    def _estimate_resistance(self):
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
                    'Rf_kN': round(r['Rf'], 2),
                    'Rw_kN': round(r['Rw'], 2),
                    'RB_kN': round(r.get('RB', 0), 2),
                    'RA_kN': round(r.get('RA', 0), 2),
                    'Rt_kN': round(r['Rt'], 2),
                    'Froude': round(r.get('Froude_number', 0), 4),
                    'Reynolds': f"{r.get('Reynolds_number', 0):.2e}",
                    'Cf': round(r.get('Cf', 0), 6),
                    'Cw': round(r.get('Cw', 0), 6),
                    'Effective_Power_kW': round(r.get('Pe_kW', 0), 1),
                    'k1': round(r.get('form_factor_k1', 1), 4),
                    'iE': round(r.get('iE', 0), 1),
                }
            except Exception as e:
                print(f"⚠️ Holtrop adapter error: {e}")
        return {'Rf_kN': 0, 'Rw_kN': 0, 'Rt_kN': 0, 'RA_kN': 0,
                'Froude': 0, 'Reynolds': '0', 'Cf': 0, 'Cw': 0,
                'Effective_Power_kW': 0}

    # ── Public API ────────────────────────────────────────────────────
    def update_hull_geometry(self, stl_path, vessel_data=None):
        """Load hull and rebuild physics interaction."""
        if vessel_data:
            loa   = float(vessel_data.get('loa', 190))
            beam  = float(vessel_data.get('beam', 32))
            draft = float(vessel_data.get('draft', 12.5))
            self.ship_Cb = float(vessel_data.get('cb', 0.82))
            self.ship_speed = float(vessel_data.get('speed', 12))
            self.ship_L = 1.5
            self.ship_B = 1.5 * beam / max(loa, 1)
            self.ship_T = 1.5 * draft / max(loa, 1)
            self._vessel_data = dict(vessel_data)

        # Rebuild physics
        self._rebuild_flow()
        self._build_domain()

        # Remove old hull
        if self.hull_mesh_item:
            self.gl_widget.removeItem(self.hull_mesh_item)
            self.hull_mesh_item = None
        if self.waterline_item:
            self.gl_widget.removeItem(self.waterline_item)
            self.waterline_item = None

        if not stl_path or not os.path.exists(stl_path) or not HAS_STL:
            return

        try:
            hull = stl_mesh.Mesh.from_file(stl_path)
            verts = hull.vectors.reshape(-1, 3).astype(np.float32)
            nf = len(hull.vectors)
            faces = np.arange(nf * 3).reshape(-1, 3).astype(np.uint32)

            # Centre & scale
            verts -= verts.mean(axis=0)
            ext = verts.max(0) - verts.min(0)
            verts *= self.ship_L / max(ext[0], 1e-3)
            verts[:, 0] += 0.5 * self.ship_L
            verts[:, 2] -= verts[:, 2].max() + self.ship_T

            self.hull_vertices = verts.copy()
            self.hull_faces = faces.copy() # Store for PyJet

            # Colours: anti-fouling red below WL, steel grey above
            fz = verts[faces[:, 0], 2]
            below = fz < 0
            cols = np.zeros((nf, 4), dtype=np.float32)
            cols[below]  = [0.50, 0.12, 0.10, 0.92]
            cols[~below] = [0.42, 0.44, 0.50, 0.92]

            self.hull_mesh_item = gl.GLMeshItem(
                vertexes=verts, faces=faces, faceColors=cols,
                smooth=True, drawEdges=False, shader='shaded')
            self.gl_widget.addItem(self.hull_mesh_item)



            # Waterline contour
            self._draw_waterline(verts, faces)
            print(f"✅ Hull loaded ({nf} faces)")
        except Exception as e:
            print(f"⚠️ STL error: {e}")

    def _draw_waterline(self, verts, faces):
        pts = []
        for tri in faces:
            v = [verts[tri[0]], verts[tri[1]], verts[tri[2]]]
            for i in range(3):
                a, b = v[i], v[(i+1)%3]
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

    def update_plot(self, results):
        """Seed physics-based particles and start simulation."""
        L, B, T = self.ship_L, self.ship_B, self.ship_T
        Fr = self._froude()

        # Check which engine we are using
        use_pyjet = getattr(self, 'btn_pyjet', None) and self.btn_pyjet.isChecked()

        if use_pyjet and HAS_PYJET and hasattr(self, 'hull_faces'):
            if not isinstance(self.flow, PyJetFlowField):
                self.flow = PyJetFlowField(self.hull_vertices, self.hull_faces, L, B, T, self.ship_speed)
        elif use_pyjet == False and isinstance(self.flow, PyJetFlowField):
            self._rebuild_flow() # Revert to numpy potential flow

        if use_pyjet:
            # PyJet will handle particles entirely. Clear numpy dummy particles
            self.p_spray_pos = np.zeros((0, 3), dtype=np.float32)
            self.p_wake_pos = np.zeros((0, 3), dtype=np.float32)
        else:
            # ── 1. Streamline flow particles (underwater) ─────────────────
            nf = 8000
            # Seed from inlet plane, spread in Y-Z
            px = np.random.uniform(0.7 * L, 1.0 * L, nf)  # start upstream (ahead of bow)
            py = np.random.normal(0, B * 0.7, nf)
            py = np.clip(py, -1.4 * B, 1.4 * B)
            pz = np.random.uniform(-1.2 * T, -0.01 * T, nf)
    
            # Pre-deflect around hull at seed
            if getattr(self, 'flow', None) and not isinstance(self.flow, PyJetFlowField):
                dy, dz = self.flow.deflect(px, py, pz)
                py += dy * 30; pz += dz * 30
    
            Cp = self.flow.pressure(px, py, pz) if self.flow else np.zeros(nf)
            self.p_flow_pos = np.column_stack([px, py, pz]).astype(np.float32)
            self.p_flow_col = self._cp_color(Cp)
    
            # ── 2. Bow spray ──────────────────────────────────────────────
            ns = 600
            sx = np.random.normal(0.44 * L, 0.04 * L, ns)
            sy = np.random.normal(0, B * 0.25, ns)
            sz = np.random.exponential(T * 0.12, ns) * (1 + 4 * Fr)
            br = np.random.uniform(0.8, 1.0, ns)
            self.p_spray_pos = np.column_stack([sx, sy, sz]).astype(np.float32)
            self.p_spray_col = np.column_stack([br, br, br*.95, np.full(ns, .4)]).astype(np.float32)
    
            # ── 3. Wake tracers ───────────────────────────────────────────
            nw = 2500
            wx = np.random.uniform(-4 * L, -0.05 * L, nw)
            wy_max = np.abs(wx - 0.5 * L) * np.tan(KELVIN_HALF)
            wy = np.random.uniform(-1, 1, nw) * wy_max
            wz = self.flow.elevation(wx, wy) if self.flow else np.zeros(nw)
            wz += np.random.uniform(-0.003 * T, 0.005 * T, nw)
            d = np.sqrt((wx - 0.5*L)**2 + wy**2 + 1e-6)
            ints = np.interp(d, [0, 3*L], [0.95, 0.15])
            self.p_wake_pos = np.column_stack([wx, wy, wz]).astype(np.float32)
            self.p_wake_col = np.column_stack([ints*.7, ints*.85, ints*.5, np.full(nw, .5)]).astype(np.float32)
    
            # Push to GPU
            self.scat_flow.setData(pos=self.p_flow_pos, color=self.p_flow_col)
            self.scat_spray.setData(pos=self.p_spray_pos, color=self.p_spray_col)
            self.scat_wake.setData(pos=self.p_wake_pos, color=self.p_wake_col)

        self.resistance_data = self._estimate_resistance()
        self._update_hud()
        if not self.timer.isActive():
            self.timer.start(16)

    def _cp_color(self, Cp):
        """Pressure → CFD rainbow: red(stag) → green(free) → blue(suction)."""
        n = len(Cp)
        t = np.clip(Cp, -1, 1) * 0.5 + 0.5
        c = np.zeros((n, 4), dtype=np.float32)
        c[:, 0] = np.clip(1.6 * t - 0.3, 0, 1) * 0.85
        c[:, 1] = np.clip(1 - 3 * np.abs(t - 0.45), 0, 1) * 0.75 + 0.1
        c[:, 2] = np.clip(1.4 - 1.8 * t, 0, 1) * 0.85
        c[:, 3] = 0.7
        return c

    # ── HUD ───────────────────────────────────────────────────────────
    def _update_hud(self):
        r = self.resistance_data
        if not r:
            return
        txt = (f"⚓ Direnç Analizi (Holtrop-Mennen 1984)\n"
               f"──────────────────────────────\n"
               f"  Sürtünme (1+k1)  Rf = {r['Rf_kN']} kN\n"
               f"  Dalga Direnci    Rw = {r['Rw_kN']} kN\n")
        if r.get('RA_kN', 0) > 0:
            txt += f"  Korelasyon       RA = {r['RA_kN']} kN\n"
        txt += (f"  Toplam Direnç    Rt = {r['Rt_kN']} kN\n"
                f"  Efektif Güç     Pe = {r['Effective_Power_kW']} kW\n"
                f"──────────────────────────────\n"
                f"  Froude No.       Fr = {r['Froude']}\n"
                f"  Reynolds No.     Re = {r['Reynolds']}\n"
                f"  Cf = {r['Cf']}   Cw = {r['Cw']}")
        if 'k1' in r:
            txt += f"\n  (1+k1) = {r['k1']}  iE = {r.get('iE','-')}°"
        self.lbl_resistance.setText(txt)
        self.lbl_resistance.adjustSize()
        self.lbl_resistance.show()
        self.lbl_resistance.move(10, self.height() - self.lbl_resistance.height() - 50)

    # ── Animation (per-frame physics) ─────────────────────────────────
    def _tick(self):
        dt = 0.04
        self._time += dt
        spd = max(self.ship_speed / 25.0, 0.3) * dt
        L, B, T = self.ship_L, self.ship_B, self.ship_T

        if HAS_PYJET and isinstance(getattr(self, 'flow', None), PyJetFlowField):
            # PyJet physics update
            self.flow.step()
            particles = self.flow.get_particles()
            if particles is not None and len(particles) > 0:
                # Color based on Z height
                col = np.zeros((len(particles), 4), dtype=np.float32)
                col[:, 0] = 0.1; col[:, 1] = 0.5; col[:, 2] = 0.9; col[:, 3] = 0.7
                self.scat_flow.setData(pos=particles, color=col)
                self.p_flow_pos = particles # For HUD point count logging
        else:
            # ── Flow particles: advect through velocity field ─────────────
            if self.p_flow_pos is not None and self.flow:
                p = self.p_flow_pos
                ux, uy, uz = self.flow.velocity(p[:, 0], p[:, 1], p[:, 2])
    
                # Free-stream (backward) + perturbation
                p[:, 0] -= spd + ux * dt * 0.5
                p[:, 1] += uy * dt * 0.8
                p[:, 2] += uz * dt * 0.4
    
                # Hard-body collision with hull
                dy, dz = self.flow.deflect(p[:, 0], p[:, 1], p[:, 2])
                p[:, 1] += dy
                p[:, 2] += dz
    
                # Force particles to stay inside hull collision boundary
                inside = self.flow.is_inside_hull(p[:, 0], p[:, 1])
                if np.any(inside):
                    hb = self.flow.hull_halfbeam(p[inside, 0])
                    p[inside, 1] = np.sign(p[inside, 1] + 1e-8) * (hb + 0.02 * B)
    
                # Keep underwater
                p[:, 2] = np.minimum(p[:, 2], -0.005 * T)
    
                # Recycle escaped particles at inlet
                esc = (p[:, 0] < -1.5 * L) | (np.abs(p[:, 1]) > 1.5 * B) | (p[:, 2] < -1.5 * T)
                nr = np.sum(esc)
                if nr > 0:
                    p[esc, 0] = np.random.uniform(0.7 * L, 1.0 * L, nr)
                    p[esc, 1] = np.random.normal(0, B * 0.5, nr)
                    p[esc, 2] = np.random.uniform(-1.2 * T, -0.01 * T, nr)
    
                # Colour from pressure
                Cp = self.flow.pressure(p[:, 0], p[:, 1], p[:, 2])
                self.p_flow_col = self._cp_color(Cp)
                self.scat_flow.setData(pos=p, color=self.p_flow_col)
    
            # ── Spray (rise + drift) ──────────────────────────────────────
            if self.p_spray_pos is not None:
                self.p_spray_pos[:, 2] += 0.005
                self.p_spray_pos[:, 0] -= spd * 0.3
                self.p_spray_pos[:, 1] += np.random.normal(0, 0.002, len(self.p_spray_pos))
                esc = (self.p_spray_pos[:, 2] > T * 3) | (self.p_spray_pos[:, 0] < -L)
                nr = np.sum(esc)
                if nr > 0:
                    self.p_spray_pos[esc, 0] = np.random.normal(0.44*L, 0.04*L, nr)
                    self.p_spray_pos[esc, 1] = np.random.normal(0, B*0.25, nr)
                    self.p_spray_pos[esc, 2] = np.random.exponential(T*0.05, nr)
                self.scat_spray.setData(pos=self.p_spray_pos, color=self.p_spray_col)
    
            # ── Wake tracers ──────────────────────────────────────────────
            if self.p_wake_pos is not None:
                self.p_wake_pos[:, 0] -= spd * 0.4
                self.p_wake_pos[:, 1] += np.sign(self.p_wake_pos[:, 1]) * 0.0008
                esc = self.p_wake_pos[:, 0] < -4.5 * L
                nr = np.sum(esc)
                if nr > 0:
                    self.p_wake_pos[esc, 0] = np.random.uniform(-0.05*L, 0.4*L, nr)
                    self.p_wake_pos[esc, 1] = np.random.uniform(-0.15*B, 0.15*B, nr)
                    self.p_wake_pos[esc, 2] = np.random.uniform(-0.005*T, 0.005*T, nr)
                self.scat_wake.setData(pos=self.p_wake_pos, color=self.p_wake_col)


        self.frame_count += 1



    # ── FPS / Light ───────────────────────────────────────────────────
    def _fps_update(self):
        fps = self.frame_count; self.frame_count = 0
        np_count = sum(len(a) for a in [self.p_flow_pos, self.p_spray_pos, self.p_wake_pos] if a is not None)
        self.lbl_stats.setText(
            f"FPS: {fps}\n"
            f"Physics: {'PyJet FLIP 3D' if type(getattr(self, 'flow', None)).__name__ == 'PyJetFlowField' else 'Potential Flow + Kelvin Wake'}\n"
            f"Particles: {np_count}")
        self.lbl_stats.adjustSize()
        self.lbl_stats.move(self.width() - self.lbl_stats.width() - 10, 10)

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
