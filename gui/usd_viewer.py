"""
USD Viewer Panel — Stage Inspector & Prim Browser
===================================================

Provides a PyQt6 widget for inspecting USD/USDA hull geometry files.
When pxr (usd-core) is available, provides full stage traversal;
otherwise falls back to text-mode USDA parsing.

Responsibilities:
  1. Display USD stage hierarchy and prim properties
  2. Browse prim tree and inspect attributes
  3. Export/load USDA files
  4. Optional flow field overlay (if pxr available)

Usage:
    panel = USDViewerPanel()
    panel.load_usd_stage("path/to/hull.usda")
"""

import os
from typing import Dict, Optional, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QGroupBox, QComboBox,
    QSplitter, QTextEdit, QSlider, QCheckBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor


# ── Optional USD imports ─────────────────────────────────────────────────────
try:
    from pxr import Usd, UsdGeom, Sdf, Vt, Gf
    HAS_USD = True
except ImportError:
    HAS_USD = False


class USDStageManager:
    """
    Manages a USD stage: loading, prim traversal, material assignment,
    and flow field overlay.

    Works with or without the pxr library — falls back to USDA text parsing.
    """

    def __init__(self):
        self.stage = None
        self.stage_path: Optional[str] = None
        self.prim_tree: List[Dict] = []

    def load(self, usd_path: str) -> bool:
        """Load a USD stage from file."""
        self.stage_path = usd_path

        if HAS_USD:
            try:
                self.stage = Usd.Stage.Open(usd_path)
                self.prim_tree = self._traverse_prims(self.stage.GetPseudoRoot())
                return True
            except Exception as e:
                print(f"⚠️ USD stage load error: {e}")
                return False
        else:
            # Fallback: parse USDA as text
            return self._parse_usda_text(usd_path)

    def _traverse_prims(self, prim) -> List[Dict]:
        """Recursively traverse USD stage prims."""
        result = []
        for child in prim.GetChildren():
            info = {
                'name': child.GetName(),
                'path': str(child.GetPath()),
                'type': child.GetTypeName(),
                'properties': {},
                'children': self._traverse_prims(child),
            }
            # Collect key properties
            for prop in child.GetProperties():
                try:
                    val = prop.Get()
                    info['properties'][prop.GetName()] = str(val)[:100]
                except Exception:
                    pass
            result.append(info)
        return result

    def _parse_usda_text(self, usda_path: str) -> bool:
        """Parse USDA text file for prim hierarchy (no pxr required)."""
        if not os.path.exists(usda_path):
            return False

        try:
            with open(usda_path, 'r', encoding='utf-8') as f:
                content = f.read()

            self.prim_tree = []
            lines = content.split('\n')
            current_path = []

            for line in lines:
                stripped = line.strip()
                if stripped.startswith('def '):
                    # Parse: def TypeName "PrimName"
                    parts = stripped.split('"')
                    prim_type = stripped.split()[1] if len(stripped.split()) > 1 else 'Unknown'
                    prim_name = parts[1] if len(parts) > 1 else 'unnamed'
                    current_path.append(prim_name)
                    self.prim_tree.append({
                        'name': prim_name,
                        'path': '/' + '/'.join(current_path),
                        'type': prim_type,
                        'properties': {},
                        'children': [],
                    })
                elif stripped == '}' and current_path:
                    current_path.pop()

            return True
        except Exception as e:
            print(f"⚠️ USDA parse error: {e}")
            return False

    def add_flow_field_prim(self, flow_data: Dict) -> bool:
        """
        Add or update a flow field visualisation prim on the USD stage.

        Creates a PointInstancer or Points prim with velocity/pressure attributes.
        """
        if not HAS_USD or self.stage is None:
            return False

        try:
            # Define flow field prim
            flow_path = Sdf.Path("/Hull_Xform/FlowField")
            flow_prim = self.stage.DefinePrim(flow_path, "Points")
            points_attr = flow_prim.CreateAttribute("points", Sdf.ValueTypeNames.Point3fArray)

            # Build points from grid
            import numpy as np
            X = flow_data.get('X')
            Y = flow_data.get('Y')
            if X is not None and Y is not None:
                pts = []
                for i in range(0, X.shape[0], 4):
                    for j in range(0, X.shape[1], 4):
                        pts.append(Gf.Vec3f(float(X[i, j]), float(Y[i, j]), 0.0))
                points_attr.Set(Vt.Vec3fArray(pts))

            self.stage.GetRootLayer().Save()
            return True
        except Exception as e:
            print(f"⚠️ Flow field prim error: {e}")
            return False

    def update_material(self, prim_path: str, color: tuple = (0.15, 0.35, 0.65)):
        """Update display color of a prim."""
        if not HAS_USD or self.stage is None:
            return

        try:
            prim = self.stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                color_attr = prim.GetAttribute("primvars:displayColor")
                if color_attr:
                    color_attr.Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
                    self.stage.GetRootLayer().Save()
        except Exception:
            pass

    def get_stage_info(self) -> Dict:
        """Get metadata about the loaded stage."""
        info = {
            'path': self.stage_path or 'None',
            'has_pxr': HAS_USD,
            'prim_count': self._count_prims(self.prim_tree),
        }

        if HAS_USD and self.stage:
            root = self.stage.GetRootLayer()
            info['up_axis'] = UsdGeom.GetStageUpAxis(self.stage)
            info['meters_per_unit'] = UsdGeom.GetStageMetersPerUnit(self.stage)

        return info

    def _count_prims(self, tree: List[Dict]) -> int:
        count = len(tree)
        for node in tree:
            count += self._count_prims(node.get('children', []))
        return count


class USDViewerPanel(QWidget):
    """
    GUI panel for USD stage inspection and prim browsing.

    Layout:
      ┌─────────────────────────────────────────┐
      │  📐 USD Stage Viewer                      │
      ├──────────────┬──────────────────────────┤
      │  Prim Tree   │  Properties / Info       │
      │              │                          │
      │  Hull_Xform  │  Type: Mesh              │
      │   ├─ Hull    │  Points: 12,345          │
      │   └─ Flow    │  Material: UsdPreview    │
      ├──────────────┴──────────────────────────┤
      │  Actions: Load | Export | Refresh       │
      └─────────────────────────────────────────┘
    """

    sync_requested = pyqtSignal(dict)
    stage_loaded   = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stage_manager = USDStageManager()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ── Header ──
        header = QLabel("📐 USD Stage Viewer")
        header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        header.setStyleSheet("color: #58a6ff;")
        layout.addWidget(header)

        # Status indicator
        self.status_label = QLabel()
        self._update_status()
        layout.addWidget(self.status_label)

        # ── Splitter (Tree + Info) ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Prim tree
        tree_group = QGroupBox("Stage Hierarchy")
        tree_layout = QVBoxLayout(tree_group)
        self.prim_tree = QTreeWidget()
        self.prim_tree.setHeaderLabels(["Prim", "Type"])
        self.prim_tree.setColumnWidth(0, 150)
        self.prim_tree.itemClicked.connect(self._on_prim_selected)
        tree_layout.addWidget(self.prim_tree)
        splitter.addWidget(tree_group)

        # Right: Properties
        info_group = QGroupBox("Properties")
        info_layout = QVBoxLayout(info_group)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFont(QFont("Consolas", 9))
        self.info_text.setStyleSheet("background: #0d1117; color: #c9d1d9;")
        info_layout.addWidget(self.info_text)
        splitter.addWidget(info_group)

        splitter.setSizes([200, 300])
        layout.addWidget(splitter)

        # ── Controls ──
        controls = QHBoxLayout()

        self.btn_load = QPushButton("📂 Load USD")
        self.btn_load.setStyleSheet(
            "QPushButton { background: #1f6feb; color: white; padding: 6px 12px; "
            "border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #388bfd; }")
        self.btn_load.clicked.connect(self._on_load_clicked)
        controls.addWidget(self.btn_load)

        self.btn_sync = QPushButton("🔄 Sync State")
        self.btn_sync.setStyleSheet(
            "QPushButton { background: #238636; color: white; padding: 6px 12px; "
            "border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #2ea043; }")
        self.btn_sync.clicked.connect(self._on_sync_clicked)
        controls.addWidget(self.btn_sync)

        self.btn_export = QPushButton("💾 Export USDA")
        self.btn_export.setStyleSheet(
            "QPushButton { background: #30363d; color: #c9d1d9; padding: 6px 12px; "
            "border-radius: 4px; }"
            "QPushButton:hover { background: #484f58; }")
        self.btn_export.clicked.connect(self._on_export_clicked)
        controls.addWidget(self.btn_export)

        layout.addLayout(controls)

        # ── Display Options ──
        options_group = QGroupBox("Display")
        options_layout = QHBoxLayout(options_group)

        self.chk_wireframe = QCheckBox("Wireframe")
        self.chk_wireframe.setStyleSheet("color: #c9d1d9;")
        options_layout.addWidget(self.chk_wireframe)

        self.chk_normals = QCheckBox("Normals")
        self.chk_normals.setStyleSheet("color: #c9d1d9;")
        options_layout.addWidget(self.chk_normals)

        self.chk_flow_overlay = QCheckBox("Flow Overlay")
        self.chk_flow_overlay.setChecked(True)
        self.chk_flow_overlay.setStyleSheet("color: #c9d1d9;")
        options_layout.addWidget(self.chk_flow_overlay)

        self.combo_material = QComboBox()
        self.combo_material.addItems(["UsdPreviewSurface", "Wireframe"])
        self.combo_material.setStyleSheet("color: #c9d1d9; background: #21262d;")
        options_layout.addWidget(QLabel("Material:"))
        options_layout.addWidget(self.combo_material)

        layout.addWidget(options_group)

    def _update_status(self):
        """Update connection status indicator."""
        if HAS_USD:
            self.status_label.setText("✅ USD (pxr) Available")
            self.status_label.setStyleSheet("color: #238636; font-size: 10px;")
        else:
            self.status_label.setText("ℹ️ USD (pxr) Not Installed — Text Mode")
            self.status_label.setStyleSheet("color: #d29922; font-size: 10px;")

    def load_usd_stage(self, usd_path: str):
        """Load and display a USD stage."""
        success = self.stage_manager.load(usd_path)
        self.prim_tree.clear()

        if success:
            self._populate_tree(self.stage_manager.prim_tree, self.prim_tree)
            info = self.stage_manager.get_stage_info()
            self.info_text.setPlainText(
                f"Stage: {info['path']}\n"
                f"Prims: {info['prim_count']}\n"
                f"USD Library: {'pxr' if info['has_pxr'] else 'text-mode'}\n"
            )
            self.stage_loaded.emit(usd_path)
        else:
            self.info_text.setPlainText(f"❌ Failed to load: {usd_path}")

    def _populate_tree(self, prims: List[Dict],
                       parent_widget):
        """Recursively populate QTreeWidget from prim data."""
        for prim_info in prims:
            if isinstance(parent_widget, QTreeWidget):
                item = QTreeWidgetItem(parent_widget)
            else:
                item = QTreeWidgetItem(parent_widget)

            item.setText(0, prim_info['name'])
            item.setText(1, prim_info.get('type', ''))
            item.setData(0, Qt.ItemDataRole.UserRole, prim_info)

            # Recurse children
            for child in prim_info.get('children', []):
                self._populate_tree([child], item)

    def _on_prim_selected(self, item, column):
        """Show properties when a prim is selected."""
        prim_info = item.data(0, Qt.ItemDataRole.UserRole)
        if prim_info:
            text = f"Path: {prim_info.get('path', '?')}\n"
            text += f"Type: {prim_info.get('type', '?')}\n"
            text += f"{'─' * 30}\n"
            for k, v in prim_info.get('properties', {}).items():
                text += f"  {k}: {v}\n"
            self.info_text.setPlainText(text)

    def _on_load_clicked(self):
        """Handle Load USD button."""
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "USD Dosyası Aç", "",
            "USD Files (*.usda *.usd *.usdc);;All Files (*)")
        if path:
            self.load_usd_stage(path)

    def _on_sync_clicked(self):
        """Sync current GUI state to USD stage."""
        self.sync_requested.emit({
            'wireframe': self.chk_wireframe.isChecked(),
            'show_normals': self.chk_normals.isChecked(),
            'flow_overlay': self.chk_flow_overlay.isChecked(),
            'material': self.combo_material.currentText(),
        })

    def _on_export_clicked(self):
        """Export current stage to USDA."""
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "USDA Kaydet", "",
            "USD ASCII (*.usda);;All Files (*)")
        if path and HAS_USD and self.stage_manager.stage:
            try:
                self.stage_manager.stage.GetRootLayer().Export(path)
                self.info_text.setPlainText(f"✅ Exported to: {path}")
            except Exception as e:
                self.info_text.setPlainText(f"❌ Export failed: {e}")

    def sync_flow_field(self, flow_data: Dict):
        """Overlay CFD flow field data onto the USD stage."""
        if self.chk_flow_overlay.isChecked():
            success = self.stage_manager.add_flow_field_prim(flow_data)
            if success:
                self.info_text.append("\n✅ Flow field synced to USD stage.")
            else:
                self.info_text.append("\nℹ️ Flow field sync skipped (no pxr).")
