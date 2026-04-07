import sys
import os
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTreeWidget, QTreeWidgetItem, QSplitter, QLabel, 
                             QFrame, QMenu, QLineEdit, QFormLayout, 
                             QGroupBox, QStackedWidget, QPushButton, QTextEdit,
                             QScrollArea, QSizePolicy, QTreeView, QFrame, QComboBox, 
                             QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar,
                             QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
from gui.styles import DARK_THEME_QSS
from gui.styles import DARK_THEME_QSS
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List, Optional, Dict
# Note: Ensure gui/ribbon.py and gui/bottom_panel.py exist
# --- STABLE IMPORTS ---
from gui.ribbon import RibbonWidget
from gui.bottom_panel import BottomPanel
from gui.model_builder_model import SmartModel, NodeType
from agents.multi_objective_optimizer import MultiObjectiveOptimizer
from agents.asset_manager import AssetManager  
from agents.climate_guardian import ClimateGuardian  

# --- VOLATILE IMPORTS (3D / OpenGL / pyqtgraph) ---
try:
    from gui.cfd_widget import CFDVisualizationWidget
    from gui.model_viewer_3d import ModelViewer3D
except ImportError as e:
    # Fallback if 3D rendering imports fail (e.g. missing pyqtgraph or PyOpenGL)
    import traceback
    traceback.print_exc()
    print(f"3D GÖRSELLEŞTİRME YÜKLENEMEDİ: {e}")
    print("Warning: Custom 3D widgets not found, using placeholders.")
    
    class CFDVisualizationWidget(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout()
            lbl = QLabel("CFD Ekranı (Kütüphane Eksik - 'pyqtgraph' veya 'PyOpenGL' yükleyin)")
            lbl.setStyleSheet("color: red; font-weight: bold;")
            layout.addWidget(lbl)
            self.setLayout(layout)
        def update_plot(self, *args, **kwargs): pass
        
    class ModelViewer3D(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout()
            lbl = QLabel("3D Gösterici (Kütüphane Eksik - 'PyOpenGL' yükleyin)")
            lbl.setStyleSheet("color: red; font-weight: bold;")
            layout.addWidget(lbl)
            self.setLayout(layout)
        def update_vessel_hull(self, *args): pass
        def load_usd(self, *args): pass
        def load_stl(self, *args): pass
        def add_shape(self, *args): pass


try:
    from core.geometry.hull_adapter import RetrosimHullAdapter
    HAS_GEOMETRY_ENGINE = True
except ImportError:
    HAS_GEOMETRY_ENGINE = False
    print("Warning: Geometry Engine not available.")

class TrainingWorker(QThread):
    """
    Eğitimi arka planda çalıştırır.
    """
    def __init__(self, agent, config):
        super().__init__()
        self.agent = agent
        self.config = config

    def run(self):
        # Ajanın eğitim fonksiyonunu başlat
        # Bu işlem uzun sürse bile arayüz donmaz
        self.agent.train_models(self.config)

class CFDWorker(QThread):
    """
    CFD Analizini arka planda çalıştırır.
    """
    def __init__(self, agent, vessel_data):
        super().__init__()
        self.agent = agent
        self.vessel_data = vessel_data

    def run(self):
        self.agent.run_analysis(self.vessel_data)

class GeometryWorker(QThread):
    """
    Arka planda parametrik gövde mesh'i üretir.
    RetrosimHullAdapter kullanarak UI parametrelerinden STL oluşturur.
    """
    finished_signal = pyqtSignal(str)    # stl_path
    error_signal = pyqtSignal(str)       # error message
    progress_signal = pyqtSignal(int, str)  # (percent, message)

    def __init__(self, vessel_data: dict, output_path: str = None):
        super().__init__()
        self.vessel_data = vessel_data
        self.output_path = output_path

    def run(self):
        try:
            if not HAS_GEOMETRY_ENGINE:
                self.error_signal.emit("Geometry Engine yüklü değil (core.geometry modülü bulunamadı).")
                return

            self.progress_signal.emit(10, "Parametrik gövde geometrisi hesaplanıyor...")

            adapter = RetrosimHullAdapter()
            adapter.set_from_ui(self.vessel_data)

            self.progress_signal.emit(30, "Design Vector oluşturuldu. Mesh üretiliyor...")

            # Generate STL
            stl_path = adapter.generate_stl(
                output_path=self.output_path,
                n_stations=31,
                n_waterlines=15
            )

            # Generate USD (Omniverse)
            usd_path = None
            if self.output_path:
                usd_path = self.output_path.replace('.stl', '.usda')
            
            usd_path = adapter.generate_usda(
                output_path=usd_path,
                n_stations=31,
                n_waterlines=15
            )

            # Update vessel data references
            if isinstance(self.vessel_data, dict):
                self.vessel_data['stl_path'] = stl_path
                self.vessel_data['usd_path'] = usd_path
            elif hasattr(self.vessel_data, 'stl_path'):
                self.vessel_data.stl_path = stl_path
                self.vessel_data.usd_path = usd_path

            self.progress_signal.emit(70, "STL ve USD (Omniverse) mesh kaydedildi. Volumetrik hesaplar yapılıyor...")

            # Extract features for logging
            features = adapter.extract_ml_features()
            vol = features.get('displaced_volume', 0)
            wsa = features.get('wetted_surface_area', 0)
            cb = features.get('Cb_actual', 0)

            self.progress_signal.emit(90, f"Hacim: {vol:.0f} m³ | WSA: {wsa:.0f} m² | Cb: {cb:.3f}")
            self.progress_signal.emit(100, "Geometri üretimi tamamlandı!")
            self.finished_signal.emit(stl_path)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(str(e))

class ClimateWorker(QThread):
    """Climate Guardian analizini arka planda çalıştırır."""
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, agent, vessel_data, target_year):
        super().__init__()
        self.agent = agent
        self.vessel_data = vessel_data
        self.target_year = target_year
    
    def run(self):
        try:
            self.progress_signal.emit(10, "İklim projeksiyonları hesaplanıyor...")
            
            # 1. Environmental conditions projection
            env_conditions = self.agent.project_environmental_conditions(self.target_year)
            self.progress_signal.emit(30, "Çevresel koşullar hesaplandı...")
            
            # 2. Regulatory changes projection
            regulations = self.agent.project_regulatory_changes(self.target_year)
            self.progress_signal.emit(50, "Düzenleyici değişiklikler hesaplandı...")
            
            # 3. Vessel performance impact
            performance = self.agent.project_vessel_performance_impact(self.vessel_data, self.target_year)
            self.progress_signal.emit(70, "Gemi performans etkisi hesaplandı...")
            
            # 4. Climate risk assessment
            risk_assessment = self.agent.calculate_climate_risk_assessment(self.vessel_data)
            self.progress_signal.emit(90, "Risk değerlendirmesi tamamlandı...")
            
            # 5. Temporal analysis (full 2025-2050)
            temporal_analysis = self.agent.generate_temporal_analysis(
                self.vessel_data, 
                start_year=2025, 
                end_year=2050
            )
            
            results = {
                'target_year': self.target_year,
                'environmental_conditions': env_conditions,
                'regulations': regulations,
                'performance_impact': performance,
                'risk_assessment': risk_assessment,
                'temporal_analysis': temporal_analysis
            }
            
            self.progress_signal.emit(100, "İklim analizi tamamlandı!")
            self.finished_signal.emit(results)
            
        except Exception as e:
            self.error_signal.emit(str(e))

# --- DATACLASS DEFINITIONS ---
@dataclass
class VesselData:
    name: str = "M/V SmartCAPEX"
    type: str = "Bulk Carrier"
    dwt: int = 55000
    loa: float = 190.0
    lbp: float = 182.0
    beam: float = 32.2
    draft: float = 12.5
    freeboard: float = 5.5
    depth: float = 18.0
    cb: float = 0.82
    cp: float = 0.84
    cm: float = 0.98
    bow_height: float = 4.0
    stern_height: float = 2.5
    bulb_length: float = 5.5
    bulb_radius: float = 2.2
    stern_shape: float = 0.8
    prop_dia: float = 6.5
    prop_blades: int = 4
    rudder_h: float = 7.0
    speed: float = 12.5
    engine_power: int = 8500
    sfoc: float = 175.0
    opex: int = 4500
    value: int = 15000000
    cii: str = "C"
    eedi: float = 12.5
    age: int = 10
    
    # EANN Environmental & Operational inputs
    wave_height: float = 1.0
    wind_speed: float = 10.0
    current_speed: float = 0.0
    sea_state: int = 3
    load_factor: float = 0.85
    fuel_type: str = "HFO"
    engine_efficiency: float = 0.45
    
    # Performance metrics
    fuel_consumption: float = 15.0
    co2_emission: float = 45.0
    cii_score: float = 4.2
    
    stl_path: Optional[str] = None
    usd_path: Optional[str] = None
    selected_retrofit: List[str] = field(default_factory=list)

@dataclass
class SurrogateConfig:
    model: str = "EANN (Physics-Informed)"
    epochs: int = 1000
    lr: float = 0.001
    anxiety: float = 0.1
    confidence: float = 0.2
    data_path: Optional[str] = None

@dataclass
class OptimizerConfig:
    pop_size: int = 50
    generations: int = 100
    algo: str = "NSGA-II"

@dataclass
class ClimateConfig:
    target_year: int = 2030
    scenario: str = "RCP 4.5"
    carbon_tax: int = 100

@dataclass
class CFDConfig:
    resolution: int = 50
    show_pressure: bool = True
    show_streamlines: bool = True
    inlet_velocity: float = 12.0
    fluid_density: float = 1025.0
    kinematic_viscosity: float = 1.18e-6
    domain_x_mult: float = 5.0
    domain_y_mult: float = 2.0
    domain_z_mult: float = 2.0

@dataclass
class RetrofitConfig:
    selected_retrofit: List[str] = field(default_factory=list)

@dataclass
class RunStatus:
    status: str = "Ready"

@dataclass
class Model3DConfig:
    pass

@dataclass
class AdvancedAnalysisConfig:
    """TOPSIS ve IPSO gelişmiş analiz konfigürasyonu"""
    # TOPSIS Ağırlıkları (toplam = 1.0)
    weight_economic: float = 0.40
    weight_environmental: float = 0.35
    weight_operational: float = 0.25
    
    # IPSO Parametreleri
    ipso_particles: int = 30
    ipso_iterations: int = 50
    
    # Analiz Seçenekleri
    use_topsis: bool = True
    use_ipso: bool = False
    sensitivity_analysis: bool = True

# SETTINGS PANEL
class SettingsManager(QWidget):
    """
    Seçilen NodeType'a göre sağ taraftaki formu dinamik olarak çizer.
    
    Asset Manager Integration:
    - Vessel data validation on field change
    - Template loading for quick setup
    - Auto-imputation for missing data
    - Data quality indicators
    """
    data_changed = pyqtSignal(int, dict) # node_type, data
    quality_updated = pyqtSignal(float, float)  # quality, completeness

    def __init__(self):
        super().__init__()
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(self.main_layout)
        
        # === ASSET MANAGER (Vessel Data Intelligence Layer) ===
        try:
            self.asset_manager = AssetManager()
        except:
            self.asset_manager = None
            print("Warning: AssetManager could not be initialized")
        
        # PROJE VERİSİ (Simülasyon Veritabanı - Hafızada tutulur)
        self.data_store = {
            NodeType.VESSEL: VesselData(),
            NodeType.SURROGATE: SurrogateConfig(),
            NodeType.OPTIMIZER: OptimizerConfig(),
            NodeType.CLIMATE: ClimateConfig(),
            NodeType.CFD: CFDConfig(),
            NodeType.RETROFIT: RetrofitConfig(),
            NodeType.MODEL_3D: Model3DConfig(),
            NodeType.RUN: RunStatus(),
            NodeType.ADVANCED_ANALYSIS: AdvancedAnalysisConfig()
        }
        
        # UI References for dynamic updates
        self.lbl_data_quality = None
        self.btn_impute = None
        self.combo_template = None

    def load_settings(self, node_type, node_text):
        """Dışarıdan çağrılan ana fonksiyon: Ekranı temizle ve yeni formu çiz"""
        self.current_node_type = node_type
        
        # 1. Eski formu temizle
        self._clear_layout()
        
        # 2. Başlık Ekle
        title = QLabel(f"🛠️ {node_text} Ayarları")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #42a5f5; margin-bottom: 10px;")
        self.main_layout.addWidget(title)
        
        # 3. İlgili veri sözlüğünü al (Yoksa boş obje)
        data = self.data_store.get(node_type, None)
        if data is None:
             # Fallback fallback
             print(f"Warning: No dataclass found for node type {node_type}")
             return

        # 4. Tipe göre formu oluştur
        if node_type == NodeType.VESSEL:
            self._form_vessel(data)
        elif node_type == NodeType.SURROGATE:
            self._form_surrogate(data)
        elif node_type == NodeType.OPTIMIZER:
            self._form_optimizer(data)
        elif node_type == NodeType.CLIMATE:
            self._form_climate(data)
        elif node_type == NodeType.CFD:
            self._form_cfd(data)
        elif node_type == NodeType.RETROFIT:
            self._form_retrofit(data)
        elif node_type == NodeType.MODEL_3D:
            self._form_model_3d(data)
        elif node_type == NodeType.RUN:
            self._view_run_status(data)
        elif node_type == NodeType.ADVANCED_ANALYSIS:
            self._form_advanced_analysis(data)
        else:
            lbl = QLabel("Bu öğe için ayar bulunmuyor.")
            lbl.setStyleSheet("color: gray; font-style: italic;")
            self.main_layout.addWidget(lbl)
    
    def _open_file_dialog(self, data):
        from PyQt6.QtWidgets import QFileDialog # Import'u buraya veya en başa ekleyebilirsin
        fname, _ = QFileDialog.getOpenFileName(self, "Veri Seti Seç", "", "Data Files (*.csv *.xlsx)")
        if fname:
            if hasattr(data, "data_path"):
                data.data_path = fname
                self.lbl_path.setText(fname)
                self.lbl_path.setStyleSheet("color: #8fce00; font-size: 10px;")
            else:
                print("Warning: data_path field missing")

    def _open_stl_dialog(self, data):
        from PyQt6.QtWidgets import QFileDialog
        fname, _ = QFileDialog.getOpenFileName(self, "STL Dosyası Seç", "", "3D Files (*.stl)")
        if fname:
            if hasattr(data, "stl_path"):
                data.stl_path = fname
            print(f"STL Path Selected: {fname}")
            self.load_settings(NodeType.VESSEL, "Gemi Veri Girişi")

    def _open_usd_dialog(self, data):
        from PyQt6.QtWidgets import QFileDialog
        fname, _ = QFileDialog.getOpenFileName(self, "USD Dosyası Seç", "", "USD Files (*.usd *.usdc *.usda *.usdz)")
        if fname:
            if hasattr(data, "usd_path"):
                data.usd_path = fname
            if hasattr(data, "stl_path"):
                data.stl_path = None # Clear STL
            print(f"USD Path Selected: {fname}")
            
            self.load_settings(NodeType.VESSEL, "Gemi Veri Girişi")

    # --- ÖZEL FORM TASARIMLARI ---
    
    def _form_vessel(self, data):
        # === ASSET MANAGER: Quick Actions (Template + Impute + Quality) ===
        if self.asset_manager:
            form_quick = self._create_group("🚀 Hızlı Başlangıç")
            
            # Template Selector
            template_names = self.asset_manager.get_template_display_names()
            template_keys = [''] + list(template_names.keys())
            template_labels = ['(Şablon Seç)'] + list(template_names.values())
            
            self.combo_template = QComboBox()
            self.combo_template.addItems(template_labels)
            self.combo_template.setStyleSheet("background-color: #2b2b2b; color: white; border: 1px solid #42a5f5; padding: 5px;")
            self.combo_template.currentIndexChanged.connect(lambda idx: self._on_template_selected(idx, template_keys, data))
            form_quick.addRow("📋 Gemi Şablonu:", self.combo_template)
            
            # Auto-Impute Button
            self.btn_impute = QPushButton("🧮 Eksik Verileri Otomatik Tamamla")
            self.btn_impute.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; padding: 8px; margin-top: 5px;")
            self.btn_impute.setCursor(Qt.CursorShape.PointingHandCursor)
            self.btn_impute.clicked.connect(lambda: self._on_impute_click(data))
            form_quick.addRow(self.btn_impute)
            
            # Data Quality Indicator
            report = self.asset_manager.get_data_quality_report(data)
            quality_text = f"📊 Kalite: {report['quality_score']}% | Tamamlanma: {report['completeness']}%"
            if report['ready_for_analysis']:
                quality_color = "#4CAF50"  # Green
                quality_icon = "✅"
            elif report['quality_score'] >= 50:
                quality_color = "#FFC107"  # Yellow
                quality_icon = "⚠️"
            else:
                quality_color = "#f44336"  # Red
                quality_icon = "❌"
            
            self.lbl_data_quality = QLabel(f"{quality_icon} {quality_text}")
            self.lbl_data_quality.setStyleSheet(f"color: {quality_color}; font-size: 11px; font-weight: bold; padding: 5px; border: 1px dashed {quality_color};")
            form_quick.addRow(self.lbl_data_quality)
            
            # Recommendations
            if report['recommendations']:
                for rec in report['recommendations'][:2]:  # Show max 2 recommendations
                    lbl_rec = QLabel(rec)
                    lbl_rec.setWordWrap(True)
                    lbl_rec.setStyleSheet("color: #aaa; font-size: 10px; font-style: italic;")
                    form_quick.addRow(lbl_rec)
        
        # 1. Fiziksel Boyutlar
        form_phys = self._create_group("Gemi Fiziksel Boyutları")
        self._add_text(form_phys, "Gemi Adı", data, "name")
        self._add_combo(form_phys, "Gemi Tipi", data, "type", ["Bulk Carrier", "Container", "Tanker", "General Cargo", "Coaster", "Yelkenli", "Fastbot"])
        self._add_int(form_phys, "DWT (ton)", data, "dwt")
        self._add_float(form_phys, "LOA (Boy)", data, "loa")
        self._add_float(form_phys, "LBP (Dikmeler Arası)", data, "lbp")
        self._add_float(form_phys, "Genişlik (Beam)", data, "beam")
        draft_w = self._add_float(form_phys, "Draft (T) [Su Çekimi]", data, "draft")
        freeboard_w = self._add_float(form_phys, "Fribord (f) [Su Üstü Kısım]", data, "freeboard")
        depth_w = self._add_float(form_phys, "Derinlik (Depth - D)", data, "depth")
        self._add_float(form_phys, "Blok Katsayısı (Cb)", data, "cb")
        
        # --- Draft + Fribord = Depth Mantığı (UI Kilitleme) ---
        if depth_w and draft_w and freeboard_w:
            depth_w.setReadOnly(True)
            depth_w.setStyleSheet("background-color: #1a1a1a; color: #888; border: 1px solid #333;")
            def update_depth(*args):
                new_depth = draft_w.value() + freeboard_w.value()
                depth_w.setValue(new_depth)
                data.depth = new_depth
            draft_w.valueChanged.connect(update_depth)
            freeboard_w.valueChanged.connect(update_depth)
            # Init update
            update_depth()
        self._add_float(form_phys, "Prizmatik Katsayı (Cp)", data, "cp")
        self._add_float(form_phys, "Midship Katsayısı (Cm)", data, "cm")
        self._add_float(form_phys, "Baş Kasara Yük. (m)", data, "bow_height")
        self._add_float(form_phys, "Kıç Kasara Yük. (m)", data, "stern_height")
        self._add_float(form_phys, "Bulb Boyu (m)", data, "bulb_length")
        self._add_float(form_phys, "Bulb Yarıçapı (m)", data, "bulb_radius")
        self._add_float(form_phys, "Kıç Formu (0-1)", data, "stern_shape")
        self._add_float(form_phys, "Pervane Çapı (m)", data, "prop_dia")
        self._add_int(form_phys, "Pervane Kanat Sayısı", data, "prop_blades")
        self._add_float(form_phys, "Dümen Yüksekliği (m)", data, "rudder_h")
        self._add_int(form_phys, "Yaş (Yıl)", data, "age")
        
        # 3D Model Import (STL)
        self.btn_import_stl = QPushButton("🛳️ 3D Model Import (.stl)")
        self.btn_import_stl.setStyleSheet("background-color: #444; color: white; border: 1px solid #666; padding: 6px; margin-top: 5px;")
        self.btn_import_stl.clicked.connect(lambda: self._open_stl_dialog(data))
        form_phys.addRow(self.btn_import_stl)
        
        if getattr(data, 'stl_path', None):
            lbl_stl = QLabel(f"Yüklü (STL): {os.path.basename(data.stl_path)}")
            lbl_stl.setStyleSheet("color: #8fce00; font-size: 10px;")
            form_phys.addRow(lbl_stl)

        # 3D Model Import (USD)
        self.btn_import_usd = QPushButton("🛸 3D Model Import (.usd)")
        self.btn_import_usd.setStyleSheet("background-color: #444; color: white; border: 1px solid #666; padding: 6px; margin-top: 5px;")
        self.btn_import_usd.clicked.connect(lambda: self._open_usd_dialog(data))
        form_phys.addRow(self.btn_import_usd)

        if getattr(data, 'usd_path', None):
            lbl_usd = QLabel(f"Yüklü (USD): {os.path.basename(data.usd_path)}")
            lbl_usd.setStyleSheet("color: #00ccff; font-size: 10px;")
            form_phys.addRow(lbl_usd)

        # === GEOMETRY ENGINE: Parametrik Gövde Üretimi ===
        form_geom = self._create_group("⚙️ Parametrik Gövde Üretimi (Ship-D)")

        self.btn_generate_hull = QPushButton("🛳️ PARAMETRİK GÖVDE ÜRET (45-Vektör)")
        self.btn_generate_hull.setStyleSheet("""
            QPushButton {
                background-color: #00796B;
                color: white;
                font-weight: bold;
                padding: 12px;
                margin-top: 5px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #00897B;
            }
        """)
        self.btn_generate_hull.setCursor(Qt.CursorShape.PointingHandCursor)
        form_geom.addRow(self.btn_generate_hull)

        # Geometry Progress Bar
        self.geometry_progress_bar = QProgressBar()
        self.geometry_progress_bar.setValue(0)
        self.geometry_progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.geometry_progress_bar.setStyleSheet("""
            QProgressBar {border: 0px; background-color: #333; color: white; height: 15px; border-radius: 4px;}
            QProgressBar::chunk {background-color: #00796B; border-radius: 4px;}
        """)
        form_geom.addRow(self.geometry_progress_bar)

        # Geometry Result Label
        self.geometry_result_label = QLabel("")
        self.geometry_result_label.setWordWrap(True)
        self.geometry_result_label.setStyleSheet("color: #aaa; font-size: 10px; margin-top: 5px;")
        form_geom.addRow(self.geometry_result_label)

        geom_info = QLabel(
            "💡 Bu düğme, yukarıdaki fiziksel boyutlardan 45 parametreli Ship-D Design Vector'ü "
            "oluşturur ve B-spline tabanlı 3D gövde mesh'i (STL) üretir.\n"
            "Üretilen mesh otomatik olarak 3D görüntüleyiciye yüklenir."
        )
        geom_info.setWordWrap(True)
        geom_info.setStyleSheet("color: #666; font-size: 9px; font-style: italic; padding: 5px; border-left: 2px solid #00796B;")
        form_geom.addRow(geom_info)

        # 2. Performans ve Tahrik
        form_perf = self._create_group("Performans ve Tahrik")
        self._add_float(form_perf, "Dizayn Hızı (Knots)", data, "speed")
        self._add_int(form_perf, "Makine Gücü (kW)", data, "engine_power")
        self._add_float(form_perf, "SFOC (g/kWh)", data, "sfoc")
        self._add_float(form_perf, "Motor Verimi", data, "engine_efficiency")
        self._add_combo(form_perf, "Yakıt Tipi", data, "fuel_type", ["HFO", "MDO", "LNG"])
        self._add_float(form_perf, "Mevcut Yakıt Tük. (t/gün)", data, "fuel_consumption")
        self._add_float(form_perf, "Mevcut CO2 Emisyonu (t/gün)", data, "co2_emission")

        # 2.5 Çevresel ve Operasyonel (EANN için)
        form_env = self._create_group("Çevre ve Operasyon (EANN Ön İzleme)")
        self._add_float(form_env, "Dalga Yüksekliği (m)", data, "wave_height")
        self._add_float(form_env, "Rüzgar Hızı (knots)", data, "wind_speed")
        self._add_float(form_env, "Akıntı Hızı (knots)", data, "current_speed")
        self._add_int(form_env, "Deniz Durumu (Sea State)", data, "sea_state")
        self._add_float(form_env, "Yük Faktörü (0-1)", data, "load_factor")

        # 3. Ekonomi ve Çevre
        form_econ = self._create_group("Ekonomi ve Düzenlemeler")
        self._add_int(form_econ, "Yıllık OPEX ($)", data, "opex")
        self._add_int(form_econ, "Gemi Değeri ($)", data, "value")
        self._add_combo(form_econ, "CII Derecesi", data, "cii", ["A", "B", "C", "D", "E"])
        self._add_float(form_econ, "CII Skoru", data, "cii_score")
        self._add_float(form_econ, "EEDI Değeri", data, "eedi")
        
        # === Export/Import Butonları ===
        form_io = self._create_group("📂 Veri Yönetimi")
        
        # Export Button
        self.btn_export_vessel = QPushButton("💾 Gemi Verilerini Kaydet (JSON)")
        self.btn_export_vessel.setStyleSheet("""
            QPushButton {
                background-color: #1565C0; 
                color: white; 
                font-weight: bold; 
                padding: 10px; 
                margin-top: 3px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.btn_export_vessel.setCursor(Qt.CursorShape.PointingHandCursor)
        form_io.addRow(self.btn_export_vessel)
        
        # Import Button
        self.btn_import_vessel = QPushButton("📁 Gemi Verilerini Yükle (JSON)")
        self.btn_import_vessel.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32; 
                color: white; 
                font-weight: bold; 
                padding: 10px; 
                margin-top: 3px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
        """)
        self.btn_import_vessel.setCursor(Qt.CursorShape.PointingHandCursor)
        form_io.addRow(self.btn_import_vessel)
        
        # Info Label
        io_info = QLabel("💡 JSON formatında gemi verilerini paylaşabilir veya yedekleyebilirsiniz.")
        io_info.setWordWrap(True)
        io_info.setStyleSheet("color: #888; font-size: 9px; font-style: italic;")
        form_io.addRow(io_info)
    
    def _on_template_selected(self, index, template_keys, data):
        """Handle template selection from dropdown"""
        if index <= 0 or not self.asset_manager:
            return
        
        template_key = template_keys[index]
        try:
            # Apply template to dataclass
            self.asset_manager.apply_template_to_dataclass(template_key, data)
            
            # Refresh the form to show new values
            self.load_settings(NodeType.VESSEL, "Gemi Veri Girişi")
            
            # Emit data changed signal
            self.data_changed.emit(self.current_node_type, asdict(data))
            
            print(f"✅ Şablon uygulandı: {template_key}")
        except Exception as e:
            print(f"❌ Şablon uygulama hatası: {e}")
    
    def _on_impute_click(self, data):
        """Handle auto-impute button click"""
        if not self.asset_manager:
            return
        
        try:
            # Impute missing data
            self.asset_manager.impute_dataclass(data)
            
            # Refresh the form
            self.load_settings(NodeType.VESSEL, "Gemi Veri Girişi")
            
            # Emit data changed signal
            self.data_changed.emit(self.current_node_type, asdict(data))
            
            # Show confirmation
            report = self.asset_manager.get_data_quality_report(data)
            print(f"✅ Veri tamamlandı! Kalite: {report['quality_score']}%, Tamamlanma: {report['completeness']}%")
            
        except Exception as e:
            print(f"❌ Veri tamamlama hatası: {e}")
    
    def _update_quality_display(self, data):
        """Update the data quality label"""
        if not self.asset_manager or not self.lbl_data_quality:
            return
        
        report = self.asset_manager.get_data_quality_report(data)
        quality_text = f"📊 Kalite: {report['quality_score']}% | Tamamlanma: {report['completeness']}%"
        
        if report['ready_for_analysis']:
            quality_color = "#4CAF50"
            quality_icon = "✅"
        elif report['quality_score'] >= 50:
            quality_color = "#FFC107"
            quality_icon = "⚠️"
        else:
            quality_color = "#f44336"
            quality_icon = "❌"
        
        self.lbl_data_quality.setText(f"{quality_icon} {quality_text}")
        self.lbl_data_quality.setStyleSheet(f"color: {quality_color}; font-size: 11px; font-weight: bold; padding: 5px; border: 1px dashed {quality_color};")

    def _form_surrogate(self, data):
        """Surrogate (Yapay Zeka) Ayar Formu"""
        form = self._create_group("Veri Seti ve Model")
        
        # 1. DOSYA YÜKLEME BUTONU
        self.btn_load = QPushButton("📁 Veri Seti Yükle (.csv / .xlsx)")
        self.btn_load.setStyleSheet("background-color: #444; color: white; border: 1px solid #666; padding: 6px;")
        # Lambda ile tıklama olayını bağlıyoruz
        self.btn_load.clicked.connect(lambda: self._open_file_dialog(data))
        form.addRow(self.btn_load)
        
        # Dosya yolunu gösteren etiket
        self.lbl_path = QLabel(getattr(data, "data_path", "Dosya seçilmedi") or "Dosya seçilmedi")
        self.lbl_path.setStyleSheet("color: #aaa; font-size: 10px; font-style: italic;")
        form.addRow(self.lbl_path)

        # 2. MODEL PARAMETRELERİ
        # Model Tipi
        self._add_combo(form, "Model Tipi", data, "model", 
                       ["EANN (Emotional Neural Network)", "Kriging (Gaussian Process)", "Random Forest"])
        
        # EANN Parametreleri
        self._add_int(form, "Epoch Sayısı", data, "epochs")
        self._add_float(form, "Kaygı (Anxiety)", data, "anxiety")
        self._add_float(form, "Güven (Confidence)", data, "confidence")
        
        # Checkbox
        chk = QCheckBox("GPU Hızlandırma Kullan")
        chk.setChecked(True)
        form.addRow("", chk)

        # 3. İLERLEME ÇUBUĞU (Progress Bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setStyleSheet("""
            QProgressBar {border: 0px; background-color: #333; color: white; height: 15px; border-radius: 4px;}
            QProgressBar::chunk {background-color: #2E8B57; border-radius: 4px;}
        """)
        self.main_layout.addWidget(self.progress_bar)

        # 4. EĞİTİM BUTONU (İşte eksik olan parça bu!)
        self.btn_train = QPushButton("🚀 MODELİ EĞİT")
        self.btn_train.setObjectName("btn_train")
        self.btn_train.setCursor(Qt.CursorShape.PointingHandCursor)
        self.main_layout.addWidget(self.btn_train)

    def _form_optimizer(self, data):
        form = self._create_group("Genetik Algoritma Ayarları")
        
        self._add_combo(form, "Algoritma", data, "algo", ["NSGA-II"])
        self._add_int(form, "Popülasyon", data, "pop_size")
        self._add_int(form, "Jenerasyon", data, "generations")

    def _form_climate(self, data):
        form = self._create_group("İklim Projeksiyonu")
        
        self._add_int(form, "Hedef Yıl", data, "target_year")
        self._add_combo(form, "Senaryo", data, "scenario", ["RCP 2.6", "RCP 4.5", "RCP 8.5"])
        self._add_int(form, "Karbon Vergisi ($)", data, "carbon_tax")
        
        # Analiz Parametreleri Grubu
        form2 = self._create_group("Analiz Seçenekleri")
        
        chk_env = QCheckBox("Çevresel Koşul Projeksiyonu (Dalga, Rüzgar)")
        chk_env.setChecked(True)
        form2.addRow(chk_env)
        
        chk_reg = QCheckBox("Düzenleyici Değişiklik Projeksiyonu (IMO, ETS)")
        chk_reg.setChecked(True)
        form2.addRow(chk_reg)
        
        chk_perf = QCheckBox("Gemi Performans Etkisi Analizi")
        chk_perf.setChecked(True)
        form2.addRow(chk_perf)
        
        chk_risk = QCheckBox("Kapsamlı Risk Değerlendirmesi")
        chk_risk.setChecked(True)
        form2.addRow(chk_risk)
        
        # İlerleme Çubuğu
        self.climate_progress_bar = QProgressBar()
        self.climate_progress_bar.setValue(0)
        self.climate_progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.climate_progress_bar.setStyleSheet("""
            QProgressBar {border: 0px; background-color: #333; color: white; height: 15px; border-radius: 4px;}
            QProgressBar::chunk {background-color: #00897b; border-radius: 4px;}
        """)
        self.main_layout.addWidget(self.climate_progress_bar)
        
        # Analiz Butonu
        self.btn_run_climate = QPushButton("🌍 İKLİM ANALİZİNİ BAŞLAT")
        self.btn_run_climate.setStyleSheet("""
            QPushButton {
                background-color: #00897b; 
                color: white; 
                font-weight: bold; 
                padding: 12px; 
                margin-top: 10px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #00695c;
            }
        """)
        self.main_layout.addWidget(self.btn_run_climate)
        
        # Sonuç Gösterim Alanı
        self.climate_result_label = QLabel("")
        self.climate_result_label.setWordWrap(True)
        self.climate_result_label.setStyleSheet("color: #aaa; font-size: 11px; margin-top: 10px; padding: 10px; border: 1px dashed #444;")
        self.main_layout.addWidget(self.climate_result_label)

    def _form_cfd(self, data):
        form = self._create_group("CFD Analiz Parametreleri")
        self._add_combo(form, "Fizik Motoru (Solver)", data, "solver", ["NVIDIA Modulus 3D FNO", "Legacy 2D PINN"])
        self._add_int(form, "Çözünürlük (Grid)", data, "resolution")
        
        chk_p = QCheckBox("Basınç Konturlarını Göster")
        chk_p.setChecked(getattr(data, "show_pressure", True))
        chk_p.toggled.connect(lambda v: setattr(data, "show_pressure", v))
        form.addRow(chk_p)
        
        chk_s = QCheckBox("Akış Çizgilerini Göster")
        chk_s.setChecked(getattr(data, "show_streamlines", True))
        chk_s.toggled.connect(lambda v: setattr(data, "show_streamlines", v))
        form.addRow(chk_s)
        
        # --- BOUNDARY CONDITIONS ---
        form_bc = self._create_group("Sınır Koşulları (Boundary Conditions)")
        self._add_float(form_bc, "Inlet Hızı (m/s)", data, "inlet_velocity")
        self._add_float(form_bc, "Akışkan Yoğunluğu (kg/m3)", data, "fluid_density")
        self._add_text(form_bc, "K. Viskozite (m2/s)", data, "kinematic_viscosity") # Scientific notation via text
        
        box_info = QLabel("CFD Domain Bounding Box (Gemi boyutlarına oranla):")
        box_info.setStyleSheet("color: #888; font-style: italic; font-size: 10px;")
        form_bc.addRow(box_info)
        
        self._add_float(form_bc, "Uzunluk Çarpanı (X)", data, "domain_x_mult")
        self._add_float(form_bc, "Genişlik Çarpanı (Y)", data, "domain_y_mult")
        self._add_float(form_bc, "Yükseklik Çarpanı (Z)", data, "domain_z_mult")
        
        self.btn_run_cfd = QPushButton("🌊 PINN CFD ANALİZİNİ BAŞLAT")
        self.btn_run_cfd.setStyleSheet("background-color: #007acc; color: white; font-weight: bold; padding: 12px; margin-top: 10px;")
        self.main_layout.addWidget(self.btn_run_cfd)

    def _form_model_3d(self, data):
        """3D Model Tasarımı Ayar Formu"""
        form = self._create_group("3D Asset Management")
        
        # Get vessel data for path sharing
        vessel_data = self.data_store.get(NodeType.VESSEL, {})
        
        # USD Import
        self.btn_import_usd_3d = QPushButton("🛸 Import USD / Omniverse (.usd)")
        self.btn_import_usd_3d.setStyleSheet("background-color: #1e88e5; color: white; font-weight: bold; padding: 10px;")
        self.btn_import_usd_3d.clicked.connect(lambda: self._open_usd_dialog(vessel_data))
        form.addRow(self.btn_import_usd_3d)
        
        if getattr(vessel_data, 'usd_path', None):
            lbl_u = QLabel(f"Active USD: {os.path.basename(vessel_data.usd_path)}")
            lbl_u.setStyleSheet("color: #00ccff; font-size: 11px;")
            form.addRow(lbl_u)

        # STL Import
        self.btn_import_stl_3d = QPushButton("🛳️ Import Legacy STL (.stl)")
        self.btn_import_stl_3d.setStyleSheet("background-color: #444; color: white; padding: 8px;")
        self.btn_import_stl_3d.clicked.connect(lambda: self._open_stl_dialog(vessel_data))
        form.addRow(self.btn_import_stl_3d)

        if getattr(vessel_data, 'stl_path', None):
            lbl_s = QLabel(f"Active STL: {os.path.basename(vessel_data.stl_path)}")
            lbl_s.setStyleSheet("color: #8fce00; font-size: 11px;")
            form.addRow(lbl_s)
            
        # Add basic design info
        info = QLabel("\n💡 İpucu: USD dosyaları Omniverse ve diğer modern 3D yazılımlarla tam uyumludur. "
                     "STL yerine USD kullanımı, yüzey normalleri ve performans açısından daha sağlıklıdır.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #888; font-size: 10px; font-style: italic; margin-top: 15px;")
        self.main_layout.addWidget(info)

    def _form_retrofit(self, data):
        form = self._create_group("Retrofit Bileşen Seçimi")
        
        # 'data' burada RetrofitConfig (veya VesselData?) -> Aslında NodeType.RETROFIT -> RetrofitConfig
        current_selection = getattr(data, 'selected_retrofit', [])
            
        components = [
            ('propeller_high_eff', 'Pervane: Yüksek Verimli Pervane (%4 Tasarruf)'),
            ('pbcf', 'Pervane: PBCF (Propeller Boss Cap Fin) (%1.5 Tasarruf)'),
            ('bulbous_bow', 'Bulb: Optimize Edilmiş Bulbous Bow (%3.5 Tasarruf)'),
            ('shaft_generator', 'Şaft: Şaft Jeneratörü Sistemi (%5 Tasarruf)'),
            ('engine_derating', 'Makine: Engine De-rating / Tuning (%2.5 Tasarruf)'),
            ('hull_coating', 'Gövde: Premium Anti-fouling Kaplama (%3 Tasarruf)'),
            ('flettner_rotor', 'Yenilikçi: Flettner Rotor (Rüzgar) (%12 Tasarruf)')
        ]
        
        for comp_id, label in components:
            chk = QCheckBox(label)
            chk.setChecked(comp_id in current_selection)
            # Closure for connecting correctly
            chk.toggled.connect(lambda checked, cid=comp_id: self._update_retrofit_selection(data, cid, checked))
            form.addRow(chk)
            
        info = QLabel("💡 Birden fazla bileşen seçildiğinde tasarruf kümülatif hesaplanır.")
        info.setStyleSheet("color: gray; font-size: 10px; font-style: italic;")
        form.addRow(info)

    def _update_retrofit_selection(self, data, comp_id, checked):
        # Update Dataclass
        current_list = getattr(data, 'selected_retrofit', [])
        if checked:
            if comp_id not in current_list:
                current_list.append(comp_id)
        else:
            if comp_id in current_list:
                current_list.remove(comp_id)
        
        # Trigger update explicitly since list mutation might not trigger setters
        setattr(data, 'selected_retrofit', current_list) 
        print(f"Retrofit Selection Updated: {current_list}")
        self.data_changed.emit(self.current_node_type, asdict(data))

    def _view_run_status(self, data):
        # Run sayfası form değil, bilgi ekranıdır
        info = QLabel("Analiz başlatılmaya hazır.\nTüm parametreler kontrol edildi.")
        info.setStyleSheet("color: #8fce00; border: 1px dashed #555; padding: 10px;")
        self.main_layout.addWidget(info)
        
        self.btn_run_analysis = QPushButton("🚀 ANALİZİ BAŞLAT")
        self.btn_run_analysis.setObjectName("btn_compute")
        self.btn_run_analysis.setFixedHeight(50)
        self.main_layout.addWidget(self.btn_run_analysis)

    def _form_advanced_analysis(self, data):
        """TOPSIS ve IPSO Gelişmiş Karar Desteği Ayar Formu"""
        
        # === TOPSIS AYARLARI ===
        form_topsis = self._create_group("📊 TOPSIS - Çok Kriterli Karar Analizi")
        
        # Ağırlık ayarları
        info_weights = QLabel("Kriter ağırlıkları (toplam 1.0 olmalı):")
        info_weights.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        form_topsis.addRow(info_weights)
        
        self._add_float(form_topsis, "💰 Ekonomik Ağırlık", data, "weight_economic")
        self._add_float(form_topsis, "🌍 Çevresel Ağırlık", data, "weight_environmental")
        self._add_float(form_topsis, "⚙️ Operasyonel Ağırlık", data, "weight_operational")
        
        # TOPSIS açıklama
        topsis_info = QLabel(
            "TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)\n"
            "En iyi senaryoyu belirlemek için tüm kriterleri dengeli değerlendirir."
        )
        topsis_info.setWordWrap(True)
        topsis_info.setStyleSheet("color: #666; font-size: 9px; padding: 5px; border-left: 2px solid #42a5f5;")
        form_topsis.addRow(topsis_info)
        
        # TOPSIS Butonu
        self.btn_run_topsis = QPushButton("📊 TOPSIS ANALİZİ ÇALIŞTIR")
        self.btn_run_topsis.setStyleSheet("""
            QPushButton {
                background-color: #5c6bc0; 
                color: white; 
                font-weight: bold; 
                padding: 12px; 
                margin-top: 5px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #3f51b5;
            }
        """)
        self.main_layout.addWidget(self.btn_run_topsis)
        
        # === IPSO AYARLARI ===
        form_ipso = self._create_group("🔄 IPSO - Pareto Optimizasyonu")
        
        self._add_int(form_ipso, "Parçacık Sayısı (Particles)", data, "ipso_particles")
        self._add_int(form_ipso, "İterasyon Sayısı", data, "ipso_iterations")
        
        # IPSO açıklama
        ipso_info = QLabel(
            "IPSO (Improved Particle Swarm Optimization)\n"
            "Çok amaçlı optimizasyon ile Pareto-optimal çözümler bulur.\n"
            "⚠️ Hesaplama süresi uzun olabilir."
        )
        ipso_info.setWordWrap(True)
        ipso_info.setStyleSheet("color: #666; font-size: 9px; padding: 5px; border-left: 2px solid #ff9800;")
        form_ipso.addRow(ipso_info)
        
        # IPSO Butonu
        self.btn_run_ipso = QPushButton("🔄 IPSO OPTİMİZASYONU ÇALIŞTIR")
        self.btn_run_ipso.setStyleSheet("""
            QPushButton {
                background-color: #ff9800; 
                color: white; 
                font-weight: bold; 
                padding: 12px; 
                margin-top: 5px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
        """)
        self.main_layout.addWidget(self.btn_run_ipso)
        
        # === HASSASLIK ANALİZİ ===
        form_sens = self._create_group("📈 Hassasiyet Analizi")
        
        chk_sens = QCheckBox("Parametre Hassasiyet Analizi Yap")
        chk_sens.setChecked(getattr(data, "sensitivity_analysis", True))
        chk_sens.toggled.connect(lambda v: setattr(data, "sensitivity_analysis", v))
        form_sens.addRow(chk_sens)
        
        sens_info = QLabel("Yakıt fiyatı, navlun oranı ve karbon vergisi değişimlerinin NPV'ye etkisini analiz eder.")
        sens_info.setWordWrap(True)
        sens_info.setStyleSheet("color: #666; font-size: 9px;")
        form_sens.addRow(sens_info)
        
        self.btn_run_sensitivity = QPushButton("📈 HASSASİYET ANALİZİ")
        self.btn_run_sensitivity.setStyleSheet("""
            QPushButton {
                background-color: #26a69a; 
                color: white; 
                font-weight: bold; 
                padding: 10px; 
                margin-top: 5px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #00897b;
            }
        """)
        self.main_layout.addWidget(self.btn_run_sensitivity)
        
        # === İLERLEME ÇUBUĞU ===
        self.advanced_progress_bar = QProgressBar()
        self.advanced_progress_bar.setValue(0)
        self.advanced_progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.advanced_progress_bar.setStyleSheet("""
            QProgressBar {border: 0px; background-color: #333; color: white; height: 15px; border-radius: 4px;}
            QProgressBar::chunk {background-color: #5c6bc0; border-radius: 4px;}
        """)
        self.main_layout.addWidget(self.advanced_progress_bar)
        
        # === SONUÇ ALANI ===
        self.advanced_result_label = QLabel("")
        self.advanced_result_label.setWordWrap(True)
        self.advanced_result_label.setStyleSheet(
            "color: #aaa; font-size: 11px; margin-top: 10px; padding: 10px; border: 1px dashed #444;"
        )
        self.main_layout.addWidget(self.advanced_result_label)
        
        # === PDF RAPOR OLUŞTUR ===
        form_report = self._create_group("📄 Rapor Oluştur")
        
        report_info = QLabel(
            "Tüm analiz sonuçlarını profesyonel PDF formatında indirin.\n"
            "Kapak sayfası, grafikler ve tavsiyeler içerir."
        )
        report_info.setWordWrap(True)
        report_info.setStyleSheet("color: #666; font-size: 9px; padding: 5px;")
        form_report.addRow(report_info)
        
        self.btn_generate_pdf = QPushButton("📄 PDF RAPOR OLUŞTUR")
        self.btn_generate_pdf.setStyleSheet("""
            QPushButton {
                background-color: #7B1FA2; 
                color: white; 
                font-weight: bold; 
                padding: 12px; 
                margin-top: 5px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #9C27B0;
            }
        """)
        self.btn_generate_pdf.setCursor(Qt.CursorShape.PointingHandCursor)
        self.main_layout.addWidget(self.btn_generate_pdf)

    # --- YARDIMCI METOTLAR (Kod tekrarını önlemek için) ---
    
    def _create_group(self, title):
        group = QGroupBox(title)
        layout = QFormLayout()
        group.setLayout(layout)
        self.main_layout.addWidget(group)
        return layout

    def _add_text(self, layout, label, data, key):
        val = getattr(data, key, "")
        widget = QLineEdit(str(val))
        self._style(widget)
        # Veri değişince attr güncelle ve sinyal gönder
        def on_change(val):
            setattr(data, key, val)
            self.data_changed.emit(self.current_node_type, asdict(data) if is_dataclass(data) else data)
        widget.textChanged.connect(on_change)
        layout.addRow(label + ":", widget)

    def _add_int(self, layout, label, data, key):
        widget = QSpinBox()
        self._style(widget)
        widget.setRange(0, 2147483647) # Int max (approx 2 Billion)
        val = getattr(data, key, 0)
        widget.setValue(int(val) if val else 0)
        def on_change(val):
            setattr(data, key, val)
            self.data_changed.emit(self.current_node_type, asdict(data) if is_dataclass(data) else data)
        widget.valueChanged.connect(on_change)
        layout.addRow(label + ":", widget)

    def _add_float(self, layout, label, data, key):
        widget = QDoubleSpinBox()
        self._style(widget)
        widget.setRange(0, 999999.99)
        val = getattr(data, key, 0.0)
        widget.setValue(float(val) if val else 0.0)
        def on_change(val):
            setattr(data, key, val)
            self.data_changed.emit(self.current_node_type, asdict(data) if is_dataclass(data) else data)
        widget.valueChanged.connect(on_change)
        layout.addRow(label + ":", widget)
        return widget

    def _add_combo(self, layout, label, data, key, options):
        widget = QComboBox()
        self._style(widget)
        widget.addItems(options)
        val = getattr(data, key, "")
        widget.setCurrentText(str(val))
        def on_change(val):
            setattr(data, key, val)
            self.data_changed.emit(self.current_node_type, asdict(data) if is_dataclass(data) else data)
        widget.currentTextChanged.connect(on_change)
        layout.addRow(label + ":", widget)

    def _style(self, widget):
        widget.setStyleSheet("background-color: #2b2b2b; color: white; border: 1px solid #555; padding: 3px;")

    def _clear_layout(self):
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    def _get_schema(self, agent_type):
        """Hangi kutucukların çıkacağını belirler"""
        schema = [] # Herkes için ortak olanları kaldırdım, isteğe bağlı ekleyebilirsin
        
        if agent_type == "Gemi":
            schema = [
                {"key": "label", "label": "Gemi Adı", "type": "text"},
                {"key": "dwt", "label": "DWT (ton)", "type": "int"},
                {"key": "speed", "label": "Hız (knots)", "type": "float"},
            ]
        elif agent_type == "Guardian":
            schema = [
                {"key": "year", "label": "Hedef Yıl", "type": "int"},
                {"key": "scenario", "label": "Senaryo", "type": "combo", "opts": ["RCP 4.5", "RCP 8.5"]},
            ]
        else:
            # Tanımsız tipler için standart kutu
            schema = [{"key": "desc", "label": "Açıklama", "type": "text"}]
            
        return schema

    def _create_row(self, field, data):
        key = field["key"]
        val = data.get(key, "") 
        widget = None

        if field["type"] == "text" or field["type"] == "float":
            widget = QLineEdit(str(val))
            widget.setStyleSheet("background-color: #2b2b2b; color: white; border: 1px solid #555;")
            widget.textChanged.connect(lambda t: data.update({key: t}))

        elif field["type"] == "int":
            widget = QSpinBox()
            widget.setStyleSheet("background-color: #2b2b2b; color: white; border: 1px solid #555;")
            widget.setRange(0, 9999999)
            try: widget.setValue(int(val))
            except: widget.setValue(0)
            widget.valueChanged.connect(lambda v: data.update({key: v}))

        elif field["type"] == "combo":
            widget = QComboBox()
            widget.setStyleSheet("background-color: #2b2b2b; color: white; border: 1px solid #555;")
            widget.addItems(field.get("opts", []))
            widget.setCurrentText(str(val))
            widget.currentTextChanged.connect(lambda t: data.update({key: t}))

        if widget:
            self.current_form.addRow(field["label"] + ":", widget)

    def _clear_layout(self):
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            widget = item.widget()
            if widget: widget.deleteLater()

class RunPage(QWidget):
    """Analizi Başlatma Sayfası"""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        header = QLabel("▶️ Simulation Run")
        # Inline style removed for global QSS
        layout.addWidget(header)
        
        btn = QPushButton("Compute")
        btn.setObjectName("btn_compute") # For specific QSS styling
        
        log = QTextEdit()
        log.setPlaceholderText("Simulation configuration summary...")
        
        layout.addWidget(btn)
        layout.addWidget(log)
        layout.addStretch()
        self.setLayout(layout)

# --- SABİT DEĞİŞKENLER (ENUMS) ---
class NodeType:
    PROJECT = 1
    VESSEL = 2      # <-- Hatayı çözen satır bu
    SURROGATE = 3
    OPTIMIZER = 4
    CLIMATE = 5
    RUN = 6
    GROUP = 7
    CFD = 8
    MODEL_3D = 9
    RETROFIT = 10
    ADVANCED_ANALYSIS = 11  # TOPSIS + IPSO Gelişmiş Analiz

# --- ANA PENCERE SINIFI ---
class SmartCAPEXMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- PENCERE AYARLARI ---
        self.setWindowTitle("SmartCAPEX AI - Maritime Retrofit Decision Support")
        self.resize(1600, 1000)
        self.setMinimumSize(1200, 800)
        
        # --- MENU BAR KURULUMU ---
        # self._setup_menu_bar() # Menü çubuğu özellikleri ağaca/ribbona taşındı
        
        # --- MODEL & VIEW SETUP (DÜZELTİLDİ) ---
        # QTreeView yerine QTreeWidget kullanıyoruz
        self.tree_view = QTreeWidget()
        self.tree_view.setFixedWidth(280)
        self.tree_view.setHeaderLabel("Model Explorer") # Artık hata vermez
        self.tree_view.setHeaderHidden(True) # İstersen başlığı gizleyebilirsin
        
        # Context Menu Setup
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.open_context_menu)
        
        # Selection Signal (QTreeWidget için daha basit sinyal)
        self.tree_view.itemSelectionChanged.connect(self.on_selection_changed)

        # --- AĞAÇ YAPISI (TREE NODES) ---
        
        # 1. Kök Düğüm (Root)
        self.root = QTreeWidgetItem(self.tree_view, ["SmartCAPEX Project"])
        self.root.setData(0, Qt.ItemDataRole.UserRole, NodeType.PROJECT) # Veri saklama
        self.root.setExpanded(True)
        
        # 2. Ana Ajanlar (Sadece bunlar varsayılan olarak var)
        self.item_vessel = QTreeWidgetItem(self.root, ["Gemi Veri Girişi"])
        self.item_vessel.setData(0, Qt.ItemDataRole.UserRole, NodeType.VESSEL)
        
        self.item_surrogate = QTreeWidgetItem(self.root, ["Surrogate Modeler"])
        self.item_surrogate.setData(0, Qt.ItemDataRole.UserRole, NodeType.SURROGATE)
        
        self.item_optimizer = QTreeWidgetItem(self.root, ["Optimizer"])
        self.item_optimizer.setData(0, Qt.ItemDataRole.UserRole, NodeType.OPTIMIZER)
        
        self.item_climate = QTreeWidgetItem(self.root, ["Climate Guardian"])
        self.item_climate.setData(0, Qt.ItemDataRole.UserRole, NodeType.CLIMATE)
        
        # Ek özellikler alt düğümler olarak eklendi
        self.item_cfd = QTreeWidgetItem(self.item_surrogate, ["🌊 PINN CFD Analizi"])
        self.item_cfd.setData(0, Qt.ItemDataRole.UserRole, NodeType.CFD)
        
        self.item_run = QTreeWidgetItem(self.item_climate, ["▶️ Analiz / Run"])
        self.item_run.setData(0, Qt.ItemDataRole.UserRole, NodeType.RUN)
        
        self.item_advanced = QTreeWidgetItem(self.item_climate, ["⚡ Gelişmiş Analiz (TOPSIS/IPSO)"])
        self.item_advanced.setData(0, Qt.ItemDataRole.UserRole, NodeType.ADVANCED_ANALYSIS)
        
        self.item_model_3d = None
        self.item_retrofit = None
        
        # --- ANA LAYOUT KURULUMU ---
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # 1. TOP: RIBBON (Eğer sınıfın yoksa burayı yorum satırı yap)
        self.ribbon = RibbonWidget()
        self.ribbon.action_triggered.connect(self.on_ribbon_action)
        main_layout.addWidget(self.ribbon)
        
        # 2. MIDDLE: SPLITTER
        self.middle_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- COL 1: MODEL BUILDER ---
        self.col_model = QWidget()
        col1_layout = QVBoxLayout()
        col1_layout.setContentsMargins(0, 0, 0, 0)
        lbl_agents = QLabel(" 🤖 Intellectual Agents")
        lbl_agents.setObjectName("HeaderLabel")
        col1_layout.addWidget(lbl_agents)
        col1_layout.addWidget(self.tree_view) # Ağacı buraya ekliyoruz
        self.col_model.setLayout(col1_layout)
        
        # --- COL 2: SETTINGS (GÜNCELLENDİ) ---
        self.col_settings = QWidget()
        col2_layout = QVBoxLayout()
        col2_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_settings = QLabel(" 🛠️ Settings & Parameters")
        lbl_settings.setObjectName("HeaderLabel")
        col2_layout.addWidget(lbl_settings)
        
        # --- YENİ SETTINGS MANAGER BURAYA GELİYOR ---
        self.settings_manager = SettingsManager()
        self.settings_manager.data_changed.connect(self.on_settings_data_changed)
        
        # Scroll Area içine koyalım ki form uzun olursa taşmasın
        scroll = QScrollArea()
        scroll.setWidget(self.settings_manager)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame) # Çerçeve çizgisini kaldır
        
        col2_layout.addWidget(scroll)
        self.col_settings.setLayout(col2_layout)
        
        # --- COL 3: GRAPHICS ---
        self.col_graphics = QWidget()
        col3_layout = QVBoxLayout()
        col3_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_header = QLabel(" 📊 Visualization & Analysis")
        lbl_header.setObjectName("HeaderLabel")
        col3_layout.addWidget(lbl_header)
        
        # Graphics Area as Stacked Widget
        self.graphics_stack = QStackedWidget()
        self.graphics_stack.setObjectName("GraphicsStack")
        
        self.placeholder_gfx = QLabel("Graphics Area")
        self.placeholder_gfx.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_gfx.setStyleSheet("background-color: #222; color: #555;")
        
        self.cfd_widget = CFDVisualizationWidget()
        self.model_3d_widget = ModelViewer3D()
        
        self.graphics_stack.addWidget(self.placeholder_gfx)
        self.graphics_stack.addWidget(self.cfd_widget)
        self.graphics_stack.addWidget(self.model_3d_widget)
        
        col3_layout.addWidget(self.graphics_stack)
        
        self.col_graphics.setLayout(col3_layout)
        
        # SPLITTER EKLEME
        self.middle_splitter.addWidget(self.col_model)
        self.middle_splitter.addWidget(self.col_settings)
        self.middle_splitter.addWidget(self.col_graphics)
        
        # Stretch faktörleri (genişleme oranları)
        self.middle_splitter.setStretchFactor(0, 1)  # Model Builder - küçük
        self.middle_splitter.setStretchFactor(1, 2)  # Settings - orta
        self.middle_splitter.setStretchFactor(2, 4)  # Graphics - büyük
        
        # Başlangıç boyutları (piksel cinsinden)
        # Toplam genişlik yaklaşık 1600px olduğunu varsayarak:
        # Sol: 220px, Orta: 350px, Sağ: kalan (yaklaşık 1030px)
        self.middle_splitter.setSizes([220, 350, 1030])
        
        # Minimum boyutları ayarla (paneller çok küçülmesin)
        self.col_model.setMinimumWidth(180)
        self.col_settings.setMinimumWidth(280)
        self.col_graphics.setMinimumWidth(400)
        
        # Splitter tutamağını daha görünür yap
        self.middle_splitter.setHandleWidth(5)
        
        main_layout.addWidget(self.middle_splitter)

        # 3. BOTTOM PANEL (Eğer sınıf yoksa yorum satırı)
        self.bottom_panel = BottomPanel()
        main_layout.addWidget(self.bottom_panel)

        from agents.surrogate_modeler import SurrogateModeler 
        from agents.pinn_cfd_agent import PINNCFDAgent
        from agents.modulus_agent import ModulusCFDAgent
        
        # Regulatory Agent (IMO CII + EU ETS monitoring)
        try:
            from agents.regulatory_agent import RegulatoryAgent
            self.regulatory_agent = RegulatoryAgent()
            print("✅ Regulatory Agent (CII/ETS) entegre edildi.")
        except Exception as e:
            self.regulatory_agent = None
            print(f"⚠️ Regulatory Agent yüklenemedi: {e}")
        
        self.surrogate_modeler = SurrogateModeler()
        self.cfd_agent = ModulusCFDAgent()
        self.modulus_agent = self.cfd_agent  # same instance
        self.omniverse_streamer = None
        self.optimizer = MultiObjectiveOptimizer(surrogate_model=self.surrogate_modeler)
        
        # Connect CFD Agent Signals
        self.cfd_agent.progress_signal.connect(self.update_progress)
        self.cfd_agent.finished_signal.connect(self.on_cfd_finished)
        self.modulus_agent.progress_signal.connect(self.update_progress)
        self.modulus_agent.finished_signal.connect(self.on_modulus_finished)

        # ── Regulatory Status Bar (EU ETS + CII live indicators) ──
        self._setup_regulatory_status_bar()


    def _setup_regulatory_status_bar(self):
        """Add regulatory status indicators to the main window status bar."""
        if not self.regulatory_agent:
            return

        try:
            status_bar = self.statusBar()
            status_bar.setStyleSheet(
                "QStatusBar { background: #161b22; color: #c9d1d9; "
                "border-top: 1px solid #30363d; font-size: 11px; }"
            )

            # Get initial regulatory summary
            vessel_obj = self.settings_manager.data_store.get(NodeType.VESSEL) if hasattr(self, 'settings_manager') else None
            vessel_data = asdict(vessel_obj) if vessel_obj and is_dataclass(vessel_obj) else {}

            summary = self.regulatory_agent.get_summary(vessel_data)

            # CII indicator
            cii_text = f"CII: ↓{summary['cii_reduction_pct']:.0f}% ({summary['year']})"
            self.cii_status_label = QLabel(cii_text)
            self.cii_status_label.setStyleSheet(
                "color: #58a6ff; padding: 2px 8px; font-weight: bold;")
            status_bar.addPermanentWidget(self.cii_status_label)

            # ETS price indicator
            ets_text = f"EU ETS: €{summary['ets_price_eur']:.0f}/tCO₂"
            if summary.get('ets_live'):
                ets_text += " 🔴"
            self.ets_status_label = QLabel(ets_text)
            self.ets_status_label.setStyleSheet(
                "color: #d29922; padding: 2px 8px; font-weight: bold;")
            status_bar.addPermanentWidget(self.ets_status_label)

            # EEDI phase
            eedi_text = f"EEDI: {summary['eedi_phase'].replace('_', ' ').title()}"
            self.eedi_status_label = QLabel(eedi_text)
            self.eedi_status_label.setStyleSheet(
                "color: #238636; padding: 2px 8px;")
            status_bar.addPermanentWidget(self.eedi_status_label)

            # Alerts count
            n_alerts = summary.get('alerts_count', 0)
            if n_alerts > 0:
                alert_text = f"⚠ {n_alerts} Uyarı"
                alert_label = QLabel(alert_text)
                alert_label.setStyleSheet(
                    "color: #da3633; padding: 2px 8px; font-weight: bold;")
                status_bar.addPermanentWidget(alert_label)

        except Exception as e:
            print(f"⚠️ Regulatory status bar hatası: {e}")

    def _setup_menu_bar(self):
        """Üst menüyü (Menu Bar) oluşturur ve ek özellikleri ekler."""
        menubar = self.menuBar()
        
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #2b2b2b;
                color: white;
                font-weight: bold;
            }
            QMenuBar::item:selected {
                background-color: #3f51b5;
            }
            QMenu {
                background-color: #333;
                color: white;
                border: 1px solid #555;
            }
            QMenu::item:selected {
                background-color: #5c6bc0;
            }
        """)

        feature_menu = menubar.addMenu("🌟 Ek Özellikler Ekle")
        
        features = [
            ("▶️ Analiz / Run", NodeType.RUN),
            ("🌊 PINN CFD Analizi", NodeType.CFD),
            ("🛸 3D Model Tasarımı", NodeType.MODEL_3D),
            ("⚙️ Retrofit Wizard", NodeType.RETROFIT),
            ("⚡ Gelişmiş Analiz (TOPSIS/IPSO)", NodeType.ADVANCED_ANALYSIS)
        ]
        
        for name, n_type in features:
            action = QAction(name, self)
            action.triggered.connect(lambda checked, n=name, t=n_type: self.add_feature_to_agent(n, t))
            feature_menu.addAction(action)

    def add_feature_to_agent(self, feature_name, node_type):
        """Seçili olan ajana/düğüme, istenen alt özelliği ekler."""
        selected_items = self.tree_view.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(
                self, 
                "Seçim Yapılmadı", 
                "Lütfen özelliği eklemek istediğiniz bir ajanı (Örn: Surrogate Modeler) sol menüden seçin."
            )
            return
            
        parent_item = selected_items[0]
        
        new_item = QTreeWidgetItem(parent_item, [feature_name])
        new_item.setData(0, Qt.ItemDataRole.UserRole, node_type)
        
        if node_type == NodeType.RUN: self.item_run = new_item
        elif node_type == NodeType.CFD: self.item_cfd = new_item
        elif node_type == NodeType.MODEL_3D: self.item_model_3d = new_item
        elif node_type == NodeType.RETROFIT: self.item_retrofit = new_item
        elif node_type == NodeType.ADVANCED_ANALYSIS: self.item_advanced = new_item
        
        parent_item.setExpanded(True)
        new_item.setSelected(True)
        
        if hasattr(self, 'bottom_panel'):
            self.bottom_panel.update_log(f"Yeni ajan özelliği eklendi: {feature_name}")

    def on_train_click(self):
        """Eğit butonuna basılınca çalışır"""
        
        config = {
            "model": "EANN (Emotional Neural Network)", # veya "Kriging"
            "epochs": 100,
            "anxiety": 0.1,
            "confidence": 0.2
            # "data_path": "veri.csv" # Eğer dosya seçtiysen buraya ekle
        }
        
        # Log alanına bilgi ver
        if hasattr(self, 'bottom_panel'):
            # Eğer Log panelin varsa oraya yaz
            print("Eğitim Başlatılıyor...") 
        
        # 3. Thread (İş Parçacığı) Oluştur ve Başlat
        self.worker = TrainingWorker(self.surrogate_modeler, config)
        
        # --- SİNYALLERİ BAĞLA (ÇOK ÖNEMLİ) ---
        # Ajan -> Arayüz (İlerleme Çubuğu Güncelle)
        self.surrogate_modeler.progress_signal.connect(self.update_progress)
        
        # Ajan -> Arayüz (Bittiğinde)
        self.surrogate_modeler.finished_signal.connect(self.on_training_finished)
        
        # Ajan -> Arayüz (Hata Olursa)
        self.surrogate_modeler.error_signal.connect(lambda err: print(f"HATA: {err}"))
        
        # İşçiyi Çalıştır
        self.worker.start()

    def update_progress(self, val, msg):
        """İlerleme çubuğunu günceller"""
        try:
            # SettingsManager içindeki progress bar'a ulaş
            if hasattr(self.settings_manager, 'progress_bar'):
                self.settings_manager.progress_bar.setValue(val)
        except (RuntimeError, AttributeError):
            # Eğer widget silinmişse (sayfa değişmişse) hata verme
            pass
        
        # Terminale de yaz ki çalıştığını görelim
        print(f"Progress: {val}% - {msg}")

    def on_training_finished(self, results):
        """Eğitim bitince çalışır"""
        print("✅ EĞİTİM TAMAMLANDI!")
        print(f"Sonuçlar: {results}")
        
        # Grafiği çizdirebilirsin
        # self.plot_results(results)



    def on_run_analysis_click(self):
        """Genel Yatırım ve Senaryo Analizini başlatır.

        Bu method Holtrop-Mennen direnç verisini alır, efektif gücü (Pe)
        yakıt tüketimine çevirir ve finansal analize enjekte eder.

        Flow: Hull Geometry → Holtrop Rt → Pe → Fuel (t/day) → NPV
        """
        print("🚀 Analiz Başlatılıyor...")

        # 1. Tüm verileri topla
        v_obj = self.settings_manager.data_store.get(NodeType.VESSEL)
        r_obj = self.settings_manager.data_store.get(NodeType.RETROFIT)
        c_obj = self.settings_manager.data_store.get(NodeType.CLIMATE)

        vessel_data = asdict(v_obj) if is_dataclass(v_obj) else (v_obj or {})
        retrofit_data = asdict(r_obj) if is_dataclass(r_obj) else (r_obj or {})
        climate_data = asdict(c_obj) if is_dataclass(c_obj) else (c_obj or {})

        # 2. Verileri birleştir
        full_data = {**vessel_data, **retrofit_data, **climate_data}

        # ─── 3. Holtrop-Mennen Direnç → Yakıt Tüketimi ──────────────
        # Convert Effective Power (Pe kW) to fuel consumption (t/day)
        # using: fuel = Pe / (η_prop × η_shaft) × SFC × 24 / 1e6
        try:
            from core.geometry.hull_adapter import RetrosimHullAdapter
            adapter = RetrosimHullAdapter()
            adapter.set_from_ui(full_data)
            speed = float(full_data.get('speed', 12.0))
            resistance = adapter.predict_total_resistance(speed)

            Pe_kW = resistance.get('Pe_kW', 0)
            Rt_kN = resistance.get('Rt', 0)
            Rw_kN = resistance.get('Rw', 0)
            Fr    = resistance.get('Froude_number', 0)
            Cw    = resistance.get('Cw', 0)

            # Propulsive efficiency chain
            eta_prop  = 0.65   # propeller open-water efficiency
            eta_shaft = 0.98   # shaft losses
            eta_hull  = 0.85   # hull efficiency (1-t)/(1-w)
            eta_total = eta_prop * eta_shaft * eta_hull  # ~0.54

            # Brake power (kW)
            Pb_kW = Pe_kW / eta_total if eta_total > 0 else 0

            # Specific Fuel Consumption: 180 g/kWh for modern slow-speed diesel
            SFC = 180  # g/kWh
            fuel_tons_day = Pb_kW * SFC * 24 / 1e6  # tons/day

            # CO2 emission: 3.114 t-CO2 per t-fuel (IMO default for HFO)
            co2_tons_day = fuel_tons_day * 3.114

            # Inject into vessel_data for financial analysis
            if fuel_tons_day > 0:
                full_data['fuel_consumption'] = round(fuel_tons_day, 2)
                full_data['co2_emission'] = round(co2_tons_day, 2)
                full_data['Pe_kW'] = round(Pe_kW, 1)
                full_data['Rt_kN'] = round(Rt_kN, 2)
                full_data['Rw_kN'] = round(Rw_kN, 2)
                full_data['Froude_number'] = round(Fr, 4)
                full_data['Cw'] = Cw
                print(f"  🌊 Holtrop-Mennen: Rt={Rt_kN:.1f} kN, "
                      f"Pe={Pe_kW:.0f} kW, Pb={Pb_kW:.0f} kW")
                print(f"  ⛽ Yakıt: {fuel_tons_day:.1f} t/gün, "
                      f"CO₂: {co2_tons_day:.1f} t/gün")
            else:
                print("  ⚠️ Holtrop Pe=0, kullanıcı girişi kullanılıyor.")
        except Exception as e:
            print(f"  ⚠️ Holtrop hesabı yapılamadı: {e}")

        print(f"Analiz Parametreleri: fuel={full_data.get('fuel_consumption','?')} t/day")

        # 4. Analizi çalıştır (fuel_consumption artık Holtrop tabanlı)
        results = self.optimizer.optimize_scenarios(vessel_data=full_data)

        print("✅ Analiz Tamamlandı!")
        self.on_analysis_finished(results)

    def on_analysis_finished(self, results):
        """Analiz bitince sonuçları ekrana basar veya grafik çizer"""
        print("--- ANALİZ SONUÇLARI ---")
        for scenario, data in results.items():
            if isinstance(data, dict) and 'npv' in data:
                print(f"Senaryo: {scenario} | NPV: {data['npv']:.2f} $ | Env Score: {data['environmental_score']:.2f}")
        
        # Gelecek adımda buraya grafik çizdirme eklenecek
        if hasattr(self, 'bottom_panel'):
            self.bottom_panel.update_log("Analiz başarıyla tamamlandı. Sonuçlar terminalde ve loglarda.")

    def on_cfd_finished(self, results):
        """CFD analizi bitince görselleştirmeyi günceller"""
        print("✅ PINN CFD Analizi Tamamlandı!")
        self.cfd_widget.update_plot(results) # Keep 2D updated in background
        self.model_3d_widget.update_cfd_results(results)
        # Force switch to 3D view
        self.graphics_stack.setCurrentWidget(self.model_3d_widget)

    def on_modulus_finished(self, results):
        """Modulus CFD analizi bitince Omniverse'e yansıt"""
        print("✅ NVIDIA Modulus Analizi Tamamlandı!")
        if hasattr(self, 'bottom_panel'):
            self.bottom_panel.update_log("Modulus Inference %100 tamamlandı. Omniverse'e veri aktarılıyor...")
            
        # Omniverse Streamer'ı başlat (USD dosyamız üzerinden)
        vessel_obj = self.settings_manager.data_store.get(NodeType.VESSEL)
        usd_path = getattr(vessel_obj, 'usd_path', None)
        
        if usd_path and os.path.exists(usd_path):
            from utils.omniverse_streamer import OmniverseStreamer
            if not self.omniverse_streamer:
                self.omniverse_streamer = OmniverseStreamer(usd_path)
            
            # Dalga yüksekliğini ve gemi hızını UI'dan al (varsayılan değerler ver)
            env_obj = self.settings_manager.data_store.get(NodeType.CLIMATE)
            wave_height = getattr(env_obj, 'mean_wave_height', 2.0) if env_obj else 2.0
            ship_speed = getattr(vessel_obj, 'service_speed', 14.0) if vessel_obj else 14.0
            
            # Omniverse canlı akışını başlat
            self.omniverse_streamer.update_environmental_state(wave_height, ship_speed)
            self.omniverse_streamer.inject_modulus_cfd_results(None, None)
            
            if hasattr(self, 'bottom_panel'):
                self.bottom_panel.update_log("🌊 Omniverse Digital Twin Live Sync güncellendi. (Animasyonlar USD dosyasına yazıldı)")
        else:
            if hasattr(self, 'bottom_panel'):
                self.bottom_panel.update_log("⚠️ USD dosyası bulunamadığı için Omniverse Stream atlandı. Lütfen parametrik gövde üretin.")

    def on_climate_analysis_click(self):
        """Climate Guardian analizini başlatır"""
        print("🌍 İklim Analizi Başlatılıyor...")
        
        # Gemi verisini al
        v_obj = self.settings_manager.data_store.get(NodeType.VESSEL)
        c_obj = self.settings_manager.data_store.get(NodeType.CLIMATE)
        
        vessel_data = asdict(v_obj) if is_dataclass(v_obj) else (v_obj or {})
        climate_data = asdict(c_obj) if is_dataclass(c_obj) else (c_obj or {})
        
        target_year = climate_data.get('target_year', 2030)
        
        # Climate Guardian instance oluştur (henüz yoksa)
        if not hasattr(self, 'climate_guardian'):
            self.climate_guardian = ClimateGuardian()
        
        self.climate_guardian.base_carbon_tax = climate_data.get('carbon_tax', 100)
        
        # İlerleme çubuğunu sıfırla
        if hasattr(self.settings_manager, 'climate_progress_bar'):
            self.settings_manager.climate_progress_bar.setValue(0)
        
        # Worker oluştur ve başlat
        self.climate_worker = ClimateWorker(self.climate_guardian, vessel_data, target_year)
        self.climate_worker.progress_signal.connect(self.update_climate_progress)
        self.climate_worker.finished_signal.connect(self.on_climate_finished)
        self.climate_worker.error_signal.connect(lambda err: print(f"❌ İklim Analizi Hatası: {err}"))
        self.climate_worker.start()
    
    def update_climate_progress(self, val, msg):
        """İklim analizi ilerleme çubuğunu günceller"""
        try:
            if hasattr(self.settings_manager, 'climate_progress_bar'):
                self.settings_manager.climate_progress_bar.setValue(val)
        except (RuntimeError, AttributeError):
            pass
        print(f"Climate Progress: {val}% - {msg}")
    
    def on_climate_finished(self, results):
        """İklim analizi tamamlandığında sonuçları gösterir"""
        print("✅ İKLİM ANALİZİ TAMAMLANDI!")
        
        # Sonuçları formatla
        target_year = results.get('target_year', 2030)
        risk = results.get('risk_assessment', {})
        env = results.get('environmental_conditions', {})
        perf = results.get('performance_impact', {})
        
        risk_level = risk.get('risk_level', 'Bilinmiyor')
        risk_score = risk.get('overall_risk_score', 0) * 100
        
        # Renk belirle
        if risk_score < 30:
            risk_color = "#4CAF50"  # Yeşil
            risk_icon = "✅"
        elif risk_score < 60:
            risk_color = "#FFC107"  # Sarı
            risk_icon = "⚠️"
        else:
            risk_color = "#f44336"  # Kırmızı
            risk_icon = "🔴"
        
        # Sonuç metnini oluştur
        result_text = f"""
<b style='color: {risk_color};'>{risk_icon} {target_year} Yılı İklim Risk Değerlendirmesi</b>

<b>Risk Seviyesi:</b> {risk_level}
<b>Risk Skoru:</b> {risk_score:.1f}%

<b>🌊 Çevresel Koşullar:</b>
• Ortalama Dalga Yüksekliği: {env.get('wave_height', 'N/A')} m
• Ortalama Rüzgar Hızı: {env.get('wind_speed', 'N/A')} knot
• Deniz Durumu: {env.get('sea_state', 'N/A')}

<b>⚡ Performans Etkisi:</b>
• Direnç Cezası: +{perf.get('resistance_penalty', 0)*100:.1f}%
• Yakıt Tüketimi Artışı: +{perf.get('fuel_increase', 0)*100:.1f}%
• Hız Kaybı: -{perf.get('speed_loss', 0)*100:.1f}%

<b>📋 Adaptasyon Önerileri:</b>
"""
        # Önerileri ekle
        measures = risk.get('adaptation_measures', [])
        for i, measure in enumerate(measures[:5], 1):
            result_text += f"  {i}. {measure}\n"
        
        # Sonucu etikete yaz
        if hasattr(self.settings_manager, 'climate_result_label'):
            self.settings_manager.climate_result_label.setText(result_text)
            self.settings_manager.climate_result_label.setStyleSheet(f"""
                color: #ddd; 
                font-size: 11px; 
                margin-top: 10px; 
                padding: 15px; 
                border: 2px solid {risk_color};
                border-radius: 8px;
                background-color: #1e1e1e;
            """)
        
        # Log paneline de yaz
        if hasattr(self, 'bottom_panel'):
            self.bottom_panel.update_log(f"İklim analizi tamamlandı. {target_year} yılı risk skoru: {risk_score:.1f}%")
        
        print(f"Sonuçlar: Risk={risk_level}, Skor={risk_score:.1f}%")

    # ====== GELİŞMİŞ ANALİZ FONKSİYONLARI (TOPSIS / IPSO) ======
    
    def on_topsis_analysis_click(self):
        """TOPSIS analizi başlatır"""
        print("📊 TOPSIS ANALİZİ BAŞLATIYOR...")
        
        # UI güncelle
        if hasattr(self.settings_manager, 'advanced_progress_bar'):
            self.settings_manager.advanced_progress_bar.setValue(10)
        if hasattr(self.settings_manager, 'advanced_result_label'):
            self.settings_manager.advanced_result_label.setText("TOPSIS analizi başlatılıyor...")
        
        # Gemi verilerini al
        v_obj = self.settings_manager.data_store.get(NodeType.VESSEL)
        vessel_data = asdict(v_obj) if is_dataclass(v_obj) else (v_obj or {})
        
        # Ağırlıkları al
        adv_obj = self.settings_manager.data_store.get(NodeType.ADVANCED_ANALYSIS)
        if adv_obj:
            weights = {
                'economic': getattr(adv_obj, 'weight_economic', 0.4),
                'environmental': getattr(adv_obj, 'weight_environmental', 0.35),
                'operational': getattr(adv_obj, 'weight_operational', 0.25)
            }
        else:
            weights = {'economic': 0.4, 'environmental': 0.35, 'operational': 0.25}
        
        try:
            # Progress
            if hasattr(self.settings_manager, 'advanced_progress_bar'):
                self.settings_manager.advanced_progress_bar.setValue(30)
            
            # Önce senaryoları oluştur
            self.optimizer.create_base_scenarios(vessel_data)
            
            if hasattr(self.settings_manager, 'advanced_progress_bar'):
                self.settings_manager.advanced_progress_bar.setValue(50)
            
            # TOPSIS analizini çalıştır
            results = self.optimizer.topsis_decision(vessel_data, weights)
            
            if hasattr(self.settings_manager, 'advanced_progress_bar'):
                self.settings_manager.advanced_progress_bar.setValue(80)
            
            # Sonuçları formatla
            if results and 'ranking' in results:
                ranking = results['ranking']  # Liste: ['Scenario1', 'Scenario2', ...]
                scores = results.get('scores', {})  # Dict: {'Scenario1': 0.85, ...}
                
                result_text = "<b style='color: #5c6bc0;'>📊 TOPSIS Analiz Sonuçları</b><br><br>"
                result_text += "<b>Senaryo Sıralaması:</b><br>"
                
                for i, scenario_name in enumerate(ranking[:5], 1):
                    score = scores.get(scenario_name, 0)
                    if i == 1:
                        result_text += f"🥇 <b>{scenario_name}</b>: {score:.4f}<br>"
                    elif i == 2:
                        result_text += f"🥈 {scenario_name}: {score:.4f}<br>"
                    elif i == 3:
                        result_text += f"🥉 {scenario_name}: {score:.4f}<br>"
                    else:
                        result_text += f"   {i}. {scenario_name}: {score:.4f}<br>"
                
                best_scenario = ranking[0] if ranking else "Bilinmiyor"
                result_text += f"<br><b style='color: #4CAF50;'>✅ Önerilen Senaryo: {best_scenario}</b>"
                
                if hasattr(self.settings_manager, 'advanced_result_label'):
                    self.settings_manager.advanced_result_label.setText(result_text)
            else:
                if hasattr(self.settings_manager, 'advanced_result_label'):
                    self.settings_manager.advanced_result_label.setText(
                        "<span style='color: #FFC107;'>⚠️ Sonuç bulunamadı. Gemi verilerini kontrol edin.</span>"
                    )
            
            if hasattr(self.settings_manager, 'advanced_progress_bar'):
                self.settings_manager.advanced_progress_bar.setValue(100)
            
            if hasattr(self, 'bottom_panel'):
                self.bottom_panel.update_log("TOPSIS analizi tamamlandı.")
                
        except Exception as e:
            print(f"TOPSIS Hatası: {e}")
            if hasattr(self.settings_manager, 'advanced_result_label'):
                self.settings_manager.advanced_result_label.setText(
                    f"<span style='color: #f44336;'>❌ TOPSIS Hatası: {str(e)}</span>"
                )
            if hasattr(self.settings_manager, 'advanced_progress_bar'):
                self.settings_manager.advanced_progress_bar.setValue(0)

    def on_ipso_optimization_click(self):
        """IPSO Pareto optimizasyonu başlatır"""
        print("🔄 IPSO OPTİMİZASYONU BAŞLATIYOR...")
        
        if hasattr(self.settings_manager, 'advanced_progress_bar'):
            self.settings_manager.advanced_progress_bar.setValue(5)
        if hasattr(self.settings_manager, 'advanced_result_label'):
            self.settings_manager.advanced_result_label.setText(
                "IPSO optimizasyonu başlatılıyor... (Bu işlem uzun sürebilir)"
            )
        
        # Gemi verilerini al
        v_obj = self.settings_manager.data_store.get(NodeType.VESSEL)
        vessel_data = asdict(v_obj) if is_dataclass(v_obj) else (v_obj or {})
        
        # IPSO parametrelerini al
        adv_obj = self.settings_manager.data_store.get(NodeType.ADVANCED_ANALYSIS)
        n_particles = getattr(adv_obj, 'ipso_particles', 30) if adv_obj else 30
        n_iterations = getattr(adv_obj, 'ipso_iterations', 50) if adv_obj else 50
        
        try:
            if hasattr(self.settings_manager, 'advanced_progress_bar'):
                self.settings_manager.advanced_progress_bar.setValue(20)
            
            # Önce senaryoları oluştur
            self.optimizer.create_base_scenarios(vessel_data)
            
            if hasattr(self.settings_manager, 'advanced_progress_bar'):
                self.settings_manager.advanced_progress_bar.setValue(35)
            
            # IPSO optimizasyonunu çalıştır
            results = self.optimizer.pareto_optimization_ipso(
                vessel_data,
                objectives=['npv', 'env_score', 'op_score'],
                n_particles=n_particles,
                n_iterations=n_iterations
            )
            
            if hasattr(self.settings_manager, 'advanced_progress_bar'):
                self.settings_manager.advanced_progress_bar.setValue(90)
            
            # Sonuçları formatla
            if results and 'best_design' in results:
                design_params = results.get('design_parameters', {})
                best_obj = results.get('best_objective', 0)
                
                result_text = "<b style='color: #ff9800;'>🔄 IPSO Optimizasyon Sonuçları</b><br><br>"
                result_text += f"<b>Parçacık Sayısı:</b> {n_particles}<br>"
                result_text += f"<b>İterasyon:</b> {n_iterations}<br>"
                result_text += f"<b>En İyi Objektif Değer:</b> {best_obj:.4f}<br><br>"
                
                result_text += "<b>Optimal Tasarım Parametreleri:</b><br>"
                result_text += f"• Yakıt Verimliliği Faktörü: {design_params.get('fuel_efficiency_factor', 0):.3f}<br>"
                result_text += f"• Bakım Kalitesi: {design_params.get('maintenance_quality', 0):.3f}<br>"
                result_text += f"• Operasyonel Günler: {design_params.get('operational_days', 0):.0f} gün/yıl<br>"
                
                # Yakınsama bilgisi
                convergence = results.get('convergence_curve', [])
                if convergence:
                    improvement = (convergence[0] - convergence[-1]) / abs(convergence[0]) * 100 if convergence[0] != 0 else 0
                    result_text += f"<br><b style='color: #4CAF50;'>📉 Yakınsama İyileşmesi: {improvement:.1f}%</b>"
                
                if hasattr(self.settings_manager, 'advanced_result_label'):
                    self.settings_manager.advanced_result_label.setText(result_text)
            else:
                if hasattr(self.settings_manager, 'advanced_result_label'):
                    self.settings_manager.advanced_result_label.setText(
                        "<span style='color: #FFC107;'>⚠️ IPSO sonucu bulunamadı. Gemi verilerini kontrol edin.</span>"
                    )
            
            if hasattr(self.settings_manager, 'advanced_progress_bar'):
                self.settings_manager.advanced_progress_bar.setValue(100)
                
            if hasattr(self, 'bottom_panel'):
                self.bottom_panel.update_log("IPSO optimizasyonu tamamlandı.")
                
        except Exception as e:
            print(f"IPSO Hatası: {e}")
            if hasattr(self.settings_manager, 'advanced_result_label'):
                self.settings_manager.advanced_result_label.setText(
                    f"<span style='color: #f44336;'>❌ IPSO Hatası: {str(e)}</span>"
                )
            if hasattr(self.settings_manager, 'advanced_progress_bar'):
                self.settings_manager.advanced_progress_bar.setValue(0)

    def on_sensitivity_analysis_click(self):
        """Parametre hassasiyet analizi başlatır"""
        print("📈 HASSASİYET ANALİZİ BAŞLATIYOR...")
        
        if hasattr(self.settings_manager, 'advanced_progress_bar'):
            self.settings_manager.advanced_progress_bar.setValue(10)
        if hasattr(self.settings_manager, 'advanced_result_label'):
            self.settings_manager.advanced_result_label.setText("Hassasiyet analizi başlatılıyor...")
        
        # Gemi verilerini al
        v_obj = self.settings_manager.data_store.get(NodeType.VESSEL)
        vessel_data = asdict(v_obj) if is_dataclass(v_obj) else (v_obj or {})
        
        try:
            if hasattr(self.settings_manager, 'advanced_progress_bar'):
                self.settings_manager.advanced_progress_bar.setValue(20)
            
            # Önce senaryoları oluştur
            self.optimizer.create_base_scenarios(vessel_data)
            
            if hasattr(self.settings_manager, 'advanced_progress_bar'):
                self.settings_manager.advanced_progress_bar.setValue(40)
            
            # Hassasiyet analizini çalıştır
            results = self.optimizer.sensitivity_analysis_extended(vessel_data)
            
            if hasattr(self.settings_manager, 'advanced_progress_bar'):
                self.settings_manager.advanced_progress_bar.setValue(80)
            
            # Sonuçları formatla
            if results:
                result_text = "<b style='color: #26a69a;'>📈 Hassasiyet Analizi Sonuçları</b><br><br>"
                
                # Yakıt fiyatı hassasiyeti
                if 'fuel_price' in results:
                    fps = results['fuel_price']
                    result_text += "<b>Yakıt Fiyatı Etkisi ($/ton):</b><br>"
                    for price, scenarios in fps.items():
                        retrofit_npv = scenarios.get('retrofit', 0)
                        result_text += f"• ${price}: Retrofit NPV ${retrofit_npv/1e6:.2f}M<br>"
                    result_text += "<br>"
                
                # Karbon vergisi hassasiyeti
                if 'carbon_tax' in results:
                    cts = results['carbon_tax']
                    result_text += "<b>Karbon Vergisi Etkisi ($/ton CO2):</b><br>"
                    for tax, scenarios in cts.items():
                        retrofit_npv = scenarios.get('retrofit', 0)
                        result_text += f"• ${tax}: Retrofit NPV ${retrofit_npv/1e6:.2f}M<br>"
                    result_text += "<br>"
                
                # İskonto oranı hassasiyeti
                if 'discount_rate' in results:
                    drs = results['discount_rate']
                    result_text += "<b>İskonto Oranı Etkisi:</b><br>"
                    for rate, scenarios in drs.items():
                        retrofit_npv = scenarios.get('retrofit', 0)
                        result_text += f"• {rate*100:.0f}%: Retrofit NPV ${retrofit_npv/1e6:.2f}M<br>"
                
                if hasattr(self.settings_manager, 'advanced_result_label'):
                    self.settings_manager.advanced_result_label.setText(result_text)
            else:
                if hasattr(self.settings_manager, 'advanced_result_label'):
                    self.settings_manager.advanced_result_label.setText(
                        "<span style='color: #FFC107;'>⚠️ Hassasiyet sonucu bulunamadı.</span>"
                    )
            
            if hasattr(self.settings_manager, 'advanced_progress_bar'):
                self.settings_manager.advanced_progress_bar.setValue(100)
                
            if hasattr(self, 'bottom_panel'):
                self.bottom_panel.update_log("Hassasiyet analizi tamamlandı.")
                
        except Exception as e:
            print(f"Hassasiyet Hatası: {e}")
            if hasattr(self.settings_manager, 'advanced_result_label'):
                self.settings_manager.advanced_result_label.setText(
                    f"<span style='color: #f44336;'>❌ Hassasiyet Hatası: {str(e)}</span>"
                )
            if hasattr(self.settings_manager, 'advanced_progress_bar'):
                self.settings_manager.advanced_progress_bar.setValue(0)

    # ====== EXPORT/IMPORT VESSEL DATA ======
    
    def on_export_vessel_click(self):
        """Gemi verilerini JSON olarak kaydet"""
        from PyQt6.QtWidgets import QFileDialog
        
        # Veriyi al
        v_obj = self.settings_manager.data_store.get(NodeType.VESSEL)
        vessel_data = asdict(v_obj) if is_dataclass(v_obj) else (v_obj or {})
        
        # Varsayılan dosya adı
        vessel_name = vessel_data.get('name', 'vessel').replace('/', '_').replace(' ', '_')
        default_name = f"{vessel_name}_data.json"
        
        # Save dialog
        filepath, _ = QFileDialog.getSaveFileName(
            self, 
            "Gemi Verilerini Kaydet", 
            default_name,
            "JSON Files (*.json)"
        )
        
        if filepath:
            try:
                # AssetManager'ın export fonksiyonunu kullan
                self.asset_manager.export_data(vessel_data, filepath, format='json')
                
                # Başarı mesajı
                QMessageBox.information(
                    self, 
                    "Başarılı", 
                    f"✅ Gemi verileri kaydedildi:\n{filepath}"
                )
                
                if hasattr(self, 'bottom_panel'):
                    self.bottom_panel.update_log(f"Gemi verileri kaydedildi: {filepath}")
                    
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Hata", 
                    f"❌ Kaydetme hatası:\n{str(e)}"
                )

    def on_import_vessel_click(self):
        """JSON'dan gemi verilerini yükle"""
        from PyQt6.QtWidgets import QFileDialog
        
        # Open dialog
        filepath, _ = QFileDialog.getOpenFileName(
            self, 
            "Gemi Verilerini Yükle", 
            "",
            "JSON Files (*.json)"
        )
        
        if filepath:
            try:
                # AssetManager'ın import fonksiyonunu kullan
                imported_data = self.asset_manager.import_data(filepath)
                
                if imported_data:
                    # Mevcut VesselData'yı güncelle
                    v_obj = self.settings_manager.data_store.get(NodeType.VESSEL)
                    
                    for key, value in imported_data.items():
                        if hasattr(v_obj, key):
                            setattr(v_obj, key, value)
                    
                    # Formu yeniden çiz
                    self.settings_manager.load_settings(NodeType.VESSEL, "Gemi Veri Girişi")
                    
                    # 3D modeli güncelle
                    vessel_data = asdict(v_obj) if is_dataclass(v_obj) else (v_obj or {})
                    self.model_3d_widget.update_vessel_hull(vessel_data)
                    
                    # Başarı mesajı
                    QMessageBox.information(
                        self, 
                        "Başarılı", 
                        f"✅ Gemi verileri yüklendi:\n{imported_data.get('name', 'Gemi')}"
                    )
                    
                    if hasattr(self, 'bottom_panel'):
                        self.bottom_panel.update_log(f"Gemi verileri yüklendi: {filepath}")
                else:
                    QMessageBox.warning(
                        self, 
                        "Uyarı", 
                        "⚠️ Dosya boş veya geçersiz format."
                    )
                    
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Hata", 
                    f"❌ Yükleme hatası:\n{str(e)}"
                )

    # ====== PDF RAPOR OLUŞTURMA ======
    
    def on_generate_pdf_click(self):
        """PDF rapor oluştur ve kaydet"""
        from PyQt6.QtWidgets import QFileDialog
        
        print("📄 PDF RAPOR OLUŞTURULUYOR...")
        
        # Progress göster
        if hasattr(self.settings_manager, 'advanced_progress_bar'):
            self.settings_manager.advanced_progress_bar.setValue(10)
        if hasattr(self.settings_manager, 'advanced_result_label'):
            self.settings_manager.advanced_result_label.setText("PDF rapor oluşturuluyor...")
        
        try:
            # Report generator import
            from utils.report_generator import SmartCAPEXReportGenerator
            
            # Gemi verilerini al
            v_obj = self.settings_manager.data_store.get(NodeType.VESSEL)
            vessel_data = asdict(v_obj) if is_dataclass(v_obj) else (v_obj or {})
            
            if hasattr(self.settings_manager, 'advanced_progress_bar'):
                self.settings_manager.advanced_progress_bar.setValue(30)
            
            # Analiz sonuçlarını topla
            analysis_results = self._collect_analysis_results()
            
            if hasattr(self.settings_manager, 'advanced_progress_bar'):
                self.settings_manager.advanced_progress_bar.setValue(50)
            
            # Kaydet dialogu
            vessel_name = vessel_data.get('name', 'Vessel').replace('/', '_').replace(' ', '_')
            default_name = f"SmartCAPEX_Report_{vessel_name}.pdf"
            
            filepath, _ = QFileDialog.getSaveFileName(
                self, 
                "PDF Raporu Kaydet", 
                default_name,
                "PDF Files (*.pdf)"
            )
            
            if filepath:
                if hasattr(self.settings_manager, 'advanced_progress_bar'):
                    self.settings_manager.advanced_progress_bar.setValue(70)
                
                # Rapor oluştur
                generator = SmartCAPEXReportGenerator()
                output_path = generator.generate_report(vessel_data, analysis_results, filepath)
                
                if hasattr(self.settings_manager, 'advanced_progress_bar'):
                    self.settings_manager.advanced_progress_bar.setValue(100)
                
                # Başarı mesajı
                if hasattr(self.settings_manager, 'advanced_result_label'):
                    self.settings_manager.advanced_result_label.setText(
                        f"<b style='color: #9C27B0;'>📄 PDF Rapor Oluşturuldu!</b><br><br>"
                        f"Dosya: {os.path.basename(output_path)}<br>"
                        f"Konum: {os.path.dirname(output_path)}"
                    )
                
                QMessageBox.information(
                    self, 
                    "Başarılı", 
                    f"✅ PDF rapor oluşturuldu:\n{output_path}"
                )
                
                if hasattr(self, 'bottom_panel'):
                    self.bottom_panel.update_log(f"PDF rapor oluşturuldu: {output_path}")
            else:
                if hasattr(self.settings_manager, 'advanced_progress_bar'):
                    self.settings_manager.advanced_progress_bar.setValue(0)
                if hasattr(self.settings_manager, 'advanced_result_label'):
                    self.settings_manager.advanced_result_label.setText("Rapor oluşturma iptal edildi.")
                    
        except Exception as e:
            print(f"PDF Hatası: {e}")
            import traceback
            traceback.print_exc()
            
            if hasattr(self.settings_manager, 'advanced_progress_bar'):
                self.settings_manager.advanced_progress_bar.setValue(0)
            
            QMessageBox.critical(
                self, 
                "Hata", 
                f"❌ PDF oluşturma hatası:\n{str(e)}"
            )
    
    def _collect_analysis_results(self) -> Dict:
        """Mevcut analiz sonuçlarını topla"""
        import numpy as np
        
        results = {}
        
        # NPV karşılaştırması (örnek veriler - gerçek analizden alınabilir)
        results['npv_comparison'] = [-500000, 1200000, 2500000]
        
        # Nakit akışı (20 yıl)
        results['cash_flow'] = list(np.cumsum(np.random.randn(21) * 200000 + 150000))
        
        # ROI bileşenleri
        results['roi_breakdown'] = [45, 25, 15, 15]
        
        # Özet metrikler
        results['total_npv'] = 2500000
        results['payback_years'] = 4.5
        results['irr'] = 18.5
        results['investment'] = 850000
        results['co2_reduction'] = 25
        
        # Tavsiye
        results['recommendation'] = 'RETROFIT'
        
        # Senaryo bilgileri
        results['scenarios'] = {
            'do_nothing': {'npv': -500000, 'risk': 'Yüksek'},
            'retrofit': {'npv': 1200000, 'risk': 'Orta'},
            'new_build': {'npv': 2500000, 'risk': 'Düşük'}
        }
        
        return results

    # ====== GEOMETRY ENGINE (PARAMETRİK GÖVDE ÜRETİMİ) ======

    def on_generate_hull_click(self):
        """Parametrik gövde üretimini başlatır (GeometryWorker)."""
        print("🛳️ PARAMETRİK GÖVDE ÜRETİMİ BAŞLATIYOR...")

        # Gemi verilerini al
        v_obj = self.settings_manager.data_store.get(NodeType.VESSEL)
        vessel_data = asdict(v_obj) if is_dataclass(v_obj) else (v_obj or {})

        # İlerleme çubuğunu sıfırla
        if hasattr(self.settings_manager, 'geometry_progress_bar'):
            self.settings_manager.geometry_progress_bar.setValue(0)
        if hasattr(self.settings_manager, 'geometry_result_label'):
            self.settings_manager.geometry_result_label.setText("Parametrik gövde üretimi başlatılıyor...")

        # Output path
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'geometry')
        os.makedirs(output_dir, exist_ok=True)
        vessel_name = vessel_data.get('name', 'vessel').replace('/', '_').replace(' ', '_')
        output_path = os.path.join(output_dir, f"{vessel_name}_hull.stl")

        # Worker oluştur ve başlat
        self.geometry_worker = GeometryWorker(vessel_data, output_path)
        self.geometry_worker.progress_signal.connect(self.on_geometry_progress)
        self.geometry_worker.finished_signal.connect(self.on_geometry_finished)
        self.geometry_worker.error_signal.connect(self.on_geometry_error)
        self.geometry_worker.start()

    def on_geometry_progress(self, val, msg):
        """Geometri üretim ilerleme çubuğunu günceller."""
        try:
            if hasattr(self.settings_manager, 'geometry_progress_bar'):
                self.settings_manager.geometry_progress_bar.setValue(val)
            if hasattr(self.settings_manager, 'geometry_result_label'):
                self.settings_manager.geometry_result_label.setText(msg)
        except (RuntimeError, AttributeError):
            pass
        print(f"Geometry Progress: {val}% - {msg}")

    def on_geometry_finished(self, stl_path):
        """Geometri üretimi tamamlandığında çağrılır."""
        print(f"✅ GÖVDE GEOMETRİSİ ÜRETİLDİ: {stl_path}")
        usd_path = stl_path.replace('.stl', '.usda') if stl_path else None

        # Vessel data'ya STL ve USD path'lerini kaydet (Frontend-Backend senkronu)
        v_obj = self.settings_manager.data_store.get(NodeType.VESSEL)
        if v_obj:
            if hasattr(v_obj, 'stl_path'):
                v_obj.stl_path = stl_path
            if hasattr(v_obj, 'usd_path'):
                v_obj.usd_path = usd_path

        # v_obj → vessel_data dönüşümü (BUG FIX: önceki sürümde tanımsızdı)
        vessel_data = asdict(v_obj) if is_dataclass(v_obj) else (v_obj if isinstance(v_obj, dict) else {})

        # 3D görüntüleyiciye yükle (Öncelikli USD, yoksa legacy STL)
        try:
            if usd_path and os.path.exists(usd_path):
                self.model_3d_widget.load_usd(usd_path)
            else:
                self.model_3d_widget.load_stl(stl_path)
        except Exception as e:
            print(f"⚠️ 3D Model yükleme hatası: {e}")
            self.model_3d_widget.load_stl(stl_path)
            
        self.graphics_stack.setCurrentWidget(self.model_3d_widget)

        # Phase 3: Holtrop-Mennen Resistance + Ship-D DL Inference
        speed = vessel_data.get('speed', 12.0)
        ai_msg = ""
        try:
            # Multi-fidelity prediction: PointNet++ → EANN → Holtrop
            hydro_results = getattr(self, 'surrogate_modeler').predict_hydrodynamics(vessel_data, speed)
            Rt = hydro_results.get('Rt_holtrop', 0)
            Rw = hydro_results.get('Rw_holtrop', 0)
            Pe = hydro_results.get('Pe_kW', 0)
            Fr = hydro_results.get('Froude_number', 0)
            k1 = hydro_results.get('form_factor_k1', 0)
            src = hydro_results.get('source', 'N/A')
            pc_nodes = hydro_results.get('point_cloud_nodes', 2048)
            ai_msg = (
                f"\n\n🌊 [Holtrop-Mennen 1984] Direnç Analizi:"
                f"\n  Rt = {Rt:.1f} kN | Rw = {Rw:.1f} kN | "
                f"Pe = {Pe:.0f} kW"
                f"\n  Fr = {Fr:.4f} | (1+k1) = {k1:.3f} | Kaynak: {src}"
                f"\n\n🤖 [Ship-D Deep Learning]:"
                f"\n  Nokta Bulutu ({pc_nodes} nodes) çıkarıldı."
            )
            # Store for financial analysis
            self._last_hydro_results = hydro_results
        except Exception as e:
            print(f"DL Inference Error: {e}")
            ai_msg = f"\n\n🤖 [Ship-D Deep Learning] Model henüz eğitilmedi."

        # Sonuç etiketini güncelle
        if hasattr(self.settings_manager, 'geometry_result_label'):
            self.settings_manager.geometry_result_label.setText(
                f"✅ STL ve USD mesh üretildi.{ai_msg}"
            )
            self.settings_manager.geometry_result_label.setStyleSheet(
                "color: #4CAF50; font-size: 11px; font-weight: bold; margin-top: 5px;"
            )

        # Log paneline yaz
        if hasattr(self, 'bottom_panel'):
            self.bottom_panel.update_log(f"Parametrik gövde üretildi ve DL tahmini yapıldı.")

        # QMessageBox ile bildirim
        QMessageBox.information(
            self,
            "Başarılı",
            f"✅ Parametrik gövde mesh'i üretildi!\n\n"
            f"Dosya: {os.path.basename(stl_path)}\n"
            f"Konum: {os.path.dirname(stl_path)}{ai_msg}\n\n"
            f"3D görüntüleyiciye otomatik yüklendi."
        )

    def on_geometry_error(self, error_msg):
        """Geometri üretim hatası."""
        print(f"❌ GEOMETRİ HATASI: {error_msg}")

        if hasattr(self.settings_manager, 'geometry_progress_bar'):
            self.settings_manager.geometry_progress_bar.setValue(0)
        if hasattr(self.settings_manager, 'geometry_result_label'):
            self.settings_manager.geometry_result_label.setText(f"❌ Hata: {error_msg}")
            self.settings_manager.geometry_result_label.setStyleSheet(
                "color: #f44336; font-size: 11px; margin-top: 5px;"
            )

        QMessageBox.critical(
            self,
            "Geometri Hatası",
            f"❌ Parametrik gövde üretimi başarısız:\n\n{error_msg}"
        )

    def on_settings_data_changed(self, node_type, data):
        """SettingsManager'dan gelen veri değişikliklerini yakalar"""
        if node_type == NodeType.VESSEL or node_type == NodeType.RETROFIT:
            # Gemi veya Retrofit verisi değiştiyse 3D modeli anlık güncelle
            # NOT: data parametresi zaten asdict() yapılmış halde geliyor (helper metodlardan ötürü)
            # Ancak biz tam kaynağı da alabiliriz.
            
            v_obj = self.settings_manager.data_store.get(NodeType.VESSEL)
            r_obj = self.settings_manager.data_store.get(NodeType.RETROFIT)
            
            vessel_data = asdict(v_obj) if is_dataclass(v_obj) else (v_obj or {})
            retrofit_data = asdict(r_obj) if is_dataclass(r_obj) else (r_obj or {})
            
            # Retrofit listesini ana gemi verisiyle senkronize et (Visualization için)
            vessel_data['selected_retrofit'] = retrofit_data.get('selected_retrofit', [])
            
            self.model_3d_widget.update_vessel_hull(vessel_data)

    # --- SİNYAL YÖNETİMİ (GÜNCELLENDİ) ---
    def on_selection_changed(self):
        """Ağaca tıklayınca çalışır"""
        items = self.tree_view.selectedItems()
        if not items: return
        
        current = items[0]
        node_text = current.text(0)
        # NodeType verisini al (VESSEL mi, CLIMATE mi?)
        node_type = current.data(0, Qt.ItemDataRole.UserRole)
        
        # Debug için yazdıralım
        print(f"Form Yükleniyor: {node_text} (ID: {node_type})")
        
        # YÖNETİCİYE FORMU ÇİZMESİNİ SÖYLE
        self.settings_manager.load_settings(node_type, node_text)
        
        # 3D Görüntüleyici varsayılan olarak hepsi için sağda kalsın (CFD hariç istersen)
        if node_type == NodeType.CFD:
            self.graphics_stack.setCurrentWidget(self.cfd_widget)
        else:
            self.graphics_stack.setCurrentWidget(self.model_3d_widget)
        
        # EĞER GEMİ VERİSİ SEÇİLDİYSE 3D MODELİ GÜNCELLE
        if node_type == NodeType.VESSEL:
            v_obj = self.settings_manager.data_store.get(NodeType.VESSEL)
            vessel_data = asdict(v_obj) if is_dataclass(v_obj) else (v_obj or {})
            
            # Check for Custom USD/STL
            if vessel_data.get('usd_path'):
                self.model_3d_widget.load_usd(vessel_data['usd_path'])
            elif vessel_data.get('stl_path'):
                self.model_3d_widget.load_stl(vessel_data['stl_path'])
            else:
                self.model_3d_widget.update_vessel_hull(vessel_data)
            
            # Export/Import butonlarını bağla
            if hasattr(self.settings_manager, 'btn_export_vessel'):
                try: self.settings_manager.btn_export_vessel.clicked.disconnect()
                except: pass
                self.settings_manager.btn_export_vessel.clicked.connect(self.on_export_vessel_click)
            
            if hasattr(self.settings_manager, 'btn_import_vessel'):
                try: self.settings_manager.btn_import_vessel.clicked.disconnect()
                except: pass
                self.settings_manager.btn_import_vessel.clicked.connect(self.on_import_vessel_click)
            
            # Geometri Üret butonunu bağla
            if hasattr(self.settings_manager, 'btn_generate_hull'):
                try: self.settings_manager.btn_generate_hull.clicked.disconnect()
                except: pass
                self.settings_manager.btn_generate_hull.clicked.connect(self.on_generate_hull_click)
        
        # EĞER SURROGATE SEÇİLDİYSE BUTONU BAĞLA
        if node_type == NodeType.SURROGATE:
            # SettingsManager içindeki butonu bul ve fonksiyonumuza bağla
            if hasattr(self.settings_manager, 'btn_train'):
                # Önce eski bağlantıları temizle (tekrar tekrar bağlanmasın)
                try: self.settings_manager.btn_train.clicked.disconnect()
                except: pass
                
                # Yeni bağlantıyı yap
                self.settings_manager.btn_train.clicked.connect(self.on_train_click)
                
        # EĞER CFD SEÇİLDİYSE BUTONU BAĞLA
        if node_type == NodeType.CFD:
            if hasattr(self.settings_manager, 'btn_run_cfd'):
                try: self.settings_manager.btn_run_cfd.clicked.disconnect()
                except: pass
                self.settings_manager.btn_run_cfd.clicked.connect(self.on_run_cfd_click)

        # EĞER RUN SEÇİLDİYSE BUTONU BAĞLA
        if node_type == NodeType.RUN:
            if hasattr(self.settings_manager, 'btn_run_analysis'):
                try: self.settings_manager.btn_run_analysis.clicked.disconnect()
                except: pass
                self.settings_manager.btn_run_analysis.clicked.connect(self.on_run_analysis_click)
        
        # EĞER CLIMATE SEÇİLDİYSE BUTONU BAĞLA
        if node_type == NodeType.CLIMATE:
            if hasattr(self.settings_manager, 'btn_run_climate'):
                try: self.settings_manager.btn_run_climate.clicked.disconnect()
                except: pass
                self.settings_manager.btn_run_climate.clicked.connect(self.on_climate_analysis_click)

        # EĞER 3D MODEL SEÇİLDİYSE mantığı tamamen kaldırıldı (hep görünür)
        
        # EĞER GELİŞMİŞ ANALİZ SEÇİLDİYSE BUTONLARI BAĞLA
        if node_type == NodeType.ADVANCED_ANALYSIS:
            if hasattr(self.settings_manager, 'btn_run_topsis'):
                try: self.settings_manager.btn_run_topsis.clicked.disconnect()
                except: pass
                self.settings_manager.btn_run_topsis.clicked.connect(self.on_topsis_analysis_click)
            
            if hasattr(self.settings_manager, 'btn_run_ipso'):
                try: self.settings_manager.btn_run_ipso.clicked.disconnect()
                except: pass
                self.settings_manager.btn_run_ipso.clicked.connect(self.on_ipso_optimization_click)
            
            if hasattr(self.settings_manager, 'btn_run_sensitivity'):
                try: self.settings_manager.btn_run_sensitivity.clicked.disconnect()
                except: pass
                self.settings_manager.btn_run_sensitivity.clicked.connect(self.on_sensitivity_analysis_click)
            
            # PDF Rapor Butonu
            if hasattr(self.settings_manager, 'btn_generate_pdf'):
                try: self.settings_manager.btn_generate_pdf.clicked.disconnect()
                except: pass
                self.settings_manager.btn_generate_pdf.clicked.connect(self.on_generate_pdf_click)

    def _ensure_geometry_exists(self):
        """
        Geometri (STL/USD) yoksa senkron olarak üretir.
        Returns: (stl_path, usd_path) tuple
        """
        v_obj = self.settings_manager.data_store.get(NodeType.VESSEL)
        stl_path = getattr(v_obj, 'stl_path', None)
        usd_path = getattr(v_obj, 'usd_path', None)

        if stl_path and os.path.exists(stl_path):
            print(f"✅ Mevcut geometri kullanılıyor: {stl_path}")
            return stl_path, usd_path

        # Geometri yoksa otomatik üret
        print("⚙️ Geometri bulunamadı — otomatik üretiliyor...")
        if not HAS_GEOMETRY_ENGINE:
            print("⚠️ Geometry Engine yüklü değil, geometri üretilemiyor.")
            return None, None

        vessel_data = asdict(v_obj) if is_dataclass(v_obj) else (v_obj if isinstance(v_obj, dict) else {})

        adapter = RetrosimHullAdapter()
        adapter.set_from_ui(vessel_data)

        # Output path
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'geometry')
        os.makedirs(output_dir, exist_ok=True)
        vessel_name = vessel_data.get('name', 'vessel').replace('/', '_').replace(' ', '_')
        stl_out = os.path.join(output_dir, f"{vessel_name}_hull.stl")
        usd_out = stl_out.replace('.stl', '.usda')

        try:
            stl_path = adapter.generate_stl(output_path=stl_out, n_stations=31, n_waterlines=15)
            usd_path = adapter.generate_usda(output_path=usd_out, n_stations=31, n_waterlines=15)

            # Vessel objesine kaydet
            if hasattr(v_obj, 'stl_path'):
                v_obj.stl_path = stl_path
            if hasattr(v_obj, 'usd_path'):
                v_obj.usd_path = usd_path

            print(f"✅ Geometri otomatik üretildi: {stl_path}")
        except Exception as e:
            print(f"⚠️ Otomatik geometri üretim hatası: {e}")
            stl_path, usd_path = None, None

        return stl_path, usd_path

    def on_run_cfd_click(self):
        """Kullanıcının girdiği Sınır Koşulları ile Dalga Direnci (CFD) Analizini Başlatır.
        Geometri yoksa otomatik üretir ve analize dahil eder."""
        print("🌊 DALGA DİRENCİ (CFD) ANALİZİ BAŞLATILIYOR...")

        # ── 1. Geometriyi Garantile ──
        stl_path, usd_path = self._ensure_geometry_exists()
        if stl_path:
            geom_msg = f"Gemi geometrisi dahil edildi: {os.path.basename(stl_path)}"
        else:
            geom_msg = "Geometri üretilemedi — varsayılan eliptik yaklaşım kullanılacak."

        QMessageBox.information(
            self, "PINN CFD Ajanı",
            f"Dalga Direnci (CFD) analizi başlatılıyor.\n{geom_msg}"
        )

        # ── 2. Verileri Topla ──
        cfd_obj = self.settings_manager.data_store.get(NodeType.CFD)
        cfd_data = asdict(cfd_obj) if is_dataclass(cfd_obj) else (cfd_obj or {})
        v_obj = self.settings_manager.data_store.get(NodeType.VESSEL)
        vessel_data = asdict(v_obj) if is_dataclass(v_obj) else (v_obj or {})

        # Geometri path'lerini vessel_data'ya ekle (ajanlar için)
        vessel_data['stl_path'] = stl_path
        vessel_data['usd_path'] = usd_path

        combined_data = {**vessel_data, **cfd_data}

        inlet_v = cfd_data.get('inlet_velocity', vessel_data.get('speed', 12.0))
        resolution = cfd_data.get('resolution', 50)

        # ── 3. CFD Widget'a Geometri Mesh ve Gemi Boyutları Yükle ──
        self.cfd_widget.ship_speed = float(vessel_data.get('speed', 12.0))
        self.cfd_widget.update_hull_geometry(stl_path, vessel_data=vessel_data)

        # ── 4. Solver'a Göre Analizi Başlat ──
        solver = combined_data.get('solver', 'NVIDIA Modulus 3D FNO')
        if "Modulus" in solver:
            if hasattr(self, 'bottom_panel'):
                self.bottom_panel.update_log(f"Modulus AI Inference başlatılıyor... ({geom_msg})")
            self.modulus_agent.run_inference(combined_data, combined_data)
        else:
            if hasattr(self, 'bottom_panel'):
                self.bottom_panel.update_log(f"Legacy PINN CFD başlatılıyor... ({geom_msg})")
            self.cfd_worker = CFDWorker(self.cfd_agent, combined_data)
            self.cfd_worker.start()

        # ── 5. CFD Vizualizasyon Widgetına Geç ──
        self.graphics_stack.setCurrentWidget(self.cfd_widget)

        # Akış parçacıkları animasyonunu başlat (dummy sonuç — gerçek sonuç callback ile gelir)
        results = {
            'X': None, 'Y': None, 'U': None, 'V': None, 'P': None,
            'speed': inlet_v,
            'resolution': resolution,
            'domain_x': cfd_data.get('domain_x_mult', 5.0),
            'domain_y': cfd_data.get('domain_y_mult', 2.0),
            'domain_z': cfd_data.get('domain_z_mult', 2.0),
        }
        if hasattr(self, 'cfd_widget'):
            self.cfd_widget.update_plot(results)

        if hasattr(self, 'bottom_panel'):
            self.bottom_panel.update_log(f"CFD Analizi {inlet_v} m/s hızında yüklendi. (RTX Vizyon)")

    def on_ribbon_action(self, action_name):
        """Handle ribbon actions based on current view"""
        print(f"Ribbon Action: {action_name}")
        
        if "Block" in action_name:
            self.model_3d_widget.add_shape("cube")
        elif "Sphere" in action_name:
            self.model_3d_widget.add_shape("sphere")
        elif "Add\nComponent" in action_name:
            self.model_3d_widget.add_shape("figure")
        
        # Omniverse / USD Actions
        elif "USD\nLOD" in action_name:
            lvl = int(action_name.split("LOD")[-1].strip())
            for node in self.model_3d_widget.scene.node_list:
                if isinstance(node, USDHull): node.set_lod(lvl)
            self.model_3d_widget.update()
        elif "Swap\nY-Z Axis" in action_name:
            for node in self.model_3d_widget.scene.node_list:
                if isinstance(node, USDHull): node.swap_axes()
            self.model_3d_widget.update()
        elif "Optimize\nMesh" in action_name:
            for node in self.model_3d_widget.scene.node_list:
                if isinstance(node, USDHull): node.optimize_geometry()
            self.model_3d_widget.update() or self.bottom_panel.update_log("USD Mesh Optimized.")
        elif "Retrofit\nWizard" in action_name:
            # Settings panel'de Retrofit Wizard'ı aç
            self.settings_manager.load_settings(NodeType.RETROFIT, "Retrofit Wizard")
            self.graphics_stack.setCurrentWidget(self.model_3d_widget)

        if any(x in action_name for x in ["Block", "Sphere", "Cylinder", "Add", "USD", "Axis", "Optimize", "Retrofit"]):
            self.graphics_stack.setCurrentWidget(self.model_3d_widget)

    def open_context_menu(self, position):
        item = self.tree_view.itemAt(position)
        if not item:
            return
            
        node_type = item.data(0, Qt.ItemDataRole.UserRole)
        
        menu = QMenu()
        del_action = QAction("Sil", self)
        
        if node_type == NodeType.PROJECT:
            del_action.setEnabled(False)
        else:
            del_action.triggered.connect(lambda: self.delete_current_node(item))
            
        menu.addAction(del_action)
        menu.exec(self.tree_view.viewport().mapToGlobal(position))

    def delete_current_node(self, item):
        # QTreeWidget'tan öğe silme
        parent = item.parent()
        if parent:
            parent.removeChild(item)
        else:
            # Root ise (gerçi yukarıda engelledik ama)
            index = self.tree_view.indexOfTopLevelItem(item)
            self.tree_view.takeTopLevelItem(index)

    def on_tree_click(self, item, col):
        """
        Ağaç menüsünde (Sol Panel) bir şeye tıklandığında çalışır.
        """
        # 1. Tıklanan satırın üzerindeki yazıyı al (Örn: "Gemi Veri Girişi")
        node_name = item.text(0)
        
        # 2. Sağ taraftaki ayar panelini güncelle
        # NOT: Senin yeni kodunda fonksiyonun adı 'load_settings' olduğu için burayı değiştirdik.
        self.settings_manager.load_settings(node_name)
    
    def update_settings_view(self, node_text):
        """
        Sağ alt paneldeki ayar formunu günceller.
        Eski 'if/else' ve 'stack' mantığı yerine, yeni dinamik yöneticiyi kullanır.
        """
        # Yöneticiye sadece ismi veriyoruz, gerisini o hallediyor.
        self.settings_manager.load_settings(node_text)


"""
class SettingsManager(QWidget):
    def __init__(self):
        super().__init__()
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(self.main_layout)
        self.current_form = None 
        
        # PROJE VERİSİ (Normalde veritabanından gelecek)
        self.project_data = {
            "SmartCAPEX Project": {"type": "Project", "label": "Ana Proje"},
            "Gemi Veri Girişi": {"type": "Gemi", "label": "M/V SmartCAPEX", "dwt": 50000},
            "Surrogate Modeler": {"type": "Surrogate", "label": "Yapay Zeka Modeli"},
            "Optimizer": {"type": "Optimizer", "pop": 50},
            "Climate Guardian": {"type": "Guardian", "year": 2030},
            "Analiz / Run": {"type": "Run"}
        }

    def load_settings(self, agent_id):
        # Ana pencereden çağrılan fonksiyon
        # 1. Veriyi Bul
        if agent_id in self.project_data:
            data = self.project_data[agent_id]
            agent_type = data.get("type", "Generic")
        else:
            # Bilinmeyen bir şeye tıklandıysa boş veri oluştur
            data = {}
            agent_type = "Generic"

        # 2. Temizlik
        self._clear_layout()

        # 3. Başlık
        header = QLabel(f"⚙️ {agent_type} Ayarları: {agent_id}")
        header.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px; color: #42a5f5;")
        self.main_layout.addWidget(header)

        # 4. Form Alanı
        group = QGroupBox("Parametreler")
        self.current_form = QFormLayout()
        
        # 5. Şemayı Al ve Çiz
        fields = self._get_schema(agent_type)
        for field in fields:
            self._create_row(field, data)

        group.setLayout(self.current_form)
        self.main_layout.addWidget(group)
"""