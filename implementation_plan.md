# Entegre Edilmemiş Backend Özelliklerinin GUI'ye Aktarımı

## Audit Sonuçları

Projedeki tüm backend modülleri, GUI dosyaları ve sinyal bağlantıları kapsamlı olarak incelendi. Aşağıdaki **5 önemli backend özelliği** var ancak GUI'ye hiç bağlanmamış:

### Tespit Edilen Boşluklar

| # | Backend Özelliği | Dosya | GUI Durumu |
|---|------------------|-------|------------|
| 1 | **FFDHullMorpher** — FFD deformasyon, dataset üretimi, geometri doğrulama | `FFDHullMorpher.py` + `hull_adapter.py` | ❌ GUI'de hiçbir yerde referans yok |
| 2 | **OmniversePanel** — USD stage viewer, prim tree, material controls | `gui/omniverse_panel.py` | ❌ Widget mevcut ama hiçbir layout'a eklenmemiş |
| 3 | **PointNet Agent** — 3D point cloud tabanlı direnç tahmini | `agents/pointnet_agent.py` | ⚠️ Dolaylı çağrılıyor (`predict_hydrodynamics` içinden) ama kendi tree node'u ve ayar formu yok |
| 4 | **OpenFOAM Bridge** — OpenFOAM case dosyası üretimi | `utils/openfoam_generator.py` | ❌ GUI'de hiçbir buton veya bağlantı yok |
| 5 | **Design Vector Validation** — Ship-D parametrelerini doğrulama | `hull_adapter.py:validate_design_vector()` | ❌ GUI'de buton veya gösterge yok |

---

## Zaten Entegre Olan Özellikler (Sorun Yok)

| Özellik | GUI Entegrasyonu |
|---------|-----------------|
| Surrogate Modeler (EANN eğitimi) | ✅ Tree node + btn_train + progress |
| Climate Guardian (İklim analizi) | ✅ Tree node + btn_run_climate + results |
| Multi-Objective Optimizer | ✅ Tree node + btn_run_analysis |
| TOPSIS / IPSO Analizi | ✅ Tree node + butonlar + sonuç alanı |
| Hassasiyet Analizi | ✅ Buton + handler |
| PDF Rapor Oluşturma | ✅ Buton + handler |
| PINN CFD / Modulus Agent | ✅ Tree node + btn_run_cfd + 3D viz |
| Regulatory Agent (CII/ETS) | ✅ Status bar + indicators |
| Geometri Üretimi (STL/USD) | ✅ btn_generate_hull + worker |
| Holtrop-Mennen Direnci | ✅ on_geometry_finished + analysis |
| Vessel I/O (JSON Export/Import) | ✅ Butonlar + handlers |

---

## Proposed Changes

### 1. FFD Hull Morpher — Yeni Tree Node + Settings Form

FFD deformasyon, dataset üretimi ve geometri doğrulama arayüzü eklenecek.

#### [MODIFY] [model_builder_model.py](file:///c:/Users/abdur/Desktop/Scap/SmartCAPEX_AI_KIM_2003/gui/model_builder_model.py)
- Yeni `NodeType`: `FFD_MORPHER = "FFD Hull Morpher"`

#### [MODIFY] [main_window.py](file:///c:/Users/abdur/Desktop/Scap/SmartCAPEX_AI_KIM_2003/gui/main_window.py)
- **Dataclass**: `FFDConfig` (n_control_points, sigma, n_samples, ship_type, symmetry)
- **Tree Node**: `item_ffd` → Surrogate Modeler altında "🔧 FFD Hull Morpher" 
- **Settings Form**: `_form_ffd_morpher(data)`
  - Kontrol parametreleri: lattice [nx, ny, nz], sigma, max_displacement
  - Ship type seçimi (tanker, container, bulk_carrier...)
  - Üç buton:
    - `btn_ffd_deform` → "🔧 FFD Deformasyon Uygula"
    - `btn_ffd_dataset` → "📊 Dataset Üret (N Varyant)"  
    - `btn_ffd_validate` → "✅ Geometri Doğrula"
- **Handlers**:
  - `on_ffd_deform_click()` — Rastgele μ vektörü ile hull deformu
  - `on_ffd_dataset_click()` — N varyant STL üretir
  - `on_ffd_validate_click()` — Mevcut geometriyi literatür sınırlarıyla doğrular
- **Signal Connections**: in `on_selection_changed()`

---

### 2. OmniversePanel — Graphics Stack'e Ekleme

Widget kodu hazır, sadece layout'a bağlanması gerekiyor.

#### [MODIFY] [main_window.py](file:///c:/Users/abdur/Desktop/Scap/SmartCAPEX_AI_KIM_2003/gui/main_window.py)
- `from gui.omniverse_panel import OmniversePanel` import
- `self.omniverse_panel = OmniversePanel()` oluştur
- `self.graphics_stack.addWidget(self.omniverse_panel)` ekle
- **Tree Node**: `item_omniverse` → Kök altında "🌐 Omniverse / Digital Twin"
- **NodeType**: `OMNIVERSE = "Omniverse"`
- Yeni `NodeType.OMNIVERSE` seçildiğinde `graphics_stack` Omniverse paneline geçsin
- Geometri üretildiğinde otomatik USD yükle: `self.omniverse_panel.load_usd_stage(usd_path)`

---

### 3. PointNet Agent — Ayar Formu Ekleme

PointNet dolaylı olarak `predict_hydrodynamics` ile çağrılıyor ancak kendi konfigürasyon paneli yok.

#### [MODIFY] [main_window.py](file:///c:/Users/abdur/Desktop/Scap/SmartCAPEX_AI_KIM_2003/gui/main_window.py)
- **Dataclass**: `PointNetConfig` (num_points, model_path, batch_size)
- **NodeType**: `POINTNET = "PointNet Agent"`
- **Tree Node**: `item_pointnet` → Surrogate Modeler altında "🧩 PointNet++ Agent"
- **Settings Form**: `_form_pointnet(data)` — nokta bulutu boyutu, model dosyası, batch size
- **Buton**: `btn_extract_pointcloud` → "🧩 Point Cloud Çıkar"
- **Handler**: `on_extract_pointcloud_click()` — Hull'dan point cloud çıkarıp log'a yazdırır

---

### 4. OpenFOAM Bridge — Case Export Butonu

OpenFOAM case dosyası üretim arayüzü.

#### [MODIFY] [main_window.py](file:///c:/Users/abdur/Desktop/Scap/SmartCAPEX_AI_KIM_2003/gui/main_window.py)
- **Buton**: CFD Settings formu (`_form_cfd`) içine "📂 OpenFOAM Case Oluştur" butonu
- **Handler**: `on_export_openfoam_click()` — `utils/openfoam_generator.py` kullanarak case dosyaları üretir
- **Signal Connection**: CFD node seçildiğinde bağla

---

### 5. Design Vector Validation — Doğrulama Butonu

Geometri üretmeden önce parametreleri doğrulama.

#### [MODIFY] [main_window.py](file:///c:/Users/abdur/Desktop/Scap/SmartCAPEX_AI_KIM_2003/gui/main_window.py)
- **Buton**: Gemi Veri Girişi formu (`_form_vessel`) içine "✅ Parametreleri Doğrula" butonu
- **Handler**: `on_validate_design_vector_click()` — Tüm parametreleri Ship-D sınırlarıyla karşılaştırır
- **Sonuç**: Doğrulama raporu `geometry_result_label`'a yazılır (yeşil/sarı/kırmızı)

---

## Dosya Değişiklik Özeti

| Dosya | Değişiklik |
|-------|-----------|
| `gui/model_builder_model.py` | +3 yeni NodeType (FFD_MORPHER, OMNIVERSE, POINTNET) |
| `gui/main_window.py` | +3 dataclass, +3 tree node, +5 settings form, +7 handler, +import değişiklikleri |

## Verification Plan

### Automated Tests
```bash
# 1. GUI Launch — tüm yeni node'lar görülüyor mu?
python main_gui.py

# 2. FFD Morpher — tree'den tıkla, form görüntüle, deform yap
# 3. OmniversePanel — tree'den tıkla, panel göster, USD yükle
# 4. PointNet — tree'den tıkla, point cloud çıkar
# 5. OpenFOAM — CFD settings'te buton görünür mü?
# 6. Validate DV — Vessel settings'te buton çalışıyor mu?
```

### Manual Verification
- Her yeni tree node tıklanabilir ve doğru form gösteriliyor
- Her buton doğru handler'ı tetikliyor
- Hata durumları bildiriliyor (eksik geometri, PyGeM yüklü değil, vb.)
