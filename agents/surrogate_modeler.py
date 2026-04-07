"""
Surrogate Modeler (EANN + Kriging) - PyQt6 Integrated
=====================================================
Arayüz ile haberleşen, Sinyal (Signal) tabanlı yapı.
PyTorch-based Emotional ANN (EANN) with Hormonal Modulation.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Optional, List, Tuple

# Geometry Engine Integration
try:
    from core.geometry.hull_adapter import RetrosimHullAdapter
    HAS_GEOMETRY = True
except ImportError:
    HAS_GEOMETRY = False

# PointNet++ Agent Integration
try:
    from agents.pointnet_agent import PointNetAgent
    HAS_POINTNET = True
except ImportError:
    HAS_POINTNET = False

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# PyQt6 Imports (ARAYÜZ İÇİN GEREKLİ KISIM)
from PyQt6.QtCore import QObject, pyqtSignal

# Sklearn Imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

# SMT Imports (Kriging)
try:
    from smt.surrogate_models import KRG
    SMT_AVAILABLE = True
except ImportError:
    SMT_AVAILABLE = False


# --- Determine the best available device ---
def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        if torch.backends.mps.is_available():
            return torch.device('mps')
    except AttributeError:
        pass
    return torch.device('cpu')

DEVICE = _get_device()


# ============================================================
# Physics-Informed Loss Function
# ============================================================
class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_physics=0.1):
        super().__init__()
        self.mse = nn.SmoothL1Loss()  # Huber Loss for robust gradients
        self.lambda_physics = lambda_physics

    def forward(self, y_pred, y_true, physics_penalty):
        # Loss = MSE(Cw_pred, Cw_true) + λ * Physics_Penalty
        return self.mse(y_pred, y_true) + self.lambda_physics * torch.mean(physics_penalty)

# ============================================================
# PyTorch EANN Model (nn.Module)
# ============================================================
class EmotionalANN(nn.Module):
    """
    Emotional Artificial Neural Network with Hormonal Modulation (Ha, Hb, Hc).
    Scientific Basis: Aljahdali et al. (2025)

    Ha (Anxiety): High wave/wind stress
    Hb (Confidence): Steady state efficiency
    Hc (Stress): Engine load / total resistance stress
    """

    def __init__(self, n_features: int, n_targets: int,
                 vessel_indices: List[int], env_indices: List[int]):
        super().__init__()
        self.vessel_indices = vessel_indices
        self.env_indices = env_indices
        n_vessel = len(vessel_indices)
        n_env = len(env_indices)

        # --- HORMONAL ENGINE (Ha, Hb, Hc) ---
        self.ha_layer = nn.Sequential(nn.Linear(n_env, 1), nn.Sigmoid())       # Anxiety
        self.hb_layer = nn.Sequential(nn.Linear(n_env, 1), nn.Sigmoid())       # Confidence
        self.hc_layer = nn.Sequential(nn.Linear(n_features, 1), nn.Sigmoid())  # Stress

        # --- PRIMARY PATHWAY (Vessel) ---
        self.vessel_bn = nn.BatchNorm1d(n_vessel)
        self.vessel_fc1 = nn.Linear(n_vessel, 128)
        self.vessel_fc2 = nn.Linear(128, 64)

        # --- SECONDARY PATHWAY (Environmental) ---
        self.env_bn = nn.BatchNorm1d(n_env)
        self.env_fc1 = nn.Linear(n_env, 64)

        # --- EMOTIONAL MODULATION ---
        self.modulator = nn.Linear(3, 64)  # 3 hormones -> 64

        # --- ATTENTION FUSION ---
        self.attn_v = nn.Linear(64, 64)
        self.attn_e = nn.Linear(64, 64)
        self.attn_gate = nn.Linear(128, 1)  # concat -> scalar gate

        # --- FUSION HEAD ---
        self.fusion_fc = nn.Linear(128, 128)

        # --- GLOBAL HORMONAL GAIN ---
        self.gain_layer = nn.Sequential(nn.Linear(3, 1), nn.Softplus())

        # --- OUTPUT ---
        self.output_layer = nn.Linear(128, n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split features
        vessel_feats = x[:, self.vessel_indices]
        env_feats = x[:, self.env_indices]

        # Hormones
        ha = self.ha_layer(env_feats)                   # (B, 1)
        hb = self.hb_layer(1.0 - env_feats)             # (B, 1)
        hc = self.hc_layer(x)                            # (B, 1)
        hormones = torch.cat([ha, hb, hc], dim=1)       # (B, 3)

        # Vessel pathway
        v = self.vessel_bn(vessel_feats)
        v = torch.relu(self.vessel_fc1(v))
        v = torch.relu(self.vessel_fc2(v))               # (B, 64)

        # Environmental pathway
        e = self.env_bn(env_feats)
        e1 = torch.relu(self.env_fc1(e))                 # (B, 64)

        # Emotional modulation
        mod = torch.tanh(self.modulator(hormones))       # (B, 64)
        e_mod = e1 * mod                                 # (B, 64)

        # Attention
        att_v = torch.tanh(self.attn_v(v))               # (B, 64)
        att_e = torch.tanh(self.attn_e(e_mod))           # (B, 64)
        concat = torch.cat([att_v, att_e], dim=1)        # (B, 128)
        gate = torch.sigmoid(self.attn_gate(concat))     # (B, 1)

        v_w = v * gate
        e_w = e_mod * (1.0 - gate)

        fusion = torch.cat([v_w, e_w], dim=1)            # (B, 128)
        fusion = torch.relu(self.fusion_fc(fusion))       # (B, 128)

        # Global hormonal gain
        gain = self.gain_layer(hormones)                  # (B, 1)
        fusion = fusion * gain                            # (B, 128)

        return self.output_layer(fusion)                  # (B, n_targets)


# ============================================================
# Emotional Learning Rate Scheduler (PyTorch)
# ============================================================
class EmotionalLRScheduler:
    """
    Adjusts learning rate based on Anxiety and Confidence parameters.
    Wraps a PyTorch optimizer.
    """
    def __init__(self, optimizer: optim.Optimizer,
                 start_lr: float = 0.001,
                 anxiety: float = 0.1,
                 confidence: float = 0.1):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.anxiety = anxiety
        self.confidence = confidence
        self.best_loss = float('inf')

        # Set initial LR
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.start_lr

    def step(self, val_loss: float):
        """Call at end of each epoch with validation loss."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.confidence = min(0.5, self.confidence + 0.02)
            self.anxiety = max(0.01, self.anxiety - 0.01)
            factor = 1.0 + (self.confidence * 0.1)
        else:
            self.anxiety = min(0.9, self.anxiety + 0.05)
            self.confidence = max(0.0, self.confidence - 0.05)
            factor = 1.0 - (self.anxiety * 0.2)

        for pg in self.optimizer.param_groups:
            old_lr = pg['lr']
            pg['lr'] = max(1e-6, min(old_lr * factor, 0.01))


# ============================================================
# Surrogate Modeler Agent (QObject)
# ============================================================
class SurrogateModeler(QObject):
    """
    Physics-Informed Surrogate Model — PyTorch Backend
    """
    # SİNYALLER (GUI BUNLARI DİNLEYECEK)
    progress_signal = pyqtSignal(int, str)   # (Yüzde, Mesaj)
    finished_signal = pyqtSignal(dict)       # (Sonuçlar)
    error_signal = pyqtSignal(str)           # (Hata Mesajı)

    MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

    def __init__(self, vessel_id: str = "default_vessel"):
        super().__init__()  # QObject init
        self.vessel_id = vessel_id
        self.models: Dict = {}
        self.scalers: Dict = {}
        self.is_trained = False
        self.training_history = None

        self.FUEL_TYPE_MAPPING = {'HFO': 0, 'MDO': 1, 'LNG': 2, '0': 0, '1': 1, '2': 2}

        self.feature_names = [
            'dwt', 'age', 'length', 'breadth', 'draft', 'speed',
            'wave_height', 'wind_speed', 'current_speed', 'sea_state',
            'load_factor', 'fuel_type', 'engine_efficiency'
        ]

        self.target_names = [
            'fuel_consumption', 'co2_emission', 'resistance_penalty',
            'cii_score', 'eedi_score', 'operational_efficiency'
        ]

        self._compute_feature_indices()
        self.regime_detector = None

        # Training data stats for drift detection
        self._train_mean: Optional[np.ndarray] = None
        self._train_std: Optional[np.ndarray] = None

        # Geometry Engine (Hull Adapter)
        self.hull_adapter: Optional['RetrosimHullAdapter'] = None
        if HAS_GEOMETRY:
            try:
                self.hull_adapter = RetrosimHullAdapter()
                print("✅ Geometry Engine (RetrosimHullAdapter) entegre edildi.")
            except Exception as e:
                print(f"⚠️ Geometry Engine yüklenemedi: {e}")

        # PointNet++ Agent (fast Cw prediction from point cloud)
        self.pointnet_agent = None
        self.load_pointnet_model()

    def _compute_feature_indices(self):
        vessel_features = ['dwt', 'age', 'length', 'breadth', 'draft',
                           'speed', 'load_factor', 'engine_efficiency']
        self.vessel_indices = [self.feature_names.index(f) for f in vessel_features]

        env_features = ['wave_height', 'wind_speed', 'sea_state']
        self.env_indices = [self.feature_names.index(f) for f in env_features]

    # ----------------------------------------------------------
    # Data Generation / Loading
    # ----------------------------------------------------------
    def generate_training_data(self, num_samples: int = 2000) -> pd.DataFrame:
        self.progress_signal.emit(5, "Sentetik fizik verisi üretiliyor...")
        np.random.seed(42)
        data = {
            'dwt': np.random.uniform(3000, 8000, num_samples),
            'age': np.random.uniform(5, 25, num_samples),
            'length': np.random.uniform(80, 130, num_samples),
            'breadth': np.random.uniform(14, 20, num_samples),
            'draft': np.random.uniform(5, 8, num_samples),
            'speed': np.random.uniform(8, 16, num_samples),
            'wave_height': np.random.uniform(0.5, 4.0, num_samples),
            'wind_speed': np.random.uniform(5, 25, num_samples),
            'current_speed': np.random.uniform(-2, 2, num_samples),
            'sea_state': np.random.randint(1, 7, num_samples),
            'load_factor': np.random.uniform(0.3, 1.0, num_samples),
            'fuel_type': np.random.randint(0, 3, num_samples),
            'engine_efficiency': np.random.uniform(0.35, 0.50, num_samples)
        }
        df = pd.DataFrame(data)

        # Low-fi empirical formulas
        base_consumption = (df['dwt'] / 1000) * (df['speed'] / 10) ** 3 * 0.5
        age_factor = 1 + (df['age'] / 100)
        weather_factor = 1 + (df['wave_height'] / 10) + (df['wind_speed'] / 100)
        efficiency_factor = (0.45 / df['engine_efficiency']) ** 0.5

        df['fuel_consumption'] = (base_consumption * age_factor * weather_factor
                                  * efficiency_factor / df['load_factor'])

        emission_factors = {0: 3.1, 1: 3.2, 2: 2.75}
        df['co2_emission'] = df['fuel_consumption'] * df['fuel_type'].map(emission_factors)

        df['resistance_penalty'] = (df['wave_height'] * 5
                                    + df['wind_speed'] * 0.5
                                    + abs(df['current_speed']) * 2)

        reference_consumption = (df['dwt'] / 1000) * 10
        df['cii_score'] = (df['fuel_consumption'] / reference_consumption) * 100
        df['eedi_score'] = (df['co2_emission'] / (df['dwt'] * df['speed'])) * 1000
        df['operational_efficiency'] = ((1 / (1 + df['resistance_penalty'] / 100))
                                        * df['load_factor'])
        return df

    def load_ship_d_dataset(self, csv_path: str) -> pd.DataFrame:
        """
        Load and map Ship-D dataset CSV to model features.
        Expects columns: B/T, L/B, Cb, Fr, Cw, Rt, ...
        Missing model features are filled with sensible defaults.
        """
        self.progress_signal.emit(5, f"Ship-D verisi okunuyor: {os.path.basename(csv_path)}...")
        raw = pd.read_csv(csv_path)

        # Build mapping — adapt column names as needed
        df = pd.DataFrame()
        df['dwt'] = raw.get('DWT', np.random.uniform(3000, 8000, len(raw)))
        df['age'] = raw.get('age', np.full(len(raw), 10))
        df['length'] = raw.get('L', raw.get('LOA', np.random.uniform(80, 130, len(raw))))
        df['breadth'] = raw.get('B', np.random.uniform(14, 20, len(raw)))
        df['draft'] = raw.get('T', np.random.uniform(5, 8, len(raw)))
        df['speed'] = raw.get('Vs', raw.get('speed', np.random.uniform(8, 16, len(raw))))
        df['wave_height'] = raw.get('Hs', np.random.uniform(0.5, 4.0, len(raw)))
        df['wind_speed'] = raw.get('Vw', np.random.uniform(5, 25, len(raw)))
        df['current_speed'] = raw.get('Vc', np.random.uniform(-2, 2, len(raw)))
        df['sea_state'] = raw.get('sea_state', np.random.randint(1, 7, len(raw)))
        df['load_factor'] = raw.get('load_factor', np.random.uniform(0.3, 1.0, len(raw)))
        df['fuel_type'] = raw.get('fuel_type', np.zeros(len(raw), dtype=int))
        df['engine_efficiency'] = raw.get('eta', np.random.uniform(0.35, 0.50, len(raw)))

        # Targets from Ship-D if available
        if 'Rt' in raw.columns:
            df['resistance_penalty'] = raw['Rt']
        else:
            df['resistance_penalty'] = df['wave_height'] * 5 + df['wind_speed'] * 0.5

        base_cons = (df['dwt'] / 1000) * (df['speed'] / 10) ** 3 * 0.5
        df['fuel_consumption'] = raw.get('fuel_consumption', base_cons)
        df['co2_emission'] = df['fuel_consumption'] * 3.1
        ref = (df['dwt'] / 1000) * 10
        df['cii_score'] = (df['fuel_consumption'] / ref) * 100
        df['eedi_score'] = (df['co2_emission'] / (df['dwt'] * df['speed'] + 1e-6)) * 1000
        df['operational_efficiency'] = (1 / (1 + df['resistance_penalty'] / 100)) * df['load_factor']

        self.progress_signal.emit(10, f"Ship-D verisi yüklendi: {len(df)} kayıt")
        return df

    # ----------------------------------------------------------
    # Model Building
    # ----------------------------------------------------------
    def build_emotional_ann_torch(self) -> EmotionalANN:
        """Constructs PyTorch-based EANN with hormonal modulation."""
        model = EmotionalANN(
            n_features=len(self.feature_names),
            n_targets=len(self.target_names),
            vessel_indices=self.vessel_indices,
            env_indices=self.env_indices
        )
        return model.to(DEVICE)

    # ----------------------------------------------------------
    # Drift Detection
    # ----------------------------------------------------------
    def detect_drift(self, X: np.ndarray, threshold: float = 3.0) -> dict:
        """
        Check if input features fall outside the training distribution.
        Uses z-score based detection.
        Returns dict with 'is_drifted' bool and 'details'.
        """
        if self._train_mean is None or self._train_std is None:
            return {'is_drifted': False, 'details': 'No training stats available'}

        z_scores = np.abs((X - self._train_mean) / (self._train_std + 1e-8))
        max_z = z_scores.max(axis=1)
        drifted_mask = max_z > threshold
        drifted_features = []
        if drifted_mask.any():
            for i in range(X.shape[1]):
                col_z = z_scores[:, i]
                if (col_z > threshold).any():
                    drifted_features.append(self.feature_names[i])

        return {
            'is_drifted': bool(drifted_mask.any()),
            'drifted_ratio': float(drifted_mask.mean()),
            'drifted_features': drifted_features,
            'max_z_score': float(max_z.max())
        }

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
    def train_models(self, config: Dict = None):
        """
        GUI Thread tarafından çağrıldığında arayüzü güncelleyerek eğitim yapar.
        config: Arayüzden gelen parametreler (epoch, lr, anxiety vb.)
        """
        try:
            self.progress_signal.emit(0, "Hazırlık yapılıyor...")

            if config is None:
                config = {}
            epochs = int(config.get('epochs', 100))
            lr = float(config.get('lr', 0.002))
            anxiety = float(config.get('anxiety', 0.1))
            confidence = float(config.get('confidence', 0.1))
            data_path = config.get('data_path', None)

            # --- Data Preparation ---
            if data_path:
                self.progress_signal.emit(10, f"Veri okunuyor: {os.path.basename(data_path)}")
                if data_path.endswith('.csv'):
                    data_df = pd.read_csv(data_path)
                else:
                    data_df = pd.read_excel(data_path)
            else:
                data_df = self.generate_training_data()

            X = data_df[self.feature_names].values
            y = data_df[self.target_names].values

            # Split
            self.progress_signal.emit(15, "Veri bölünüyor ve ölçekleniyor...")
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.4, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42)

            self.scalers['features'] = StandardScaler()
            self.scalers['targets'] = MinMaxScaler()

            X_train_s = self.scalers['features'].fit_transform(X_train)
            X_val_s = self.scalers['features'].transform(X_val)
            X_test_s = self.scalers['features'].transform(X_test)

            y_train_s = self.scalers['targets'].fit_transform(y_train)
            y_val_s = self.scalers['targets'].transform(y_val)
            y_test_s = self.scalers['targets'].transform(y_test)

            # Store training stats for drift detection
            self._train_mean = X_train.mean(axis=0)
            self._train_std = X_train.std(axis=0)

            # --- 1. EANN TRAINING (PyTorch) ---
            self.progress_signal.emit(20, "EANN Mimarisi oluşturuluyor (PyTorch)...")
            model = self.build_emotional_ann_torch()
            self.models['eann'] = model

            # Phase 3: Physics-Informed Custom Loss Integration
            criterion = PhysicsInformedLoss(lambda_physics=0.15) 
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
            scheduler = EmotionalLRScheduler(optimizer, start_lr=lr,
                                             anxiety=anxiety, confidence=confidence)

            # DataLoaders
            train_ds = TensorDataset(
                torch.tensor(X_train_s, dtype=torch.float32),
                torch.tensor(y_train_s, dtype=torch.float32))
            val_ds = TensorDataset(
                torch.tensor(X_val_s, dtype=torch.float32),
                torch.tensor(y_val_s, dtype=torch.float32))

            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

            # Early stopping state
            best_val_loss = float('inf')
            patience = 15
            patience_counter = 0
            best_state_dict = None

            for epoch in range(epochs):
                # --- Train ---
                model.train()
                epoch_loss = 0.0
                for xb, yb in train_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    optimizer.zero_grad()
                    pred = model(xb)
                    
                    # Compute synthetic physics penalty proxy (e.g. displacement violation)
                    # For realistic implementation, this evaluates Navier-Stokes or Continuity constraints
                    physics_penalty = torch.relu(pred[:, 0] - xb[:, 0] * 1.5) ** 2 
                    
                    loss = criterion(pred, yb, physics_penalty)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * xb.size(0)
                epoch_loss /= len(train_ds)

                # --- Validate ---
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        pred = model(xb)
                        physics_p = torch.zeros_like(pred[:, 0]) # Placeholder for val
                        val_loss += criterion(pred, yb, physics_p).item() * xb.size(0)
                val_loss /= len(val_ds)

                # Emotional LR adjustment
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.progress_signal.emit(
                            70, f"Early stopping @ epoch {epoch+1} | val_loss: {best_val_loss:.4f}")
                        break

                # Progress signal
                pct = 20 + int((epoch / epochs) * 50)
                self.progress_signal.emit(
                    pct, f"EANN Eğitiliyor... Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

            # Restore best weights
            if best_state_dict is not None:
                model.load_state_dict(best_state_dict)

            # Save EANN model
            self._save_eann_model(model)

            # --- 2. KRIGING (SMT) ---
            if SMT_AVAILABLE and config.get('model') == 'Kriging (Gaussian Process)':
                self.progress_signal.emit(70, "Kriging Matrisleri Hesaplanıyor (SMT)...")
                self.models['kriging'] = []
                total_targets = len(self.target_names)

                for i, target_name in enumerate(self.target_names):
                    p = 70 + int((i / total_targets) * 20)
                    self.progress_signal.emit(p, f"Kriging Eğitiliyor: {target_name}")

                    krg = KRG(theta0=[1e-2] * X.shape[1], print_prediction=False)
                    krg.set_training_values(X_train_s, y_train_s[:, i])
                    krg.train()
                    self.models['kriging'].append(krg)

            # --- 3. BASELINE (RF) ---
            self.progress_signal.emit(90, "Random Forest Baseline eğitiliyor...")
            self.models['rf'] = MultiOutputRegressor(
                RandomForestRegressor(n_estimators=50, n_jobs=-1))
            self.models['rf'].fit(X_train_s, y_train_s)

            self.is_trained = True

            # --- RESULTS ---
            self.progress_signal.emit(95, "Test seti üzerinde skor hesaplanıyor...")
            scores = {}

            # EANN test loss
            model.eval()
            X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE)
            y_test_t = torch.tensor(y_test_s, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                test_pred = model(X_test_t)
                physics_p = torch.zeros_like(test_pred[:, 0])
                scores['eann_loss'] = float(criterion(test_pred, y_test_t, physics_p).item())

            scores['rf_r2'] = self.models['rf'].score(X_test_s, y_test_s)

            self.progress_signal.emit(100, "Eğitim Başarıyla Tamamlandı!")
            self.finished_signal.emit(scores)

        except Exception as e:
            self.error_signal.emit(str(e))
            import traceback
            traceback.print_exc()

    # ----------------------------------------------------------
    # Model Save / Load
    # ----------------------------------------------------------
    def _save_eann_model(self, model: nn.Module):
        """Save EANN model weights as .pt file."""
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        path = os.path.join(self.MODEL_DIR, f"{self.vessel_id}_eann_model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'vessel_indices': self.vessel_indices,
            'env_indices': self.env_indices,
            'n_features': len(self.feature_names),
            'n_targets': len(self.target_names),
        }, path)
        print(f"💾 EANN model kaydedildi: {path}")

    def _load_eann_model(self) -> Optional[EmotionalANN]:
        """Load EANN model weights from .pt file."""
        path = os.path.join(self.MODEL_DIR, f"{self.vessel_id}_eann_model.pt")
        if not os.path.exists(path):
            return None
        try:
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            model = EmotionalANN(
                n_features=ckpt['n_features'],
                n_targets=ckpt['n_targets'],
                vessel_indices=ckpt['vessel_indices'],
                env_indices=ckpt['env_indices']
            ).to(DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            print(f"✅ EANN model yüklendi: {path}")
            return model
        except Exception as e:
            print(f"⚠️ EANN model yüklenemedi: {e}")
            return None

    # ----------------------------------------------------------
    # Predict
    # ----------------------------------------------------------
    def predict(self, vessel_data, model_type: str = 'eann'):
        if not self.is_trained:
            return {t: 0.0 for t in self.target_names}

        # Prepare input
        if isinstance(vessel_data, dict):
            v_data = vessel_data.copy()
            if 'length' not in v_data: v_data['length'] = v_data.get('loa', 0)
            if 'breadth' not in v_data: v_data['breadth'] = v_data.get('beam', 0)
            if isinstance(v_data.get('fuel_type'), str):
                v_data['fuel_type'] = self.FUEL_TYPE_MAPPING.get(v_data['fuel_type'], 0)
            X = np.array([[v_data.get(f, 0) for f in self.feature_names]])
        else:
            X = np.array(vessel_data).reshape(1, -1)

        # Drift detection warning
        drift = self.detect_drift(X)
        if drift['is_drifted']:
            print(f"⚠️ DRIFT DETECTED: z={drift['max_z_score']:.1f}, "
                  f"features={drift['drifted_features']}")

        X_scaled = self.scalers['features'].transform(X)

        # Model selection
        if model_type == 'eann' or 'EANN' in model_type:
            model = self.models['eann']
            model.eval()
            X_t = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                y_pred_s = model(X_t).cpu().numpy()
        elif 'Kriging' in model_type and SMT_AVAILABLE and 'kriging' in self.models:
            preds = []
            for krg in self.models['kriging']:
                preds.append(krg.predict_values(X_scaled)[0, 0])
            y_pred_s = np.array([preds])
        else:
            # Fallback to RF
            y_pred_s = self.models['rf'].predict(X_scaled)

        y_pred = self.scalers['targets'].inverse_transform(y_pred_s)
        return {target: float(y_pred[0][i]) for i, target in enumerate(self.target_names)}

    # ----------------------------------------------------------
    # PointNet++ Model Loading
    # ----------------------------------------------------------
    def load_pointnet_model(self):
        """Auto-detect and load PointNet++ model (.pth/.onnx) from models/."""
        if not HAS_POINTNET:
            return
        try:
            self.pointnet_agent = PointNetAgent(num_points=2048)
            if self.pointnet_agent.is_trained:
                print("✅ PointNet++ Cw prediction model entegre edildi.")
            else:
                print("ℹ️ PointNet++ model yüklü ama eğitilmemiş.")
        except Exception as e:
            print(f"⚠️ PointNet++ yüklenemedi: {e}")
            self.pointnet_agent = None

    # ----------------------------------------------------------
    # Hydrodynamics Prediction (Geometry-Integrated)
    # ----------------------------------------------------------
    def predict_hydrodynamics(self, hull_params: Dict, speed: float) -> Dict:
        """
        Multi-fidelity total resistance prediction for MCDM input.

        Priority cascade:
          1. PointNet++ Cw (if trained, ~0.01s)
          2. EANN inference (if trained, ~0.05s)
          3. Holtrop-Mennen fallback

        Combines hull geometry features (via RetrosimHullAdapter) with
        the highest-fidelity available model.

        Args:
            hull_params: UI vessel data dictionary.
            speed: Ship speed in knots.

        Returns:
            Dictionary with Rt prediction and volumetric features.
        """
        result = {'speed': speed, 'source': 'fallback'}

        # 1. Geometry features from Hull Adapter
        if self.hull_adapter and HAS_GEOMETRY:
            try:
                self.hull_adapter.set_from_ui(hull_params)
                ml_features = self.hull_adapter.extract_ml_features()
                
                # SHIP-D Phase 3: Fast Cw prediction via PointNet++ extraction
                point_cloud = self.hull_adapter.extract_point_cloud(num_points=2048)
                result['point_cloud_nodes'] = point_cloud.shape[0]
                
                result['geometry_features'] = ml_features
                result['displaced_volume'] = ml_features.get('displaced_volume', 0)
                result['wetted_surface_area'] = ml_features.get('wetted_surface_area', 0)
                result['Cb_actual'] = ml_features.get('Cb_actual', 0)

                # Full Holtrop-Mennen resistance from geometry
                resistance = self.hull_adapter.predict_total_resistance(speed)
                result['Rt_holtrop'] = resistance.get('Rt', 0)
                result['Rw_holtrop'] = resistance.get('Rw', 0)
                result['Cw_holtrop'] = resistance.get('Cw', 0)
                result['Cf_holtrop'] = resistance.get('Cf', 0)
                result['Froude_number'] = resistance.get('Froude_number', 0)
                result['Pe_kW'] = resistance.get('Pe_kW', 0)
                result['form_factor_k1'] = resistance.get('form_factor_k1', 0)
                result['iE'] = resistance.get('iE', 0)
                result['source'] = 'geometry+holtrop'

                # --- Design Vector Validation (Ship-D bounds) ---
                validation = self.hull_adapter.validate_design_vector()
                result['dv_valid'] = validation.get('valid', True)
                result['dv_warnings'] = len(validation.get('warnings', []))

            except Exception as e:
                print(f"⚠️ Geometry feature extraction hatası: {e}")

        # 2. PointNet++ Cw prediction (highest fidelity, ~0.01s)
        if self.pointnet_agent and self.pointnet_agent.is_trained:
            try:
                if 'point_cloud_nodes' in result:
                    # Use already extracted point cloud
                    point_cloud = self.hull_adapter.extract_point_cloud(num_points=2048)
                    pn_pred = self.pointnet_agent.predict_from_point_cloud(point_cloud)
                    result['Cw_pointnet'] = pn_pred.get('Cw', 0)
                    result['Cf_pointnet'] = pn_pred.get('Cf', 0)
                    result['Ct_pointnet'] = pn_pred.get('Ct', 0)
                    result['source'] = 'geometry+pointnet'
                    print(f"🧠 PointNet++ Cw={pn_pred['Cw']:.6f}")
            except Exception as e:
                print(f"⚠️ PointNet++ inference hatası: {e}")

        # 3. EANN inference (if trained)
        if self.is_trained:
            try:
                eann_pred = self.predict(hull_params, model_type='eann')
                result['eann_predictions'] = eann_pred
                result['fuel_consumption'] = eann_pred.get('fuel_consumption', 0)
                result['resistance_penalty'] = eann_pred.get('resistance_penalty', 0)
                if 'pointnet' not in result.get('source', ''):
                    result['source'] = 'geometry+eann' if 'geometry' in result.get('source', '') else 'eann'
            except Exception as e:
                print(f"⚠️ EANN inference hatası: {e}")

        return result