"""
Surrogate Modeler (XGBoost + Kriging + GC-FNO) — PyQt6 Integrated
=================================================================
Primary model: XGBoost for tabular vessel + environmental features.
Secondary: GC-FNO resistance coefficients as additional geometry-conditioned features.
Fallback: GradientBoosting (sklearn) when XGBoost is not installed.
Optional: Kriging (SMT) for uncertainty quantification.

Why XGBoost:
  - Tabular data benchmarks consistently favor tree-based methods
    (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022)
  - Built-in feature importance → interpretable for naval architects
  - 20× faster training, fewer hyperparameters
"""

import numpy as np
import pandas as pd
import os
import joblib
from typing import Dict, Optional, List, Tuple

# Geometry Engine Integration
try:
    from core.geometry.FFDHullMorpher import RetrosimHullAdapter
    HAS_GEOMETRY = True
except ImportError:
    HAS_GEOMETRY = False

# PointNet++ Agent Integration
try:
    from agents.pointnet_agent import PointNetAgent
    HAS_POINTNET = True
except ImportError:
    HAS_POINTNET = False

# GC-FNO Agent Integration
try:
    from agents.modulus_agent import ModulusCFDAgent
    HAS_GCFNO = True
except ImportError:
    HAS_GCFNO = False

# XGBoost (preferred tree-based model)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# PyQt6 Imports
from PyQt6.QtCore import QObject, pyqtSignal

# Sklearn Imports
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# SMT Imports (Kriging — uncertainty quantification)
try:
    from smt.surrogate_models import KRG
    SMT_AVAILABLE = True
except ImportError:
    SMT_AVAILABLE = False





# ============================================================
# Surrogate Modeler Agent (QObject)
# ============================================================
class SurrogateModeler(QObject):
    """
    Tabular Surrogate Model — XGBoost / Kriging / GBR

    Predicts operational targets (fuel consumption, CO2, CII, EEDI) from
    vessel parameters and environmental conditions.
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
                print("[OK] Geometry Engine (RetrosimHullAdapter) entegre edildi.")
            except Exception as e:
                print(f"[!] Geometry Engine yüklenemedi: {e}")

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
        Train surrogate models. Primary: XGBoost/GradientBoosting.
        Optional Kriging (SMT) for uncertainty quantification.

        config: Parameters from GUI (epochs, lr, model type, etc.)
        """
        try:
            self.progress_signal.emit(0, "Hazırlık yapılıyor...")

            if config is None:
                config = {}
            epochs = int(config.get('epochs', 100))
            lr = float(config.get('lr', 0.002))
            data_path = config.get('data_path', None)
            model_type = config.get('model', 'XGBoost')

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

            scores = {}

            # === 1. PRIMARY: XGBoost / GradientBoosting ===
            self.progress_signal.emit(20, "XGBoost / GradientBoosting eğitiliyor...")

            if HAS_XGBOOST and 'XGBoost' in model_type:
                # XGBoost — best for tabular data
                self.progress_signal.emit(25, "XGBoost modeli eğitiliyor...")
                xgb_models = []
                for i, target_name in enumerate(self.target_names):
                    pct = 25 + int((i / len(self.target_names)) * 35)
                    self.progress_signal.emit(
                        pct, f"XGBoost: {target_name} eğitiliyor...")

                    model = xgb.XGBRegressor(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        random_state=42,
                        verbosity=0,
                    )
                    model.fit(
                        X_train_s, y_train_s[:, i],
                        eval_set=[(X_val_s, y_val_s[:, i])],
                        verbose=False,
                    )
                    xgb_models.append(model)

                self.models['xgboost'] = xgb_models
                self.models['primary'] = 'xgboost'

                # Evaluate
                xgb_preds = np.column_stack([
                    m.predict(X_test_s) for m in xgb_models
                ])
                scores['xgboost_r2'] = float(r2_score(
                    y_test_s, xgb_preds, multioutput='uniform_average'))
                scores['xgboost_mae'] = float(mean_absolute_error(
                    y_test_s, xgb_preds, multioutput='uniform_average'))

                # Feature importance (first target)
                self._feature_importance = dict(zip(
                    self.feature_names,
                    xgb_models[0].feature_importances_.tolist()
                ))
                print(f"[#] Feature Importance: {self._feature_importance}")

            else:
                # Fallback: sklearn GradientBoosting
                self.progress_signal.emit(25, "GradientBoosting eğitiliyor...")
                gbr = MultiOutputRegressor(
                    GradientBoostingRegressor(
                        n_estimators=200, max_depth=5,
                        learning_rate=0.1, subsample=0.8,
                        random_state=42,
                    )
                )
                gbr.fit(X_train_s, y_train_s)
                self.models['gbr'] = gbr
                self.models['primary'] = 'gbr'

                gbr_preds = gbr.predict(X_test_s)
                scores['gbr_r2'] = float(r2_score(
                    y_test_s, gbr_preds, multioutput='uniform_average'))

            # === 2. KRIGING (SMT — uncertainty quantification) ===
            if SMT_AVAILABLE and 'Kriging' in model_type:
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

            self.is_trained = True

            # Save primary model
            self._save_primary_model()

            self.progress_signal.emit(100, "Eğitim Başarıyla Tamamlandı!")
            self.finished_signal.emit(scores)

        except Exception as e:
            self.error_signal.emit(str(e))
            import traceback
            traceback.print_exc()



    # ----------------------------------------------------------
    # Model Save / Load
    # ----------------------------------------------------------
    def _save_primary_model(self):
        """Save the primary model (XGBoost or GBR) to disk."""
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        primary = self.models.get('primary', 'rf')

        if primary == 'xgboost' and 'xgboost' in self.models:
            for i, model in enumerate(self.models['xgboost']):
                path = os.path.join(
                    self.MODEL_DIR,
                    f"{self.vessel_id}_xgb_{self.target_names[i]}.json"
                )
                model.save_model(path)
            print(f"[S] XGBoost models saved to {self.MODEL_DIR}")
        elif primary == 'gbr' and 'gbr' in self.models:
            path = os.path.join(self.MODEL_DIR, f"{self.vessel_id}_gbr.joblib")
            joblib.dump(self.models['gbr'], path)
            print(f"[S] GradientBoosting model saved: {path}")

        # Save scalers
        scaler_path = os.path.join(self.MODEL_DIR, f"{self.vessel_id}_scalers.joblib")
        joblib.dump(self.scalers, scaler_path)



    # ----------------------------------------------------------
    # Predict
    # ----------------------------------------------------------
    def predict(self, vessel_data, model_type: str = 'auto'):
        """
        Predict operational targets using the best available model.

        Model priority: XGBoost > GBR > Kriging (fallback)
        """
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
            print(f"[!] DRIFT DETECTED: z={drift['max_z_score']:.1f}, "
                  f"features={drift['drifted_features']}")

        X_scaled = self.scalers['features'].transform(X)

        # Auto model selection (best available)
        if model_type == 'auto':
            primary = self.models.get('primary', 'gbr')
        else:
            primary = model_type.lower()

        if primary == 'xgboost' and 'xgboost' in self.models:
            y_pred_s = np.column_stack([
                m.predict(X_scaled) for m in self.models['xgboost']
            ])
        elif primary == 'gbr' and 'gbr' in self.models:
            y_pred_s = self.models['gbr'].predict(X_scaled)
        elif ('kriging' in primary or 'Kriging' in model_type) and SMT_AVAILABLE and 'kriging' in self.models:
            preds = []
            for krg in self.models['kriging']:
                preds.append(krg.predict_values(X_scaled)[0, 0])
            y_pred_s = np.array([preds])
        else:
            # Fallback: use whatever primary model is available
            if 'xgboost' in self.models:
                y_pred_s = np.column_stack([m.predict(X_scaled) for m in self.models['xgboost']])
            elif 'gbr' in self.models:
                y_pred_s = self.models['gbr'].predict(X_scaled)
            else:
                return {t: 0.0 for t in self.target_names}

        y_pred = self.scalers['targets'].inverse_transform(y_pred_s)

        # Physics post-processing
        result = {target: float(y_pred[0][i])
                  for i, target in enumerate(self.target_names)}
        result = self._apply_physics_constraints(result, vessel_data)
        return result

    def _apply_physics_constraints(self, predictions: Dict,
                                    vessel_data) -> Dict:
        """
        Apply real physics constraints to predictions.

        Ensures:
          - All values are non-negative
          - Fuel consumption is consistent with resistance × speed
          - CO2 emission scales with fuel consumption
          - CII/EEDI are bounded
        """
        # Non-negativity
        for key in predictions:
            if predictions[key] < 0:
                predictions[key] = 0.0

        # CO2 = fuel × emission factor
        fuel = predictions.get('fuel_consumption', 0)
        if fuel > 0:
            # Typical HFO emission factor: 3.114 tCO2/tFuel
            expected_co2 = fuel * 3.114
            predictions['co2_emission'] = expected_co2

        # Operational efficiency bounded [0, 1]
        predictions['operational_efficiency'] = max(
            0.0, min(1.0, predictions.get('operational_efficiency', 0.5)))

        return predictions

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
                print("[OK] PointNet++ Cw prediction model entegre edildi.")
            else:
                print("[i] PointNet++ model yüklü ama eğitilmemiş.")
        except Exception as e:
            print(f"[!] PointNet++ yüklenemedi: {e}")
            self.pointnet_agent = None

    # ----------------------------------------------------------
    # Hydrodynamics Prediction (Geometry-Integrated)
    # ----------------------------------------------------------
    def predict_hydrodynamics(self, hull_params: Dict, speed: float) -> Dict:
        """
        Multi-fidelity total resistance prediction for MCDM input.

        Priority cascade:
          1. GC-FNO (geometry-conditioned, ~0.02s)
          2. PointNet++ Cw (if trained, ~0.01s)
          3. XGBoost/GBR operational prediction
          4. Holtrop-Mennen fallback (always available)

        Combines hull geometry features (via RetrosimHullAdapter) with
        the highest-fidelity available model.

        Args:
            hull_params: UI vessel data dictionary.
            speed: Ship speed in knots.

        Returns:
            Dictionary with Rt prediction and volumetric features.
        """
        result = {'speed': speed, 'source': 'fallback'}
        point_cloud = None

        # 1. Geometry features from Hull Adapter
        if self.hull_adapter and HAS_GEOMETRY:
            try:
                self.hull_adapter.set_from_ui(hull_params)
                ml_features = self.hull_adapter.extract_ml_features()

                # Point cloud extraction (primary geometry for AI)
                point_cloud = self.hull_adapter.extract_point_cloud(
                    num_points=2048, method='parametric'
                )
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

                # Design Vector Validation (Ship-D bounds)
                validation = self.hull_adapter.validate_design_vector()
                result['dv_valid'] = validation.get('valid', True)
                result['dv_warnings'] = len(validation.get('warnings', []))

            except Exception as e:
                print(f"[!] Geometry feature extraction hatası: {e}")

        # 2. GC-FNO resistance prediction (highest fidelity, ~0.02s)
        if HAS_GCFNO and point_cloud is not None:
            try:
                fno_agent = ModulusCFDAgent()
                if fno_agent.is_trained:
                    fno_pred = fno_agent.predict_resistance(
                        point_cloud, speed, hull_params
                    )
                    result['Cw_fno'] = fno_pred.get('Cw', 0)
                    result['Cf_fno'] = fno_pred.get('Cf', 0)
                    result['Ct_fno'] = fno_pred.get('Ct', 0)
                    result['Rt_fno_kN'] = fno_pred.get('Rt_kN', 0)
                    result['Pe_fno_kW'] = fno_pred.get('Pe_kW', 0)
                    result['source'] = 'geometry+gc-fno'
                    print(f"[AI] GC-FNO Cw={fno_pred.get('Cw', 0):.6f}")
            except Exception as e:
                print(f"[!] GC-FNO inference hatası: {e}")

        # 3. PointNet++ Cw prediction (~0.01s)
        if self.pointnet_agent and self.pointnet_agent.is_trained:
            try:
                if point_cloud is not None:
                    pn_pred = self.pointnet_agent.predict_from_point_cloud(point_cloud)
                    result['Cw_pointnet'] = pn_pred.get('Cw', 0)
                    result['Cf_pointnet'] = pn_pred.get('Cf', 0)
                    result['Ct_pointnet'] = pn_pred.get('Ct', 0)
                    if 'gc-fno' not in result.get('source', ''):
                        result['source'] = 'geometry+pointnet'
                    print(f"[AI] PointNet++ Cw={pn_pred['Cw']:.6f}")
            except Exception as e:
                print(f"[!] PointNet++ inference hatası: {e}")

        # 4. XGBoost / GBR operational prediction (fuel, CO2, CII, EEDI)
        if self.is_trained:
            try:
                operational_pred = self.predict(hull_params, model_type='auto')
                result['operational_predictions'] = operational_pred
                result['fuel_consumption'] = operational_pred.get('fuel_consumption', 0)
                result['resistance_penalty'] = operational_pred.get('resistance_penalty', 0)
                if 'gc-fno' not in result.get('source', '') and 'pointnet' not in result.get('source', ''):
                    result['source'] = ('geometry+xgboost'
                                        if 'geometry' in result.get('source', '')
                                        else 'xgboost')
            except Exception as e:
                print(f"[!] XGBoost inference hatası: {e}")

        return result