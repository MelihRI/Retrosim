"""
Surrogate Modeler (EANN Core) - Physics-Informed Surrogate Model
=================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt


class SurrogateModeler:
    """
    Physics-Informed Surrogate Model for Ship Performance Prediction
    
    ÖĞRENİM: Docstring best practices
    - Açıklama
    - Attributes (class variables)
    - Methods (public vs private)
    """
    
    def __init__(self, vessel_id: str = "default_vessel"):
        """
        ÖĞRENİM: Type hints kullanımı
        - vessel_id: str = default value ile
        - IDE'ler autocomplete yapabilir
        - Kod daha okunabilir
        """
        self.vessel_id = vessel_id
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.training_history = None  # History'yi saklayalım
        
        # ÖĞRENİM: Configuration as constants
        self.FUEL_TYPE_MAPPING = {
            'HFO': 0, 'MDO': 1, 'LNG': 2,
            '0': 0, '1': 1, '2': 2
        }
        
        # Feature ve target tanımları
        self.feature_names = [
            'dwt', 'age', 'length', 'breadth', 'draft', 'speed',
            'wave_height', 'wind_speed', 'current_speed', 'sea_state',
            'load_factor', 'fuel_type', 'engine_efficiency'
        ]
        
        self.target_names = [
            'fuel_consumption', 'co2_emission', 'resistance_penalty',
            'cii_score', 'eedi_score', 'operational_efficiency'
        ]
        
        # ÖĞRENİM: Computed properties - feature groups için indeksler
        self._compute_feature_indices()
        
        # Regime detector'ı ayrı tutacağız (visualization için)
        self.regime_detector = None
    
    def _compute_feature_indices(self):
        """
        ÖĞRENİM: Private method (underscore ile başlar)
        - Internal kullanım için
        - API değil, implementation detail
        """
        # Vessel characteristics indices
        vessel_features = ['dwt', 'age', 'length', 'breadth', 'draft', 
                          'speed', 'load_factor', 'engine_efficiency']
        self.vessel_indices = [
            self.feature_names.index(f) for f in vessel_features
        ]
        
        # Environmental features indices
        env_features = ['wave_height', 'wind_speed', 'sea_state']
        self.env_indices = [
            self.feature_names.index(f) for f in env_features
        ]
        
        print(f"✓ Vessel indices: {self.vessel_indices}")
        print(f"✓ Environmental indices: {self.env_indices}")
    
    def generate_training_data(self, num_samples: int = 5000) -> pd.DataFrame:
        """
        ÖĞRENİM: Type hints in/out
        num_samples: int (input)
        -> pd.DataFrame (output)
        """
        np.random.seed(42)
        
        # ÖĞRENİM: Vectorized operations (numpy)
        # Loop yerine array operations → çok daha hızlı
        
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
        
        # ÖĞRENİM: Physics-based calculations
        # Basit formüllerle gerçekçi veri üret
        base_consumption = (df['dwt'] / 1000) * (df['speed'] / 10) ** 3 * 0.5
        age_factor = 1 + (df['age'] / 100)
        weather_factor = 1 + (df['wave_height'] / 10) + (df['wind_speed'] / 100)
        efficiency_factor = (0.45 / df['engine_efficiency']) ** 0.5
        
        df['fuel_consumption'] = (base_consumption * age_factor * 
                                 weather_factor * efficiency_factor / 
                                 df['load_factor'])
        
        # CO2 emissions
        emission_factors = {0: 3.1, 1: 3.2, 2: 2.75}
        df['co2_emission'] = df['fuel_consumption'] * df['fuel_type'].map(emission_factors)
        
        # Resistance penalty
        df['resistance_penalty'] = (df['wave_height'] * 5 + 
                                   df['wind_speed'] * 0.5 + 
                                   abs(df['current_speed']) * 2)
        
        # CII Score
        reference_consumption = (df['dwt'] / 1000) * 10
        df['cii_score'] = (df['fuel_consumption'] / reference_consumption) * 100
        
        # EEDI Score
        df['eedi_score'] = (df['co2_emission'] / (df['dwt'] * df['speed'])) * 1000
        
        # Operational efficiency
        df['operational_efficiency'] = ((1 / (1 + df['resistance_penalty'] / 100)) * 
                                       df['load_factor'])
        
        return df
    
    def build_emotional_ann(self) -> keras.Model:
        """
        Build EANN architecture with proper index handling
        
        ÖĞRENİM: Method returns keras.Model
        Type hint yardımcı olur
        """
        # ============ INPUT LAYER ============
        inputs = keras.Input(
            shape=(len(self.feature_names),), 
            name='vessel_inputs'
        )
        
        print(f"✓ Building EANN with {len(self.feature_names)} input features")
        
        # ============ DUAL PATHWAY ============
        # ÖĞRENİM: Lambda layer ile feature extraction
        vessel_features = layers.Lambda(
            lambda x: tf.gather(x, self.vessel_indices, axis=1),
            name='vessel_pathway'
        )(inputs)
        
        env_features = layers.Lambda(
            lambda x: tf.gather(x, self.env_indices, axis=1),
            name='environmental_pathway'
        )(inputs)
        
        print(f"  ✓ Vessel pathway: {len(self.vessel_indices)} features")
        print(f"  ✓ Environmental pathway: {len(self.env_indices)} features")
        
        # Pathway 1: Vessel Characteristics
        v = layers.BatchNormalization(name='vessel_bn_input')(vessel_features)
        v = layers.Dense(128, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(0.01),
                        name='vessel_dense_1')(v)
        v = layers.BatchNormalization(name='vessel_bn_1')(v)
        v = layers.Dropout(0.25, name='vessel_dropout_1')(v)
        
        v = layers.Dense(96, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.01),
                        name='vessel_dense_2')(v)
        v = layers.BatchNormalization(name='vessel_bn_2')(v)
        v = layers.Dropout(0.2, name='vessel_dropout_2')(v)
        
        # Pathway 2: Environmental Features (Hormonal System)
        e = layers.BatchNormalization(name='env_bn_input')(env_features)
        
        # ÖĞRENİM: Multi-scale processing
        # Aynı input → 3 farklı representation
        e1 = layers.Dense(32, activation='relu', name='env_calm')(e)
        e2 = layers.Dense(32, activation='relu', name='env_moderate')(e)
        e3 = layers.Dense(32, activation='relu', name='env_rough')(e)
        
        # ÖĞRENİM: Hormonal gating - EANN'nin핵심
        gate_logits = layers.Dense(3, name='regime_logits')(e)
        regime_probs = layers.Softmax(name='hormonal_gate')(gate_logits)
        
        # ÖĞRENİM: Weighted combination
        # Branch çıktısı * probability → soft selection
        
        # Calm contribution
        calm_weight = layers.Lambda(
            lambda x: tf.expand_dims(x[:, 0], -1),
            name='calm_weight'
        )(regime_probs)
        calm_contrib = layers.Multiply(name='calm_modulation')([e1, calm_weight])
        
        # Moderate contribution
        moderate_weight = layers.Lambda(
            lambda x: tf.expand_dims(x[:, 1], -1),
            name='moderate_weight'
        )(regime_probs)
        moderate_contrib = layers.Multiply(name='moderate_modulation')([e2, moderate_weight])
        
        # Rough contribution
        rough_weight = layers.Lambda(
            lambda x: tf.expand_dims(x[:, 2], -1),
            name='rough_weight'
        )(regime_probs)
        rough_contrib = layers.Multiply(name='rough_modulation')([e3, rough_weight])
        
        # Combine all regimes
        e_weighted = layers.Add(name='regime_combination')([
            calm_contrib, moderate_contrib, rough_contrib
        ])
        
        # Environmental refinement
        e_final = layers.Dense(64, activation='relu', name='env_final')(e_weighted)
        e_final = layers.BatchNormalization(name='env_bn_final')(e_final)
        
        # ÖĞRENİM: Regime detector'ı ayrı model olarak sakla
        # Visualization için kullanacağız
        self.regime_detector = keras.Model(
            inputs=inputs,
            outputs=regime_probs,
            name='regime_detector'
        )
        
        # ============ FUSION LAYER ============
        # ÖĞRENİM: Attention mechanism
        # Her pathway'in önemini öğren
        
        attention_v = layers.Dense(96, activation='tanh', name='attention_vessel')(v)
        attention_e = layers.Dense(96, activation='tanh', name='attention_env')(e_final)
        
        # Attention weight: sigmoid → [0, 1]
        attention_concat = layers.concatenate([attention_v, attention_e], name='attention_concat')
        attention_weights = layers.Dense(1, activation='sigmoid', name='attention_weights')(attention_concat)
        
        # Weighted fusion
        v_attended = layers.Multiply(name='vessel_attended')([v, attention_weights])
        
        # Complement weight (1 - attention_weight)
        complement_weight = layers.Lambda(
            lambda x: 1 - x, 
            name='complement_weight'
        )(attention_weights)
        e_attended = layers.Multiply(name='env_attended')([e_final, complement_weight])
        
        # Final fusion
        combined = layers.concatenate([v_attended, e_attended], name='fusion')
        
        # ============ DECISION LAYERS ============
        combined = layers.Dense(128, activation='relu',
                               kernel_regularizer=keras.regularizers.l2(0.01),
                               name='decision_1')(combined)
        combined = layers.BatchNormalization(name='decision_bn_1')(combined)
        combined = layers.Dropout(0.2, name='decision_dropout_1')(combined)
        
        combined = layers.Dense(64, activation='relu', name='decision_2')(combined)
        combined = layers.Dropout(0.15, name='decision_dropout_2')(combined)
        
        # ============ OUTPUT LAYER ============
        outputs = layers.Dense(
            len(self.target_names),
            activation='linear',
            name='performance_outputs'
        )(combined)
        
        # ============ MODEL CREATION ============
        model = keras.Model(inputs=inputs, outputs=outputs, name='EANN_Vessel_Surrogate')
        
        # ÖĞRENİM: Learning rate schedule
        lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=0.001,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.0001
        )
        
        # ÖĞRENİM: AdamW optimizer
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.004,
            clipnorm=1.0
        )
        
        # ÖĞRENİM: Huber loss - outlier'lara robust
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.Huber(delta=0.1),
            metrics=['mae', 'mse', keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        
        print("✓ EANN model built successfully")
        print(f"  Total parameters: {model.count_params():,}")
        
        return model
    
    def train_models(self, data_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Train surrogate models with improved error handling
        
        ÖĞRENİM: Optional type hint
        data_df: Optional[pd.DataFrame] = None
        → Accepts DataFrame or None
        """
        if data_df is None:
            print("Generating training data...")
            data_df = self.generate_training_data()
        
        # Prepare data
        X = data_df[self.feature_names].values
        y = data_df[self.target_names].values
        
        # ÖĞRENİM: Train/validation/test split
        # 60/20/20 split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        print(f"✓ Data split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        self.scalers['targets'] = MinMaxScaler()
        
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        X_val_scaled = self.scalers['features'].transform(X_val)
        X_test_scaled = self.scalers['features'].transform(X_test)
        
        y_train_scaled = self.scalers['targets'].fit_transform(y_train)
        y_val_scaled = self.scalers['targets'].transform(y_val)
        y_test_scaled = self.scalers['targets'].transform(y_test)
        
        # ÖĞRENİM: Callbacks - training'i kontrol et
        os.makedirs("models", exist_ok=True)
        
        # CRITICAL FIX: ModelCheckpoint with .h5 format causes recursion with Lambda layers
        # ReduceLROnPlateau removed because it conflicts with CosineDecayRestarts schedule
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True,
                verbose=1
            )
            # NOTE: ReduceLROnPlateau removed - conflicts with LearningRateSchedule
            # Learning rate is already controlled by CosineDecayRestarts in optimizer
            
            # NOTE: ModelCheckpoint disabled due to recursion error with Lambda layers
            # Model will be saved manually after training
        ]
        
        # Train EANN
        print("\n" + "="*60)
        print("Training Emotional ANN...")
        print("="*60)
        
        self.models['eann'] = self.build_emotional_ann()
        
        history = self.models['eann'].fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=200,  # ÖĞRENİM: 3000 çok fazla, early stopping kullan
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_history = history
        
        # Train ensemble models for comparison
        print("\nTraining Random Forest...")
        self.models['rf'] = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        )
        self.models['rf'].fit(X_train_scaled, y_train_scaled)
        
        print("Training Gradient Boosting...")
        self.models['gb'] = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, random_state=42)
        )
        self.models['gb'].fit(X_train_scaled, y_train_scaled)
        
        # ÖĞRENİM: Evaluate on test set (unseen data)
        print("\n" + "="*60)
        print("Evaluation Results on Test Set:")
        print("="*60)
        
        eann_metrics = self.models['eann'].evaluate(
            X_test_scaled, y_test_scaled, verbose=0
        )
        rf_score = self.models['rf'].score(X_test_scaled, y_test_scaled)
        gb_score = self.models['gb'].score(X_test_scaled, y_test_scaled)
        
        print(f"EANN - Loss: {eann_metrics[0]:.4f}, MAE: {eann_metrics[1]:.4f}, RMSE: {eann_metrics[3]:.4f}")
        print(f"Random Forest R²: {rf_score:.4f}")
        print(f"Gradient Boosting R²: {gb_score:.4f}")
        
        self.is_trained = True
        
        # Save the trained EANN model manually (since ModelCheckpoint was disabled)
        try:
            model_path = f"models/{self.vessel_id}_eann_model.keras"
            self.models['eann'].save(model_path)
            print(f"\n✓ Model saved to: {model_path}")
        except Exception as e:
            print(f"\n⚠ Warning: Could not save model: {e}")
        
        return {
            'eann_loss': eann_metrics[0],
            'eann_mae': eann_metrics[1],
            'eann_rmse': eann_metrics[3],
            'rf_score': rf_score,
            'gb_score': gb_score,
            'history': history
        }
    
    def _normalize_fuel_type(self, fuel_type_val):
        """
        ÖĞRENİM: Defensive programming
        Handle different input types for fuel_type
        """
        # String input
        if isinstance(fuel_type_val, str):
            return self.FUEL_TYPE_MAPPING.get(fuel_type_val, 0)
        
        # Boolean input
        if isinstance(fuel_type_val, bool):
            return int(fuel_type_val)
        
        # Numeric input
        try:
            return int(fuel_type_val)
        except (ValueError, TypeError):
            print(f"Warning: Invalid fuel_type '{fuel_type_val}', defaulting to 0 (HFO)")
            return 0
    
    def predict(self, vessel_data, model_type: str = 'eann') -> Dict[str, float]:
        """
        Predict vessel performance
        
        ÖĞRENİM: Input validation & error handling
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction. Call train_models() first.")
        
        # ÖĞRENİM: Defensive copy
        if isinstance(vessel_data, dict):
            vessel_data = vessel_data.copy()
            
            # Normalize fuel_type
            if 'fuel_type' in vessel_data:
                vessel_data['fuel_type'] = self._normalize_fuel_type(vessel_data['fuel_type'])
        
        # Convert to array
        if isinstance(vessel_data, dict):
            X = np.array([[vessel_data.get(f, 0) for f in self.feature_names]])
        else:
            X = np.array(vessel_data).reshape(1, -1)
        
        # Scale
        X_scaled = self.scalers['features'].transform(X)
        
        # Predict
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self.models.keys())}")
        
        y_pred_scaled = self.models[model_type].predict(X_scaled)
        y_pred = self.scalers['targets'].inverse_transform(y_pred_scaled)
        
        # Return as dictionary
        results = {target: y_pred[0][i] for i, target in enumerate(self.target_names)}
        
        return results
    
    def analyze_regime_detection(self, vessel_data):
        """
        ÖĞRENİM: New method - regime analysis
        Hangi regime'in aktif olduğunu göster
        """
        if not self.is_trained or self.regime_detector is None:
            raise ValueError("Model must be trained first")
        
        # Prepare input
        if isinstance(vessel_data, dict):
            X = np.array([[vessel_data.get(f, 0) for f in self.feature_names]])
        else:
            X = np.array(vessel_data).reshape(1, -1)
        
        X_scaled = self.scalers['features'].transform(X)
        
        # Get regime probabilities
        regime_probs = self.regime_detector.predict(X_scaled, verbose=0)
        
        regime_names = ['Calm', 'Moderate', 'Rough']
        results = {name: prob for name, prob in zip(regime_names, regime_probs[0])}
        
        # Dominant regime
        dominant_idx = np.argmax(regime_probs[0])
        results['dominant_regime'] = regime_names[dominant_idx]
        results['confidence'] = regime_probs[0][dominant_idx]
        
        return results
    
    def plot_training_history(self):
        """
        ÖĞRENİM: Visualization method
        Training sürecini görselleştir
        """
        if self.training_history is None:
            print("No training history available")
            return
        
        history = self.training_history.history
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Loss Over Epochs', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE
        axes[0, 1].plot(history['mae'], label='Train MAE', linewidth=2)
        axes[0, 1].plot(history['val_mae'], label='Val MAE', linewidth=2)
        axes[0, 1].set_title('MAE Over Epochs', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # RMSE
        axes[1, 0].plot(history['rmse'], label='Train RMSE', linewidth=2)
        axes[1, 0].plot(history['val_rmse'], label='Val RMSE', linewidth=2)
        axes[1, 0].set_title('RMSE Over Epochs', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate (if available)
        if 'lr' in history:
            axes[1, 1].plot(history['lr'], linewidth=2, color='green')
            axes[1, 1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'LR history not available', 
                           ha='center', va='center', fontsize=12)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'models/{self.vessel_id}_training_history.png', dpi=150)
        plt.show()
        
        print(f"✓ Training history plot saved to models/{self.vessel_id}_training_history.png")
   
    
    def save_models(self, filepath: str):
        """Save trained models and scalers"""
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save TensorFlow model
        self.models['eann'].save(f"{filepath}_eann.h5")
        
        # Save sklearn models
        with open(f"{filepath}_models.pkl", 'wb') as f:
            pickle.dump({
                'rf': self.models['rf'],
                'gb': self.models['gb'],
                'scalers': self.scalers,
                'feature_names': self.feature_names,
                'target_names': self.target_names
            }, f)
    
    def load_models(self, filepath):
        """Load pre-trained models and scalers"""
        # Load TensorFlow model
        self.models['eann'] = keras.models.load_model(f"{filepath}_eann.h5")
        
        # Load sklearn models
        with open(f"{filepath}_models.pkl", 'rb') as f:
            data = pickle.load(f)
            self.models['rf'] = data['rf']
            self.models['gb'] = data['gb']
            self.scalers = data['scalers']
            self.feature_names = data['feature_names']
            self.target_names = data['target_names']
        
        self.is_trained = True


# Example usage
if __name__ == "__main__":
    # Create and train modeler
    modeler = SurrogateModeler()
    modeler.train_models()
    
    # Test prediction
    test_vessel = {
        'dwt': 5000,
        'age': 15,
        'length': 100,
        'breadth': 16,
        'draft': 6.5,
        'speed': 12,
        'wave_height': 2.0,
        'wind_speed': 15,
        'current_speed': 0.5,
        'sea_state': 3,
        'load_factor': 0.8,
        'fuel_type': 0,  # HFO
        'engine_efficiency': 0.42
    }
    
    results = modeler.predict(test_vessel)
    print("\nPrediction Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.3f}")
    
    # Regime analysis
    if modeler.regime_detector is not None:
        regime_analysis = modeler.analyze_regime_detection(test_vessel)
        print(f"\nRegime Analysis:")
        print(f"  Dominant Regime: {regime_analysis['dominant_regime']}")
        print(f"  Confidence: {regime_analysis['confidence']:.2%}")
        print(f"  Calm: {regime_analysis['Calm']:.2%}")
        print(f"  Moderate: {regime_analysis['Moderate']:.2%}")
        print(f"  Rough: {regime_analysis['Rough']:.2%}")
