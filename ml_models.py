import os
import numpy as np
import pandas as pd
import joblib
from analytical_models import ClinkerCoolerModels

class ClinkerCoolerML(ClinkerCoolerModels):
    def __init__(self, user_config_path='user_config.json', default_config_path='config_defaults.json', model_dir='trained_models'):
        super().__init__(user_config_path, default_config_path)
        self.model_dir = model_dir
        self.models = {'temperature': {}, 'power': {}}
        self.scaler = None
        self._load_registry()

    def _load_registry(self):
        try:
            scaler_path = os.path.join(self.model_dir, 'feature_scaler.joblib')
            if os.path.exists(scaler_path): self.scaler = joblib.load(scaler_path)
            self.models['temperature']['xgboost'] = joblib.load(os.path.join(self.model_dir, 'PI_TOUT_xgb_model.joblib'))
            self.models['temperature']['rf'] = joblib.load(os.path.join(self.model_dir, 'PI_TOUT_rf_model.joblib'))
            self.models['temperature']['extra_trees'] = joblib.load(os.path.join(self.model_dir, 'PI_TOUT_et_model.joblib'))
            self.models['power']['xgboost'] = joblib.load(os.path.join(self.model_dir, 'PI_POW_xgb_model.joblib'))
            self.models['power']['rf'] = joblib.load(os.path.join(self.model_dir, 'PI_POW_rf_model.joblib'))
            self.models['power']['extra_trees'] = joblib.load(os.path.join(self.model_dir, 'PI_POW_et_model.joblib'))
            print("ML Registry loaded successfully.")
        except Exception as e:
            print(f"Warning: Model registry failed to load. {e}")

    def _prepare_features(self, S, vg, m):
        """Prépare les données au format exact attendu par les modèles IA (DataFrame Pandas)."""
        vg_safe = max(vg, 0.1)
        m_safe = max(m, 0.1)
        ratio = S / (m_safe * vg_safe)
        
        # 1. Créer un dictionnaire avec les NOMS EXACTS des colonnes de votre entraînement
        feature_dict = {
            'softot': [S],
            'vg': [vg],
            'm': [m],
            # 'ratio': [ratio]
        }
        
        # 2. Convertir en DataFrame Pandas
        df = pd.DataFrame(feature_dict)
        
        # 3. Appliquer le scaler si existant, tout en conservant le format DataFrame
        if self.scaler:
            scaled_data = self.scaler.transform(df)
            # Reconstruire le DataFrame après le scaling pour garder les noms de colonnes
            df = pd.DataFrame(scaled_data, columns=df.columns)
            
        return df

    def calc_output_temperature(self, S, vg, m, t_in=None, t_amb=None, algo='analytical'):
        """Overrides the analytical temperature calculation with ML if requested."""
        if algo == 'analytical':
            return super().calc_output_temperature(S, vg, m, t_in, t_amb)
            
        X = self._prepare_features(S, vg, m)
        model = self.models['temperature'].get(algo)
        if not model: raise ValueError(f"Algorithm '{algo}' not found in registry.")
        return max(float(model.predict(X)[0]), 30.0)

    def calc_total_cost(self, S, vg, m, t_in=None, t_amb=None, alpha=1.0, beta=1.0, algo='analytical'):
        """Overrides total cost to integrate ML physics with Analytical economics."""
        if algo == 'analytical':
            # FIX: Use 'self' directly instead of 'self.analytical'
            t_out = self.calc_output_temperature(S, vg, m)
            P_elec = self.calc_electrical_power(S, vg, m)
        else:
            features = self._prepare_features(S, vg, m)
            
            # --- 1. Prédiction de la Température ---
            if algo in self.models['temperature']:
                t_out = float(self.models['temperature'][algo].predict(features)[0])
            else:
                # FIX: Use 'self' directly
                t_out = self.calc_output_temperature(S, vg, m)
                
            # --- 2. Prédiction de la Puissance ---
            if algo in self.models['power']:
                # On utilise le modèle IA pour prédire la puissance (kW)
                P_elec = float(self.models['power'][algo].predict(features)[0])
            else:
                # FIX: Fallback sur la physique si le modèle IA n'est pas trouvé
                P_elec = self.calc_electrical_power(S, vg, m)

        # --- 3. Calcul du Coût Électrique ---
        elec_price = self.config['costs']['electricity_price_kwh']
        elec_cost = P_elec * elec_price
        maint_cost = self.calc_maintenance_cost(t_out)
        
        return {
            "temperature_out": t_out,
            "electrical_cost": elec_cost,
            "maintenance_cost": maint_cost,
            "total_weighted_cost": (alpha * elec_cost) + (beta * maint_cost)
        }

    def calc_total_cost_gradient(self, S, vg, m, alpha=1.0, beta=1.0, algo='analytical', step_S=5.0, step_vg=0.5):
        """Macro-step numerical gradient for tree-based optimization."""
        if algo == 'analytical':
            return super().calc_total_cost_gradient(S, vg, m, alpha, beta)
            
        cost_S_plus = self.calc_total_cost(S + step_S, vg, m, alpha=alpha, beta=beta, algo=algo)['total_weighted_cost']
        cost_S_minus = self.calc_total_cost(S - step_S, vg, m, alpha=alpha, beta=beta, algo=algo)['total_weighted_cost']
        dC_dS = (cost_S_plus - cost_S_minus) / (2.0 * step_S)
        
        cost_vg_plus = self.calc_total_cost(S, vg + step_vg, m, alpha=alpha, beta=beta, algo=algo)['total_weighted_cost']
        cost_vg_minus = self.calc_total_cost(S, vg - step_vg, m, alpha=alpha, beta=beta, algo=algo)['total_weighted_cost']
        dC_dvg = (cost_vg_plus - cost_vg_minus) / (2.0 * step_vg)

        return {"gradient": { "dC_dS": dC_dS, "dC_dvg": dC_dvg }}