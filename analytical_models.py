import json
import math
import os
import numpy as np

class ClinkerCoolerModels:
    def __init__(self, user_config_path='user_config.json', default_config_path='config_defaults.json'):
        self.user_config_path = user_config_path
        self.default_config_path = default_config_path
        self.config = self._load_config()
        
    def _load_config(self):
        if os.path.exists(self.user_config_path):
            print(f"Loading user configuration from: {self.user_config_path}")
            with open(self.user_config_path, 'r') as file:
                return json.load(file)
        elif os.path.exists(self.default_config_path):
            print(f"No user config found. Loading defaults from: {self.default_config_path}")
            with open(self.default_config_path, 'r') as file:
                return json.load(file)
        else:
            raise FileNotFoundError(f"CRITICAL ERROR: Neither {self.user_config_path} nor {self.default_config_path} was found.")

    def save_user_config(self, new_config_data):
        try:
            with open(self.user_config_path, 'w') as file:
                json.dump(new_config_data, file, indent=4)
            self.config = new_config_data
            print(f"User configuration successfully saved to {self.user_config_path}")
            return True
        except Exception as e:
            print(f"Error saving user configuration: {e}")
            return False

    def calc_output_temperature(self, S, vg, m, t_in=None, t_amb=None):
        coeffs = self.config['temperature_model']
        var_config = self.config['process_variability']
        
        t_in = t_in if t_in is not None else var_config['clinker_inlet_temp_Tin']['nominal_mean']
        t_amb = t_amb if t_amb is not None else var_config['ambient_air_temp_Tamb']['nominal_mean']
        
        vg_safe = max(vg, 0.1)
        m_safe = max(m, 0.1)
        
        t_out = t_in * np.exp(coeffs['coeff_soufflage'] * S / (m_safe * vg_safe)) + t_amb
        return max(t_out, t_amb)

    def calc_electrical_power(self, S, vg, m):
        coeffs = self.config['power_model']
        vg_safe = max(vg, 0.1)
        term1 = coeffs['k1'] * (S ** 3)
        term2 = coeffs['k2'] - (coeffs['k3'] * (m / vg_safe))
        return max(term1 * term2, 0.0)

    def calc_electrical_cost(self, S, vg, m):
        power_kw = self.calc_electrical_power(S, vg, m)
        price_kwh = self.config['costs']['electricity_price_kwh']
        return power_kw * price_kwh

    def calc_belt_lifetime_arrhenius(self, target_temp_c, belt_config):
        arrhenius = belt_config['arrhenius']
        T_target_k = target_temp_c + 273.15
        
        ref_points = []
        for i in range(1, 4):
            key = f'reference_point_{i}'
            if key in arrhenius and arrhenius[key] is not None:
                T_k = arrhenius[key]['temperature_c'] + 273.15
                L_h = arrhenius[key]['lifetime_h']
                ref_points.append((T_k, L_h))
                
        ref_points.sort(key=lambda x: x[0])

        if len(ref_points) >= 3:
            p1, p2 = ref_points[0], ref_points[1] 
            if T_target_k > ref_points[1][0]:
                p1, p2 = ref_points[1], ref_points[2] 
            T1, L1 = p1
            T2, L2 = p2
            if abs(T1 - T2) < 0.1: return L1
            k_coeff = math.log(L1 / L2) / ((1.0 / T1) - (1.0 / T2))
            return L1 * math.exp(k_coeff * ((1.0 / T_target_k) - (1.0 / T1)))
        elif len(ref_points) >= 1:
            Ea = arrhenius.get('activation_energy_j_mol', 35000)
            T1, L1 = ref_points[0] 
            return L1 * math.exp((Ea / 8.314) * ((1.0 / T_target_k) - (1.0 / T1)))
        return 999999.0

    def calc_maintenance_cost(self, T_out):
        price_m2 = self.config['costs']['belt_price_per_m2']
        avg_area = self.config['costs']['average_replacement_area_m2']
        system_utilization = self.config.get('extraction_system', {}).get('total_utilization_rate', 0.07)
        total_weight = sum(belt.get('relative_use_rate', 0.0) for belt in self.config['belts'])
        
        total_maintenance_cost_per_hour = 0.0
        
        for belt in self.config['belts']:
            weight = belt.get('relative_use_rate', 0.0)
            if weight <= 0 or total_weight <= 0: continue
            
            workload_fraction = weight / total_weight
            U_i = system_utilization * workload_fraction
            theoretical_lifetime = self.calc_belt_lifetime_arrhenius(T_out, belt)
            actual_lifetime = theoretical_lifetime / U_i
            total_maintenance_cost_per_hour += (1.0 / actual_lifetime) * (price_m2 * avg_area)
            
        return total_maintenance_cost_per_hour

    def calc_total_cost(self, S, vg, m, t_in=None, t_amb=None, alpha=1.0, beta=1.0):
        electrical_cost = self.calc_electrical_cost(S, vg, m)
        t_out = self.calc_output_temperature(S, vg, m, t_in, t_amb)
        maintenance_cost = self.calc_maintenance_cost(t_out)
        
        return {
            "temperature_out": t_out,
            "electrical_cost": electrical_cost,
            "maintenance_cost": maintenance_cost,
            "total_weighted_cost": (alpha * electrical_cost) + (beta * maintenance_cost)
        }

    def run_monte_carlo_simulation(self, S_target, vg_target, m_target, alpha=1.0, beta=1.0, num_simulations=5000):
        var_config = self.config['process_variability']
        
        def get_noise_std(var_key):
            data = var_config[var_key]
            return data['std_total'] * math.sqrt(max(0.0, 1.0 - data['r2_assignable']))

        sigma_S = get_noise_std('soufflage_S')
        sigma_vg = get_noise_std('vitesse_grilles_vg')
        sigma_m = get_noise_std('debit_clinker_m')
        sigma_Tin = get_noise_std('clinker_inlet_temp_Tin')
        sigma_Tamb = get_noise_std('ambient_air_temp_Tamb')
        
        nom_Tin = var_config['clinker_inlet_temp_Tin']['nominal_mean']
        nom_Tamb = var_config['ambient_air_temp_Tamb']['nominal_mean']

        S_dist = np.clip(np.random.normal(S_target, sigma_S, num_simulations), var_config['soufflage_S']['min'], var_config['soufflage_S']['max'])
        vg_dist = np.clip(np.random.normal(vg_target, sigma_vg, num_simulations), var_config['vitesse_grilles_vg']['min'], var_config['vitesse_grilles_vg']['max'])
        m_dist = np.clip(np.random.normal(m_target, sigma_m, num_simulations), var_config['debit_clinker_m']['min'], var_config['debit_clinker_m']['max'])
        Tin_dist = np.clip(np.random.normal(nom_Tin, sigma_Tin, num_simulations), var_config['clinker_inlet_temp_Tin']['min'], var_config['clinker_inlet_temp_Tin']['max'])
        Tamb_dist = np.clip(np.random.normal(nom_Tamb, sigma_Tamb, num_simulations), var_config['ambient_air_temp_Tamb']['min'], var_config['ambient_air_temp_Tamb']['max'])

        results = [self.calc_total_cost(S=S_dist[i], vg=vg_dist[i], m=m_dist[i], t_in=Tin_dist[i], t_amb=Tamb_dist[i], alpha=alpha, beta=beta) for i in range(num_simulations)]
        
        costs = np.array([r['total_weighted_cost'] for r in results])
        var_95 = np.percentile(costs, 95)
        
        return {
            "num_simulations": num_simulations,
            "deterministic_cost": self.calc_total_cost(S_target, vg_target, m_target, nom_Tin, nom_Tamb, alpha, beta)['total_weighted_cost'],
            "robust_optimum_cost": np.mean(costs),
            "cost_std": np.std(costs),
            "var_95": var_95,
            "cvar_95": np.mean(costs[costs >= var_95]),
            "expected_temperature": np.mean([r['temperature_out'] for r in results]),
            "expected_electrical_cost": np.mean([r['electrical_cost'] for r in results]),
            "expected_maintenance_cost": np.mean([r['maintenance_cost'] for r in results])
        }
    
    def calc_total_cost_gradient(self, S, vg, m, alpha=1.0, beta=1.0):
        vg_safe = max(vg, 0.1)
        m_safe = max(m, 0.1)
        
        coeffs_pwr = self.config['power_model']
        price_kwh = self.config['costs']['electricity_price_kwh']
        dCelec_dS = price_kwh * 3 * coeffs_pwr['k1'] * (S ** 2) * (coeffs_pwr['k2'] - coeffs_pwr['k3'] * (m_safe / vg_safe))
        dCelec_dvg = price_kwh * coeffs_pwr['k1'] * (S ** 3) * (coeffs_pwr['k3'] * m_safe / (vg_safe ** 2))

        coeffs_temp = self.config['temperature_model']
        t_base = coeffs_temp['T_base']
        c_s = coeffs_temp['coeff_soufflage']
        exp_term = np.exp(c_s * S / (m_safe * vg_safe))
        dT_dS = t_base * exp_term * (c_s / (m_safe * vg_safe))
        dT_dvg = t_base * exp_term * (-c_s * S / (m_safe * (vg_safe ** 2)))

        t_out = self.calc_output_temperature(S, vg, m)
        dCmaint_dT = 0.0
        
        belt1 = next((b for b in self.config['belts'] if b['id'] == 1), self.config['belts'][0])
        arrhenius = belt1['arrhenius']
        T_target_k = t_out + 273.15
        
        c_belt = self.config['costs']['belt_price_per_m2'] * self.config['costs']['average_replacement_area_m2']
        system_utilization = self.config.get('extraction_system', {}).get('total_utilization_rate', 0.85)
        
        ref_points = sorted([(arrhenius[f'reference_point_{i}']['temperature_c'] + 273.15, arrhenius[f'reference_point_{i}']['lifetime_h']) for i in range(1, 4) if arrhenius.get(f'reference_point_{i}')], key=lambda x: x[0])
        
        k_local, L_theo = 0.0, 999999.0
        
        if len(ref_points) >= 3:
            p1, p2 = (ref_points[1], ref_points[2]) if T_target_k > ref_points[1][0] else (ref_points[0], ref_points[1])
            T1, L1, T2, L2 = p1[0], p1[1], p2[0], p2[1]
            if abs(T1 - T2) > 0.1:
                k_local = math.log(L1 / L2) / ((1.0 / T1) - (1.0 / T2))
                L_theo = L1 * math.exp(k_local * ((1.0 / T_target_k) - (1.0 / T1)))
        elif len(ref_points) >= 1:
            k_local = arrhenius.get('activation_energy_j_mol', 35000) / 8.314
            L_theo = ref_points[0][1] * math.exp(k_local * ((1.0 / T_target_k) - (1.0 / ref_points[0][0])))
        
        if k_local != 0.0:
            dCmaint_dT = ((c_belt * system_utilization) / max(L_theo, 0.01)) * (k_local / (T_target_k ** 2))

        return {
            "gradient": {
                "dC_dS": (alpha * dCelec_dS) + (beta * dCmaint_dT * dT_dS),
                "dC_dvg": (alpha * dCelec_dvg) + (beta * dCmaint_dT * dT_dvg)
            },
            "components": {"dCelec_dS": dCelec_dS, "dCelec_dvg": dCelec_dvg, "dCmaint_dT": dCmaint_dT, "dT_dS": dT_dS, "dT_dvg": dT_dvg}
        }

    def calc_iso_temperature_ratio(self, T_target, m):
        coeffs = self.config['temperature_model']
        t_base = coeffs['T_base']
        t_emb = coeffs.get('T_emb', 25.0)
        c_s = coeffs['coeff_soufflage']
        
        if T_target <= t_emb: raise ValueError(f"Target temperature ({T_target}°C) cannot be equal to or lower than ambient air ({t_emb}°C).")
        ratio_temp = (T_target - t_emb) / t_base
        if ratio_temp <= 0: raise ValueError("Invalid temperature target. Exceeds model base boundaries.")
        return (m / c_s) * math.log(ratio_temp)

    def evaluate_iso_temperature_scenario(self, T_target, m, S=None, vg=None):
        if S is None and vg is None: raise ValueError("Must provide either S or vg to anchor the line.")
        K_slope = self.calc_iso_temperature_ratio(T_target, m)
        
        if S is not None:
            S_calc = max(S, 0.0)
            vg_calc = max(S_calc / K_slope if K_slope != 0 else 0.1, 0.1)
        else:
            vg_calc = max(vg, 0.1)
            S_calc = max(K_slope * vg_calc, 0.0)
            
        maint_cost = self.calc_maintenance_cost(T_target)
        elec_cost = self.calc_electrical_cost(S_calc, vg_calc, m)
        
        return {
            "target_temperature": T_target,
            "iso_slope_K": K_slope,
            "S_required": S_calc,
            "vg_required": vg_calc,
            "electrical_cost": elec_cost,
            "maintenance_cost": maint_cost,
            "total_cost": elec_cost + maint_cost
        }

    def calc_kpis(self, S, vg, m):
        return {
            "specific_energy_kwh_t": self.calc_electrical_power(S, vg, m) / max(m, 0.1),
            "total_power_kw": self.calc_electrical_power(S, vg, m),
            "output_temperature_c": self.calc_output_temperature(S, vg, m)
        }

    def evaluate_belt_lifetimes(self, T_out):
        system_utilization = self.config.get('extraction_system', {}).get('total_utilization_rate', 0.85)
        total_weight = sum(b.get('relative_use_rate', 0.0) for b in self.config['belts'])
        belt_stats = []
        
        for belt in self.config['belts']:
            weight = belt.get('relative_use_rate', 0.0)
            hours_operated = belt.get('hours_already_operated', 0)
            if weight <= 0 or total_weight <= 0:
                belt_stats.append({
                    "belt_id": belt['id'], 
                    "name": belt['name'], 
                    "absolute_utilization_rate": 0.0, 
                    "hours_already_operated": hours_operated, 
                    "theoretical_lifetime_h": self.calc_belt_lifetime_arrhenius(T_out, belt), 
                    "actual_expected_lifetime_h": 9999999.0, 
                    "estimated_remaining_life_h": 9999999.0
                })
            else:
                U_i = system_utilization * (weight / total_weight)
                theo = self.calc_belt_lifetime_arrhenius(T_out, belt)
                actual = theo / U_i
                belt_stats.append({"belt_id": belt['id'], "name": belt['name'], "absolute_utilization_rate": U_i, "hours_already_operated": hours_operated, "theoretical_lifetime_h": theo, "actual_expected_lifetime_h": actual, "estimated_remaining_life_h": max(actual - hours_operated, 0.0)})
        return belt_stats

    def validate_constraints(self, S, vg, m):
        warnings = []
        limits = self.config.get('limits', {})
        if limits:
            if S > limits.get('fan_speed_total', {}).get('max', 5400): warnings.append("ALERTE: Soufflage total dépasse la capacité maximale des ventilateurs.")
            if vg > limits.get('grate_speed', {}).get('max', 30): warnings.append("ALERTE: Vitesse des grilles dépasse la limite mécanique.")
        t_out = self.calc_output_temperature(S, vg, m)
        if t_out > limits.get('clinker_temp', {}).get('output_max', 250): warnings.append(f"DANGER: Température de sortie ({t_out:.1f}°C) critique pour les bandes.")
        if self.calc_electrical_power(S, vg, m) > self.config.get('ventilators', {}).get('power_consumption', {}).get('max', 4500): warnings.append("ALERTE: Appel de puissance électrique au-dessus du seuil autorisé.")
        return warnings

if __name__ == "__main__":
    models = ClinkerCoolerModels('config_defaults.json')
    results = models.calc_total_cost(300, 15, 215)
    print("--- TEST EVALUATION ---")
    print(f"Temperature Sortie: {results['temperature_out']:.2f} °C")
    print(f"COUT TOTAL (Weighted): {results['total_weighted_cost']:.2f} MAD/h")