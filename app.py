from flask import Flask, request, jsonify
from flask_cors import CORS
from ml_models import ClinkerCoolerML
import traceback

app = Flask(__name__)
CORS(app) 

print("Initializing Clinker Cooler Engine...")
engine = ClinkerCoolerML(user_config_path='user_config.json', default_config_path='config_defaults.json')
app.config['engine_instance'] = engine
print("Engine ready.")

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online", 
        "message": "Clinker Cooler API is running.",
        "active_models": list(engine.models['temperature'].keys()) if engine.models['temperature'] else ["analytical"]
    }), 200

@app.route('/api/v1/evaluate', methods=['POST'])
def evaluate_decision():
    try:
        data = request.json
        S = float(data.get('fan_speed', 300.0))
        vg = float(data.get('grate_speed', 15.0))
        m = float(data.get('kiln_feed', 215.0))
        alpha = float(data.get('alpha', 1.0))
        beta = float(data.get('beta', 1.0))

        # --- RÉCUPÉRATION DES POIDS INDIVIDUELS (Défaut à 1.0) ---
        weights = data.get('model_weights', {})
        w_temp = weights.get('temperature', {})
        w_power = weights.get('power', {})

        algorithms = ['analytical', 'xgboost', 'rf', 'extra_trees']
        models_results = {}
        
        sum_temp, sum_elec, sum_maint = 0, 0, 0
        total_w_temp, total_w_power = 0, 0
        valid_models = 0

        for algo in algorithms:
            try:
                res = engine.calc_total_cost(S, vg, m, alpha=alpha, beta=beta, algo=algo)
                models_results[algo] = res
                
                # Extraction des poids spécifiques à ce modèle (défaut = 1.0)
                wt = float(w_temp.get(algo, 1.0))
                wp = float(w_power.get(algo, 1.0))
                
                # --- CALCUL DES SOMMES PONDÉRÉES ---
                # Si le poids est 0, le modèle n'a aucune influence sur la moyenne
                sum_temp += res['temperature_out'] * wt
                sum_maint += res['maintenance_cost'] * wt  # La maintenance dépend de la temp, donc utilise wt
                total_w_temp += wt
                
                sum_elec += res['electrical_cost'] * wp
                total_w_power += wp
                
                valid_models += 1
            except Exception as e:
                print(f"Modèle ignoré ({algo}): {e}")
                pass 

        if valid_models == 0:
            raise ValueError("Aucun modèle n'est disponible.")

        # --- CALCUL DE LA MOYENNE DE L'ENSEMBLE (Weighted Average) ---
        avg_temp = sum_temp / total_w_temp if total_w_temp > 0 else 0
        avg_maint = sum_maint / total_w_temp if total_w_temp > 0 else 0
        avg_elec = sum_elec / total_w_power if total_w_power > 0 else 0
        
        # Le coût total pondéré de l'ensemble est recalculé proprement
        avg_total = (alpha * avg_elec) + (beta * avg_maint)

        avg_results = {
            "temperature_out": avg_temp,
            "electrical_cost": avg_elec,
            "maintenance_cost": avg_maint,
            "weighted_cost": avg_total
        }

        # Calculate KPIs and Lifetimes based on the New Weighted Average
        kpis = engine.calc_kpis(S, vg, m)
        kpis['output_temperature_c'] = avg_results['temperature_out']
        lifetimes = engine.evaluate_belt_lifetimes(avg_results['temperature_out'])

        # --- CORRECTION DES ALARMES (ALIGNEMENT SUR LA MOYENNE) ---
        # 1. On récupère les avertissements de base
        raw_warnings = engine.validate_constraints(S, vg, m)
        
        # 2. On filtre pour supprimer l'alarme de température générée par le modèle physique seul
        warnings = [w for w in raw_warnings if "Température" not in w and "température" not in w and "sortie" not in w]
        
        # 3. On génère la véritable alarme basée sur la MOYENNE pondérée
        max_temp_limit = engine.config.get('limits', {}).get('clinker_temp', {}).get('output_max', 250)
        
        if avg_results['temperature_out'] > max_temp_limit:
            warnings.append(f"DANGER: La température MOYENNE de sortie ({avg_results['temperature_out']:.1f}°C) dépasse la limite critique de {max_temp_limit}°C pour les bandes.")
        # ----------------------------------------------------------

        response = {
            "status": "success",
            "evaluation": {
                "models": models_results,
                "current_point": avg_results,
                "kpis": kpis,
                "belt_lifetimes": lifetimes,
                "warnings": warnings
            }
        }
        return jsonify(response), 200

    except Exception as e:
        import traceback
        print(f"Error in /evaluate: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/v1/evaluate/iso-temp', methods=['POST'])
def evaluate_iso_temperature():
    try:
        data = request.json
        target_temp = float(data.get('target_temperature'))
        m = float(data.get('kiln_feed', 215.0))
        
        S_anchor = data.get('fan_speed')
        vg_anchor = data.get('grate_speed')
        if S_anchor is not None: S_anchor = float(S_anchor)
        if vg_anchor is not None: vg_anchor = float(vg_anchor)

        iso_results = engine.evaluate_iso_temperature_scenario(target_temp, m, S_anchor, vg_anchor)
        return jsonify({"status": "success", "iso_temperature_data": iso_results}), 200

    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        print(f"Error in /evaluate/iso-temp: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": "Internal Server Error"}), 500

@app.route('/api/v1/config', methods=['GET'])
def get_config():
    try:
        return jsonify({"status": "success", "config": engine.config}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/v1/config', methods=['POST'])
def save_config():
    try:
        success = engine.save_user_config(request.json)
        if success:
            return jsonify({"status": "success"}), 200
        else:
            return jsonify({"status": "error", "message": "Save failed."}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)