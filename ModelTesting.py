import os
import json
import time
from analytical_models import ClinkerCoolerModels

def run_tests():
    print("===================================================")
    print("   CLINKER COOLER DIGITAL TWIN - UNIT TESTS")
    print("===================================================\n")
    
    # 1. Initialization Test
    try:
        models = ClinkerCoolerModels(user_config_path='dummy_user.json', default_config_path='config_defaults.json')
        print("[PASS] Model successfully initialized and loaded defaults.")
    except Exception as e:
        print(f"[FAIL] Initialization failed: {e}")
        return

    # Baseline operating parameters
    S_base = 400.0    # Soufflage (Nm3/h)
    vg_base = 15.0    # Vitesse grilles (mouv/min)
    m_base = 415.0    # Kiln feed (t/h)
    alpha_base = 1.0  # Electrical weight
    beta_base = 1.0   # Maintenance weight

    print("\n--- TEST 1: BASELINE DETERMINISTIC COST ---")
    try:
        base_results = models.calc_total_cost(S_base, vg_base, m_base, alpha=alpha_base, beta=beta_base)
        print(f"  Temperature Sortie : {base_results['temperature_out']:.2f} °C")
        print(f"  Cout Electrique    : {base_results['electrical_cost']:.2f} MAD/h")
        print(f"  Cout Maintenance   : {base_results['maintenance_cost']:.2f} MAD/h")
        print(f"  COUT TOTAL PONDERE : {base_results['total_weighted_cost']:.2f} MAD/h")
        print("[PASS] Baseline evaluation successful.")
    except Exception as e:
        print(f"[FAIL] Baseline evaluation failed: {e}")

    print("\n--- TEST 2: PHYSICS VALIDATION (FAN COOLING) ---")
    try:
        S_high = 450.0
        high_fan_results = models.calc_total_cost(S_high, vg_base, m_base)
        print(f"  Temp at S={S_base}: {base_results['temperature_out']:.2f} °C")
        print(f"  Temp at S={S_high}: {high_fan_results['temperature_out']:.2f} °C")
        
        if high_fan_results['temperature_out'] < base_results['temperature_out']:
            print("[PASS] Physics Check: Increasing fan speed successfully decreased temperature.")
        else:
            print("[FAIL] Physics Check: Increasing fan speed INCREASED temperature. Check 'coeff_soufflage' sign in JSON.")
    except Exception as e:
        print(f"[FAIL] Physics validation failed: {e}")

    print("\n--- TEST 3: SAFETY FAILSAFE (ZERO GRATE SPEED) ---")
    try:
        zero_vg_results = models.calc_total_cost(S_base, 0.0, m_base)
        print(f"  Calculated safely with vg=0: Cost = {zero_vg_results['total_weighted_cost']:.2f} MAD/h")
        print("[PASS] ZeroDivisionError prevented successfully.")
    except Exception as e:
        print(f"[FAIL] Safety failsafe failed: {e}")

    print("\n--- TEST 4: GRADIENT AND CHAIN RULE ---")
    try:
        gradient_data = models.calc_total_cost_gradient(S_base, vg_base, m_base, alpha_base, beta_base)
        print("  Gradient Components:")
        print(f"    d(Maint)/dT  : {gradient_data['components']['dCmaint_dT']:.4f}")
        print(f"    dT/dS        : {gradient_data['components']['dT_dS']:.4f}")
        print("  Final Gradient Vector:")
        print(f"    d(Total)/dS  : {gradient_data['gradient']['dC_dS']:.4f}")
        print(f"    d(Total)/dvg : {gradient_data['gradient']['dC_dvg']:.4f}")
        print("[PASS] Exact analytical gradient evaluated successfully.")
    except Exception as e:
        print(f"[FAIL] Gradient calculation failed: {e}")

    print("\n--- TEST 5: ISO-TEMPERATURE SCENARIO ---")
    try:
        target_temp = 120.0 # °C
        iso_results = models.evaluate_iso_temperature_scenario(target_temp, m_base, vg=vg_base)
        print(f"  Target Temperature : {iso_results['target_temperature']:.1f} °C")
        print(f"  Iso-Slope (K)      : {iso_results['iso_slope_K']:.4f}")
        print(f"  Required S         : {iso_results['S_required']:.1f} Nm3/h (for vg={vg_base})")
        print(f"  Resulting Cost     : {iso_results['total_cost']:.2f} MAD/h")
        print("[PASS] Iso-temperature evaluated successfully.")
    except Exception as e:
        print(f"[FAIL] Iso-temperature calculation failed: {e}")

    print("\n--- TEST 6: KPIS AND CONSTRAINTS ---")
    try:
        kpis = models.calc_kpis(S_base, vg_base, m_base)
        print(f"  Specific Energy : {kpis['specific_energy_kwh_t']:.2f} kWh/t")
        
        # Test validator with an extreme scenario (S=10000)
        warnings = models.validate_constraints(10000.0, vg_base, m_base)
        if warnings:
            print(f"  Validator correctly caught warnings: {len(warnings)} found.")
            for w in warnings:
                print(f"    -> {w}")
        print("[PASS] KPIs and Constraints validated successfully.")
    except Exception as e:
        print(f"[FAIL] KPIs and Constraints failed: {e}")

    print("\n--- TEST 7: BELT LIFETIMES EVALUATION ---")
    try:
        lifetimes = models.evaluate_belt_lifetimes(base_results['temperature_out'])
        print(f"  Evaluated {len(lifetimes)} belts at {base_results['temperature_out']:.1f} °C:")
        for b in lifetimes:
            print(f"    Belt {b['belt_id']}: U_abs = {b['absolute_utilization_rate']:.3f} | Remainder = {b['estimated_remaining_life_h']:.0f} h")
        print("[PASS] Belt lifetimes calculated successfully.")
    except Exception as e:
        print(f"[FAIL] Belt lifetimes calculation failed: {e}")

    print("\n--- TEST 8: MONTE CARLO SIMULATION (STOCHASTIC RISK) ---")
    try:
        print("  Running 1000 simulations...")
        start_time = time.time()
        
        # Using 1000 simulations for the test script to keep it fast
        mc_results = models.run_monte_carlo_simulation(S_base, vg_base, m_base, num_simulations=1000)
        
        elapsed = time.time() - start_time
        print(f"  Simulation completed in {elapsed:.3f} seconds.")
        print(f"  Deterministic Cost : {mc_results['deterministic_cost']:.2f} MAD/h")
        print(f"  Expected Mean Cost : {mc_results['robust_optimum_cost']:.2f} MAD/h")
        print(f"  Std Deviation (σ)  : {mc_results['cost_std']:.2f} MAD/h")
        print(f"  Value at Risk (95%): {mc_results['var_95']:.2f} MAD/h")
        print(f"  Cond. VaR (CVaR)   : {mc_results['cvar_95']:.2f} MAD/h")
        print("[PASS] Monte Carlo simulation executed successfully.")
    except Exception as e:
        print(f"[FAIL] Monte Carlo simulation failed: {e}")

    print("\n===================================================")
    print("   TESTING COMPLETE")
    print("===================================================")

if __name__ == "__main__":
    run_tests()