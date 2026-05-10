
# ── EXAMPLE: How to use in Flask backend ────────────────────────

from diabetes_risk_api import load_models, predict_risk

# Load ONCE at startup
models = load_models("./model_export/")

# Example user input (from your frontend form)
user_input = {
    "age"               : 38,
    "waist_cm"          : 91.0,
    "gender"            : "female",
    "physical_activity" : "none",
    "family_history"    : True,
    "polyuria"          : True,
    "polydipsia"        : True,
    "sudden_weight_loss": False,
    "weakness"          : True,
    "polyphagia"        : False,
    "genital_thrush"    : False,
    "visual_blurring"   : False,
    "itching"           : True,
    "irritability"      : False,
    "delayed_healing"   : False,
    "partial_paresis"   : False,
    "muscle_stiffness"  : False,
    "alopecia"          : False,
    "obesity"           : True,
    "high_bp"           : True,
    "high_chol"         : False,
    "bmi"               : 31.2,
    "smoker"            : False,
    "stroke"            : False,
    "heart_disease"     : False,
    "phys_active"       : False,
    "heavy_alcohol"     : False,
    "mental_health_days": 5,
    "pcos"              : True,
    "acanthosis_nigricans": True,
    "skin_tags"         : False,
    "nafld"             : False,
    "sleep_apnea"       : False,
    "thyroid_condition" : False,
    "chronic_stress_depression": True,
    "gestational_diabetes": False,
    "poor_sleep"        : True,
    "waist_hip_ratio"   : 0.88,
}

result = predict_risk(models, user_input)

print(result["final_score_pct"])   # e.g. 91.5
print(result["category"])          # e.g. "VERY HIGH"
print(result["modifiable_drivers"])# list of strings
print(result["advice"])            # string for UI
