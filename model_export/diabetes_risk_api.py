"""
Early Diabetes Risk Prediction API
===================================
Authors : [Your Name] + [Friend's Name]
Project : Early Diabetes Risk Detection System

HOW TO USE IN A BACKEND:
    from diabetes_risk_api import load_models, predict_risk

    # Load once at startup
    models = load_models("path/to/model_export/")

    # Call for each user
    result = predict_risk(models, user_input_dict)
    # result["final_score_pct"]  → float (0-100)
    # result["category"]         → "LOW" / "MODERATE" / "HIGH" / "VERY HIGH"
    # result["modifiable_drivers"] → list of strings
    # result["advice"]           → string shown to user

INPUT FORMAT (user_input_dict):
    {
        # Basic Info
        "age"               : 38,        # years (int)
        "waist_cm"          : 91.0,      # centimetres (float)
        "gender"            : "female",  # "male" or "female"
        "physical_activity" : "none",    # "regular"/"irregular"/"none"
        "family_history"    : True,      # bool

        # Symptoms (all bool or 0/1)
        "polyuria"           : True,
        "polydipsia"         : True,
        "sudden_weight_loss" : False,
        "weakness"           : True,
        "polyphagia"         : False,
        "genital_thrush"     : False,
        "visual_blurring"    : False,
        "itching"            : True,
        "irritability"       : False,
        "delayed_healing"    : False,
        "partial_paresis"    : False,
        "muscle_stiffness"   : False,
        "alopecia"           : False,
        "obesity"            : True,

        # Lifestyle (bool or float)
        "high_bp"           : True,
        "high_chol"         : False,
        "bmi"               : 31.2,
        "smoker"            : False,
        "stroke"            : False,
        "heart_disease"     : False,
        "phys_active"       : False,
        "heavy_alcohol"     : False,
        "mental_health_days": 5,         # 0-30

        # Clinical Signs (all bool)
        "pcos"                     : True,
        "acanthosis_nigricans"     : True,
        "skin_tags"                : False,
        "nafld"                    : False,
        "sleep_apnea"              : False,
        "thyroid_condition"        : False,
        "chronic_stress_depression": True,
        "gestational_diabetes"     : False,
        "poor_sleep"               : True,
        "waist_hip_ratio"          : 0.88,  # float or None
    }
"""

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path


# ── BRFSS Age Category Map ──────────────────────────────────────
BRFSS_AGE_MAP = {
    1:(18,24,"18-24"), 2:(25,29,"25-29"), 3:(30,34,"30-34"),
    4:(35,39,"35-39"), 5:(40,44,"40-44"), 6:(45,49,"45-49"),
    7:(50,54,"50-54"), 8:(55,59,"55-59"), 9:(60,64,"60-64"),
    10:(65,69,"65-69"),11:(70,74,"70-74"),12:(75,79,"75-79"),
    13:(80,99,"80+")
}

def _age_to_brfss(age):
    for cat,(lo,hi,_) in BRFSS_AGE_MAP.items():
        if lo <= age <= hi: return cat
    return 13

def _gender_to_brfss(gender):
    return 1 if str(gender).lower() in ["male","m"] else 0


# ── IDRS Formula ────────────────────────────────────────────────
def _compute_idrs(age, waist_cm, gender, physical_activity,
                  family_history):
    bd = {}
    age_pts = 0 if age<35 else (20 if age<50 else 30)
    bd["Age"] = age_pts

    male = str(gender).lower() in ["male","m","1"]
    if male:
        waist_pts = 0 if waist_cm<90 else (10 if waist_cm<100 else 20)
    else:
        waist_pts = 0 if waist_cm<80 else (10 if waist_cm<90 else 20)
    bd["Waist circumference"] = waist_pts

    pa = str(physical_activity).lower()
    pa_pts = 0 if pa=="regular" else (20 if pa=="irregular" else 30)
    bd["Physical inactivity"] = pa_pts

    fh_pts = 20 if family_history else 0
    bd["Family history of diabetes"] = fh_pts

    score = age_pts + waist_pts + pa_pts + fh_pts
    norm  = round(score / 100.0, 4)
    cat   = ("Low" if score<30 else
             "Moderate" if score<=50 else
             "Borderline High" if score<=59 else "High")
    return score, norm, cat, bd


# ── Clinical Modifiers ──────────────────────────────────────────
def _apply_modifiers(base_score, clinical, gender):
    score, total, log = base_score, 0.0, []
    MAX = 0.20

    def add(label, amt, ev):
        nonlocal score, total
        rem = MAX - total
        if rem <= 0: return
        a = min(amt, rem)
        score = min(1.0, score + a)
        total += a
        log.append((label, round(a,3), ev))

    if clinical.get("gestational_diabetes"):
        add("History of gestational diabetes",
            0.20,"HR 5-15x, Bellamy 2009")
    if clinical.get("pcos"):
        add("PCOS",0.18,"OR 2.87, Teede 2018")
    if clinical.get("nafld"):
        add("NAFLD",0.15,"HR 2.22, Mantovani")
    if clinical.get("acanthosis_nigricans"):
        add("Acanthosis Nigricans",0.12,"88% specificity IR")
    if clinical.get("sleep_apnea"):
        add("Sleep Apnea",0.10,"RR 1.63, Lee 2013")
    if clinical.get("thyroid_condition"):
        add("Thyroid condition",0.08,"HR 1.26")
    if clinical.get("chronic_stress_depression"):
        add("Chronic stress/depression",0.07,"HR 1.37, Knol 2006")
    if clinical.get("skin_tags"):
        add("Skin tags",0.07,"IR dermatosis marker")
    if clinical.get("poor_sleep"):
        add("Poor sleep <6h",0.06,"Cortisol pathway")
    if (clinical.get("acanthosis_nigricans") and
            clinical.get("skin_tags")):
        add("AN+Skin Tags co-occurrence",0.05,"Combined IR")
    if (clinical.get("pcos") and
            clinical.get("acanthosis_nigricans")):
        add("PCOS+AN co-occurrence",0.05,"Compounded IR")

    whr = clinical.get("waist_hip_ratio")
    if whr:
        male = str(gender).lower() in ["male","m"]
        if ((male and whr>1.00) or (not male and whr>0.95)):
            add(f"Very high WHR ({whr:.2f})",0.12,"Central obesity")
        elif ((male and whr>0.90) or (not male and whr>0.85)):
            add(f"High WHR ({whr:.2f})",0.07,"Abdominal obesity")

    return round(score,4), log


# ── Risk Category ───────────────────────────────────────────────
def _get_category(score):
    if score < 0.25:
        return ("LOW","🟢",
                "Low metabolic risk. Maintain healthy lifestyle.")
    elif score < 0.45:
        return ("MODERATE","🟡",
                "Moderate risk. Lifestyle changes recommended.")
    elif score < 0.65:
        return ("HIGH","🟠",
                "Elevated risk. Consult a physician for screening.")
    else:
        return ("VERY HIGH","🔴",
                "High risk. Medical evaluation strongly advised.")


# ── Load Models ─────────────────────────────────────────────────
def load_models(model_dir: str) -> dict:
    """
    Load all trained models and config from model_export/ folder.
    Call this ONCE at server startup — not on every request.

    Parameters
    ----------
    model_dir : str — path to model_export/ folder

    Returns
    -------
    models : dict — pass this to predict_risk() on every call
    """
    p = Path(model_dir)
    models = {
        "stage2"  : joblib.load(p/"stage2_lifestyle_model.pkl"),
        "stage3"  : joblib.load(p/"stage3_symptom_model.pkl"),
        "config"  : json.loads((p/"feature_names.json").read_text())
    }
    print(f"✅ Models loaded from {model_dir}")
    return models


# ── Main Prediction Function ────────────────────────────────────
def predict_risk(models: dict, user_input: dict) -> dict:
    """
    Run full 5-stage diabetes risk prediction.

    Parameters
    ----------
    models     : dict — output of load_models()
    user_input : dict — see module docstring for format

    Returns
    -------
    result : dict with keys:
        final_score_pct  : float  — risk percentage 0-100
        category         : str    — LOW/MODERATE/HIGH/VERY HIGH
        emoji            : str    — colour emoji
        advice           : str    — one-line advice string
        component_scores : dict   — breakdown per stage
        modifiable_drivers   : list[str]
        nonmodifiable_drivers: list[str]
        modifier_log     : list[tuple]
        raw_scores       : dict
    """
    cfg      = models["config"]
    uci_cols = cfg["uci_features"]
    br_cols  = cfg["brfss_features"]
    weights  = cfg["fusion_weights"]
    thresh   = cfg["decision_threshold"]

    age      = int(user_input["age"])
    gender   = user_input.get("gender","female")

    # ── Stage 1: IDRS ───────────────────────────────────────────
    idrs_raw, idrs_norm, idrs_cat, idrs_bd = _compute_idrs(
        age=age,
        waist_cm=float(user_input.get("waist_cm", 80)),
        gender=gender,
        physical_activity=user_input.get("physical_activity","none"),
        family_history=bool(user_input.get("family_history", False))
    )

    # ── Stage 3: UCI Symptoms ───────────────────────────────────
    sym = {k: int(bool(user_input.get(k, False)))
           for k in uci_cols}
    sym_df = pd.DataFrame([sym]).reindex(columns=uci_cols,
                                          fill_value=0)
    stage3_prob = float(
        models["stage3"].predict_proba(sym_df)[0, 1]
    )

    # ── Stage 2: BRFSS Lifestyle ────────────────────────────────
    lif = {
        "HighBP"              : int(bool(user_input.get("high_bp",False))),
        "HighChol"            : int(bool(user_input.get("high_chol",False))),
        "BMI"                 : float(user_input.get("bmi", 22)),
        "Smoker"              : int(bool(user_input.get("smoker",False))),
        "Stroke"              : int(bool(user_input.get("stroke",False))),
        "HeartDiseaseorAttack": int(bool(user_input.get("heart_disease",False))),
        "PhysActivity"        : int(bool(user_input.get("phys_active",False))),
        "HvyAlcoholConsump"   : int(bool(user_input.get("heavy_alcohol",False))),
        "MentHlth"            : int(user_input.get("mental_health_days", 0)),
        "Sex"                 : _gender_to_brfss(gender),
        "Age"                 : _age_to_brfss(age),
    }
    lif_df = pd.DataFrame([lif]).reindex(columns=br_cols,
                                          fill_value=0)
    stage2_prob = float(
        models["stage2"].predict_proba(lif_df)[0, 1]
    )

    # ── Weighted fusion ─────────────────────────────────────────
    fused = (weights["idrs"]   * idrs_norm   +
             weights["stage2"] * stage2_prob +
             weights["stage3"] * stage3_prob)

    # Conflict guards
    if stage3_prob >= 0.70 and idrs_norm < 0.30:
        fused = max(fused, 0.40)
    if stage2_prob >= 0.80 and fused < 0.45:
        fused = max(fused, 0.45)

    # ── Stage 4: Clinical modifiers ─────────────────────────────
    clinical = {
        "pcos"                     : bool(user_input.get("pcos",False)),
        "acanthosis_nigricans"     : bool(user_input.get("acanthosis_nigricans",False)),
        "skin_tags"                : bool(user_input.get("skin_tags",False)),
        "nafld"                    : bool(user_input.get("nafld",False)),
        "sleep_apnea"              : bool(user_input.get("sleep_apnea",False)),
        "thyroid_condition"        : bool(user_input.get("thyroid_condition",False)),
        "chronic_stress_depression": bool(user_input.get("chronic_stress_depression",False)),
        "gestational_diabetes"     : bool(user_input.get("gestational_diabetes",False)),
        "poor_sleep"               : bool(user_input.get("poor_sleep",False)),
        "waist_hip_ratio"          : user_input.get("waist_hip_ratio", None),
    }
    final_score, mod_log = _apply_modifiers(fused, clinical, gender)
    category, emoji, advice = _get_category(final_score)

    # ── Driver attribution ──────────────────────────────────────
    NON_MOD = {"Age","Family history of diabetes"}
    mod_drv, nonmod_drv = [], []

    for feat, pts in idrs_bd.items():
        if pts > 0:
            (nonmod_drv if feat in NON_MOD
             else mod_drv).append(f"[IDRS] {feat}")

    for feat in uci_cols:
        if bool(user_input.get(feat, False)):
            mod_drv.append(
                f"[Symptom] {feat.replace('_',' ').title()}"
            )

    for feat, val in lif.items():
        if feat == "BMI" and val > 25:
            mod_drv.append(f"[Lifestyle] High BMI ({val:.1f})")
        elif feat == "Age":
            label = BRFSS_AGE_MAP.get(val,(0,0,"Unknown"))[2]
            nonmod_drv.append(f"[Lifestyle] Age {label}")
        elif feat in {"HeartDiseaseorAttack","Stroke"} and val==1:
            nonmod_drv.append(f"[Lifestyle] {feat}")
        elif val == 1:
            mod_drv.append(f"[Lifestyle] {feat}")

    for cond, boost, ev in mod_log:
        if any(x in cond.lower()
               for x in ["gestational","stroke","heart"]):
            nonmod_drv.append(f"[Clinical] {cond}")
        else:
            mod_drv.append(f"[Clinical] {cond}")

    return {
        "final_score_pct"      : round(final_score * 100, 1),
        "category"             : category,
        "emoji"                : emoji,
        "advice"               : advice,
        "component_scores"     : {
            "IDRS Clinical Baseline" : round(idrs_norm*100,1),
            "Lifestyle Risk (BRFSS)" : round(stage2_prob*100,1),
            "Symptom Burden (UCI)"   : round(stage3_prob*100,1),
            "Clinical Modifier Boost": round((final_score-fused)*100,1)
        },
        "idrs_raw"             : idrs_raw,
        "idrs_category"        : idrs_cat,
        "modifiable_drivers"   : list(dict.fromkeys(mod_drv))[:8],
        "nonmodifiable_drivers": list(dict.fromkeys(nonmod_drv))[:4],
        "modifier_log"         : mod_log,
        "raw_scores"           : {
            "idrs_norm"   : round(idrs_norm,3),
            "stage2_prob" : round(stage2_prob,3),
            "stage3_prob" : round(stage3_prob,3),
            "fused"       : round(fused,3),
            "final"       : round(final_score,3)
        }
    }
