
# Diabetes Risk Screener — Questionnaire Specification
## For Frontend/Backend Integration

This document defines every question the UI must ask the user,
the input type, validation rules, and the exact key name to use
when calling predict_risk().

---

## SECTION 1 — Basic Information

| Question | UI Element | Valid Values | API Key |
|---|---|---|---|
| How old are you? | Number input | 18–90 (integer) | `age` |
| What is your gender? | Radio button | Male / Female | `gender` |
| What is your waist circumference? (measure with a tape at navel level) | Number input | 50–150 cm (float) | `waist_cm` |
| What is your waist-to-hip ratio? (waist cm ÷ hip cm) | Number input | 0.5–1.5 (float) | `waist_hip_ratio` |
| How would you describe your physical activity? | Radio button | Regular (≥30 min daily) / Irregular / None/Sedentary | `physical_activity` → pass as "regular"/"irregular"/"none" |
| Does a parent or sibling have diabetes? | Checkbox | True / False | `family_history` |

---

## SECTION 2 — Lifestyle & Health Conditions

| Question | UI Element | Valid Values | API Key |
|---|---|---|---|
| What is your BMI? | Number input or calculator | 10–60 (float) | `bmi` |
| Have you been told by a doctor you have high blood pressure? | Checkbox | True / False | `high_bp` |
| Have you been told by a doctor you have high cholesterol? | Checkbox | True / False | `high_chol` |
| Have you ever had a heart attack or been diagnosed with heart disease? | Checkbox | True / False | `heart_disease` |
| Have you ever had a stroke? | Checkbox | True / False | `stroke` |
| Have you smoked at least 100 cigarettes in your lifetime? | Checkbox | True / False | `smoker` |
| Do you drink heavily? (men >14 drinks/week, women >7/week) | Checkbox | True / False | `heavy_alcohol` |
| Have you been physically active in the last 30 days? | Checkbox | True / False | `phys_active` |
| In the last 30 days, how many days was your mental health not good? | Slider | 0–30 (integer) | `mental_health_days` |

---

## SECTION 3 — Current Symptoms
*Ask user to check all that currently apply*

| Symptom | Plain English Label | API Key |
|---|---|---|
| Polyuria | Frequent urination | `polyuria` |
| Polydipsia | Excessive thirst | `polydipsia` |
| Polyphagia | Increased hunger | `polyphagia` |
| Sudden weight loss | Sudden unexplained weight loss | `sudden_weight_loss` |
| Weakness | Unusual weakness or fatigue | `weakness` |
| Genital thrush | Genital thrush / yeast infection | `genital_thrush` |
| Visual blurring | Blurred vision | `visual_blurring` |
| Itching | Unusual itching | `itching` |
| Irritability | Irritability / mood changes | `irritability` |
| Delayed healing | Slow wound healing | `delayed_healing` |
| Partial paresis | Muscle weakness / numbness | `partial_paresis` |
| Muscle stiffness | Muscle stiffness | `muscle_stiffness` |
| Alopecia | Hair loss | `alopecia` |
| Obesity | Clinically obese | `obesity` |

All symptom values: True / False (bool) or 1 / 0 (int)

---

## SECTION 4 — Clinical Signs & Medical History

| Question | UI Element | Valid Values | API Key |
|---|---|---|---|
| Do you have dark, velvety skin at your neck, armpits, or groin? | Checkbox | True / False | `acanthosis_nigricans` |
| Do you have skin tags? (small soft growths on skin) | Checkbox | True / False | `skin_tags` |
| Have you been diagnosed with PCOS? (females only — show conditionally) | Checkbox | True / False | `pcos` |
| Have you had gestational diabetes? (females only) | Checkbox | True / False | `gestational_diabetes` |
| Have you been diagnosed with fatty liver disease (NAFLD)? | Checkbox | True / False | `nafld` |
| Have you been diagnosed with sleep apnea? | Checkbox | True / False | `sleep_apnea` |
| Do you have a thyroid condition? | Checkbox | True / False | `thyroid_condition` |
| Do you experience chronic stress or depression? | Checkbox | True / False | `chronic_stress_depression` |
| Do you regularly get less than 6 hours of sleep? | Checkbox | True / False | `poor_sleep` |

---

## API OUTPUT — What To Display

```python
result = predict_risk(models, user_input)

result["final_score_pct"]       # float  → show as big number e.g. "73.5%"
result["category"]              # str    → "LOW"/"MODERATE"/"HIGH"/"VERY HIGH"
result["emoji"]                 # str    → 🟢 🟡 🟠 🔴
result["advice"]                # str    → one-line message to user
result["component_scores"]      # dict   → breakdown bar chart
result["modifiable_drivers"]    # list   → bullet points "what you can change"
result["nonmodifiable_drivers"] # list   → bullet points "fixed factors"
result["modifier_log"]          # list   → clinical signs that boosted score
```

## IMPORTANT NOTES FOR FRONTEND

1. PCOS and gestational diabetes questions should only show
   if gender == "Female"

2. BMI calculator helper recommended — many users won't know
   their BMI. Formula: weight(kg) / height(m)²

3. Waist-to-hip ratio helper recommended:
   Measure waist at navel, hip at widest point, divide.

4. Always show this disclaimer with results:
   "This is a non-invasive screening tool. It does NOT diagnose
    diabetes. Please consult a qualified physician."

5. Never show raw probability numbers to users —
   only show the category and percentage.
