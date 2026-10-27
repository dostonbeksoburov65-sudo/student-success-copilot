# ml/ml_model.py

import joblib
import os
import pandas as pd

# -------------------------------------------------------
# Paths — relative to project root
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _path(filename):
    return os.path.join(BASE_DIR, filename)

# -------------------------------------------------------
# Load models
# -------------------------------------------------------
model_risk  = joblib.load(_path('model_risk.pkl'))
model_label = joblib.load(_path('model_label.pkl'))
model_hours = joblib.load(_path('model_hours.pkl'))
scaler      = joblib.load(_path('scaler.pkl'))

FEATURES = ['subjects', 'deadlines', 'days_to_deadline',
            'hours_needed', 'avail_hours', 'confidence',
            'stress', 'sleep', 'gender_code']

print("ML models loaded successfully.")

# -------------------------------------------------------
# Gender encoder
# -------------------------------------------------------
def encode_gender(gender_str):
    mapping = {
        'female':             1.0,
        'non-binary':         0.5,
        'male':               0.0,
        'prefer not to say':  0.0
    }
    return mapping.get(gender_str.lower().strip(), 0.0)

# -------------------------------------------------------
# Main predict function
# -------------------------------------------------------
def predict(subjects, deadlines, days_to_deadline,
            hours_needed, avail_hours, confidence,
            stress, sleep, gender):
    """
    Input  : raw student values
    Output : dict with risk_score, risk_label, rec_hours
    """
    gender_code = encode_gender(gender)

    input_df = pd.DataFrame([[
        subjects, deadlines, days_to_deadline,
        hours_needed, avail_hours, confidence,
        stress, sleep, gender_code
    ]], columns=FEATURES)

    input_scaled = scaler.transform(input_df)

    risk_score = float(model_risk.predict(input_scaled)[0])
    risk_label = model_label.predict(input_scaled)[0]
    rec_hours  = float(model_hours.predict(input_scaled)[0])

    risk_score = round(max(0, min(100, risk_score)), 1)
    rec_hours  = round(max(5, min(45, rec_hours)), 1)

    return {
        'risk_score': risk_score,
        'risk_label': risk_label,
        'rec_hours':  rec_hours
    }

# -------------------------------------------------------
# Test
# -------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Scenario 1: Struggling Student ===")
    r = predict(
        subjects=7, deadlines=4, days_to_deadline=2,
        hours_needed=30, avail_hours=2.0, confidence=20,
        stress=9, sleep=5.0, gender='female'
    )
    print(f"  Risk Score : {r['risk_score']}/100")
    print(f"  Risk Label : {r['risk_label']}")
    print(f"  Rec. Hours : {r['rec_hours']}h/week")

    print("\n=== Scenario 2: Well-Prepared Student ===")
    r = predict(
        subjects=3, deadlines=1, days_to_deadline=6,
        hours_needed=10, avail_hours=6.0, confidence=85,
        stress=2, sleep=8.0, gender='male'
    )
    print(f"  Risk Score : {r['risk_score']}/100")
    print(f"  Risk Label : {r['risk_label']}")
    print(f"  Rec. Hours : {r['rec_hours']}h/week")