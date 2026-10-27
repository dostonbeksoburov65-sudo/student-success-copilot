
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    classification_report
)
import joblib

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'student_dataset.csv')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

def model_path(name):
    return os.path.join(BASE_DIR, f'{name}.pkl')

# -------------------------------------------------------
# FEATURES
# -------------------------------------------------------
FEATURES = [
    'subjects', 'deadlines', 'days_to_deadline',
    'hours_needed', 'avail_hours', 'confidence',
    'stress', 'sleep', 'gender_code'
]

# -------------------------------------------------------
# 1. LOAD DATASET
# -------------------------------------------------------
print("=" * 55)
print("  STUDENT SUCCESS COPILOT — MODEL TRAINING")
print("=" * 55)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
print(f"    Columns: {list(df.columns)}")

# -------------------------------------------------------
# 2. PREPARE FEATURES AND TARGETS
# -------------------------------------------------------
X = df[FEATURES]

# Regression targets
y_risk  = df['risk_score']                  # continuous 0-100
y_hours = df['hours_needed']                # continuous (study hours proxy)

# Classification target
y_label = df['risk_label']                  # LOW / MODERATE / HIGH / CRITICAL

print(f"\n[2] Features : {FEATURES}")
print(f"    Targets  : risk_score (regression), risk_label (classification), hours_needed (regression)")

# -------------------------------------------------------
# 3. TRAIN / TEST SPLIT  --  80% train, 20% test
# -------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.20

(X_train, X_test,
 y_risk_train,  y_risk_test,
 y_hours_train, y_hours_test,
 y_label_train, y_label_test) = train_test_split(
    X, y_risk, y_hours, y_label,
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = y_label        # keep class distribution balanced
)

print(f"\n[3] Train / Test split")
print(f"    Total rows  : {len(df)}")
print(f"    Train rows  : {len(X_train)}  ({int((1 - TEST_SIZE) * 100)}%)")
print(f"    Test rows   : {len(X_test)}   ({int(TEST_SIZE * 100)}%)")
print(f"    Random state: {RANDOM_STATE}")
print(f"    Stratified  : yes (balanced risk_label classes)")

# -------------------------------------------------------
# 4. SCALE FEATURES
# -------------------------------------------------------
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on train set
X_test_scaled  = scaler.transform(X_test)         # transform test with same scaler

print(f"\n[4] Feature scaling: StandardScaler fitted on training data only")

# -------------------------------------------------------
# 5a. TRAIN: RandomForestRegressor -- risk_score
# -------------------------------------------------------
print("\n[5] Training models...")
print("    -> RandomForestRegressor (risk_score) ...")

model_risk = RandomForestRegressor(
    n_estimators = 100,
    random_state = RANDOM_STATE
)
model_risk.fit(X_train_scaled, y_risk_train)

# Evaluate
y_risk_pred = model_risk.predict(X_test_scaled)
risk_mae    = round(mean_absolute_error(y_risk_test, y_risk_pred), 3)
risk_mse    = round(mean_squared_error(y_risk_test,  y_risk_pred), 3)
risk_r2     = round(r2_score(y_risk_test, y_risk_pred), 3)

print(f"\n    --- Risk Score Regression (RandomForest) ---")
print(f"    MAE  : {risk_mae}   (mean absolute error, lower is better)")
print(f"    MSE  : {risk_mse}   (mean squared error,  lower is better)")
print(f"    R2   : {risk_r2}    (1.0 = perfect fit)")

# -------------------------------------------------------
# 5b. TRAIN: RandomForestClassifier -- risk_label
# -------------------------------------------------------
print("\n    -> RandomForestClassifier (risk_label) ...")

model_label = RandomForestClassifier(
    n_estimators = 100,
    random_state = RANDOM_STATE
)
model_label.fit(X_train_scaled, y_label_train)

# Evaluate
y_label_pred = model_label.predict(X_test_scaled)
label_acc    = round(accuracy_score(y_label_test, y_label_pred), 3)
label_f1     = round(f1_score(y_label_test, y_label_pred, average='weighted'), 3)

print(f"\n    --- Risk Label Classification (RandomForest) ---")
print(f"    Accuracy : {label_acc}  (fraction of correct predictions)")
print(f"    F1 Score : {label_f1}  (weighted average across classes)")
print(f"\n    Per-class breakdown:")
print(classification_report(y_label_test, y_label_pred, zero_division=0))

# -------------------------------------------------------
# 5c. TRAIN: LinearRegression -- recommended hours
# -------------------------------------------------------
print("    -> LinearRegression (recommended study hours) ...")

model_hours = LinearRegression()
model_hours.fit(X_train_scaled, y_hours_train)

# Evaluate
y_hours_pred = model_hours.predict(X_test_scaled)
hours_mae    = round(mean_absolute_error(y_hours_test, y_hours_pred), 3)
hours_mse    = round(mean_squared_error(y_hours_test,  y_hours_pred), 3)
hours_r2     = round(r2_score(y_hours_test, y_hours_pred), 3)

print(f"\n    --- Recommended Hours Regression (LinearRegression) ---")
print(f"    MAE  : {hours_mae}")
print(f"    MSE  : {hours_mse}")
print(f"    R2   : {hours_r2}")

# -------------------------------------------------------
# 6. LIMITATIONS DISCUSSION (printed for report reference)
# -------------------------------------------------------
print("\n[6] Limitations")
print("    - Dataset is synthetic (200 rows). Real student data may differ.")
print("    - hours_needed is used as study hours proxy, not a true label.")
print("    - Gender encoding (0/0.5/1) is a simplification.")
print("    - RandomForest may overfit on small datasets without cross-validation.")
print("    - R2=1.0 on hours regression likely reflects data leakage -- treat with caution.")

# -------------------------------------------------------
# 7. SAVE MODELS AND SCALER
# -------------------------------------------------------
joblib.dump(scaler,      SCALER_PATH)
joblib.dump(model_risk,  model_path('model_risk'))
joblib.dump(model_label, model_path('model_label'))
joblib.dump(model_hours, model_path('model_hours'))

print("\n[7] Models saved to project root:")
print(f"    scaler.pkl      -> {SCALER_PATH}")
print(f"    model_risk.pkl  -> {model_path('model_risk')}")
print(f"    model_label.pkl -> {model_path('model_label')}")
print(f"    model_hours.pkl -> {model_path('model_hours')}")

print("\n" + "=" * 55)
print("  Training complete.")
print("=" * 55)
