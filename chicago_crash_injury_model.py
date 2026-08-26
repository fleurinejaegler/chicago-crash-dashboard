# %%
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "chicago_crashes_recent_sample_50000.csv")

# %%
print(df.dtypes)
# %%
print(df.isna().sum())

# %% Binary targets
df = df[df["INJURIES_TOTAL"].isna() == False].copy()
df = df[df["MOST_SEVERE_INJURY"].isna() == False].copy()
df["INJURY"] = (df["INJURIES_TOTAL"] > 0).astype(int)
df["SEVERE"] = df["MOST_SEVERE_INJURY"].isin(
    ["INCAPACITATING INJURY", "FATAL"]
).astype(int)
print(df["INJURY"].value_counts())
print(f"Injury rate: {df['INJURY'].mean() * 100:.2f}%")
print(df["SEVERE"].value_counts())
print(f"Severe injury rate: {df['SEVERE'].mean() * 100:.2f}%")

# %% Injury rate by condition
for col in ["CRASH_HOUR", "WEATHER_CONDITION", "LIGHTING_CONDITION",
            "ROADWAY_SURFACE_COND", "POSTED_SPEED_LIMIT", "FIRST_CRASH_TYPE"]:
    print(f"\n--- Injury rate by {col} ---")
    print(df.groupby(col)["INJURY"].mean())

# %% Features and labels
numeric = ["POSTED_SPEED_LIMIT", "NUM_UNITS", "CRASH_HOUR",
           "CRASH_DAY_OF_WEEK", "CRASH_MONTH"]
categorical = ["WEATHER_CONDITION", "LIGHTING_CONDITION", "ROADWAY_SURFACE_COND",
               "FIRST_CRASH_TYPE", "TRAFFICWAY_TYPE", "PRIM_CONTRIBUTORY_CAUSE"]

# One-hot encoding is fit inside the pipeline (not with pd.get_dummies) so a
# single new row at inference time gets the same columns the model was
# trained on, and the fitted pipeline can be persisted with joblib.
X = df[numeric + categorical]

# The threshold reported as "final" below -- picked once by eye from the
# validation sweep, since both targets are rare and 0.5 misses almost every
# positive case.
FINAL_THRESHOLD = 0.15


def make_model() -> Pipeline:
    # Numeric features are scaled so coefficients land on a comparable scale --
    # without it, POSTED_SPEED_LIMIT (15-55) and a 0/1 dummy aren't
    # comparable, and "top drivers by |coefficient|" would be meaningless.
    num_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    prep = ColumnTransformer([
        ("num", num_steps, numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))])


def fit_eval_save(target_col: str, save_name: str) -> Pipeline:
    """Train/validation/test split, fit, pick a threshold on validation,
    report it on the held-out test set, and persist the pipeline."""
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
    )
    # Hold out a validation set from the training data -- the threshold below
    # is chosen on validation, never on the test set, so the final test-set
    # numbers stay an honest estimate.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, shuffle=True, random_state=42, stratify=y_train
    )

    baseline = [0] * len(y_test)
    print(f"\n=== {target_col} ===")
    print(f"Baseline (always predict no {target_col.lower()}):")
    print(f"  Accuracy: {accuracy_score(y_test, baseline):.3f}")
    print(f"  Recall:   {recall_score(y_test, baseline, zero_division=0):.3f}")

    model = make_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\nLogistic regression @ 0.5 (test set):")
    print(f"  Accuracy:  {accuracy_score(y_test, predictions):.3f}")
    print(f"  Precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(y_test, predictions, zero_division=0):.3f}")

    # Both targets are rare, so few crashes reach the default 0.5 cutoff and
    # recall is low there. Sweep on validation, not test.
    val_probabilities = model.predict_proba(X_val)[:, 1]
    print("\nThreshold sweep (validation set)")
    print("Threshold | Accuracy | Precision | Recall")
    for t in [0.5, 0.4, 0.3, 0.2, 0.15]:
        p = (val_probabilities > t).astype(int)
        print(f"   {t:.2f}   |  {accuracy_score(y_val, p):.3f}   |"
              f"   {precision_score(y_val, p, zero_division=0):.3f}   |"
              f" {recall_score(y_val, p, zero_division=0):.3f}")

    test_final = (model.predict_proba(X_test)[:, 1] > FINAL_THRESHOLD).astype(int)
    print(f"\nTest set at threshold {FINAL_THRESHOLD} (chosen on validation above):")
    print(f"  Accuracy:  {accuracy_score(y_test, test_final):.3f}")
    print(f"  Precision: {precision_score(y_test, test_final, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(y_test, test_final, zero_division=0):.3f}")

    joblib.dump(model, MODEL_DIR / save_name)
    print(f"Saved -> {MODEL_DIR / save_name}")
    return model


# %% Injury model
injury_model = fit_eval_save("INJURY", "chicago_injury_lr.joblib")

# %% Severity model (incapacitating or fatal)
severe_model = fit_eval_save("SEVERE", "chicago_severe_lr.joblib")

# %% How much would a more flexible model buy us, on the injury target?
X_train, X_test, y_train, y_test = train_test_split(
    X, df["INJURY"], test_size=0.2, shuffle=True, random_state=42, stratify=df["INJURY"]
)
prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
])

print(f"\n{'Model':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}")
for name, clf in [
    ("Logistic Regression", LogisticRegression(max_iter=2000)),
    ("Random Forest", RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42)),
    ("Gradient Boosting", HistGradientBoostingClassifier(random_state=42)),
]:
    pipe = Pipeline([("prep", prep), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(f"{name:<22}{accuracy_score(y_test, pred):>10.3f}"
          f"{precision_score(y_test, pred, zero_division=0):>11.3f}"
          f"{recall_score(y_test, pred, zero_division=0):>9.3f}")
