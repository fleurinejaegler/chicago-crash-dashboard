# %%
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
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

X = df[numeric + categorical]


def make_model() -> Pipeline:
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
    """Train/test split, fit, report threshold sweep, persist the pipeline."""
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
    )

    model = make_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n=== {target_col} ===")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.3f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.3f}")

    # Both targets are rare, so few crashes reach the default 0.5 cutoff and
    # recall is low there.
    probabilities = model.predict_proba(X_test)[:, 1]
    print("\nThreshold | Accuracy | Precision | Recall")
    for t in [0.5, 0.4, 0.3, 0.2, 0.15]:
        p = (probabilities > t).astype(int)
        print(f"   {t:.2f}   |  {accuracy_score(y_test, p):.3f}   |"
              f"   {precision_score(y_test, p, zero_division=0):.3f}   |"
              f" {recall_score(y_test, p, zero_division=0):.3f}")

    joblib.dump(model, MODEL_DIR / save_name)
    print(f"Saved -> {MODEL_DIR / save_name}")
    return model


# %% Injury model
injury_model = fit_eval_save("INJURY", "chicago_injury_lr.joblib")

# %% Severity model (incapacitating or fatal)
severe_model = fit_eval_save("SEVERE", "chicago_severe_lr.joblib")