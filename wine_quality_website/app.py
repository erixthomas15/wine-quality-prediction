import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "wine_quality_random_forest.pkl")
DATA_PATH = os.path.join(BASE, "WineQT_batch_id_dataset.csv")

app = Flask(__name__)
model = joblib.load(MODEL_PATH)
dataset = pd.read_csv(DATA_PATH)

BASE_FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "ph", "sulphates", "alcohol"
]
MODEL_FEATURES = list(model.feature_names_in_)

# The supplied preprocessing is IQR capping followed by StandardScaler.
# We reproduce the same transformation from the original dataset.
def fit_preprocessor(df):
    params = {}
    for c in BASE_FEATURES:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        capped = df[c].clip(lo, hi)
        mean = capped.mean()
        scale = capped.std(ddof=0)
        params[c] = (lo, hi, mean, scale)
    return params

PREP = fit_preprocessor(dataset)

def engineered_row(raw):
    x = dict(raw)
    x["bound_sulfur_dioxide"] = x["total_sulfur_dioxide"] - x["free_sulfur_dioxide"]
    x["acidity_ratio"] = x["fixed_acidity"] / max(x["volatile_acidity"], 1e-9)
    x["alcohol_density"] = x["alcohol"] / max(x["density"], 1e-9)
    x["sugar_alcohol"] = x["residual_sugar"] / max(x["alcohol"], 1e-9)
    return x

# Fit the same type of transformation for engineered features on the source data.
eng_df = dataset[BASE_FEATURES].copy()
eng_df["bound_sulfur_dioxide"] = eng_df["total_sulfur_dioxide"] - eng_df["free_sulfur_dioxide"]
eng_df["acidity_ratio"] = eng_df["fixed_acidity"] / eng_df["volatile_acidity"].replace(0, np.nan)
eng_df["alcohol_density"] = eng_df["alcohol"] / eng_df["density"].replace(0, np.nan)
eng_df["sugar_alcohol"] = eng_df["residual_sugar"] / eng_df["alcohol"].replace(0, np.nan)
ENG_PREP = {}
for c in ["bound_sulfur_dioxide","acidity_ratio","alcohol_density","sugar_alcohol"]:
    s = eng_df[c].replace([np.inf, -np.inf], np.nan).fillna(eng_df[c].median())
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3-q1
    lo, hi = q1-1.5*iqr, q3+1.5*iqr
    capped = s.clip(lo, hi)
    ENG_PREP[c] = (lo, hi, capped.mean(), capped.std(ddof=0))

def transform(raw):
    row = engineered_row(raw)
    out = {}
    for c in BASE_FEATURES:
        lo, hi, mean, scale = PREP[c]
        v = float(np.clip(row[c], lo, hi))
        out[c] = (v - mean) / scale
    for c in ["bound_sulfur_dioxide","acidity_ratio","alcohol_density","sugar_alcohol"]:
        lo, hi, mean, scale = ENG_PREP[c]
        v = float(np.clip(row[c], lo, hi))
        out[c] = (v - mean) / scale
    return pd.DataFrame([[out[c] for c in MODEL_FEATURES]], columns=MODEL_FEATURES)

@app.route("/")
def home():
    stats = {
        "rows": len(dataset),
        "features": len(BASE_FEATURES),
        "quality_min": int(dataset.quality.min()),
        "quality_max": int(dataset.quality.max()),
        "quality_counts": {str(k): int(v) for k,v in dataset.quality.value_counts().sort_index().items()},
        "avg_alcohol": {str(k): round(float(v),2) for k,v in dataset.groupby("quality").alcohol.mean().items()},
        "avg_volatile": {str(k): round(float(v),3) for k,v in dataset.groupby("quality").volatile_acidity.mean().items()},
    }
    return render_template("index.html", stats=stats)

@app.post("/predict")
def predict():
    try:
        raw = {c: float(request.form[c]) for c in BASE_FEATURES}
        X = transform(raw)
        pred = int(model.predict(X)[0])
        probs = model.predict_proba(X)[0]
        classes = [int(c) for c in model.classes_]
        probabilities = {str(c): round(float(p)*100, 2) for c,p in zip(classes, probs)}
        return jsonify({"prediction": pred, "probabilities": probabilities})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
