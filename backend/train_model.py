"""Train the screening pipeline and produce public held-out accuracy metrics.

Run:
    python train_model.py

Produces:
    model/pipeline.joblib   -> trained preprocessing and prediction pipeline
    model/metrics.json      -> test accuracy and sample counts
"""

import json
import os

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_PATH = os.path.join(os.path.dirname(__file__), "heart.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "HeartDisease"
NUMERIC_FEATURES = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    # A handful of rows have Cholesterol == 0, which is a data-entry artifact, not a
    # real reading. Mark it missing; the training pipeline imputes from training rows.
    df["Cholesterol"] = df["Cholesterol"].replace(0, float("nan"))
    df["Cholesterol"] = pd.to_numeric(df["Cholesterol"], errors="coerce")
    return df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def main():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        random_state=43,
        stratify=y_development,
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
    }
    if HAS_XGB:
        candidates["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
        )

    validation_f1 = {}

    for name, clf in candidates.items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        validation_f1[name] = f1_score(y_validation, pipe.predict(X_validation))

    # Pick the best model by F1 (balances precision/recall, better than raw accuracy
    # for a screening task) without using the final held-out test set for selection.
    best_name = max(validation_f1, key=validation_f1.__getitem__)
    best_pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("clf", clone(candidates[best_name])),
        ]
    )
    best_pipeline.fit(X_development, y_development)
    test_accuracy = round(accuracy_score(y_test, best_pipeline.predict(X_test)), 4)

    joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "pipeline.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": test_accuracy,
                "dataset_samples": len(df),
                "test_samples": len(y_test),
            },
            f,
            indent=2,
        )

    print(f"\nHeld-out test accuracy: {test_accuracy:.1%}")
    print(f"Saved prediction pipeline -> {MODEL_DIR}/pipeline.joblib")
    print(f"Saved public metrics -> {MODEL_DIR}/metrics.json")


if __name__ == "__main__":
    main()
