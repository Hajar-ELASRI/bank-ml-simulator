"""Train and persist the three bank marketing classification pipelines."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "bank.csv"
MODELS_DIR = ROOT / "models"
TARGET = "y"
RANDOM_STATE = 42


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    """Create a fresh preprocessor so every saved model is a standalone pipeline."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def calculate_metrics(y_true: pd.Series, y_pred: list[str]) -> dict:
    """Return JSON-serializable binary classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label="yes", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label="yes", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label="yes", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["no", "yes"]).tolist(),
        "confusion_matrix_labels": ["no", "yes"],
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, sep=";")
    if TARGET not in data.columns:
        raise ValueError(f"La colonne cible '{TARGET}' est absente du dataset.")
    if data.isna().any().any():
        raise ValueError("Le dataset contient des valeurs manquantes.")

    X = data.drop(columns=TARGET)
    y = data[TARGET]
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    estimators = {
        "svm": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            # A single worker also works in restricted Windows environments.
            n_jobs=1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=600,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE,
        ),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    all_metrics: dict[str, dict] = {}

    for name, estimator in estimators.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessing", build_preprocessor(numeric_columns, categorical_columns)),
                ("model", estimator),
            ]
        )
        print(f"Entraînement de {name}...")
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        all_metrics[name] = calculate_metrics(y_test, predictions)
        joblib.dump(pipeline, MODELS_DIR / f"{name}.pkl")
        print(f"  accuracy = {all_metrics[name]['accuracy']:.4f}")

    schema = {
        "target": TARGET,
        "feature_order": X.columns.tolist(),
        "numeric_features": {
            column: {
                "min": float(X[column].min()),
                "max": float(X[column].max()),
                "default": float(X[column].median()),
                "integer": bool(pd.api.types.is_integer_dtype(X[column])),
            }
            for column in numeric_columns
        },
        "categorical_features": {
            column: sorted(X[column].astype(str).unique().tolist())
            for column in categorical_columns
        },
        "dataset_rows": len(data),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": RANDOM_STATE,
    }

    (MODELS_DIR / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (MODELS_DIR / "schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Modèles et métriques sauvegardés dans {MODELS_DIR}")


if __name__ == "__main__":
    main()
