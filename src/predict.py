"""
predict.py
==========
Inférence à partir d'un modèle sauvegardé : prédiction unitaire (une
transaction) ou en batch (un DataFrame). Utilisé par l'application
Streamlit (app/app.py).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config, features


REQUIRED_INPUT_COLUMNS = [
    config.TIME_COL,
    *[f"V{i}" for i in range(1, 29)],
    config.AMOUNT_COL,
]


def load_model(path: Path):
    """Charge un modèle sérialisé (.joblib) sauvegardé par src/train.py."""
    return joblib.load(path)


def validate_batch_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the input schema and numeric values before model inference."""
    if df.empty:
        raise ValueError("Le fichier ne contient aucune transaction.")
    if not df.columns.is_unique:
        raise ValueError("Le fichier contient des noms de colonnes en double.")

    missing = [
        column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns
    ]
    if missing:
        raise ValueError(
            "Colonnes obligatoires manquantes : " + ", ".join(missing)
        )

    validated = df.copy()
    numeric_columns = REQUIRED_INPUT_COLUMNS.copy()
    if config.TARGET_COL in validated.columns:
        numeric_columns.append(config.TARGET_COL)

    for column in numeric_columns:
        values = pd.to_numeric(validated[column], errors="coerce")
        has_missing = values.isna().any()
        has_infinite = not np.isfinite(values.to_numpy(dtype=float)).all()
        if has_missing or has_infinite:
            raise ValueError(
                f"La colonne « {column} » contient des valeurs vides ou non numériques."
            )
        validated[column] = values

    if (validated[config.AMOUNT_COL] < 0).any():
        raise ValueError("Les montants doivent être positifs ou nuls.")

    if config.TARGET_COL in validated.columns:
        if not validated[config.TARGET_COL].isin([0, 1]).all():
            raise ValueError("La colonne Class facultative doit contenir uniquement 0 ou 1.")
        validated[config.TARGET_COL] = validated[config.TARGET_COL].astype(int)

    return validated


def _model_features(df_feat: pd.DataFrame, model) -> pd.DataFrame:
    default_columns = REQUIRED_INPUT_COLUMNS + ["TransactionHour", "TransactionDay"]
    columns = list(getattr(model, "feature_names_in_", default_columns))
    missing = [column for column in columns if column not in df_feat.columns]
    if missing:
        raise ValueError(
            "Variables attendues par le modèle absentes : " + ", ".join(missing)
        )
    return df_feat[columns]


def predict_single(transaction_dict: dict, model, scaler=None, threshold: float = 0.5):
    """
    Prédit la classe et la probabilité de fraude pour UNE transaction.

    Parameters
    ----------
    transaction_dict : dict
        Doit contenir Time, Amount, V1..V28 (une valeur par variable).
    model : modèle scikit-learn / XGBoost / LightGBM entraîné.
    scaler : StandardScaler déjà ajusté sur le train (cf. preprocessing.py).
        Si fourni, on standardise Time/Amount/TransactionHour/TransactionDay
        de la même manière qu'à l'entraînement.
    threshold : seuil de décision (idéalement le seuil optimal issu de
        evaluate.plot_cost_curve, pas 0.5 par défaut).

    Returns
    -------
    (classe_predite: int, proba_fraude: float)
    """
    df = validate_batch_input(pd.DataFrame([transaction_dict]))
    df = features.create_time_features(df)

    if scaler is not None:
        cols = [c for c in config.SCALED_COLS if c in df.columns]
        df[cols] = scaler.transform(df[cols])

    df = _model_features(df, model)

    proba = model.predict_proba(df)[0, 1]
    classe = int(proba >= threshold)
    return classe, float(proba)


def predict_batch(df: pd.DataFrame, model, scaler=None, threshold: float = 0.5) -> pd.DataFrame:
    """
    Prédit classe + probabilité pour un DataFrame de transactions.

    Retourne le DataFrame d'entrée enrichi des colonnes `proba_fraude` et
    `classe_predite`.
    """
    df = validate_batch_input(df)
    df_feat = features.create_time_features(df.copy())

    if scaler is not None:
        cols = [c for c in config.SCALED_COLS if c in df_feat.columns]
        df_feat[cols] = scaler.transform(df_feat[cols])

    X = _model_features(df_feat, model)

    proba = model.predict_proba(X)[:, 1]
    result = df.copy()
    result["proba_fraude"] = proba
    result["classe_predite"] = (proba >= threshold).astype(int)
    return result
