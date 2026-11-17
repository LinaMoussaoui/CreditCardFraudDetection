"""Synthetic data and inference helpers for the simple Streamlit demo."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TRANSACTION_TYPES = (
    "paiement_carte",
    "achat_en_ligne",
    "virement",
    "retrait_especes",
)
INPUT_COLUMNS = (
    "montant",
    "type_transaction",
    "heure",
    "international",
    "nouveau_beneficiaire",
)
BOOLEAN_COLUMNS = ("international", "nouveau_beneficiaire")


def make_synthetic_transactions(
    n_samples: int = 12000, random_state: int = 42
) -> pd.DataFrame:
    """Create a reproducible, clearly synthetic dataset for model training."""
    rng = np.random.default_rng(random_state)
    transaction_types = rng.choice(
        TRANSACTION_TYPES,
        size=n_samples,
        p=[0.42, 0.24, 0.20, 0.14],
    )
    amounts = rng.lognormal(mean=4.2, sigma=1.15, size=n_samples)
    hours = rng.integers(0, 24, size=n_samples)
    international = rng.random(n_samples) < 0.08
    new_payee = rng.random(n_samples) < 0.18

    score = np.full(n_samples, -5.0)
    score += np.log1p(amounts) * 0.12
    score += np.isin(hours, [0, 1, 2, 3, 4, 5]) * 1.1
    score += international * 1.0
    score += new_payee * 1.0
    score += np.isin(transaction_types, ["virement", "retrait_especes"]) * 0.5
    score += (amounts > 1500) * 1.3
    score += (international & new_payee) * 1.5
    score += ((amounts > 1000) & new_payee) * 1.0

    fraud_probability = 1 / (1 + np.exp(-score))
    is_fraud = rng.random(n_samples) < fraud_probability

    return pd.DataFrame(
        {
            "montant": amounts.round(2),
            "type_transaction": transaction_types,
            "heure": hours,
            "international": international,
            "nouveau_beneficiaire": new_payee,
            "fraude": is_fraud.astype(int),
        }
    )


def train_demo_model() -> Pipeline:
    """Train a small classifier on reproducible simulated transactions."""
    training_data = make_synthetic_transactions()
    preprocessing = ColumnTransformer(
        [
            (
                "type",
                OneHotEncoder(handle_unknown="ignore"),
                ["type_transaction"],
            )
        ],
        remainder="passthrough",
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=160,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(training_data[list(INPUT_COLUMNS)], training_data["fraude"])
    return model


def _parse_boolean_column(values: pd.Series, column: str) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "oui": True,
        "non": False,
    }
    parsed = normalized.map(mapping)
    invalid = parsed.isna()
    if invalid.any():
        raise ValueError(
            f"« {column} » must contain true/false, yes/no, or 1/0."
        )
    return parsed.astype(bool)


def validate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize human-readable transaction columns."""
    if transactions.empty:
        raise ValueError("The CSV contains no transactions.")
    if not transactions.columns.is_unique:
        raise ValueError("The CSV has duplicate column names.")

    missing = [column for column in INPUT_COLUMNS if column not in transactions]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    validated = transactions.copy()
    for column in ("montant", "heure"):
        numeric_values = pd.to_numeric(validated[column], errors="coerce")
        if not np.isfinite(numeric_values.to_numpy(dtype=float)).all():
            raise ValueError(f"« {column} » must contain numeric values.")
        validated[column] = numeric_values

    if (validated["montant"] < 0).any():
        raise ValueError("Amounts must be zero or greater.")
    if (
        (validated["heure"] < 0)
        | (validated["heure"] > 23)
        | (validated["heure"] % 1 != 0)
    ).any():
        raise ValueError("« heure » must be a whole number from 0 to 23.")
    validated["heure"] = validated["heure"].astype(int)

    validated["type_transaction"] = (
        validated["type_transaction"].astype(str).str.strip().str.lower()
    )
    invalid_types = sorted(
        set(validated["type_transaction"]) - set(TRANSACTION_TYPES)
    )
    if invalid_types:
        raise ValueError(
            "Unknown transaction type(s): "
            + ", ".join(invalid_types)
            + ". Allowed values: "
            + ", ".join(TRANSACTION_TYPES)
            + "."
        )

    for column in BOOLEAN_COLUMNS:
        validated[column] = _parse_boolean_column(validated[column], column)

    return validated


def score_transactions(
    model: Pipeline,
    transactions: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return validated input transactions with risk scores and alert flags."""
    if not 0 <= threshold <= 1:
        raise ValueError("The alert threshold must be between 0 and 1.")
    validated = validate_transactions(transactions)
    probabilities = model.predict_proba(validated[list(INPUT_COLUMNS)])[:, 1]

    results = validated.copy()
    results["probabilite_fraude"] = probabilities
    results["alerte"] = probabilities >= threshold
    results["decision"] = np.where(
        results["alerte"], "A examiner", "Pas d'alerte"
    )
    return results


def example_transactions_csv() -> bytes:
    """Provide a compact CSV sample that can be uploaded to the app."""
    examples = pd.DataFrame(
        [
            [125.00, "paiement_carte", 14, False, False],
            [58.90, "achat_en_ligne", 19, False, True],
            [2400.00, "virement", 2, True, True],
            [35.00, "retrait_especes", 3, False, False],
            [820.00, "virement", 11, False, True],
        ],
        columns=INPUT_COLUMNS,
    )
    return examples.to_csv(index=False).encode("utf-8")
