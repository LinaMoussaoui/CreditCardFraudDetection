"""
train.py
========
Entraînement des modèles et orchestration des expériences
(modèle x stratégie de rééquilibrage), avec journalisation dans
reports/resultats.csv et sauvegarde des modèles dans models/.
"""

import time
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src import config, data_loader, features, preprocessing, evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_model(model_name: str, X_train, y_train, params: dict, class_weight=None, scale_pos_weight=None):
    """
    Instancie et entraîne un modèle selon `model_name`.

    Modèles supportés : dummy, logreg, decision_tree, random_forest,
    xgboost, lightgbm.

    `class_weight` / `scale_pos_weight` sont injectés uniquement pour la
    stratégie "class_weight" (cf. preprocessing.handle_imbalance) ; ils sont
    ignorés si la stratégie de rééquilibrage a déjà modifié les données
    (undersample / smote), pour éviter de pondérer deux fois.
    """
    params = dict(params)  # copie défensive

    if model_name == "dummy":
        model = DummyClassifier(**params)

    elif model_name == "logreg":
        if class_weight is not None:
            params["class_weight"] = class_weight
        model = LogisticRegression(**params)

    elif model_name == "decision_tree":
        if class_weight is not None:
            params["class_weight"] = class_weight
        model = DecisionTreeClassifier(**params)

    elif model_name == "random_forest":
        if class_weight is not None:
            params["class_weight"] = class_weight
        model = RandomForestClassifier(**params)

    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight
        model = XGBClassifier(**params)

    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        if class_weight is not None:
            # LightGBM accepte directement un dict class_weight
            params["class_weight"] = class_weight
        model = LGBMClassifier(**params)

    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start
    logger.info("%s entraîné en %.2fs", model_name, training_time)

    return model, training_time


def run_experiment(model_name: str, imbalance_strategy: str, params: dict, data: dict, threshold: float = 0.5):
    """
    Pipeline complet pour UNE configuration (modèle x stratégie) :
      1. Récupère les splits prétraités (déjà standardisés) depuis `data`.
      2. Applique la stratégie de rééquilibrage sur le train uniquement.
      3. Entraîne le modèle.
      4. Évalue sur le jeu de test.
      5. Journalise le résultat dans reports/resultats.csv.
      6. Sauvegarde le modèle entraîné dans models/.

    Parameters
    ----------
    data : dict
        Doit contenir les clés X_train, y_train, X_test, y_test
        (déjà nettoyés + standardisés, cf. notebooks/02_preparation.ipynb).
    """
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    amount_test = data.get("amount_test")  # montants réels (non standardisés), pour le coût métier

    X_train_res, y_train_res, extra = preprocessing.handle_imbalance(
        X_train, y_train, strategy=imbalance_strategy
    )

    model, training_time = train_model(
        model_name,
        X_train_res,
        y_train_res,
        params,
        class_weight=extra.get("class_weight") if imbalance_strategy == "class_weight" else None,
        scale_pos_weight=extra.get("scale_pos_weight") if imbalance_strategy == "class_weight" else None,
    )

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # DummyClassifier "most_frequent" n'a pas de proba informative
        y_proba = model.predict(X_test).astype(float)

    metrics = evaluate.compute_metrics(y_test, y_proba, threshold=threshold)

    # Coût métier conforme à l'énoncé officiel : le coût d'un FN est le
    # montant réel de la transaction manquée, pas un forfait (voir
    # src/evaluate.py::plot_cost_curve pour le détail et la justification).
    y_pred_at_threshold = (np.asarray(y_proba) >= threshold).astype(int)
    y_test_arr = np.asarray(y_test)
    fn_mask = (y_test_arr == 1) & (y_pred_at_threshold == 0)
    if amount_test is not None:
        cost_total = np.asarray(amount_test)[fn_mask].sum() + metrics["fp"] * config.COUT_FP
    else:
        cost_total = metrics["fn"] * config.COUT_FN_FALLBACK + metrics["fp"] * config.COUT_FP

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "imbalance_strategy": imbalance_strategy,
        "threshold": threshold,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "cost_total": cost_total,
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
        "training_time_sec": training_time,
    }
    _append_result(row)

    model_filename = config.MODELS_DIR / f"{model_name}_{imbalance_strategy}.joblib"
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_filename)
    logger.info("Modèle sauvegardé : %s", model_filename)

    return model, row


def _append_result(row: dict) -> None:
    """Ajoute une ligne de résultats à reports/resultats.csv (créé si absent)."""
    config.RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if config.RESULTS_CSV.exists():
        df_row.to_csv(config.RESULTS_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(config.RESULTS_CSV, mode="w", header=True, index=False)


def _prepare_data() -> dict:
    """
    Charge les splits temporels déjà sauvegardés (parquet), applique le
    feature engineering et la standardisation (fit sur train uniquement),
    et retourne un dict prêt pour run_experiment.
    """
    train_df = data_loader.load_processed(config.TRAIN_PATH)
    val_df = data_loader.load_processed(config.VAL_PATH)
    test_df = data_loader.load_processed(config.TEST_PATH)

    train_df = features.create_time_features(train_df)
    val_df = features.create_time_features(val_df)
    test_df = features.create_time_features(test_df)

    X_train, y_train = features.select_features(train_df)
    X_val, y_val = features.select_features(val_df)
    X_test, y_test = features.select_features(test_df)

    # On conserve les montants RÉELS (non standardisés) du test AVANT scaling :
    # nécessaires pour calculer le coût métier des faux négatifs en euros
    # (cf. evaluate.plot_cost_curve), qui n'a plus de sens une fois `Amount`
    # standardisé par le StandardScaler.
    amount_test_raw = X_test[config.AMOUNT_COL].copy().reset_index(drop=True)

    X_train, X_val, X_test, scaler = preprocessing.scale_features(X_train, X_val, X_test)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "amount_test": amount_test_raw,
        "scaler": scaler,
    }


def run_all_experiments():
    """
    Lance l'ensemble des configurations définies dans config.EXPERIMENTS
    (≥ 6 configurations modèle x stratégie, conformément au cahier des
    charges) et retourne le DataFrame consolidé des résultats.
    """
    data = _prepare_data()
    results = []
    for exp in config.EXPERIMENTS:
        logger.info("=== Expérience : %s ===", exp)
        params = config.MODEL_PARAMS[exp["model_name"]]
        _, row = run_experiment(exp["model_name"], exp["imbalance_strategy"], params, data)
        results.append(row)

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    logger.info("\n%s", results_df[["model_name", "imbalance_strategy", "pr_auc", "recall", "f1"]])
    return results_df


if __name__ == "__main__":
    run_all_experiments()
