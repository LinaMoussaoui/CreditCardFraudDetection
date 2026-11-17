"""
evaluate.py
===========
Fonctions d'évaluation adaptées à la classification déséquilibrée :
métriques, courbes ROC/PR comparées, matrice de coûts et seuil optimal,
calibration, analyse d'erreurs et interprétabilité SHAP.

⚠️ Rappel du cahier des charges : reporter la seule aire ROC est INSUFFISANT
(trop optimiste sur données déséquilibrées). Le PR-AUC est la métrique
principale ici ; le ROC-AUC est reporté en complément.
"""

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """
    Calcule l'ensemble des métriques adaptées à un problème déséquilibré,
    pour un seuil de décision donné.

    L'accuracy est calculée mais ne doit JAMAIS être utilisée seule pour
    comparer des modèles ici : avec ~99,8% de transactions légitimes, un
    modèle trivial qui prédit toujours "non-fraude" obtient une accuracy
    > 99% tout en étant inutile.
    """
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "threshold": threshold,
    }


def plot_roc_curves(models_dict: dict, X_test, y_test, save_path=None):
    """
    Trace les courbes ROC de plusieurs modèles sur un même graphique.

    models_dict : {"nom_affiché": modèle_entraîné, ...}
    """
    plt.figure(figsize=(7, 6))
    for label, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Aléatoire")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbes ROC comparées")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_pr_curves(models_dict: dict, X_test, y_test, save_path=None):
    """Trace les courbes Precision-Recall de plusieurs modèles (métrique principale)."""
    plt.figure(figsize=(7, 6))
    baseline = y_test.mean()
    for label, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        plt.plot(recall, precision, label=f"{label} (PR-AUC = {ap:.3f})")
    plt.axhline(baseline, color="k", linestyle="--", label=f"Baseline aléatoire ({baseline:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Courbes Precision-Recall comparées")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_confusion_matrices(models_dict: dict, X_test, y_test, thresholds=config.DISPLAY_THRESHOLDS, save_path=None):
    """
    Affiche une grille de matrices de confusion : une ligne par modèle,
    une colonne par seuil testé.
    """
    n_models = len(models_dict)
    n_thresh = len(thresholds)
    fig, axes = plt.subplots(n_models, n_thresh, figsize=(5 * n_thresh, 4 * n_models), squeeze=False)

    for i, (label, model) in enumerate(models_dict.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        for j, t in enumerate(thresholds):
            y_pred = (y_proba >= t).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i][j])
            axes[i][j].set_title(f"{label} — seuil={t}\nRecall={recall_score(y_test, y_pred):.3f}")
            axes[i][j].set_xlabel("Prédit")
            axes[i][j].set_ylabel("Réel")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def _costs_over_thresholds(y_true, y_proba, fn_amounts=None, cost_fn: float = config.COUT_FN_FALLBACK,
                            cost_fp: float = config.COUT_FP, thresholds=None):
    """
    Calcule le coût total pour une grille de seuils, sans tracer de graphique.
    Fonction partagée par plot_cost_curve() (notebooks) et find_optimal_threshold()
    (utilisée par l'application Streamlit, qui n'a pas besoin du graphique).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    if fn_amounts is not None:
        fn_amounts = np.asarray(fn_amounts)

    costs = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)
        n_fp = fp_mask.sum()
        if fn_amounts is not None:
            cout_fn_total = fn_amounts[fn_mask].sum()
        else:
            cout_fn_total = fn_mask.sum() * cost_fn
        costs.append(cout_fn_total + n_fp * cost_fp)

    return thresholds, np.array(costs)


def find_optimal_threshold(y_true, y_proba, fn_amounts=None, cost_fn: float = config.COUT_FN_FALLBACK,
                            cost_fp: float = config.COUT_FP):
    """
    Retourne (seuil_optimal, coût_total_minimal) sans tracer de graphique.
    Utilisé par l'application Streamlit pour proposer un seuil "recommandé"
    au lieu de 0.5 par défaut, sans dépendance à matplotlib côté app.
    """
    thresholds, costs = _costs_over_thresholds(y_true, y_proba, fn_amounts, cost_fn, cost_fp)
    best_idx = np.argmin(costs)
    return float(thresholds[best_idx]), float(costs[best_idx])


def plot_cost_curve(y_true, y_proba, fn_amounts=None, cost_fn: float = config.COUT_FN_FALLBACK,
                     cost_fp: float = config.COUT_FP, save_path=None):
    """
    Trace le COÛT TOTAL en fonction du seuil de décision, et retourne le
    seuil qui minimise ce coût.

    Conformément à l'énoncé officiel du sujet IA-DS-2026-01 : "laisser passer
    une fraude coûte le MONTANT DE LA TRANSACTION, bloquer une transaction
    légitime coûte un CLIENT MÉCONTENT". Le coût d'un faux négatif est donc
    calculé transaction par transaction à partir de son montant réel
    (`fn_amounts`, la colonne Amount NON standardisée alignée sur y_true), et
    non comme un forfait fixe. Le coût d'un faux positif reste forfaitaire
    (`cost_fp`), le "mécontentement client" n'étant pas quantifiable en euros
    directement dans ce dataset.

        Coût total(seuil) = somme(Amount des FN au seuil) + FP(seuil) * cost_fp

    Si `fn_amounts` n'est pas fourni (montants réels indisponibles), on
    retombe sur un forfait fixe `cost_fn` par faux négatif (moins fidèle à
    l'énoncé, à éviter si possible — voir config.COUT_FN_FALLBACK).
    """
    thresholds, costs = _costs_over_thresholds(y_true, y_proba, fn_amounts, cost_fn, cost_fp)
    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]
    best_cost = costs[best_idx]

    label_cost = "somme(Amount) des FN" if fn_amounts is not None else f"{cost_fn}€/FN (forfait)"

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, costs, label="Coût total")
    plt.axvline(best_threshold, color="red", linestyle="--",
                label=f"Seuil optimal = {best_threshold:.2f} (coût = {best_cost:,.0f}€)")
    plt.xlabel("Seuil de décision")
    plt.ylabel(f"Coût total (€)  [coût FN = {label_cost}, coût FP = {cost_fp}€]")
    plt.title("Coût total en fonction du seuil de décision")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

    logger.info("Seuil optimal (minimisation du coût) : %.3f (coût total = %.0f€)", best_threshold, best_cost)
    return best_threshold


def plot_calibration_curve(model, X_test, y_test, n_bins: int = 10, save_path=None):
    """
    Trace la courbe de fiabilité (calibration) : compare la probabilité
    prédite moyenne à la fréquence réelle de fraude par bin. Un modèle bien
    calibré suit la diagonale — important si les probabilités sont utilisées
    telles quelles (ex. priorisation des alertes) et pas seulement leur rang.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=n_bins, strategy="quantile")

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Modèle")
    plt.plot([0, 1], [0, 1], "k--", label="Calibration parfaite")
    plt.xlabel("Probabilité prédite moyenne")
    plt.ylabel("Fréquence observée de fraude")
    plt.title("Courbe de fiabilité (calibration)")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def analyze_errors(X_test, y_test, y_pred, y_proba, n: int = 10, amount_col: str = config.AMOUNT_COL) -> pd.DataFrame:
    """
    Analyse d'erreurs — SECTION LA PLUS DISCRIMINANTE du rapport (chapitre 5).

    Isole les faux négatifs (fraudes manquées) triés par montant décroissant
    (les plus coûteux d'abord) et les faux positifs les plus "confiants"
    (probabilité élevée alors que la transaction est légitime), pour
    illustrer concrètement les limites du modèle sur des cas réels.

    Returns
    -------
    pd.DataFrame des n cas les plus intéressants à commenter dans le rapport.
    """
    df = X_test.copy()
    df["y_true"] = np.asarray(y_test)
    df["y_pred"] = np.asarray(y_pred)
    df["y_proba"] = np.asarray(y_proba)

    fn_cases = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].sort_values(amount_col, ascending=False)
    fp_cases = df[(df["y_true"] == 0) & (df["y_pred"] == 1)].sort_values("y_proba", ascending=False)

    n_fn = min(n // 2 + n % 2, len(fn_cases))
    n_fp = min(n - n_fn, len(fp_cases))

    cases = pd.concat([
        fn_cases.head(n_fn).assign(type_erreur="Faux Négatif (fraude manquée)"),
        fp_cases.head(n_fp).assign(type_erreur="Faux Positif (transaction bloquée à tort)"),
    ])

    logger.info("Analyse d'erreurs : %s FN et %s FP sélectionnés pour le rapport", n_fn, n_fp)
    return cases


def plot_shap_summary(model, X_test, max_display: int = 15, save_path=None):
    """
    Affiche le SHAP summary plot (importance et effet des variables) pour un
    modèle à base d'arbres (Random Forest, XGBoost, LightGBM).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # Pour les classifieurs binaires, certains explainers renvoient une liste [classe0, classe1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap.summary_plot(shap_values, X_test, max_display=max_display, show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_shap_individual(model, X_test, idx: int, save_path=None):
    """
    Explique UNE alerte individuelle (transaction à l'index `idx` de X_test)
    via un waterfall plot SHAP — utilisé aussi dans l'application Streamlit.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test.iloc[[idx]])
    shap.plots.waterfall(shap_values[0], show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
