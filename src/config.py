"""
config.py
=========
Configuration centrale du projet : chemins, graine aléatoire, hyperparamètres
des modèles et constantes métier (matrice de coûts).

Centraliser ces valeurs ici garantit la reproductibilité (une seule source de
vérité) et évite de disperser des "nombres magiques" dans le code.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins du projet
# ---------------------------------------------------------------------------
# BASE_DIR = racine du dépôt (dossier parent de src/)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"

# Fichiers issus du split temporel (format parquet, cf. data_loader.save_processed)
TRAIN_PATH = PROCESSED_DATA_DIR / "train.parquet"
VAL_PATH = PROCESSED_DATA_DIR / "val.parquet"
TEST_PATH = PROCESSED_DATA_DIR / "test.parquet"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_CSV = REPORTS_DIR / "resultats.csv"

# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Découpage temporel (S1 — jamais de découpage aléatoire sur ces données)
# ---------------------------------------------------------------------------
# Le dataset couvre ~172 792 secondes (≈ 2 jours). On trie les transactions
# par Time croissant puis on coupe successivement en train / val / test.
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, validation, test

# ---------------------------------------------------------------------------
# Variable cible et colonnes
# ---------------------------------------------------------------------------
TARGET_COL = "Class"
TIME_COL = "Time"
AMOUNT_COL = "Amount"

# Colonnes à standardiser (StandardScaler fit sur train uniquement)
SCALED_COLS = ["Time", "Amount", "TransactionHour", "TransactionDay"]

# ---------------------------------------------------------------------------
# Matrice de coûts métier (S4)
# ---------------------------------------------------------------------------
# Le cahier des charges officiel (sujet IA-DS-2026-01) précise explicitement :
# "laisser passer une fraude coûte le MONTANT DE LA TRANSACTION, bloquer une
# transaction légitime coûte un CLIENT MÉCONTENT". Le coût d'un faux négatif
# n'est donc PAS un forfait fixe : il est calculé dynamiquement, transaction
# par transaction, à partir de sa colonne Amount (voir
# src/evaluate.py::plot_cost_curve, paramètre `fn_amounts`).
#
# COUT_FN_FALLBACK : coût forfaitaire utilisé UNIQUEMENT si les montants réels
#                     ne sont pas disponibles (ex. exploration rapide sans
#                     recharger Amount non standardisé). À éviter en pratique.
# COUT_FP           : coût forfaitaire d'un blocage abusif d'une transaction
#                     légitime (faux positif) : temps de vérification,
#                     insatisfaction client, coût de traitement du dossier par
#                     le service fraude. Estimé à 20€ (hypothèse à justifier/
#                     ajuster dans le rapport).
COUT_FN_FALLBACK = 500.0
COUT_FP = 20.0

# ---------------------------------------------------------------------------
# Hyperparamètres des modèles (valeurs de départ raisonnables, à affiner
# éventuellement par recherche d'hyperparamètres — hors périmètre obligatoire)
# ---------------------------------------------------------------------------
MODEL_PARAMS = {
    "dummy": {
        "strategy": "most_frequent",
    },
    "logreg": {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "decision_tree": {
        "max_depth": 8,
        "random_state": RANDOM_STATE,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 12,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "aucpr",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "lightgbm": {
        "n_estimators": 300,
        "max_depth": -1,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    },
}

# ---------------------------------------------------------------------------
# Configurations d'expériences imposées par le cahier des charges
# (≥ 6 combinaisons modèle x stratégie de rééquilibrage)
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {"model_name": "dummy", "imbalance_strategy": "class_weight"},
    {"model_name": "logreg", "imbalance_strategy": "class_weight"},
    {"model_name": "decision_tree", "imbalance_strategy": "class_weight"},
    {"model_name": "random_forest", "imbalance_strategy": "class_weight"},
    {"model_name": "random_forest", "imbalance_strategy": "smote"},
    {"model_name": "xgboost", "imbalance_strategy": "class_weight"},
    {"model_name": "xgboost", "imbalance_strategy": "smote"},
    {"model_name": "lightgbm", "imbalance_strategy": "class_weight"},
]

# Seuils utilisés pour l'affichage comparatif des matrices de confusion
DISPLAY_THRESHOLDS = [0.3, 0.5]
