"""
preprocessing.py
=================
Nettoyage, standardisation (fit sur train uniquement) et gestion du
déséquilibre des classes.

⚠️ POINTS DE VIGILANCE DU CAHIER DES CHARGES :
1. Le StandardScaler doit être ajusté (fit) UNIQUEMENT sur le train, puis
   appliqué (transform) tel quel sur val/test. Sinon, les statistiques de
   val/test "fuient" dans l'entraînement (data leakage).
2. SMOTE doit être appliqué UNIQUEMENT sur le train, et seulement APRÈS le
   split (jamais avant). Appliquer SMOTE avant le split laisserait des
   exemples synthétiques quasi identiques se retrouver à la fois en train et
   en test, ce qui fausse complètement l'évaluation (fuite grave).
"""

import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage de base :
    - vérifie et rapporte les valeurs manquantes (le dataset original n'en
      contient aucune, mais on vérifie par prudence / reproductibilité) ;
    - supprime les doublons exacts (le dataset original en contient 1081,
      probablement des transactions distinctes ayant, par coïncidence, les
      mêmes valeurs PCA arrondies -> on les retire pour éviter qu'un même
      enregistrement ne se retrouve dupliqué entre deux splits).
    """
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        logger.warning("%s valeurs manquantes détectées.", n_missing)
    else:
        logger.info("Aucune valeur manquante détectée.")

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    logger.info("Doublons supprimés : %s (sur %s lignes initiales)", n_removed, n_before)

    return df


def scale_features(X_train, X_val, X_test, cols=None):
    """
    Standardise les colonnes `cols` : le StandardScaler est ajusté (fit)
    EXCLUSIVEMENT sur X_train, puis appliqué (transform) à X_val et X_test.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
    cols : liste de colonnes à standardiser (par défaut config.SCALED_COLS)

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    cols = cols or [c for c in config.SCALED_COLS if c in X_train.columns]

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_val[cols] = scaler.transform(X_val[cols])
    X_test[cols] = scaler.transform(X_test[cols])

    logger.info("Standardisation appliquée sur les colonnes : %s (fit sur train uniquement)", cols)
    return X_train, X_val, X_test, scaler


def handle_imbalance(X_train, y_train, strategy: str = "class_weight"):
    """
    Gère le déséquilibre des classes UNIQUEMENT sur le jeu d'entraînement.

    Parameters
    ----------
    strategy : {"class_weight", "undersample", "smote"}
        - "class_weight" : ne modifie pas les données, retourne un
          dictionnaire de poids à passer aux modèles scikit-learn / boosting
          (paramètre `class_weight` ou `scale_pos_weight`).
        - "undersample"  : sous-échantillonnage aléatoire de la classe
          majoritaire (RandomUnderSampler).
        - "smote"        : sur-échantillonnage synthétique de la classe
          minoritaire (SMOTE), appliqué APRÈS le split, sur train seulement.

    Returns
    -------
    X_train_res, y_train_res, extra
        extra contient par exemple le class_weight dict quand strategy vaut
        "class_weight" (X_train_res/y_train_res sont alors inchangés).
    """
    if strategy == "class_weight":
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        class_weight = {0: 1.0, 1: n_neg / n_pos}
        scale_pos_weight = n_neg / n_pos  # équivalent pour XGBoost/LightGBM
        logger.info("Stratégie class_weight : %s (scale_pos_weight=%.1f)", class_weight, scale_pos_weight)
        return X_train, y_train, {"class_weight": class_weight, "scale_pos_weight": scale_pos_weight}

    if strategy == "undersample":
        rus = RandomUnderSampler(random_state=config.RANDOM_STATE)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        logger.info("Sous-échantillonnage : %s -> %s lignes", len(X_train), len(X_res))
        return X_res, y_res, {}

    if strategy == "smote":
        smote = SMOTE(random_state=config.RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info("SMOTE (train uniquement) : %s -> %s lignes", len(X_train), len(X_res))
        return X_res, y_res, {}

    raise ValueError(f"Stratégie de rééquilibrage inconnue : {strategy}")
