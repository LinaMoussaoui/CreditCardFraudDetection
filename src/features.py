"""
features.py
============
Feature engineering : création de variables dérivées temporelles et
sélection des features / cible pour la modélisation.
"""

import pandas as pd

from src import config


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des variables dérivées à partir de `Time` (secondes écoulées
    depuis la première transaction du dataset) :

    - TransactionHour : heure de la journée (0-23), capture les schémas
      cycliques (les fraudes peuvent être plus fréquentes à certaines heures).
    - TransactionDay : numéro du jour (0, 1, ...), capture d'éventuelles
      différences de comportement entre les jours de la période de collecte.

    Ces variables sont calculées AVANT le split ou de manière identique sur
    chaque split (elles ne dépendent que de Time, pas de statistiques
    calculées sur les données -> pas de fuite).
    """
    df = df.copy()
    seconds_per_day = 24 * 3600
    df["TransactionHour"] = (df[config.TIME_COL] % seconds_per_day) // 3600
    df["TransactionDay"] = df[config.TIME_COL] // seconds_per_day
    return df


def select_features(df: pd.DataFrame):
    """
    Sépare le DataFrame en (X, y) : features explicatives et variable cible.

    Retourne
    --------
    X : pd.DataFrame
        Toutes les colonnes sauf la cible.
    y : pd.Series
        La colonne cible (Class).
    """
    y = df[config.TARGET_COL]
    X = df.drop(columns=[config.TARGET_COL])
    return X, y
