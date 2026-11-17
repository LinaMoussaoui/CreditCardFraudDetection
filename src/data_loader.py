"""
data_loader.py
===============
Chargement du dataset brut et découpage temporel strict.

⚠️ POINT DE VIGILANCE MAJEUR DU CAHIER DES CHARGES :
Ce dataset possède un indicateur d'ordre temporel (`Time`, secondes écoulées
depuis la première transaction). Un découpage ALÉATOIRE (train_test_split
classique) est donc INTERDIT : il mélangerait le futur et le passé et
donnerait une estimation de performance trop optimiste et non réaliste d'un
système déployé en production (qui ne voit jamais le futur au moment de
l'entraînement). On utilise à la place un découpage temporel strict.
"""

from pathlib import Path
import logging

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Charge le dataset brut `creditcard.csv`.

    Si le fichier n'existe pas localement, tente de le télécharger via
    kagglehub (nécessite une configuration Kaggle valide sur la machine).

    Parameters
    ----------
    path : Path
        Emplacement attendu du fichier CSV.

    Returns
    -------
    pd.DataFrame
        Le dataset brut, avec les colonnes Time, V1..V28, Amount, Class.
    """
    path = Path(path)

    if path.exists():
        logger.info("Chargement du dataset local : %s", path)
        df = pd.read_csv(path)
    else:
        logger.info(
            "Fichier %s introuvable. Tentative de téléchargement via kagglehub...",
            path,
        )
        try:
            import kagglehub
        except ImportError as exc:
            raise RuntimeError(
                "kagglehub n'est pas installé et le fichier local est absent. "
                "Installez-le avec `pip install kagglehub` ou déposez "
                "creditcard.csv dans data/raw/ (voir data/README.md)."
            ) from exc

        download_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        csv_path = Path(download_path) / "creditcard.csv"
        df = pd.read_csv(csv_path)

        # On copie le fichier dans data/raw/ pour les prochaines exécutions
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Dataset téléchargé et copié vers %s", path)

    logger.info("Dataset chargé : %s lignes, %s colonnes", *df.shape)
    _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Vérifie que les colonnes essentielles sont présentes."""
    expected = {config.TARGET_COL, config.TIME_COL, config.AMOUNT_COL}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset : {missing}")


def temporal_split(
    df: pd.DataFrame,
    ratios: tuple = config.SPLIT_RATIOS,
    time_col: str = config.TIME_COL,
) -> tuple:
    """
    Découpage TEMPOREL STRICT (jamais aléatoire).

    On trie les transactions par ordre chronologique croissant (`Time`) puis
    on découpe successivement en train / validation / test selon les ratios
    fournis. Ainsi :
      - le modèle n'est jamais entraîné sur des transactions "futures" par
        rapport à celles sur lesquelles il est évalué ;
      - le protocole simule un déploiement réel où l'on prédit l'avenir à
        partir du passé.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset complet, avec la colonne `time_col`.
    ratios : tuple(float, float, float)
        Proportions (train, val, test), doit sommer à 1.0.
    time_col : str
        Nom de la colonne temporelle utilisée pour le tri.

    Returns
    -------
    (train_df, val_df, test_df) : tuple de pd.DataFrame
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "Les ratios doivent sommer à 1.0"

    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)

    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_df = df_sorted.iloc[:n_train]
    val_df = df_sorted.iloc[n_train : n_train + n_val]
    test_df = df_sorted.iloc[n_train + n_val :]

    logger.info(
        "Split temporel -> train: %s (%.1f%%) | val: %s (%.1f%%) | test: %s (%.1f%%)",
        len(train_df), 100 * len(train_df) / n,
        len(val_df), 100 * len(val_df) / n,
        len(test_df), 100 * len(test_df) / n,
    )
    logger.info(
        "Taux de fraude -> train: %.4f%% | val: %.4f%% | test: %.4f%%",
        100 * train_df[config.TARGET_COL].mean(),
        100 * val_df[config.TARGET_COL].mean(),
        100 * test_df[config.TARGET_COL].mean(),
    )

    return train_df, val_df, test_df


def save_processed(df: pd.DataFrame, path: Path) -> None:
    """Sauvegarde un DataFrame prétraité au format parquet (compact, typé)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Données sauvegardées : %s (%s lignes)", path, len(df))


def load_processed(path: Path) -> pd.DataFrame:
    """Recharge un split déjà sauvegardé (train/val/test) au format parquet."""
    return pd.read_parquet(path)


if __name__ == "__main__":
    df = load_raw_data()
    train_df, val_df, test_df = temporal_split(df)
    save_processed(train_df, config.TRAIN_PATH)
    save_processed(val_df, config.VAL_PATH)
    save_processed(test_df, config.TEST_PATH)
