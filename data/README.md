# Données — Détection de fraude bancaire (IA-DS-2026-01)

## Source

- **Nom du dataset** : Credit Card Fraud Detection
- **Fournisseur** : Machine Learning Group - ULB (Université Libre de Bruxelles)
- **Lien officiel** : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Contexte original** : transactions par carte bancaire réalisées par des porteurs
  européens en septembre 2013, sur une fenêtre de 2 jours.

## Licence

Le dataset est distribué sous licence **Database Contents License (DbCL) v1.0**
(licence ouverte pour le contenu de bases de données). Voir la page Kaggle pour
le détail des conditions d'utilisation et de citation.

## Volumétrie

| Élément | Valeur |
|---|---|
| Nombre de transactions | 284 807 |
| Nombre de fraudes | 492 (0,173 %) |
| Nombre de transactions légitimes | 284 315 (99,827 %) |
| Nombre de colonnes | 31 (Time, V1–V28, Amount, Class) |
| Taille du fichier CSV | ≈ 150 Mo |
| Période couverte | 2 jours (≈ 172 792 secondes) |

## Procédure de téléchargement

Le fichier `creditcard.csv` n'est **pas versionné** dans ce dépôt (voir `.gitignore`)
en raison de sa taille. Deux méthodes de récupération :

### Méthode 1 — via kagglehub (recommandée, utilisée par `src/data_loader.py`)

```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('mlg-ulb/creditcardfraud'))"
```

Le chemin retourné contient `creditcard.csv`. Copier ou lier ce fichier vers :

```
data/raw/creditcard.csv
```

### Méthode 2 — téléchargement manuel

1. Se connecter sur https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Télécharger l'archive et extraire `creditcard.csv`
3. Placer le fichier dans `data/raw/creditcard.csv`

`src/data_loader.py::load_raw_data()` gère automatiquement les deux cas : si
`data/raw/creditcard.csv` existe déjà, il est utilisé directement ; sinon le
téléchargement via `kagglehub` est déclenché.

## Reproductibilité

- Graine aléatoire fixée : `RANDOM_STATE = 42` (voir `src/config.py`)
- Aucune donnée synthétique ne remplace le dataset réel
- Le découpage train/val/test est **temporel** (basé sur `Time`), jamais aléatoire
  — voir `src/data_loader.py::temporal_split()`
