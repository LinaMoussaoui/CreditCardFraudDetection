# 💳 Détection de Fraude sur Transactions par Carte Bancaire

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25-red.svg)](https://streamlit.io/)
[![Sujet](https://img.shields.io/badge/Sujet-IA--DS--2026--01-green.svg)]()

> Projet de stage d'été 2026 — Apprentissage supervisé sur données tabulaires
> déséquilibrées, avec optimisation du seuil de décision par une fonction de
> coût métier.

## Contexte

Projet réalisé dans le cadre du stage d'été 2026 (sujet **IA-DS-2026-01**), en
4e année Ingénierie Informatique et Réseaux — option IA & Data Science —
**EMSI Casablanca**, sous l'encadrement du **Pr. Hamza Abouabid**.

Il fait suite à un projet de module "Exploration de données" réalisé plus tôt
dans le cursus, entièrement refactorisé et étendu ici pour répondre au cahier
des charges du stage : découpage temporel, traitement rigoureux du
déséquilibre, seuil de décision optimisé par une matrice de coûts, et
démonstration interactive.

## Structure du projet

```
projet-IA-DS-2026-01-Moussaoui-Lina/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/            # creditcard.csv (non versionné, voir data/README.md)
│   ├── processed/      # splits temporels train/val/test (non versionnés)
│   └── README.md
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preparation.ipynb
│   └── 03_modelisation.ipynb
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/              # modèles entraînés (.joblib, non versionnés)
├── reports/
│   ├── figures/
│   └── resultats.csv
├── app/
│   └── app.py           # démonstration Streamlit
├── livrables/
│   ├── L1_note_cadrage.md
│   ├── L2_fiche_qualification_donnees.md
│   └── annexe_IA_generative.md
└── rapport/
    ├── Rapport_academique_detection_fraude.docx
    └── generer_rapport_academique.py
```

## Dataset

- **Source** : Kaggle — [`mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Licence** : Database Contents License (DbCL) v1.0
- **Volumétrie** : 284 807 transactions, 31 variables, 492 fraudes (0,173 %)
- Détails complets : voir [`data/README.md`](data/README.md)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Placer ensuite `creditcard.csv` dans `data/raw/` (voir `data/README.md` pour
la procédure de téléchargement via `kagglehub` ou manuelle).

## Utilisation

### 1. Préparation des données (découpage temporel + features)

```bash
python -m src.data_loader
```

Génère `data/processed/{train,val,test}.parquet` à partir d'un découpage
**temporel** strict (jamais aléatoire).

### 2. Entraînement de l'ensemble des expériences

```bash
python -m src.train
```

Entraîne les ≥ 6 configurations modèle × stratégie de rééquilibrage définies
dans `src/config.py::EXPERIMENTS`, journalise les résultats dans
`reports/resultats.csv` et sauvegarde les modèles dans `models/`.

### 3. Notebooks (recommandé pour suivre le raisonnement complet)

Exécuter dans l'ordre :
```
notebooks/01_exploration.ipynb
notebooks/02_preparation.ipynb
notebooks/03_modelisation.ipynb
```

### 4. Démonstration interactive

```bash
streamlit run app/app.py
```

L'application interactive propose deux actions simples : vérifier une
transaction avec des champs lisibles (montant, type, heure, transaction
internationale et nouveau bénéficiaire), ou charger un CSV et télécharger les
résultats. Un fichier prêt à essayer est disponible dans
[`transactions_demo.csv`](transactions_demo.csv).

Cette interface est volontairement un prototype pédagogique autonome. Elle
entraîne en mémoire un petit classifieur sur des exemples synthétiques
reproductibles, sans utiliser `V1`–`V28`, le dataset Kaggle, ni les modèles de
la partie expérimentale. Les alertes ne prouvent pas une fraude et ce modèle
n'est pas destiné à des décisions bancaires réelles. Les notebooks et
expériences de détection sur le dataset Kaggle restent séparés pour conserver
la reproductibilité du travail académique.

## Résultats clés

Résultats enregistrés dans `reports/resultats.csv`, au seuil **0,5**, sur le
jeu de test temporel (42 560 transactions, dont 52 fraudes). Les valeurs
ci-dessous sont le dernier enregistrement disponible par configuration ;
elles ne représentent pas un benchmark indépendant ou répété.

| Modèle | Stratégie | PR-AUC | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Modèle | Stratégie | PR-AUC | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Dummy | class_weight | 0.0012 | 0.000 | 0.000 | 0.5000 |
| Logistic Regression | class_weight | 0.7235 | 0.865 | 0.073 | 0.9793 |
| Decision Tree | class_weight | 0.3206 | 0.731 | 0.167 | 0.8625 |
| Random Forest | class_weight | 0.7741 | 0.712 | 0.822 | 0.9604 |
| Random Forest | smote | 0.7775 | 0.750 | 0.750 | 0.9025 |
| XGBoost | class_weight | 0.7628 | 0.750 | 0.804 | 0.9756 |
| XGBoost | smote | 0.7582 | 0.769 | 0.667 | 0.9617 |
| LightGBM | class_weight | 0.7738 | 0.750 | 0.813 | 0.9757 |

La fonction de coût enregistrée additionne les montants bruts des fraudes
manquées et 20 € par faux positif. Le forfait de 20 € est une hypothèse
pédagogique, à valider avec un contexte métier. Le seuil des résultats ci-dessus
est 0,5 ; il ne s’agit pas d’un seuil optimal certifié. Tout réglage devrait
être effectué sur validation, jamais sur le jeu de test.

## Reproductibilité

- Graine aléatoire fixée : `RANDOM_STATE = 42` (`src/config.py`)
- Versions des dépendances figées (`requirements.txt`)
- Découpage temporel déterministe (aucun aléa dans `temporal_split`)
- Suivi des expériences : `reports/resultats.csv` (ou MLflow, voir
  `requirements.txt`)

## Auteurs

- **MOUSSAOUI Lina** — EMSI, 4e année Ingénierie
  Informatique et Réseaux, option IA & Data Science
- Encadrant de stage : **Pr. Hamza Abouabid**
- Année : 2025/2026

## Références

- Dal Pozzolo, A. et al., *Credit Card Fraud Detection: A Realistic Modeling
  and a Novel Learning Strategy*, IEEE TNNLS, 2017.
- Documentation scikit-learn, imbalanced-learn, XGBoost, LightGBM, SHAP.
- Dataset Kaggle : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
