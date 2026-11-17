# L2 — Fiche de qualification des données

**Projet** : IA-DS-2026-01 — Détection de fraude bancaire

## Source et lien

- Dataset : *Credit Card Fraud Detection*
- Fournisseur : Machine Learning Group - ULB
- Lien : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Licence

Database Contents License (DbCL) v1.0.

## Volumétrie

| Élément | Valeur |
|---|---|
| Lignes | 284 807 |
| Colonnes | 31 |
| Taille du fichier CSV | ≈ 150 Mo |
| Période couverte | ≈ 2 jours (172 792 secondes) |

## Description du schéma

| Colonne | Type | Description |
|---|---|---|
| `Time` | float | Secondes écoulées depuis la première transaction du jeu de données |
| `V1`–`V28` | float | Composantes issues d'une PCA sur les variables originales (anonymisées pour raisons de confidentialité) |
| `Amount` | float | Montant de la transaction (€) |
| `Class` | int (0/1) | Variable cible : 0 = légitime, 1 = fraude |

## Taux de valeurs manquantes

0 % sur l'ensemble des colonnes (dataset complet, aucune imputation nécessaire).

Doublons exacts détectés : 1 081 lignes (≈ 0,38 % du dataset) — supprimés lors
du nettoyage (`src/preprocessing.clean_data`).

## Distribution de la variable cible

| Classe | Effectif | Proportion |
|---|---|---|
| 0 (légitime) | 284 315 | 99,827 % |
| 1 (fraude) | 492 | 0,173 % |

Taux de positifs très inférieur au seuil de 5 % : le problème est fortement
déséquilibré, ce qui conditionne le choix des métriques et des stratégies de
rééquilibrage.

## Limites et biais identifiés

1. **Anonymisation par PCA** : les variables V1–V28 n'ont pas de signification
   métier directement interprétable, ce qui limite l'explication fonctionnelle
   des alertes (seule une interprétation statistique via SHAP est possible).
2. **Fenêtre temporelle très courte** (2 jours, transactions européennes de
   septembre 2013) : les schémas de fraude ont probablement évolué depuis, et
   un modèle entraîné sur ces données ne capture pas de saisonnalité longue
   (jour de semaine, mois, événements commerciaux).
3. **Déséquilibre extrême** : avec seulement 492 exemples positifs, le risque de
   sur-apprentissage sur les fraudes observées est réel, et la variance des
   estimations de performance (notamment sur des sous-échantillons de test) peut
   être élevée.
