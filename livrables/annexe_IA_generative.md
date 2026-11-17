# Annexe — Déclaration d'usage de l'IA générative

Conformément à la charte IA générative du cahier des charges IA-DS-2026-01,
cette annexe déclare les outils d'intelligence artificielle générative utilisés
au cours du projet, les tâches concernées, et les vérifications effectuées.

## Outils utilisés

| Outil | Usage principal |
|---|---|
| Claude (Anthropic) | Refactorisation du notebook initial en architecture modulaire (`src/`), génération des squelettes de notebooks, rédaction de docstrings et commentaires en français, structuration des livrables (notes de cadrage, fiches, README) |
| Cursor / ChatGPT (le cas échéant) | Complément éventuel pour la relecture de code et la mise en forme du rapport |

## Tâches concernées

- Réorganisation du code du notebook original en modules `src/*.py`
  (config, chargement des données, prétraitement, entraînement, évaluation,
  prédiction).
- Rédaction de la trame des 3 notebooks (`01_exploration`, `02_preparation`,
  `03_modelisation`) et de leurs commentaires explicatifs en français.
- Rédaction de l'application de démonstration Streamlit (`app/app.py`).
- Mise en forme du `README.md`, du `.gitignore`, des fiches livrables L1/L2 et
  de cette annexe.

## Vérifications effectuées

- **Aucun chiffre de performance produit par l'IA n'a été utilisé sans
  vérification** : les résultats numériques présentés dans le rapport
  proviennent exclusivement de l'exécution effective des notebooks sur le
  dataset réel, jamais d'une estimation ou d'une invention du modèle génératif.
- Chaque fonction générée a été relue et je suis en mesure d'expliquer son
  fonctionnement ligne par ligne en soutenance, conformément à l'exigence du
  cahier des charges.
- Les choix méthodologiques structurants (découpage temporel, matrice de
  coûts, métrique principale) ont été validés par rapport aux exigences
  explicites du sujet avant d'être implémentés.
- Le code généré a été adapté/corrigé à la main lorsque nécessaire pour
  correspondre exactement au dataset et au contexte du projet.
