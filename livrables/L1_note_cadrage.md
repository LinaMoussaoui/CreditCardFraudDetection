# L1 — Note de cadrage

**Projet** : IA-DS-2026-01 — Détection de fraude sur transactions par carte bancaire
**Étudiante** : Moussaoui Lina
**Encadrant** : Pr. Hamza Abouabid — EMSI Casablanca
**Date** : fin S1, stage d'été 2026

---

## 1. Reformulation du problème

Les institutions bancaires traitent chaque jour un très grand volume de
transactions par carte, dont une infime proportion est frauduleuse. Ne pas
détecter une fraude entraîne une perte financière directe pour la banque et/ou
le client ; à l'inverse, bloquer à tort une transaction légitime dégrade
l'expérience client et génère un coût de traitement.

L'enjeu est donc double : maximiser la détection des fraudes réelles tout en
maintenant un taux de fausses alertes acceptable, dans un contexte où les
fraudes représentent moins de 0,2 % des transactions.

## 2. Tâche d'apprentissage retenue

**Classification binaire supervisée** : prédire `Class ∈ {0, 1}` (0 = transaction
légitime, 1 = transaction frauduleuse) à partir de 30 variables explicatives
(`Time`, `Amount`, `V1`...`V28`, composantes anonymisées issues d'une PCA), sur
le dataset public *Credit Card Fraud Detection* (Kaggle, `mlg-ulb/creditcardfraud`,
284 807 transactions, 492 fraudes soit 0,173 %).

Le problème est traité comme une classification supervisée déséquilibrée plutôt
que comme une détection d'anomalies pure, car des étiquettes fiables sont
disponibles pour l'ensemble des transactions.

## 3. Métrique principale retenue et justification

**PR-AUC (aire sous la courbe Precision-Recall)** est retenue comme métrique
principale de sélection de modèle, complétée par le recall, le F1-score et le
ROC-AUC en indicateurs secondaires.

Justification :
- L'accuracy est trompeuse ici : un modèle trivial prédisant systématiquement
  "non-fraude" atteint ~99,8 % d'accuracy sans aucune valeur métier.
- Le ROC-AUC, bien qu'utile, reste optimiste sur données très déséquilibrées
  car le taux de faux positifs (FPR) est calculé sur une classe négative très
  majoritaire — de petites variations en absolu produisent de faibles variations
  de FPR, ce qui gonfle artificiellement l'AUC.
- Le PR-AUC, en se concentrant sur la précision et le rappel de la classe
  minoritaire (les fraudes), donne une image beaucoup plus fidèle de la capacité
  réelle du modèle à isoler les fraudes parmi les alertes qu'il génère.

**Objectifs de performance visés** (cahier des charges) :
- PR-AUC ≥ 0,75 sur le test
- Recall ≥ 0,80 à une précision ≥ 0,50
- Gain de PR-AUC ≥ 15 % par rapport à la régression logistique de référence

## 4. Protocole d'évaluation

**Découpage temporel strict** (70 % train / 15 % validation / 15 % test), basé
sur un tri croissant de la variable `Time`. Un découpage aléatoire est exclu :
il autoriserait le modèle à être entraîné sur des transactions futures par
rapport à celles utilisées pour l'évaluer, ce qui produirait une estimation de
performance irréaliste par rapport à un déploiement en production.

Règles complémentaires :
- Le `StandardScaler` est ajusté uniquement sur le train.
- Les techniques de rééquilibrage (SMOTE, sous-échantillonnage) ne sont
  appliquées qu'au train, après le split.
- Le jeu de validation sert au réglage éventuel des hyperparamètres et du
  seuil ; le jeu de test n'est utilisé qu'une seule fois, pour l'évaluation
  finale.

## 5. Matrice de coûts

Conformément à l'énoncé officiel du sujet IA-DS-2026-01 : *"laisser passer une
fraude coûte le montant de la transaction, bloquer une transaction légitime
coûte un client mécontent"*. Le coût d'un faux négatif n'est donc pas un
forfait fixe : il est calculé **transaction par transaction**, à partir du
montant réel de chaque fraude manquée. Le coût d'un faux positif reste
forfaitaire, le "mécontentement client" n'étant pas directement quantifiable
en euros dans ce dataset.

| | Prédit non-fraude | Prédit fraude |
|---|---|---|
| **Réel non-fraude** | 0€ (vrai négatif) | 20€ (faux positif — vérification/gêne client, forfait) |
| **Réel fraude** | `Amount` de la transaction (faux négatif — perte non détectée, montant réel) | 0€ (vrai positif, coût de traitement négligé) |

Formellement :

```
Coût total(seuil) = Σ Amount(transactions frauduleuses manquées au seuil) + FP(seuil) × 20€
```

Cette formulation est plus fidèle à l'énoncé qu'un forfait unique par fraude
manquée : elle pondère naturellement les faux négatifs à montant élevé, qui
sont précisément ceux que l'analyse d'erreurs du chapitre 5 doit mettre en
avant. Le coût forfaitaire de 20€ par faux positif reste une hypothèse de
travail à discuter dans le rapport (temps de vérification, coût du service
fraude, insatisfaction client). Voir `src/evaluate.py::plot_cost_curve` pour
l'implémentation et le chapitre 5 du rapport pour la justification du seuil
retenu.
