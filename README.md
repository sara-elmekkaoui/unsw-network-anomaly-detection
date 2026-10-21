# Détection d'anomalies IoT/réseau — Approche hybride ML + DL
### Mini-projet — Module Sécurité des systèmes informatiques — Master SIDI

Ce projet implémente et compare **4 architectures hybrides** combinant
Machine Learning classique et Deep Learning pour la détection d'anomalies
(intrusions/botnets) sur le dataset **UNSW-NB15**.

## 1. Architectures implémentées

| Modèle | Deep Learning (extraction) | Machine Learning (classification) | Principe |
|---|---|---|---|
| **H1** | Auto-Encoder | Random Forest | L'AE est entraîné uniquement sur du trafic *normal* ; l'erreur de reconstruction devient une feature d'anomalie ajoutée aux features originales, classifiées par RF. |
| **H2** | LSTM | SVM | Le LSTM apprend une représentation séquentielle des features (embedding) ; le SVM classifie à partir de cet embedding. |
| **H3** | MLP | XGBoost | Le MLP apprend des features profondes (embedding) ; XGBoost classifie à partir des features originales + embedding. |
| **H4** | CNN1D | Gradient Boosting | Le CNN1D extrait des motifs locaux dans le vecteur de features ; Gradient Boosting classifie à partir de l'embedding obtenu. |

## 2. Structure du projet

```
unsw_project/
├── config.py          # Chemins, hyperparamètres, constantes
├── preprocessing.py   # Chargement, nettoyage, encodage, normalisation, SMOTE
├── models.py           # Les 4 classes de modèles hybrides H1 à H4
├── evaluate.py          # Calcul des métriques + collecte des données pour les graphiques
├── plotting.py            # Génération des matrices de confusion et courbes ROC
├── main.py                 # Script principal à exécuter
├── requirements.txt
├── data/                  # <- À PLACER ICI vos fichiers CSV UNSW-NB15
└── results/               # Tableau de résultats (CSV) + graphiques (PNG) générés
```

## 3. Installation

```bash
pip install -r requirements.txt
```

(Testé avec Python 3.10+, TensorFlow 2.21, scikit-learn 1.8, XGBoost 2.x)

## 4. Récupération du dataset réel

Le dataset UNSW-NB15 (version prétraitée par les auteurs, déjà divisée en
train/test) est disponible ici :
- Page officielle : https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Fichiers à télécharger : `UNSW_NB15_training-set.csv` et `UNSW_NB15_testing-set.csv`

Placez ces deux fichiers dans le dossier `data/` du projet, **sans changer
leur nom**. Si ces fichiers sont absents, `main.py` génère automatiquement
un petit dataset **synthétique** de démonstration afin que vous puissiez
vérifier que le code s'exécute correctement avant d'avoir les vraies
données (un message d'avertissement vous le signale clairement dans la
console).

## 5. Exécution

```bash
python3 main.py
```

Le script va :
1. Charger et prétraiter les données (nettoyage, encodage One-Hot des
   colonnes `proto`/`service`/`state`, normalisation StandardScaler,
   rééquilibrage SMOTE sur le train).
2. Entraîner et évaluer successivement H1, H2, H3 et H4.
3. Afficher dans la console le tableau de résultats (Précision, Rappel,
   F1-Score, Accuracy, temps d'entraînement, temps d'inférence — pour
   chaque modèle et chaque classe "Avec anomalie" / "Sans anomalie").
4. Sauvegarder ce tableau dans `results/resultats_finaux.csv`.
5. Générer automatiquement les graphiques dans `results/` :
   - `confusion_matrices_all.png` — les 4 matrices de confusion côte à côte
   - `roc_curves_all.png` — les 4 courbes ROC superposées (avec AUC), idéal
     pour comparer directement les modèles dans le rapport
   - `detail_h1.png`, `detail_h2.png`, `detail_h3.png`, `detail_h4.png` —
     une figure par modèle (matrice de confusion + courbe ROC côte à côte),
     pratique si vous voulez commenter chaque modèle individuellement

## 6. Points à ajuster avant l'exécution sur le vrai dataset complet

- **`config.EPOCHS`** : 15 par défaut (suffisant pour tester) ; augmentez
  (30-50) pour de meilleures performances sur le dataset complet.
- **`config.BATCH_SIZE`** : 256 par défaut, à adapter selon votre RAM/GPU.
- Le dataset complet UNSW-NB15 contient ~175 000 lignes (train) — prévoyez
  un temps de calcul plus long que sur les données synthétiques de test.
- Si vous souhaitez une analyse multi-classes (par type d'attaque :
  Fuzzers, DoS, Exploits, Generic, etc.), utilisez la colonne
  `attack_cat` à la place de `label`, et adaptez `evaluate.py` en
  conséquence (métriques multi-classes).

## 7. Pour le rapport

- Le tableau généré dans `results/resultats_finaux.csv` correspond
  directement au tableau demandé dans l'énoncé du mini-projet.
- Pensez à ajouter dans le rapport : la matrice de confusion de chaque
  modèle (affichée dans la console pendant l'exécution), une courbe
  ROC/AUC si souhaité, et une discussion comparant les 4 architectures
  (compromis performance / temps de calcul, sensibilité au déséquilibre
  des classes, etc.).
