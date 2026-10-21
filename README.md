# Détection d’anomalies réseau — Approche hybride Machine Learning et Deep Learning

## 1. Présentation du projet

Ce projet consiste à concevoir, entraîner et comparer quatre architectures hybrides combinant le **Deep Learning (DL)** et le **Machine Learning (ML)** classique pour la détection d’anomalies réseau, notamment les intrusions et les activités malveillantes.

Les expérimentations reposent sur le dataset **UNSW-NB15**, qui contient des exemples de trafic réseau normal et malveillant.

L’objectif est d’étudier différentes stratégies combinant l’apprentissage de représentations par des réseaux de neurones et des algorithmes de classification classiques, puis de comparer leurs performances à l’aide de plusieurs métriques d’évaluation.

## 2. Architectures implémentées

Quatre architectures hybrides sont étudiées :

| Modèle | Composant Deep Learning | Classificateur Machine Learning | Principe |
|---|---|---|---|
| **H1** | Auto-Encoder | Random Forest | Combine les caractéristiques du trafic avec une représentation issue de l’apprentissage non supervisé et de l’erreur de reconstruction, selon l’implémentation. |
| **H2** | LSTM | SVM | Exploite une représentation apprise par un réseau LSTM pour effectuer la classification. |
| **H3** | MLP | XGBoost | Combine l’apprentissage de représentations par un MLP avec un classificateur XGBoost. |
| **H4** | CNN 1D | Gradient Boosting | Exploite les motifs appris par un CNN 1D pour alimenter un classificateur Gradient Boosting. |

Cette approche permet d’explorer les complémentarités entre les réseaux de neurones et les algorithmes classiques de Machine Learning.

## 3. Technologies utilisées

- **Langage :** Python
- **Deep Learning :** TensorFlow / Keras
- **Machine Learning :** Scikit-learn, XGBoost
- **Manipulation des données :** Pandas, NumPy
- **Visualisation :** Matplotlib
- **Prétraitement :** encodage des variables catégorielles, normalisation et rééquilibrage des classes avec SMOTE

## 4. Structure du projet

```text
unsw-network-anomaly-detection/
├── data/
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
├── results/
│   ├── confusion_matrices_all.png
│   ├── roc_curves_all.png
│   ├── detail_h1.png
│   ├── detail_h2.png
│   ├── detail_h3.png
│   ├── detail_h4.png
│   └── Tableau des metrics.png
├── README.md
├── config.py
├── evaluate.py
├── main.py
├── models.py
├── plotting.py
├── preprocessing.py
└── requirements.txt
```

### Description des fichiers

- `config.py` : configuration, chemins et paramètres du projet.
- `preprocessing.py` : chargement et préparation des données.
- `models.py` : définition des quatre architectures hybrides.
- `evaluate.py` : évaluation des modèles et calcul des métriques.
- `plotting.py` : génération des graphiques comparatifs.
- `main.py` : point d’entrée du programme et orchestration des étapes.
- `requirements.txt` : liste des dépendances Python.
- `data/` : fichiers du dataset utilisés pour l’entraînement et le test.
- `results/` : graphiques et résultats enregistrés.

## 5. Dataset

Le projet utilise le dataset **UNSW-NB15**, développé pour la recherche sur la détection d’intrusions réseau.

**Source officielle :** [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

Les fichiers attendus sont :

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

Placez ces fichiers dans le dossier `data/`, en conservant leurs noms d’origine.

Les résultats obtenus dépendent des données réellement chargées. Si le programme utilise un jeu de données synthétique de démonstration en l’absence des fichiers attendus, ces résultats ne doivent pas être interprétés comme des performances obtenues sur le dataset UNSW-NB15 réel.

## 6. Installation

### Prérequis

- Python compatible avec les dépendances du projet
- Git
- Les fichiers du dataset UNSW-NB15

### Étape 1 — Cloner le dépôt

```bash
git clone https://github.com/sara-elmekkaoui/unsw-network-anomaly-detection.git
cd unsw-network-anomaly-detection
```

### Étape 2 — Créer un environnement virtuel

Sous Windows avec PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Étape 3 — Installer les dépendances

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Les principales dépendances sont :

- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- XGBoost
- TensorFlow CPU

La compatibilité de TensorFlow avec la version de Python utilisée doit être vérifiée lors de l’installation.

### Étape 4 — Préparer les données

Placez les deux fichiers CSV du dataset dans le dossier `data/`.

Vérifiez que leurs noms correspondent à ceux attendus par le programme.

## 7. Exécution

Depuis le répertoire principal du projet, lancez :

```bash
python main.py
```

Le programme est conçu pour effectuer les opérations suivantes :

1. Charger et prétraiter les données.
2. Encoder les variables catégorielles et préparer les caractéristiques numériques.
3. Rééquilibrer les données d’entraînement avec SMOTE lorsque cette option est activée.
4. Entraîner et évaluer les quatre architectures hybrides.
5. Afficher un tableau récapitulatif des résultats.
6. Enregistrer les métriques calculées dans un fichier CSV.
7. Générer les graphiques d’évaluation dans le dossier `results/`.

La durée d’exécution varie selon le matériel, la configuration des modèles et la taille des données.

## 8. Métriques d’évaluation

Les modèles sont comparés à l’aide de plusieurs indicateurs :

- **Accuracy :** proportion totale de prédictions correctes.
- **Precision :** proportion de prédictions positives qui sont correctes.
- **Recall :** proportion des exemples positifs correctement détectés.
- **F1-score :** moyenne harmonique de la précision et du rappel.
- **ROC-AUC :** capacité du modèle à distinguer les classes selon différents seuils de décision.
- **Temps d’entraînement :** durée nécessaire à l’apprentissage.
- **Temps d’inférence :** durée nécessaire à la production des prédictions.

Les matrices de confusion permettent d’analyser les erreurs de classification, tandis que les courbes ROC facilitent la comparaison des capacités de discrimination.

Dans un contexte de cybersécurité, le rappel et le F1-score sont particulièrement intéressants, car les faux négatifs correspondent à des activités malveillantes qui n’ont pas été détectées.

## 9. Résultats et visualisations

Après une exécution réussie, le programme enregistre les résultats et génère plusieurs visualisations dans le dossier `results/`.

### Matrices de confusion

![Matrices de confusion des quatre modèles](results/confusion_matrices_all.png)

### Courbes ROC

![Courbes ROC comparatives des quatre modèles](results/roc_curves_all.png)

### Analyses détaillées des modèles

- [H1 — Auto-Encoder + Random Forest](results/detail_h1.png)
- [H2 — LSTM + SVM](results/detail_h2.png)
- [H3 — MLP + XGBoost](results/detail_h3.png)
- [H4 — CNN 1D + Gradient Boosting](results/detail_h4.png)

### Fichiers de résultats

| Fichier | Description |
|---|---|
| `resultats_finaux.csv` | Tableau des métriques calculées par le programme, après une exécution réussie. |
| `Tableau des metrics.png` | Visualisation du tableau des métriques, si elle a été générée et conservée dans le dépôt. |
| `confusion_matrices_all.png` | Matrices de confusion comparatives. |
| `roc_curves_all.png` | Courbes ROC comparatives. |
| `detail_h1.png` à `detail_h4.png` | Visualisations détaillées des quatre architectures. |

Les résultats numériques doivent être interprétés en fonction des données utilisées, du protocole expérimental et des conditions d’entraînement.

## 10. Perspectives d’amélioration

Plusieurs pistes peuvent prolonger ce travail :

- Optimiser les hyperparamètres des quatre architectures.
- Comparer les performances avec celles de modèles ML et DL utilisés séparément.
- Analyser les faux positifs et les faux négatifs.
- Étudier les compromis entre qualité de détection, temps d’entraînement et temps d’inférence.
- Étendre l’étude à la classification multiclasse des catégories d’attaques à partir de la variable `attack_cat`.
- Évaluer la généralisation sur d’autres jeux de données réseau.
- Renforcer la reproductibilité des expériences en documentant les paramètres et les versions des bibliothèques.

## 11. Ressources

**Dataset :** [UNSW-NB15 — Site officiel](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

**Dépôt GitHub :** [sara-elmekkaoui/unsw-network-anomaly-detection](https://github.com/sara-elmekkaoui/unsw-network-anomaly-detection)
