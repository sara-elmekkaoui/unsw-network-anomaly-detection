# Détection d’anomalies réseau — Approche hybride Machine Learning et Deep Learning

## 1. Présentation du projet

Ce projet vise à concevoir et comparer quatre architectures hybrides combinant le **Deep Learning (DL)** et le **Machine Learning (ML)** classique pour la détection d’anomalies réseau, notamment les intrusions et les activités malveillantes.

Les expérimentations s’appuient sur le dataset **UNSW-NB15**, qui contient des exemples de trafic réseau normal et malveillant.

L’objectif est d’évaluer différentes stratégies d’extraction de caractéristiques et de classification selon leurs performances de détection, leur capacité de généralisation et leur coût de calcul.

## 2. Architectures implémentées

Quatre modèles hybrides sont étudiés :

| Modèle | Extraction de caractéristiques (DL) | Classification (ML) | Principe |
|---|---|---|---|
| **H1** | Auto-Encoder | Random Forest | Exploite l’erreur de reconstruction comme indicateur supplémentaire d’anomalie, combiné aux caractéristiques originales. |
| **H2** | LSTM | SVM | Apprend une représentation des caractéristiques avec un réseau LSTM, puis utilise cette représentation pour la classification. |
| **H3** | MLP | XGBoost | Combine les caractéristiques originales et les représentations apprises par le MLP avant la classification. |
| **H4** | CNN 1D | Gradient Boosting | Extrait des motifs locaux dans les vecteurs de caractéristiques, puis effectue la classification à partir de la représentation obtenue. |

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
├── config.py
├── preprocessing.py
├── models.py
├── evaluate.py
├── plotting.py
├── main.py
├── requirements.txt
├── README.md
├── data/
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
└── results/
    ├── confusion_matrices_all.png
    ├── roc_curves_all.png
    ├── detail_h1.png
    ├── detail_h2.png
    ├── detail_h3.png
    └── detail_h4.png
```

Les fichiers Python sont organisés autour de plusieurs modules :

- `config.py` : chemins, constantes et hyperparamètres.
- `preprocessing.py` : chargement et préparation des données.
- `models.py` : implémentation des quatre architectures hybrides.
- `evaluate.py` : calcul des métriques et collecte des données d’évaluation.
- `plotting.py` : génération des graphiques comparatifs.
- `main.py` : script principal d’exécution.
- `requirements.txt` : dépendances nécessaires au projet.

Le dépôt contient également un dossier `Mini projet/` avec une copie secondaire de certains scripts et graphiques.

## 5. Dataset

Le projet utilise le dataset **UNSW-NB15**, développé pour la recherche sur la détection d’intrusions réseau.

**Page officielle :** [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

Les deux fichiers attendus sont :

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

Placez-les dans le dossier `data/` en conservant leurs noms d’origine.

Si les fichiers sont absents, le programme peut utiliser un jeu de données synthétique de démonstration, conformément au comportement prévu dans le code. Les résultats obtenus sur ces données synthétiques ne représentent pas les performances des modèles sur le dataset UNSW-NB15 réel.

## 6. Installation

### Cloner le dépôt

```bash
git clone https://github.com/sara-elmekkaoui/unsw-network-anomaly-detection.git
cd unsw-network-anomaly-detection
```

### Créer un environnement virtuel

Sous Windows :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Installer les dépendances

```bash
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

Les versions exactes utilisées peuvent varier selon l’environnement Python et les dépendances installées.

## 7. Exécution

Depuis le répertoire principal du projet, exécutez :

```bash
python main.py
```

Le programme est conçu pour effectuer les opérations suivantes :

1. Charger et prétraiter les données.
2. Encoder les variables catégorielles et normaliser les caractéristiques numériques.
3. Rééquilibrer les données d’entraînement à l’aide de SMOTE.
4. Entraîner et évaluer les quatre architectures hybrides H1, H2, H3 et H4.
5. Afficher un tableau récapitulatif des résultats dans la console.
6. Enregistrer les métriques dans un fichier CSV.
7. Générer les graphiques d’évaluation dans le dossier `results/`.

La durée d’exécution dépend notamment du matériel disponible, des hyperparamètres et de la taille du jeu de données.

## 8. Évaluation des modèles

Les architectures sont comparées à l’aide de plusieurs métriques :

- **Accuracy :** proportion totale de prédictions correctes.
- **Précision :** proportion de prédictions positives qui sont correctes.
- **Rappel :** capacité à identifier les exemples réellement positifs.
- **F1-score :** compromis entre précision et rappel.
- **ROC-AUC :** capacité à distinguer les classes selon différents seuils de décision.
- **Temps d’entraînement :** durée nécessaire à l’apprentissage.
- **Temps d’inférence :** durée nécessaire pour produire les prédictions.

Les matrices de confusion permettent d’étudier les erreurs de classification, tandis que les courbes ROC facilitent la comparaison des capacités de discrimination des modèles.

## 9. Résultats et visualisations

Après une exécution réussie de `python main.py`, le programme enregistre le tableau récapitulatif et génère les visualisations dans le dossier `results/`.

| Fichier | Description |
|---|---|
| `resultats_finaux.csv` | Tableau des métriques d’évaluation calculées par le programme. |
| `confusion_matrices_all.png` | Matrices de confusion comparatives des quatre modèles. |
| `roc_curves_all.png` | Courbes ROC comparatives avec les valeurs AUC, selon l’implémentation. |
| `detail_h1.png` | Visualisation détaillée du modèle H1. |
| `detail_h2.png` | Visualisation détaillée du modèle H2. |
| `detail_h3.png` | Visualisation détaillée du modèle H3. |
| `detail_h4.png` | Visualisation détaillée du modèle H4. |

Les résultats dépendent des données utilisées et du bon déroulement de l’entraînement et de l’évaluation.

## 10. Perspectives d’amélioration

Plusieurs pistes peuvent prolonger ce travail :

- Optimiser les hyperparamètres des quatre architectures.
- Étudier les compromis entre précision, rappel et temps de calcul.
- Analyser les faux positifs et les faux négatifs.
- Étendre l’étude à la classification multiclasse des catégories d’attaques à partir de la colonne `attack_cat`.
- Évaluer la capacité de généralisation des modèles sur d’autres jeux de données réseau.

## 11. Auteurs et ressources

**Cadre académique :** Master Systèmes d’Information Décisionnels et Imagerie (SIDI)  
**Module :** Sécurité des systèmes informatiques  
**Dataset :** [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

**Dépôt GitHub :** [unsw-network-anomaly-detection](https://github.com/sara-elmekkaoui/unsw-network-anomaly-detection)
