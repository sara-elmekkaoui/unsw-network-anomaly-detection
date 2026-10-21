import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import config


def _generate_synthetic_dataset(n_train=6000, n_test=2000, seed=config.RANDOM_STATE):

    rng = np.random.RandomState(seed)

    def make(n):
        df = pd.DataFrame({
            "id": np.arange(n),
            "dur": rng.exponential(1.0, n),
            "proto": rng.choice(["tcp", "udp", "icmp"], n),
            "service": rng.choice(["-", "http", "dns", "ftp", "smtp"], n),
            "state": rng.choice(["FIN", "CON", "INT", "REQ"], n),
            "spkts": rng.poisson(10, n),
            "dpkts": rng.poisson(8, n),
            "sbytes": rng.exponential(500, n),
            "dbytes": rng.exponential(400, n),
            "rate": rng.exponential(50, n),
            "sttl": rng.randint(0, 255, n),
            "dttl": rng.randint(0, 255, n),
            "sload": rng.exponential(1000, n),
            "dload": rng.exponential(900, n),
            "sloss": rng.poisson(1, n),
            "dloss": rng.poisson(1, n),
            "ct_srv_src": rng.poisson(3, n),
            "ct_dst_ltm": rng.poisson(3, n),
            "ct_src_dport_ltm": rng.poisson(3, n),
        })
        label = rng.choice([0, 1], size=n, p=[0.68, 0.32])
       
        df.loc[label == 1, "sbytes"] *= 3
        df.loc[label == 1, "rate"] *= 4
        df["label"] = label
        df["attack_cat"] = np.where(label == 1, "Generic", "Normal")
        return df

    return make(n_train), make(n_test)


def load_raw_data():
    """Charge les CSV réels si présents, sinon génère des données synthétiques."""
    if os.path.exists(config.TRAIN_PATH) and os.path.exists(config.TEST_PATH):
        print(f"[INFO] Chargement des données réelles depuis {config.DATA_DIR}")
        train_df = pd.read_csv(config.TRAIN_PATH)
        test_df = pd.read_csv(config.TEST_PATH)
    else:
        print("[ATTENTION] Fichiers UNSW-NB15 introuvables dans 'data/'.")
        print("            -> Génération d'un dataset SYNTHÉTIQUE de démonstration.")
        print("            Placez 'UNSW_NB15_training-set.csv' et")
        print("            'UNSW_NB15_testing-set.csv' dans le dossier data/")
        print("            pour utiliser les vraies données.")
        train_df, test_df = _generate_synthetic_dataset()
    return train_df, test_df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage de base : suppression doublons et gestion des NA."""
    df = df.drop_duplicates()
    # colonnes numériques : on remplit les NA par la médiane
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode()[0])
    return df


def build_preprocessor(df_train: pd.DataFrame, feature_cols, categorical_cols):
    """Construit un ColumnTransformer (One-Hot + StandardScaler) ajusté sur le train."""
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )
    preprocessor.fit(df_train[feature_cols])
    return preprocessor


def prepare_data(use_smote: bool = True):
    """
    Pipeline complet de préparation des données.

    Retourne :
        X_train, y_train, X_test, y_test (numpy arrays prêts pour les modèles)
    """
    train_df, test_df = load_raw_data()

    train_df = clean_dataframe(train_df)
    test_df = clean_dataframe(test_df)

    drop_cols = [c for c in [config.ID_COL, config.TARGET_COL, config.ATTACK_CAT_COL] if c in train_df.columns]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    categorical_cols = [c for c in config.CATEGORICAL_COLS if c in feature_cols]

    y_train = train_df[config.TARGET_COL].astype(int).values
    y_test = test_df[config.TARGET_COL].astype(int).values

    preprocessor = build_preprocessor(train_df, feature_cols, categorical_cols)

    X_train = preprocessor.transform(train_df[feature_cols])
    X_test = preprocessor.transform(test_df[feature_cols])

    print(f"[INFO] Dimensions avant SMOTE -> X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"[INFO] Répartition des classes (train) : {np.bincount(y_train)}")

    if use_smote:
        sm = SMOTE(random_state=config.RANDOM_STATE)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"[INFO] Dimensions après SMOTE -> X_train: {X_train.shape}")
        print(f"[INFO] Répartition des classes après SMOTE : {np.bincount(y_train)}")

    return X_train.astype(np.float32), y_train, X_test.astype(np.float32), y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = prepare_data()
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
