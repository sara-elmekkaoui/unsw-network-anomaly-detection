import os
import pandas as pd

import config
from preprocessing import prepare_data
from models import get_all_models
from evaluate import build_results_table
from plotting import generate_all_figures


def main():
    print("=" * 70)
    print("Détection d'anomalies IoT/réseau - Approche hybride ML + DL")
    print("Dataset : UNSW-NB15")
    print("=" * 70)

    X_train, y_train, X_test, y_test = prepare_data(use_smote=True)

    models = get_all_models()

    results_df, all_extras = build_results_table(models, X_train, y_train, X_test, y_test)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", None)
    print("\n" + "=" * 70)
    print("TABLEAU FINAL DES RESULTATS")
    print("=" * 70)
    print(results_df)

    out_path = os.path.join(config.RESULTS_DIR, "resultats_finaux.csv")
    results_df.to_csv(out_path, encoding="utf-8-sig")
    print(f"\n[INFO] Résultats sauvegardés dans : {out_path}")

    print("\n" + "=" * 70)
    print("GENERATION DES GRAPHIQUES")
    print("=" * 70)
    generate_all_figures(all_extras, output_dir=config.RESULTS_DIR)
    print(f"\n[INFO] Toutes les figures sont dans : {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()