import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    
    print(f"\n[INFO] Entraînement du modèle {model.name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)  # probabilité de la classe "anomalie"

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    print(f"[INFO] {model.name} -> Accuracy={acc:.4f} | AUC={roc_auc:.4f} | "
          f"Train time={model.train_time_:.2f}s | Predict time={model.predict_time_:.2f}s")
    print(f"[INFO] Matrice de confusion (lignes=réel, colonnes=prédit) :\n{cm}")

    rows = [
        {
            "Modele": model.name,
            "Classe": "Sans anomalie",
            "Precision": round(precision[0], 4),
            "Rappel": round(recall[0], 4),
            "F1-Score": round(f1[0], 4),
            "Accuracy": round(acc, 4),
            "Temps_entrainement_s": round(model.train_time_, 2),
            "Temps_inference_s": round(model.predict_time_, 4),
        },
        {
            "Modele": model.name,
            "Classe": "Avec anomalie",
            "Precision": round(precision[1], 4),
            "Rappel": round(recall[1], 4),
            "F1-Score": round(f1[1], 4),
            "Accuracy": round(acc, 4),
            "Temps_entrainement_s": round(model.train_time_, 2),
            "Temps_inference_s": round(model.predict_time_, 4),
        },
    ]

    extra = {
        "name": model.name,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc,
    }
    return rows, extra


def build_results_table(models, X_train, y_train, X_test, y_test):
    all_rows = []
    all_extras = []
    for model in models:
        rows, extra = evaluate_model(model, X_train, y_train, X_test, y_test)
        all_rows.extend(rows)
        all_extras.append(extra)
    df = pd.DataFrame(all_rows)
    df = df.set_index(["Modele", "Classe"])
    return df, all_extras
