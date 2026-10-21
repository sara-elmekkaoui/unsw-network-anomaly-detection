import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

import config

CLASS_NAMES = ["Sans anomalie", "Avec anomalie"]


def _plot_confusion_matrix_on_ax(ax, cm, title):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Prédit", fontsize=9)
    ax.set_ylabel("Réel", fontsize=9)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(
                j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center", va="center", color=color, fontsize=8,
            )
    return im


def plot_all_confusion_matrices(all_extras, save_path):
    n = len(all_extras)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, extra in zip(axes, all_extras):
        _plot_confusion_matrix_on_ax(ax, extra["confusion_matrix"], extra["name"])
    # masquer les sous-graphiques inutilisés s'il y en a
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Matrices de confusion des modèles hybrides", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Figure sauvegardée : {save_path}")


def plot_combined_roc_curves(all_extras, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for extra, color in zip(all_extras, colors):
        ax.plot(
            extra["fpr"], extra["tpr"], color=color, lw=2,
            label=f"{extra['name']} (AUC = {extra['auc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1, label="Aléatoire (AUC = 0.5)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR)")
    ax.set_title("Courbes ROC — comparaison des modèles hybrides", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Figure sauvegardée : {save_path}")


def plot_individual_model_figure(extra, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    _plot_confusion_matrix_on_ax(axes[0], extra["confusion_matrix"], "Matrice de confusion")

    axes[1].plot(extra["fpr"], extra["tpr"], color="#1f77b4", lw=2,
                 label=f"AUC = {extra['auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Taux de faux positifs (FPR)")
    axes[1].set_ylabel("Taux de vrais positifs (TPR)")
    axes[1].set_title("Courbe ROC")
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle(extra["name"], fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Figure sauvegardée : {save_path}")


def generate_all_figures(all_extras, output_dir=None):
    
    output_dir = output_dir or config.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    plot_all_confusion_matrices(
        all_extras, os.path.join(output_dir, "confusion_matrices_all.png")
    )
    plot_combined_roc_curves(
        all_extras, os.path.join(output_dir, "roc_curves_all.png")
    )
    for extra in all_extras:
        safe_name = (
            extra["name"].split(" (")[0].strip().replace(" ", "_").lower()
        )  # ex: "h1"
        plot_individual_model_figure(
            extra, os.path.join(output_dir, f"detail_{safe_name}.png")
        )
