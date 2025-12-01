import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    recall_score,
)
from sklearn.calibration import calibration_curve

# ───────────── KONFIGURACJA ŚCIEŻEK ─────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # .../IWUM-Projekt-1/Modele_interpretowalne
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..")) # .../IWUM-Projekt-1

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, "zbiór_7.csv")
PREPROC_DIR = os.path.join(PROJECT_ROOT, "EDA", "preprocesing_pipelines")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "wykresy_oceny_jakosci")

os.makedirs(PLOTS_DIR, exist_ok=True)


# =====================================================================
#                 FUNKCJE POMOCNICZE / METRYKI DODATKOWE
# =====================================================================

def calculate_ks_statistic(y_true, y_pred_proba):
    df = pd.DataFrame({"y": y_true, "p": y_pred_proba}).sort_values("p")
    pos = df.loc[df["y"] == 1, "p"]
    neg = df.loc[df["y"] == 0, "p"]

    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    from scipy.stats import ks_2samp
    ks_stat, _ = ks_2samp(pos, neg)
    return ks_stat


def print_recall_specificity(y_true, y_prob, threshold, model_name, dataset_name):
    """
    Liczy i wypisuje recall (TPR) oraz specificity (TNR) dla zadanego progu.
    """
    y_pred = (y_prob >= threshold).astype(int)

    rec = recall_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    print(f"\n=== {model_name} – {dataset_name} (threshold={threshold:.2f}) ===")
    print(f"Recall (TPR):      {rec:.4f}")
    print(f"Specificity (TNR): {spec:.4f}")


# =====================================================================
#                            WYKRESY
# =====================================================================

def plot_roc(models, savepath=None):
    """ROC curves dla obu modeli (val + test)."""
    if savepath is None:
        savepath = os.path.join(PLOTS_DIR, "roc_logit_tree.png")

    plt.figure(figsize=(7, 6))
    for name, y_val, p_val, y_test, p_test in models:
        fpr_val, tpr_val, _ = roc_curve(y_val, p_val)
        fpr_test, tpr_test, _ = roc_curve(y_test, p_test)

        auc_val = roc_auc_score(y_val, p_val)
        auc_test = roc_auc_score(y_test, p_test)

        plt.plot(fpr_val, tpr_val, label=f"{name} – Val (AUC={auc_val:.3f})")
        plt.plot(fpr_test, tpr_test, label=f"{name} – Test (AUC={auc_test:.3f})")

    plt.plot([0, 1], [0, 1], "--", label="Losowy model")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve – Logit (WoE) vs Decision Tree")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()


def plot_pr(models, savepath=None):
    """Precision–Recall curves dla obu modeli (val + test)."""
    if savepath is None:
        savepath = os.path.join(PLOTS_DIR, "pr_logit_tree.png")

    plt.figure(figsize=(7, 6))

    for name, y_val, p_val, y_test, p_test in models:
        prec_val, rec_val, _ = precision_recall_curve(y_val, p_val)
        prec_test, rec_test, _ = precision_recall_curve(y_test, p_test)

        ap_val = average_precision_score(y_val, p_val)
        ap_test = average_precision_score(y_test, p_test)

        plt.plot(rec_val, prec_val, label=f"{name} – Val (AP={ap_val:.3f})")
        plt.plot(rec_test, prec_test, label=f"{name} – Test (AP={ap_test:.3f})")

    baseline = models[0][1].mean()
    plt.hlines(baseline, 0, 1, linestyles="--", label=f"Baseline ({baseline:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve – Logit (WoE) vs Decision Tree")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()


def plot_calibration(models, savepath=None, n_bins=10):
    """Calibration / reliability plot dla obu modeli (val + test)."""
    if savepath is None:
        savepath = os.path.join(PLOTS_DIR, "calibration_logit_tree.png")

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "--", label="Idealna kalibracja")

    for name, y_val, p_val, y_test, p_test in models:
        pt_val, pp_val = calibration_curve(y_val, p_val, n_bins=n_bins)
        pt_test, pp_test = calibration_curve(y_test, p_test, n_bins=n_bins)

        plt.plot(pp_val, pt_val, "o-", label=f"{name} – Val")
        plt.plot(pp_test, pt_test, "s-", label=f"{name} – Test")

    plt.xlabel("Średnie przewidziane PD (bin)")
    plt.ylabel("Rzeczywista częstość defaultów")
    plt.title("Calibration – Logit (WoE) vs Decision Tree")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()


def plot_hist(models, out_dir=None):
    """Histogramy PD dla good/bad osobno dla każdego modelu i zbioru."""
    if out_dir is None:
        out_dir = PLOTS_DIR

    for name, y_val, p_val, y_test, p_test in models:
        for ds_name, y, p in [("val", y_val, p_val), ("test", y_test, p_test)]:
            plt.figure(figsize=(7, 6))
            df = pd.DataFrame({"y": y, "p": p})

            plt.hist(
                df[df["y"] == 0]["p"],
                bins=20,
                alpha=0.6,
                density=True,
                label="Good",
            )
            plt.hist(
                df[df["y"] == 1]["p"],
                bins=20,
                alpha=0.6,
                density=True,
                label="Bad",
            )

            plt.xlabel("Przewidywane PD")
            plt.ylabel("Gęstość")
            plt.title(f"Histogram PD – {name} – {ds_name}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            fname = f"hist_{name}_{ds_name}.png"
            plt.savefig(os.path.join(out_dir, fname), dpi=150)
            plt.close()


# =====================================================================
#                                MAIN
# =====================================================================

def main():
    print(" Wczytywanie danych i modeli...")
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["default"])
    y = df["default"]

    # modele
    logit = joblib.load(os.path.join(MODELS_DIR, "best_logistic_regression_woe.pkl"))
    tree = joblib.load(os.path.join(MODELS_DIR, "best_decision_tree.pkl"))

    # preprocessing
    preproc_logit = joblib.load(os.path.join(PREPROC_DIR, "preprocessing_logit_woe.pkl"))
    preproc_tree = joblib.load(os.path.join(PREPROC_DIR, "preprocessing_tree.pkl"))

    # podział danych (jak w innych plikach)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # transformacje
    X_val_logit = preproc_logit.transform(X_val)
    X_test_logit = preproc_logit.transform(X_test)

    X_val_tree = preproc_tree.transform(X_val)
    X_test_tree = preproc_tree.transform(X_test)

    # predykcje
    p_val_logit = logit.predict_proba(X_val_logit)[:, 1]
    p_test_logit = logit.predict_proba(X_test_logit)[:, 1]

    p_val_tree = tree.predict_proba(X_val_tree)[:, 1]
    p_test_tree = tree.predict_proba(X_test_tree)[:, 1]

    # pakujemy modele do listy dla wygody
    MODELE = [
        ("Logit", y_val, p_val_logit, y_test, p_test_logit),
        ("Tree",  y_val, p_val_tree,  y_test, p_test_tree),
    ]

    # =====================================================================
    #        RECALL I SPECIFICITY DLA WYBRANEGO PROGU (np. 0.2)
    # =====================================================================
    threshold = 0.20
    print("\n>>> Metryki dla progu decyzyjnego PD =", threshold)

    # Logit – val i test
    print_recall_specificity(y_val,  p_val_logit, threshold, "Logit", "VAL")
    print_recall_specificity(y_test, p_test_logit, threshold, "Logit", "TEST")

    # Tree – val i test
    print_recall_specificity(y_val,  p_val_tree, threshold, "Tree", "VAL")
    print_recall_specificity(y_test, p_test_tree, threshold, "Tree", "TEST")

    # =====================================================================
    #                          WYKRESY
    # =====================================================================
    print("\n Rysuję ROC...")
    plot_roc(MODELE)

    print(" Rysuję PR...")
    plot_pr(MODELE)

    print(" Rysuję calibration...")
    plot_calibration(MODELE)

    print(" Rysuję histogramy PD...")
    plot_hist(MODELE)

    print(" Wygenerowano wykresy dla logitu i drzewca!")
    print(f"   Pliki zapisane w: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
