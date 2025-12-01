import os
import pandas as pd
import numpy as np
import joblib
# reszta importów sklearn, warnings, itd.

# ───────────── KONFIGURACJA ŚCIEŻEK ─────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # .../IWUM-Projekt-1/Modele_interpretowalne
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..")) # .../IWUM-Projekt-1

DATA_PATH = os.path.join(PROJECT_ROOT, "zbiór_7.csv")
PREPROC_DIR = os.path.join(PROJECT_ROOT, "EDA", "preprocesing_pipelines")

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "model_results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)
from scipy.stats import ks_2samp
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
#                            CUSTOM METRICS
# =====================================================================

def gini_from_auc(auc):
    return 2 * auc - 1

def calculate_ks_statistic(y_true, y_pred_proba):
    """Kolmogorov-Smirnov statistic."""
    data = pd.DataFrame({"y": y_true, "p": y_pred_proba}).sort_values("p")

    pos_probs = data.loc[data["y"] == 1, "p"]
    neg_probs = data.loc[data["y"] == 0, "p"]

    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return np.nan

    ks_stat, _ = ks_2samp(pos_probs, neg_probs)
    return ks_stat


# =====================================================================
#                        GRIDY HIPERPARAMETRÓW
# =====================================================================

def create_logistic_regression_grid():
    """
    Logit na WoE — legalne kombinacje penalty/solver:
    - L2 + lbfgs / newton-cg
    - L1 + saga / liblinear
    - Elasticnet + saga
    """
    base_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )

    param_grid = [
        {
            "penalty": ["l2"],
            "solver": ["lbfgs", "newton-cg"],
            "C": [0.01, 0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
        },
        {
            "penalty": ["l1"],
            "solver": ["liblinear", "saga"],
            "C": [0.01, 0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
        },
        {
            "penalty": ["elasticnet"],
            "solver": ["saga"],
            "l1_ratio": [0.3, 0.5, 0.7],
            "C": [0.01, 0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
        },
    ]

    return base_model, param_grid


def create_decision_tree_grid():
    """
    Drzewo interpretowalne (płytkie) + pruning.
    """
    model = DecisionTreeClassifier(random_state=42)

    param_grid = {
        "max_depth": [3, 4, 5, 7, 10],
        "min_samples_split": [20, 50, 100],
        "min_samples_leaf": [20, 50, 100],
        "criterion": ["gini", "entropy"],
        "class_weight": [None, "balanced"],
        "ccp_alpha": [0.0, 0.001, 0.01],
    }

    return model, param_grid


# =====================================================================
#                           EWALUACJA MODELI
# =====================================================================

def evaluate_model(model, X, y, model_name="Model", dataset_name="val"):
    y_pred_proba = model.predict_proba(X)[:, 1]

    roc = roc_auc_score(y, y_pred_proba)
    
    return {
        "model_name": model_name,
        "dataset": dataset_name,
        "roc_auc": roc,
        "gini": 2 * roc - 1,
        "pr_auc": average_precision_score(y, y_pred_proba),
        "ks": calculate_ks_statistic(y, y_pred_proba),
        "log_loss": log_loss(y, y_pred_proba),
        "brier": brier_score_loss(y, y_pred_proba),
    }

def print_evaluation_table(results):
    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("              WYNIKI MODELI (VAL + TEST)")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    return df


# =====================================================================
#                         GRIDSEARCH DLA MODELU
# =====================================================================

def train_with_gridsearch(
    model, param_grid, X_train, y_train, model_name="Model", cv=5
):
    print("\n" + "=" * 80)
    print(f" GridSearch: {model_name}")
    print("=" * 80)

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",  #  tylko ROC-AUC, żadnych custom scorerów
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )

    gs.fit(X_train, y_train)

    print("\nNajlepsze parametry:")
    print(gs.best_params_)
    print(f"Najlepszy ROC-AUC CV: {gs.best_score_:.4f}")

    return gs.best_estimator_, gs


# =====================================================================
#                                MAIN
# =====================================================================

def main():
    print(" Wczytywanie danych...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["default"])
    y = df["default"]

    # Podział jak w EDA.py — 60/20/20
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print("\n Ładowanie pipeline’ów...")
    tree_preproc = joblib.load(os.path.join(PREPROC_DIR, "preprocessing_tree.pkl"))
    logit_preproc = joblib.load(os.path.join(PREPROC_DIR, "preprocessing_logit_woe.pkl"))

    print("\n Transformacja danych dla drzewa...")
    X_train_tree = tree_preproc.transform(X_train)
    X_val_tree = tree_preproc.transform(X_val)
    X_test_tree = tree_preproc.transform(X_test)

    print("\n Transformacja danych dla logitu...")
    X_train_logit = logit_preproc.transform(X_train)
    X_val_logit = logit_preproc.transform(X_val)
    X_test_logit = logit_preproc.transform(X_test)

    # ============================
    #       GRIDSEARCH LOGIT
    # ============================
    logit_model, logit_grid = create_logistic_regression_grid()
    best_logit, gs_logit = train_with_gridsearch(
        logit_model, logit_grid, X_train_logit, y_train, "Logit (WoE)", cv=5
    )

    # ============================
    #       GRIDSEARCH DRZEWO
    # ============================
    tree_model, tree_grid = create_decision_tree_grid()
    best_tree, gs_tree = train_with_gridsearch(
        tree_model, tree_grid, X_train_tree, y_train, "Decision Tree", cv=5
    )

    # ============================
    #            EWALUACJA
    # ============================
    results = []

    # logit
    results.append(evaluate_model(best_logit, X_val_logit, y_val, "Logit_WoE", "val"))
    results.append(evaluate_model(best_logit, X_test_logit, y_test, "Logit_WoE", "test"))

    # drzewo
    results.append(evaluate_model(best_tree, X_val_tree, y_val, "DecisionTree", "val"))
    results.append(evaluate_model(best_tree, X_test_tree, y_test, "DecisionTree", "test"))

    df_results = print_evaluation_table(results)

    # ============================
    #             ZAPIS
    # ============================
    print("\n Zapisujemy modele...")

    joblib.dump(best_logit, os.path.join(MODELS_DIR, "best_logistic_regression_woe.pkl"))
    joblib.dump(best_tree, os.path.join(MODELS_DIR, "best_decision_tree.pkl"))
    
    df_results.to_csv(
        os.path.join(RESULTS_DIR, "model_evaluation_results.csv"),
        index=False,
    )
    
    pd.DataFrame(gs_logit.cv_results_).to_csv(
        os.path.join(RESULTS_DIR, "grid_results_logit_woe.csv"),
        index=False,
    )
    pd.DataFrame(gs_tree.cv_results_).to_csv(
        os.path.join(RESULTS_DIR, "grid_results_tree.csv"),
        index=False,
    )
    
    print("\n================ BETA COEFFICIENTS ================\n")

    # pobierz nazwy zmiennych po transformacji WOE + DropColumns
    woe_feature_names = logit_preproc.get_feature_names_out()
    
    # ale DropColumnsTransformer uciął kolumny — więc
    # pobieramy REALNE nazwy cech po transformacji
    X_logit_df = pd.DataFrame(X_train_logit)
    feature_names = list(X_logit_df.columns)
    
    # współczynniki
    betas = best_logit.coef_[0]
    intercept = best_logit.intercept_[0]
    
    print(f"Intercept (β0): {intercept:.6f}\n")
    
    for fname, beta in zip(feature_names, betas):
        print(f"{fname:40s}  β = {beta:.6f}")

    print("Zapisano wszystkie modele i wyniki.")

    return best_logit, best_tree, df_results


if __name__ == "__main__":
    main()
