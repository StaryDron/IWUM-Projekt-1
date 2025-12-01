import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
import re
import ast
warnings.filterwarnings("ignore")

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)
from sklearn.neural_network import MLPClassifier
from scipy.stats import ks_2samp, randint, uniform
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# KONFIGURACJA SCIEZEK
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

#  DODAJ TE LINIE - dodaj folder EDA do sys.path
EDA_DIR = os.path.join(PROJECT_ROOT, "EDA")
if EDA_DIR not in sys.path:
    sys.path.append(EDA_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "zbiór_7.csv")
PREPROC_DIR = os.path.join(PROJECT_ROOT, "EDA", "preprocesing_pipelines")
MODELS_DIR = os.path.join(BASE_DIR, "models_blackbox")
RESULTS_DIR = os.path.join(BASE_DIR, "blackbox_results")
SHAP_DIR = os.path.join(RESULTS_DIR, "shap_plots")
LIME_DIR = os.path.join(RESULTS_DIR, "lime_explanations")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(LIME_DIR, exist_ok=True)
def calculate_ks_statistic(y_true, y_pred_proba):
    data = pd.DataFrame({"y": y_true, "p": y_pred_proba}).sort_values("p")
    pos_probs = data.loc[data["y"] == 1, "p"]
    neg_probs = data.loc[data["y"] == 0, "p"]
    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return np.nan
    ks_stat, _ = ks_2samp(pos_probs, neg_probs)
    return ks_stat

def create_xgboost_grid():
    model = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    param_distributions = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.2),
        "min_child_weight": randint(1, 10),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha": uniform(0, 1),
        "reg_lambda": uniform(0, 2),
        "scale_pos_weight": [1, 2, 3],
    }
    return model, param_distributions

def create_lightgbm_grid():
    model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    param_distributions = {
        "n_estimators": randint(100, 500),
        "num_leaves": randint(20, 100),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.2),
        "min_data_in_leaf": randint(20, 200),
        "feature_fraction": uniform(0.6, 0.4),
        "bagging_fraction": uniform(0.6, 0.4),
        "bagging_freq": [5],
        "lambda_l1": uniform(0, 1),
        "lambda_l2": uniform(0, 2),
        "scale_pos_weight": [1, 2, 3],
    }
    return model, param_distributions

def create_mlp_grid():
    model = MLPClassifier(
        random_state=42,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
    )
    param_distributions = {
        "hidden_layer_sizes": [(50,), (100,), (150,), (50, 25), (100, 50), (150, 75)],
        "activation": ["relu", "tanh"],
        "alpha": uniform(0.0001, 0.01),
        "learning_rate_init": uniform(0.001, 0.01),
        "batch_size": [32, 64, 128],
    }
    return model, param_distributions

def evaluate_model(model, X, y, model_name="Model", dataset_name="val"):
    y_pred_proba = model.predict_proba(X)[:, 1]
    return {
        "model_name": model_name,
        "dataset": dataset_name,
        "roc_auc": roc_auc_score(y, y_pred_proba),
        "pr_auc": average_precision_score(y, y_pred_proba),
        "ks": calculate_ks_statistic(y, y_pred_proba),
        "log_loss": log_loss(y, y_pred_proba),
        "brier": brier_score_loss(y, y_pred_proba),
    }

def print_evaluation_table(results):
    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print(" WYNIKI MODELI BLACK-BOX (VAL + TEST)")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    return df

def train_with_randomized_search(model, param_distributions, X_train, y_train, model_name="Model", n_iter=50, cv=5):
    print("\n" + "=" * 80)
    print(f"Tuning {model_name} z RandomizedSearchCV")
    print("=" * 80)
    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True,
    )
    rs.fit(X_train, y_train)
    print("\nNajlepsze parametry:")
    print(rs.best_params_)
    print(f"Najlepszy ROC-AUC CV: {rs.best_score_:.4f}")
    cv_results = pd.DataFrame(rs.cv_results_)
    best_idx = rs.best_index_
    train_score = cv_results.loc[best_idx, "mean_train_score"]
    val_score = cv_results.loc[best_idx, "mean_test_score"]
    print(f"Train ROC-AUC: {train_score:.4f}, Val ROC-AUC: {val_score:.4f}")
    print(f"Overfitting gap: {train_score - val_score:.4f}")
    return rs.best_estimator_, rs

import re
import matplotlib.pyplot as plt

def generate_shap_explanations(model, X_train, X_test, feature_names, model_name):
    print(f"\nGenerowanie wyjasnien SHAP dla {model_name}...")

    # Poprawka base_score dla XGBoost, jeśli potrzebna (jeśli używasz XGBoost)
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        base_score = booster.attr("base_score")
        if base_score is not None:
            if isinstance(base_score, str):
                try:
                    base_score = ast.literal_eval(base_score)
                except Exception:
                    pass
            if isinstance(base_score, (list, tuple, np.ndarray)):
                base_score = base_score[0]
            base_score = float(base_score)
            booster.set_param("base_score", base_score)

    import shap
    import matplotlib.pyplot as plt

    if hasattr(model, "get_booster") or hasattr(model, "booster_"):
        explainer = shap.TreeExplainer(model)
    else:
        background = shap.sample(X_train, 100)
        explainer = shap.KernelExplainer(model.predict_proba, background)

    shap_values = explainer.shap_values(X_test[:500])

    # Obsługa różnych formatów shap_values (lista po klasach lub ndarray 3D)
    if isinstance(shap_values, list):
        shap_values_class = shap_values[1]  # wybierz SHAP dla klasy pozytywnej (indeks 1)
    elif len(shap_values.shape) == 3:
        shap_values_class = shap_values[..., 1]  # wybierz SHAP dla klasy pozytywnej
    else:
        shap_values_class = shap_values

    mean_abs_shap = np.abs(shap_values_class).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-3:][::-1]

    # wykres podsumowujący (bar)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_class, X_test[:500], feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, f"{model_name}_summary_bar.png"), dpi=150)
    plt.close()

    # wykres typu beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_class, X_test[:500], feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, f"{model_name}_beeswarm.png"), dpi=150)
    plt.close()

    # wykresy zależności dla 3 najważniejszych cech
    for idx in top_features_idx:
        shap.dependence_plot(idx, shap_values_class, X_test[:500], feature_names=feature_names, interaction_index=None, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, f"{model_name}_dependence_{feature_names[idx]}.png"), dpi=150)
        plt.close()

    print(f"Wykresy SHAP zapisane w {SHAP_DIR}")

    return explainer, shap_values_class


def generate_lime_explanations(model, X_train, X_test, y_test, feature_names, model_name, n_instances=5):
    print(f"\nGenerowanie wyjasnien LIME dla {model_name}...")
    
    # Upewnij się, że dane to dense numpy array
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
        
    if hasattr(X_train, "values"):
        X_train = X_train.values
    if hasattr(X_test, "values"):
        X_test = X_test.values
        
    # Upewnij się, że feature_names to lista
    feature_names = list(feature_names)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["No Default", "Default"],
        mode="classification",
        random_state=42,
        discretize_continuous=True
    )

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    tp_idx = np.where((y_test == 1) & (y_pred == 1))[0]
    tn_idx = np.where((y_test == 0) & (y_pred == 0))[0]
    fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]

    instances = []
    labels = []

    if len(tp_idx) > 0:
        instances.append(tp_idx[0])
        labels.append("True_Positive")
    if len(tn_idx) > 0:
        instances.append(tn_idx[0])
        labels.append("True_Negative")
    if len(fp_idx) > 0:
        instances.append(fp_idx[0])
        labels.append("False_Positive")
    if len(fn_idx) > 0:
        instances.append(fn_idx[0])
        labels.append("False_Negative")

    lime_explanations = []

    for i, (idx, label) in enumerate(zip(instances, labels)):
        # explain_instance wymaga pojedynczej instancji jako 1D array
        exp = explainer.explain_instance(X_test[idx], model.predict_proba, num_features=10)
        
        exp.save_to_file(os.path.join(LIME_DIR, f"{model_name}_{label}_instance_{idx}.html"))
        
        fig = exp.as_pyplot_figure()
        fig.tight_layout()
        plt.savefig(os.path.join(LIME_DIR, f"{model_name}_{label}_instance_{idx}.png"), dpi=150, bbox_inches="tight")
        plt.close()

        lime_explanations.append({
            "instance_idx": idx,
            "label": label,
            "true_class": y_test[idx],
            "predicted_class": y_pred[idx],
            "predicted_proba": y_pred_proba[idx],
            "explanation": exp.as_list(),
        })

    lime_df = pd.DataFrame(lime_explanations)
    lime_df.to_csv(os.path.join(LIME_DIR, f"{model_name}_lime_explanations.csv"), index=False)
    print(f"Wyjasnienia LIME zapisane w {LIME_DIR}")

    return lime_explanations

def main():
    print("Wczytywanie danych...")
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["default"])
    y = df["default"]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print("\nLadowanie pipeline preprocessingu dla modeli nieinterpretowalnych...")
    preproc = joblib.load(os.path.join(PREPROC_DIR, "preprocessing_blackbox.pkl"))
    print("\nTransformacja danych...")
    X_train_proc = preproc.transform(X_train)
    X_val_proc = preproc.transform(X_val)
    X_test_proc = preproc.transform(X_test)
    feature_names = preproc.get_feature_names_out()
    
    xgb_model, xgb_grid = create_xgboost_grid()
    best_xgb, rs_xgb = train_with_randomized_search(xgb_model, xgb_grid, X_train_proc, y_train, "XGBoost", n_iter=50, cv=5)
    
    lgbm_model, lgbm_grid = create_lightgbm_grid()
    best_lgbm, rs_lgbm = train_with_randomized_search(lgbm_model, lgbm_grid, X_train_proc, y_train, "LightGBM", n_iter=50, cv=5)
    
    mlp_model, mlp_grid = create_mlp_grid()
    best_mlp, rs_mlp = train_with_randomized_search(mlp_model, mlp_grid, X_train_proc, y_train, "MLP", n_iter=30, cv=5)
    
    results = []
    results.append(evaluate_model(best_xgb, X_val_proc, y_val, "XGBoost", "val"))
    results.append(evaluate_model(best_xgb, X_test_proc, y_test, "XGBoost", "test"))
    results.append(evaluate_model(best_lgbm, X_val_proc, y_val, "LightGBM", "val"))
    results.append(evaluate_model(best_lgbm, X_test_proc, y_test, "LightGBM", "test"))
    results.append(evaluate_model(best_mlp, X_val_proc, y_val, "MLP", "val"))
    results.append(evaluate_model(best_mlp, X_test_proc, y_test, "MLP", "test"))
    df_results = print_evaluation_table(results)
    
    print("\n" + "=" * 80)
    print("GENEROWANIE WYSJASNIEN SHAP")
    print("=" * 80)
    shap_xgb = generate_shap_explanations(best_xgb, X_train_proc, X_test_proc, feature_names, "XGBoost")
    shap_lgbm = generate_shap_explanations(best_lgbm, X_train_proc, X_test_proc, feature_names, "LightGBM")
    shap_mlp = generate_shap_explanations(best_mlp, X_train_proc, X_test_proc, feature_names, "MLP")
    
    print("\n" + "=" * 80)
    print("GENEROWANIE WYJASNIEN LIME")
    print("=" * 80)
    lime_xgb = generate_lime_explanations(best_xgb, X_train_proc, X_test_proc, y_test.values, feature_names, "XGBoost")
    lime_lgbm = generate_lime_explanations(best_lgbm, X_train_proc, X_test_proc, y_test.values, feature_names, "LightGBM")
    lime_mlp = generate_lime_explanations(best_mlp, X_train_proc, X_test_proc, y_test.values, feature_names, "MLP")
    
    print("\nZapisywanie modeli i wynikow...")
    joblib.dump(best_xgb, os.path.join(MODELS_DIR, "best_xgboost.pkl"))
    joblib.dump(best_lgbm, os.path.join(MODELS_DIR, "best_lightgbm.pkl"))
    joblib.dump(best_mlp, os.path.join(MODELS_DIR, "best_mlp.pkl"))
    df_results.to_csv(os.path.join(RESULTS_DIR, "blackbox_evaluation_results.csv"), index=False)
    pd.DataFrame(rs_xgb.cv_results_).to_csv(os.path.join(RESULTS_DIR, "grid_results_xgboost.csv"), index=False)
    pd.DataFrame(rs_lgbm.cv_results_).to_csv(os.path.join(RESULTS_DIR, "grid_results_lightgbm.csv"), index=False)
    pd.DataFrame(rs_mlp.cv_results_).to_csv(os.path.join(RESULTS_DIR, "grid_results_mlp.csv"), index=False)
    
    print("Zapisano wszystkie modele black-box i wyniki.")
    print(f"\nWyniki w: {RESULTS_DIR}")
    print(f"Wykresy SHAP w: {SHAP_DIR}")
    print(f"Wyjasnienia LIME w: {LIME_DIR}")
    return best_xgb, best_lgbm, best_mlp, df_results

if __name__ == "__main__":
    main()