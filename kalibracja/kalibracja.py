import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

# =============================================================================
# 1. KONFIGURACJA ŚCIEŻEK
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# [FIX] Dodajemy folder EDA do sys.path, aby joblib widział definicje klas
EDA_DIR = os.path.join(PROJECT_ROOT, "EDA")
if EDA_DIR not in sys.path:
    sys.path.insert(0, EDA_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


DATA_PATH = os.path.join(PROJECT_ROOT, "zbiór_7.csv")
PREPROC_DIR = os.path.join(PROJECT_ROOT, "EDA", "preprocesing_pipelines")
MODELS_INTERP_DIR = os.path.join(PROJECT_ROOT, "Modele_interpretowalne", "models")
MODELS_BLACKBOX_DIR = os.path.join(PROJECT_ROOT, "Modele_nieinterpretowalne", "models_blackbox")

OUTPUT_DIR = CURRENT_DIR
IMG_DIR = os.path.join(OUTPUT_DIR, "plots_separate") # Nowy folder na oddzielne wykresy
os.makedirs(IMG_DIR, exist_ok=True)

TARGET_MEAN_PD = 0.04

# =============================================================================
# 2. KLASY KALIBRATORÓW
# =============================================================================

class BetaCalibration(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.lr = LogisticRegression(C=999999999, solver='lbfgs')

    def fit(self, X_probs, y):
        eps = 1e-15
        p = np.clip(X_probs, eps, 1 - eps)
        l_p = np.log(p)
        l_1_p = -np.log(1 - p)
        X_trans = np.column_stack([l_p, l_1_p])
        self.lr.fit(X_trans, y)
        return self

    def predict_proba(self, X_probs):
        eps = 1e-15
        p = np.clip(X_probs, eps, 1 - eps)
        l_p = np.log(p)
        l_1_p = -np.log(1 - p)
        X_trans = np.column_stack([l_p, l_1_p])
        return self.lr.predict_proba(X_trans)


class CalibrationInTheLarge(BaseEstimator, ClassifierMixin):
    def __init__(self, target_mean=0.04):
        self.target_mean = target_mean
        self.delta = 0.0

    def fit(self, X_probs, y=None):
        eps = 1e-15
        p = np.clip(X_probs, eps, 1 - eps)
        logits = np.log(p / (1 - p))

        def objective(delta):
            shifted_logits = logits + delta
            shifted_probs = 1 / (1 + np.exp(-shifted_logits))
            return np.mean(shifted_probs) - self.target_mean

        try:
            self.delta = brentq(objective, -10, 10)
        except ValueError:
            self.delta = 0.0
        return self

    def predict_proba(self, X_probs):
        eps = 1e-15
        p = np.clip(X_probs, eps, 1 - eps)
        logits = np.log(p / (1 - p))
        shifted_logits = logits + self.delta
        new_probs = 1 / (1 + np.exp(-shifted_logits))
        return np.column_stack([1 - new_probs, new_probs])

# =============================================================================
# 3. FUNKCJE POMOCNICZE - METRYKI I WYKRESY (ZMODYFIKOWANE)
# =============================================================================

def compute_metrics(y_true, y_prob, model_name="Model"):
    n = len(y_true)
    base_prob = np.mean(y_true)
    brier = brier_score_loss(y_true, y_prob)
    
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    
    reliability = 0.0
    resolution = 0.0
    ece = 0.0
    
    for i in range(n_bins):
        mask = bin_indices == i
        count = np.sum(mask)
        if count > 0:
            prob_avg = np.mean(y_prob[mask])
            true_avg = np.mean(y_true[mask])
            reliability += count * (prob_avg - true_avg)**2
            resolution += count * (true_avg - base_prob)**2
            ece += np.abs(prob_avg - true_avg) * (count / n)
            
    reliability /= n
    resolution /= n
    
    try:
        df = pd.DataFrame({'y': y_true, 'p': y_prob})
        df['bucket'] = pd.qcut(df['p'], n_bins, duplicates='drop')
        ace = df.groupby('bucket').apply(lambda x: np.abs(x['p'].mean() - x['y'].mean())).mean()
    except:
        ace = np.nan

    return {
        "Method": model_name,
        "Avg_PD": np.mean(y_prob),
        "ECE": ece,
        "ACE": ace,
        "Brier": brier,
        "Rel": reliability,
        "Res": resolution
    }

def plot_single_reliability(y_true, y_prob, title, filename):
    """Generuje wykres reliability z histogramem na drugiej osi Y."""
    fig, ax1 = plt.subplots(figsize=(7, 6))

    # --- OŚ LEWA (Reliability: 0.0 - 1.0) ---
    ax1.plot([0, 1], [0, 1], "k:", label="Perfect", alpha=0.6)
    
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax1.plot(mean_pred, frac_pos, "s-", label="Model", color='navy', linewidth=2, markersize=6)
    
    ax1.set_ylabel("Fraction of Positives (Reliability)", color='navy')
    ax1.set_ylim([-0.05, 1.05])
    ax1.tick_params(axis='y', labelcolor='navy')
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.grid(True, alpha=0.3)

    # --- OŚ PRAWA (Histogram: Liczebność) ---
    ax2 = ax1.twinx()  # Druga oś współdzieląca X
    
    # Rysujemy histogram z przezroczystością
    ax2.hist(y_prob, range=(0, 1), bins=10, histtype="stepfilled", 
             color="gray", alpha=0.2, label="Distribution")
    
    ax2.set_ylabel("Count (Histogram)", color="gray")
    ax2.tick_params(axis='y', labelcolor="gray")
    
    # Legenda - łączymy wpisy z obu osi
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.title(title)
    
    save_path = os.path.join(IMG_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_single_histogram(y_prob, title, filename):
    """Generuje histogram dla pojedynczej serii danych."""
    plt.figure(figsize=(8, 5))
    
    plt.hist(y_prob, bins=50, alpha=0.7, color='steelblue', 
             edgecolor='black', label="PD Distribution", density=True)
    
    # Target line
    plt.axvline(TARGET_MEAN_PD, color='red', linestyle='--', linewidth=2, 
                label=f'Target {TARGET_MEAN_PD}')
    
    plt.title(title)
    plt.xlabel("Predicted Probability (PD)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(IMG_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def find_file(directory, pattern):
    if not os.path.exists(directory): return None
    files = os.listdir(directory)
    for f in files:
        if pattern.lower() in f.lower() and f.endswith('.pkl'):
            return os.path.join(directory, f)
    pkls = [f for f in files if f.endswith('.pkl')]
    if pkls: return os.path.join(directory, pkls[0])
    return None

# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    print(">>> [1/6] Wczytywanie danych...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Brak pliku: {DATA_PATH}")
        
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["default"])
    y = df["default"]
    
    model_logit = joblib.load(os.path.join(MODELS_INTERP_DIR, "best_logistic_regression_woe.pkl"))
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(">>> [2/6] Wczytywanie Preprocessingu...")
    
    # Logit
    path_pre_logit = os.path.join(PREPROC_DIR, "preprocessing_logit_woe.pkl")
    if os.path.exists(path_pre_logit):
        try:
            pre_logit = joblib.load(path_pre_logit)
            X_val_logit = pre_logit.transform(X_val)
            X_test_logit = pre_logit.transform(X_test)
            X_val_logit   = pd.DataFrame(X_val_logit,   columns=model_logit.feature_names_in_)
            X_test_logit  = pd.DataFrame(X_test_logit,  columns=model_logit.feature_names_in_)
        except Exception as e:
            print(f"![ERROR] Logit Preproc: {e}")
            return
    else:
        X_val_logit, X_test_logit = X_val, X_test

    # Blackbox
    path_pre_bb = os.path.join(PREPROC_DIR, "preprocessing_blackbox.pkl")
    if os.path.exists(path_pre_bb):
        try:
            pre_bb = joblib.load(path_pre_bb)
            X_val_bb = pre_bb.transform(X_val)
            X_test_bb = pre_bb.transform(X_test)
        except Exception as e:
            print(f"![ERROR] Blackbox Preproc: {e}")
            return
    else:
        X_val_bb, X_test_bb = X_val, X_test

    print(">>> [3/6] Wczytywanie Modeli...")
    
    # Logit
    path_logit = find_file(MODELS_INTERP_DIR, "logit") or find_file(MODELS_INTERP_DIR, "logistic")
    if path_logit:
      #  model_logit = joblib.load(path_logit)
        p_val_logit = model_logit.predict_proba(X_val_logit)[:, 1]
        p_test_logit = model_logit.predict_proba(X_test_logit)[:, 1]
    else:
        print("![ERROR] Brak modelu Logit.")
        p_val_logit, p_test_logit = np.zeros(len(y_val)), np.zeros(len(y_test))

    # XGBoost
    path_xgb = find_file(MODELS_BLACKBOX_DIR, "xgboost") or find_file(MODELS_BLACKBOX_DIR, "boost")
    if path_xgb:
        model_xgb = joblib.load(path_xgb)
        try:
            p_val_xgb = model_xgb.predict_proba(X_val_bb)[:, 1]
            p_test_xgb = model_xgb.predict_proba(X_test_bb)[:, 1]
        except:
            p_val_xgb = model_xgb.predict_proba(np.array(X_val_bb))[:, 1]
            p_test_xgb = model_xgb.predict_proba(np.array(X_test_bb))[:, 1]
    else:
        print("![ERROR] Brak modelu XGBoost.")
        p_val_xgb, p_test_xgb = np.zeros(len(y_val)), np.zeros(len(y_test))

    models_to_calibrate = [
        ("Logit", p_val_logit, p_test_logit),
        ("XGBoost", p_val_xgb, p_test_xgb)
    ]

    results_table = []

    print(">>> [4/6] Kalibracja...")

    for name, p_val, p_test in models_to_calibrate:
        if np.sum(p_val) == 0: continue

        print(f"   ... Przetwarzanie: {name}")
        
        # Definicja metod i predykcji
        methods_map = {}
        
        # 0. Original
        methods_map["Original"] = p_test
        results_table.append(compute_metrics(y_test, p_test, f"{name}_Original"))
        
        # 1. Platt
        platt = LogisticRegression(C=99999, solver='lbfgs')
        platt.fit(p_val.reshape(-1, 1), y_val)
        p_test_platt = platt.predict_proba(p_test.reshape(-1, 1))[:, 1]
        methods_map["Platt"] = p_test_platt
        results_table.append(compute_metrics(y_test, p_test_platt, f"{name}_Platt"))
        
        # 2. Isotonic
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(p_val, y_val)
        p_test_iso = iso.predict(p_test)
        methods_map["Isotonic"] = p_test_iso
        results_table.append(compute_metrics(y_test, p_test_iso, f"{name}_Isotonic"))
        
        # 3. Beta
        beta = BetaCalibration()
        beta.fit(p_val, y_val)
        p_test_beta = beta.predict_proba(p_test)[:, 1]
        methods_map["Beta"] = p_test_beta
        results_table.append(compute_metrics(y_test, p_test_beta, f"{name}_Beta"))
        
        # 4. Iso + Large 4%
        p_val_iso = iso.predict(p_val)
        cal_large = CalibrationInTheLarge(target_mean=TARGET_MEAN_PD)
        cal_large.fit(p_val_iso)
        p_test_final = cal_large.predict_proba(p_test_iso)[:, 1]
        methods_map["Iso_Large4%"] = p_test_final
        results_table.append(compute_metrics(y_test, p_test_final, f"{name}_Iso+Large4%"))

        # >>> GENEROWANIE ODDZIELNYCH WYKRESÓW <<<
        print(f"       Generowanie wykresów w {IMG_DIR}...")
        for method_name, prob_arr in methods_map.items():
            # Bezpieczna nazwa pliku
            safe_method = method_name.replace(" ", "").replace("+", "_").replace("%", "")
            
            # 1. Reliability Curve
            plot_single_reliability(
                y_test, 
                prob_arr, 
                title=f"Reliability: {name} - {method_name}", 
                filename=f"rel_{name}_{safe_method}.png"
            )
            
            # 2. Histogram
            plot_single_histogram(
                prob_arr, 
                title=f"PD Hist: {name} - {method_name}", 
                filename=f"hist_{name}_{safe_method}.png"
            )

    print(">>> [5/6] Zapis tabeli wyników...")
    df_results = pd.DataFrame(results_table)
    cols = ["Avg_PD", "ECE", "ACE", "Brier", "Rel", "Res"]
    for c in cols:
        if c in df_results.columns:
            df_results[c] = df_results[c].round(5)
            
    csv_path = os.path.join(OUTPUT_DIR, "wyniki_kalibracji.csv")
    df_results.to_csv(csv_path, index=False)
    
    print("\n" + "="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    print(f"Wykresy (każdy osobno) zapisano w: {IMG_DIR}")
    print(">>> Zakończono.")

if __name__ == "__main__":
    main()
