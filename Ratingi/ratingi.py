# rating_pipeline.py
"""
Pipeline do:
- wczytania modeli (logit WoE + XGBoost),
- policzenia PD na train/val/test,
- zbudowania ratingów (AAA...CCC) na podstawie PD,
- wygenerowania tabel ratingowych i tabel decyzyjnych.

Zakładamy, że wejściowe modele zwracają już "PD" (docelowo: skalibrowane).
Na razie można używać PD z niekalibrowanego logitu.
"""
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# ============================================================
#                   KONFIGURACJA ŚCIEŻEK
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # .../IWUM-Projekt-1/Ratingi
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..")) # .../IWUM-Projekt-1

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


DATA_PATH = os.path.join(PROJECT_ROOT, "zbiór_7.csv")

# interpretowalny logit + jego preproc WoE
LOGIT_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "Modele_interpretowalne",
    "models",
    "best_logistic_regression_woe.pkl",
)
LOGIT_PREPROC_PATH = os.path.join(
    PROJECT_ROOT,
    "EDA",
    "preprocesing_pipelines",
    "preprocessing_logit_woe.pkl",
)

# black-box XGBoost (tu przyjmuję strukturę podobną jak w repo)
XGB_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "Modele_nieinterpretowalne",
    "models_blackbox",
    "best_xgboost.pkl",
)
# TODO: jeśli XGBoost ma swój pipeline/preproc, dodaj tu ścieżkę:
XGB_PREPROC_PATH = os.path.join(
    PROJECT_ROOT,
    "EDA",
    "preprocesing_pipelines",
    "preprocessing_blackbox.pkl",  
)

RESULTS_DIR = os.path.join(BASE_DIR, "rating_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Nazwy ratingów – rosnące ryzyko (AAA = najlepszy, CCC = najgorszy)
RATING_LABELS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]


# ============================================================
#                     Wczytanie danych
# ============================================================

def load_data():
    """
    Wczytuje pełny zbiór i robi podział 60/20/20 (train/val/test),
    spójny z resztą projektu.
    """
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["default"])
    y = df["default"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_models():
    """
    Ładuje:
    - logit interpretowalny + pipeline WoE
    - XGBoost + pipeline 
    """
    # logit
    logit_model = joblib.load(LOGIT_MODEL_PATH)
    logit_preproc = joblib.load(LOGIT_PREPROC_PATH)

    # XGBoost 
    if os.path.exists(XGB_MODEL_PATH):
        xgb_model = joblib.load(XGB_MODEL_PATH)
    else:
        xgb_model = None

    if os.path.exists(XGB_PREPROC_PATH):
        xgb_preproc = joblib.load(XGB_PREPROC_PATH)
    else:
        xgb_preproc = None

    return logit_model, logit_preproc, xgb_model, xgb_preproc


# ============================================================
#                Predykcja PD dla modeli
# ============================================================

def predict_pd_logit(logit_model, logit_preproc, X):
    """
    Zwraca przewidywane PD dla logitu.
    """
    X_tr = logit_preproc.transform(X)
    pd_hat = logit_model.predict_proba(X_tr)[:, 1]
    return pd_hat


def predict_pd_xgb(xgb_model, xgb_preproc, X):
    """
    Zwraca przewidywane PD dla XGBoost.
    """
    if xgb_model is None:
        return None

    if xgb_preproc is not None:
        X_tr = xgb_preproc.transform(X)
    else:
        X_tr = X

    if hasattr(xgb_model, "predict_proba"):
        pd_hat = xgb_model.predict_proba(X_tr)[:, 1]
    else:
        # niektóre implementacje zwracają bezpośrednio PD
        pd_hat = xgb_model.predict(X_tr)
    return pd_hat


# ============================================================
#           Budowa ratingów na podstawie PD
# ============================================================

def build_rating_bins_by_quantiles(pd_train, n_classes=7):
    """
    Wyznacza progi ratingów na podstawie kwantyli PD z TRAIN.

    Zwraca tablicę krawędzi [b0, b1, ..., b_n], gdzie:
    - b0 = 0.0
    - b_n = 1.0
    """
    quantiles = np.linspace(0, 1, n_classes + 1)
    bin_edges = np.quantile(pd_train, quantiles)

    # upewniamy się, że zakres jest cały [0,1]
    bin_edges[0] = 0.0
    bin_edges[-1] = 1.0

    # małe zabezpieczenie przed duplikatami progów
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) - 1 < n_classes:
        # jeśli duplikaty, mamy mniej "slotów" ratingowych,
        # więc skracamy listę RATING_LABELS przy mapowaniu
        print(" Ostrzeżenie: duplikujące się progi ratingów (mało zróżnicowane PD).")
    print(bin_edges)
    return bin_edges


def assign_ratings(pd_hat, bin_edges, labels):
    """
    Przypisuje ratingi na podstawie PD i progów.

    pd_hat   : wektor PD
    bin_edges: krawędzie przediałów (rosnące)
    labels   : list[str], np. ["AAA", "AA", ..., "CCC"]

    Zwraca Series dtype=category.
    """
    # jeśli z powodu duplikatów progów mamy mniej przedziałów
    n_intervals = len(bin_edges) - 1
    if n_intervals != len(labels):
        labels = labels[:n_intervals]

    ratings = pd.cut(
        pd_hat,
        bins=bin_edges,
        labels=labels,
        right=False,   # lewostronnie domknięte: [b_i, b_{i+1})
        include_lowest=True,
    )
    return ratings


def rating_summary(y_true, pd_hat, ratings, model_name, dataset_name):
    """
    Podsumowanie ratingów:
    - liczebność
    - liczba bad
    - bad rate
    - średnie PD

    Zwraca DataFrame + wypisuje na ekran.
    """
    df = pd.DataFrame({
        "y": y_true,
        "pd": pd_hat,
        "rating": ratings,
    })

    summary = (
        df.groupby("rating")
          .agg(
              n_obs=("y", "size"),
              n_bad=("y", "sum"),
              bad_rate=("y", "mean"),
              avg_pd=("pd", "mean"),
          )
          .reset_index()
    )

    print("\n" + "=" * 70)
    print(f"RATING SUMMARY – {model_name} – {dataset_name}")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("=" * 70)

    return summary


# ============================================================
#          Funkcje do progów decyzyjnych / tabel decyzyjnych
# ============================================================


profit_good_accepted=0.15   # +15% na dobrym kredycie
loss_bad_accepted=0.5      # -50% na złym kredycie
cost_reject_good= 0.06     # utrata ~40% potencjalnego zysku
profit_reject_bad=0.2      # uniknięcie 40% potencjalnej straty


def expected_profit(
    y_true,
    pd_hat,
    threshold,
    profit_good_accepted=0.15,   # +15% na dobrym kredycie
    loss_bad_accepted=0.50,      # -50% na złym kredycie
    frac_aux=0.4                 # ułamek dla utraconego zysku / unikniętej straty
):
    """
    Liczy oczekiwany zysk portfela dla danego progu PD.

    Znaczenie:
    - y_true = 0 -> dobry klient
    - y_true = 1 -> zły klient (default)
    - akceptujemy jeśli PD <= threshold

    Przypadki:
    - good & accepted   -> +profit_good_accepted
    - bad  & accepted   -> -loss_bad_accepted
    - good & rejected   -> cost_reject_good  (ujemny)
    - bad  & rejected   -> profit_reject_bad (dodatni)
    """

    cost_reject_good = -frac_aux * profit_good_accepted   # np. -0.06
    profit_reject_bad = frac_aux * loss_bad_accepted      # np. +0.20

    y_true = np.asarray(y_true)
    pd_hat = np.asarray(pd_hat)

    accept = pd_hat <= threshold
    reject = ~accept

    good = (y_true == 0)
    bad  = (y_true == 1)

    n_A_good = np.sum(accept & good)
    n_A_bad  = np.sum(accept & bad)
    n_R_good = np.sum(reject & good)
    n_R_bad  = np.sum(reject & bad)

    return (
        n_A_good * profit_good_accepted
        - n_A_bad  * loss_bad_accepted
        + n_R_bad  * profit_reject_bad
        + n_R_good * cost_reject_good
    )


def decision_table(y_true, pd_hat, thresholds):
    """
    Buduje tabelę decyzyjną dla różnych progów PD:
    - udział zaakceptowanych / odrzuconych
    - bad rate w portfelu zaakceptowanym / odrzuconym
    - liczby TP, FP, FN, TN

    Zwraca DataFrame.
    """
    rows = []
    n = len(y_true)

    for thr in thresholds:
        y_pred = (pd_hat <= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        accepted = tp + fp
        rejected = tn + fn

        row = {
            "threshold": thr,
            "accept_rate": accepted / n,
            "reject_rate": rejected / n,
            "bad_rate_accepted": fp / accepted if accepted > 0 else np.nan,
            "bad_rate_rejected": fn / rejected if rejected > 0 else np.nan,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def build_cost_curve(
    y_true,
    pd_hat,
    thresholds,
    model_name="Model",
    save_path=None,
    profit_good_accepted=0.15,
    loss_bad_accepted=0.50,
    frac_aux=0.4
):
    """
    Buduje cost curve: próg PD -> oczekiwany zysk.
    """

    profits = []
    for thr in thresholds:
        prof = expected_profit(
            y_true,
            pd_hat,
            thr,
            profit_good_accepted=profit_good_accepted,
            loss_bad_accepted=loss_bad_accepted,
            frac_aux=frac_aux,
        )
        profits.append(prof)

    curve_df = pd.DataFrame({
        "threshold": thresholds,
        "expected_profit": profits,
    })

    if save_path is not None:
        plt.figure()
        plt.plot(thresholds, profits, marker="o")
        plt.xlabel("Próg PD (akceptujemy jeśli PD ≤ próg)")
        plt.ylabel("Oczekiwany zysk (jednostki umowne)")
        plt.title(f"Cost curve – {model_name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    return curve_df


# ============================================================
#                           MAIN
# ============================================================

def main():
    # 1. Dane
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # 2. Modele
    logit_model, logit_preproc, xgb_model, xgb_preproc = load_models()

    # 3. PD z logitu (tu docelowo możesz wstawić PD po kalibracji)
    pd_train_logit = predict_pd_logit(logit_model, logit_preproc, X_train)
    pd_val_logit   = predict_pd_logit(logit_model, logit_preproc, X_val)
    pd_test_logit  = predict_pd_logit(logit_model, logit_preproc, X_test)

    # 4. (opcjonalnie) PD z XGBoost
    if xgb_model is not None:
        pd_train_xgb = predict_pd_xgb(xgb_model, xgb_preproc, X_train)
        pd_val_xgb   = predict_pd_xgb(xgb_model, xgb_preproc, X_val)
        pd_test_xgb  = predict_pd_xgb(xgb_model, xgb_preproc, X_test)
    else:
        pd_train_xgb = pd_val_xgb = pd_test_xgb = None

    # 5. Budowa progów ratingowych na podstawie PD z TRAIN (logit)
    bin_edges = build_rating_bins_by_quantiles(
        pd_train_logit,
        n_classes=len(RATING_LABELS)
    )

    # 6. Przypisanie ratingów dla logitu
    ratings_train_logit = assign_ratings(pd_train_logit, bin_edges, RATING_LABELS)
    ratings_val_logit   = assign_ratings(pd_val_logit,   bin_edges, RATING_LABELS)
    ratings_test_logit  = assign_ratings(pd_test_logit,  bin_edges, RATING_LABELS)

    # 7. Podsumowania ratingowe (logit)
    summary_train_logit = rating_summary(
        y_train, pd_train_logit, ratings_train_logit,
        model_name="Logit_WoE",
        dataset_name="TRAIN",
    )
    summary_val_logit = rating_summary(
        y_val, pd_val_logit, ratings_val_logit,
        model_name="Logit_WoE",
        dataset_name="VAL",
    )
    summary_test_logit = rating_summary(
        y_test, pd_test_logit, ratings_test_logit,
        model_name="Logit_WoE",
        dataset_name="TEST",
    )

    # 8. (opcjonalnie) te same ratingi dla XGBoost – używamy TYCH SAMYCH progów PD
    if pd_train_xgb is not None:
        ratings_train_xgb = assign_ratings(pd_train_xgb, bin_edges, RATING_LABELS)
        ratings_val_xgb   = assign_ratings(pd_val_xgb,   bin_edges, RATING_LABELS)
        ratings_test_xgb  = assign_ratings(pd_test_xgb,  bin_edges, RATING_LABELS)

        summary_train_xgb = rating_summary(
            y_train, pd_train_xgb, ratings_train_xgb,
            model_name="XGBoost",
            dataset_name="TRAIN",
        )
        summary_val_xgb = rating_summary(
            y_val, pd_val_xgb, ratings_val_xgb,
            model_name="XGBoost",
            dataset_name="VAL",
        )
        summary_test_xgb = rating_summary(
            y_test, pd_test_xgb, ratings_test_xgb,
            model_name="XGBoost",
            dataset_name="TEST",
        )
    else:
        summary_train_xgb = summary_val_xgb = summary_test_xgb = None

    # 9. Tabele decyzyjne dla logitu (np. na WALIDACJI)
    thresholds = np.linspace(0.02, 0.98, 50)  # zakres PD do analizy
    decision_val_logit = decision_table(y_val, pd_val_logit, thresholds)
    decision_test_logit = decision_table(y_test, pd_test_logit, thresholds)

    print("\nDECISION TABLE – Logit – VAL")
    print(decision_val_logit.to_string(index=False))

    print("\nDECISION TABLE – Logit – TEST")
    print(decision_test_logit.to_string(index=False))

     # 9b. Tabele decyzyjne dla XGBoost (jeśli model istnieje)
    if pd_val_xgb is not None:
        decision_val_xgb = decision_table(y_val, pd_val_xgb, thresholds)
        decision_test_xgb = decision_table(y_test, pd_test_xgb, thresholds)

        print("\nDECISION TABLE – XGBoost – VAL")
        print(decision_val_xgb.to_string(index=False))

        print("\nDECISION TABLE – XGBoost – TEST")
        print(decision_test_xgb.to_string(index=False))
    else:
        decision_val_xgb = decision_test_xgb = None

    # 9c. Cost curves – logit
    cost_curve_val_logit = build_cost_curve(
        y_val, pd_val_logit, thresholds,
        model_name="Logit_WoE",
        save_path=os.path.join(RESULTS_DIR, "cost_curve_logit_val.png"),
    )
    cost_curve_test_logit = build_cost_curve(
        y_test, pd_test_logit, thresholds,
        model_name="Logit_WoE",
        save_path=os.path.join(RESULTS_DIR, "cost_curve_logit_test.png"),
    )

    # Cost curves – XGBoost (jeśli jest)
    if pd_val_xgb is not None:
        cost_curve_val_xgb = build_cost_curve(
            y_val, pd_val_xgb, thresholds,
            model_name="XGBoost",
            save_path=os.path.join(RESULTS_DIR, "cost_curve_xgb_val.png"),
        )
        cost_curve_test_xgb = build_cost_curve(
            y_test, pd_test_xgb, thresholds,
            model_name="XGBoost",
            save_path=os.path.join(RESULTS_DIR, "cost_curve_xgb_test.png"),
        )

    # 10. Zapis wyników do CSV (żeby można było wciągnąć do raportu / Excela)
    summary_train_logit.to_csv(
        os.path.join(RESULTS_DIR, "rating_summary_logit_train.csv"),
        index=False,
    )
    summary_val_logit.to_csv(
        os.path.join(RESULTS_DIR, "rating_summary_logit_val.csv"),
        index=False,
    )
    summary_test_logit.to_csv(
        os.path.join(RESULTS_DIR, "rating_summary_logit_test.csv"),
        index=False,
    )

    decision_val_logit.to_csv(
        os.path.join(RESULTS_DIR, "decision_table_logit_val.csv"),
        index=False,
    )
    decision_test_logit.to_csv(
        os.path.join(RESULTS_DIR, "decision_table_logit_test.csv"),
        index=False,
    )

    if summary_train_xgb is not None:
        summary_train_xgb.to_csv(
            os.path.join(RESULTS_DIR, "rating_summary_xgb_train.csv"),
            index=False,
        )
        summary_val_xgb.to_csv(
            os.path.join(RESULTS_DIR, "rating_summary_xgb_val.csv"),
            index=False,
        )
        summary_test_xgb.to_csv(
            os.path.join(RESULTS_DIR, "rating_summary_xgb_test.csv"),
            index=False,
        )
        if decision_val_xgb is not None:
            decision_val_xgb.to_csv(
                os.path.join(RESULTS_DIR, "decision_table_xgb_val.csv"),
                index=False,
            )
            decision_test_xgb.to_csv(
                os.path.join(RESULTS_DIR, "decision_table_xgb_test.csv"),
                index=False,
            )

    print("\nZapisano tabele ratingowe i decyzyjne do:", RESULTS_DIR)


if __name__ == "__main__":
    main()
