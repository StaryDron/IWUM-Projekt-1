import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,  # PR AUC
    f1_score,
    brier_score_loss,
    roc_curve
)

# ============================================================
#                KONFIGURACJA ŚCIEŻEK
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # .../Modele_interpretowalne
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..")) # .../IWUM-Projekt-1

DATA_PATH = os.path.join(PROJECT_ROOT, "zbiór_7.csv")
PREPROC_DIR = os.path.join(PROJECT_ROOT, "EDA", "preprocesing_pipelines")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_data():
    """Wczytuje dane i robi podział 60/20/20 jak w innych skryptach."""
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


def load_models_and_preproc():
    """Ładuje logit i pipeline WoE."""
    logit_path = os.path.join(MODELS_DIR, "best_logistic_regression_woe.pkl")
    preproc_logit_path = os.path.join(PREPROC_DIR, "preprocessing_logit_woe.pkl")

    if not os.path.exists(logit_path):
        raise FileNotFoundError(f"Brak modelu logitu: {logit_path}")
    if not os.path.exists(preproc_logit_path):
        raise FileNotFoundError(f"Brak pipeline'u logitowego: {preproc_logit_path}")

    logit = joblib.load(logit_path)
    preproc_logit = joblib.load(preproc_logit_path)

    return logit, preproc_logit


def to_dataframe(X, col_names):
    """Upewniamy się, że mamy DataFrame z nazwami kolumn."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=col_names)


# ============================================================
#                FUNKCJE METRYK MODELU
# ============================================================

def ks_statistic(y_true, y_proba):
    """
    Statystyka KS = max|TPR - FPR| po wszystkich progach.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ks = np.max(tpr - fpr)
    return ks


def compute_metrics(y_true, y_proba, threshold=0.5):
    """
    Liczy podstawowe miary jakości dla modelu binarnego.

    y_true  : wektor 0/1
    y_proba : prawdopodobieństwa klasy pozytywnej (default=1)
    threshold : próg do policzenia F1 (domyślnie 0.5)
    """
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)  # PR AUC
    gini = 2 * roc_auc - 1
    brier = brier_score_loss(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)
    ks = ks_statistic(y_true, y_proba)

    return {
        "ROC AUC": roc_auc,
        "PR AUC": pr_auc,
        "Gini": gini,
        "KS": ks,
        "Brier score": brier,
        "F1 (thr=0.5)": f1,
    }


def print_metrics(name, metrics_dict):
    print(f"\n===== {name} =====")
    for k, v in metrics_dict.items():
        print(f"{k:15s}: {v:0.4f}")


# ============================================================
#                 GŁÓWNY PRZEPŁYW
# ============================================================

if __name__ == "__main__":
    # 1. Dane
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # 2. Model i preprocessing
    logit, preproc_logit = load_models_and_preproc()

    # Podgląd hiperparametrów aktualnego logitu (z grid searcha),
    # przyda się później, gdy będziemy trenować modele z podzbiorem cech.
    print("Hiperparametry aktualnego modelu logistycznego:")
    print(logit.get_params())

    # 3. Transformacja WOE – używamy TYLKO .transform, pipeline jest już dopasowany na trainie
    X_train_woe = preproc_logit.transform(X_train)
    X_val_woe   = preproc_logit.transform(X_val)
    X_test_woe  = preproc_logit.transform(X_test)

    # (opcjonalnie – nazwy kolumn po WOE; przydadzą się do selekcji zmiennych)
    try:
        woe_feature_names = preproc_logit.get_feature_names_out()
    except AttributeError:
        # jeśli pipeline nie ma tej metody, to na razie zostawiamy None
        woe_feature_names = None

    # 4. Predykcje aktualnego modelu na zbiorach (główny baseline – TEST)
  #  y_train_proba = logit.predict_proba(X_train_woe)[:, 1]
  #  y_val_proba   = logit.predict_proba(X_val_woe)[:, 1]
 #   y_test_proba  = logit.predict_proba(X_test_woe)[:, 1]

    # 5. Metryki
  ##  train_metrics = compute_metrics(y_train, y_train_proba)
  #  val_metrics   = compute_metrics(y_val,   y_val_proba)
  #  test_metrics  = compute_metrics(y_test,  y_test_proba)

   # print_metrics("TRAIN – aktualny logit WOE", train_metrics)
  #  print_metrics("VAL   – aktualny logit WOE", val_metrics)
  #  print_metrics("TEST  – aktualny logit WOE", test_metrics)

    # ============================================================
    #        SELEKCJA ZMIENNYCH: LR + BIC NA TRAIN
    # ============================================================
    
    # 1. Upewniamy się, że mamy DataFrame z nazwami kolumn
    if woe_feature_names is None:
        n_features = X_train_woe.shape[1]
        woe_feature_names = [f"var_{i}" for i in range(n_features)]
    
    X_train_woe_df = pd.DataFrame(X_train_woe, columns=woe_feature_names)
    X_val_woe_df   = pd.DataFrame(X_val_woe,   columns=woe_feature_names)
    X_test_woe_df  = pd.DataFrame(X_test_woe,  columns=woe_feature_names)
    X_train_woe_df = X_train_woe_df.astype(float)
    X_val_woe_df   = X_val_woe_df.astype(float)
    X_test_woe_df  = X_test_woe_df.astype(float)
    
    
    def fit_logit_sm(X, y):
        """
        Dopasowuje klasyczną (nieregularyzowaną) regresję logistyczną w statsmodels.
        Wymuszamy floaty, bo statsmodels nie lubi dtype=object/category.
        """
        # rzutujemy na numpy float
        X_np = np.asarray(X, dtype=float)
        y_np = np.asarray(y, dtype=float)
    
        # dokładamy wyraz wolny
        X_const = sm.add_constant(X_np, has_constant="add")
    
        model = sm.Logit(y_np, X_const)
        res = model.fit(disp=False, maxiter=200)
        return res
    
    
    print("\n=== Dopasowuję pełny model (statsmodels.Logit) na TRAIN dla LR-testów... ===")
    full_res = fit_logit_sm(X_train_woe_df, y_train)
    ll_full = full_res.llf
    print(f"Log-likelihood pełnego modelu: {ll_full:.4f}")
    print(f"Liczba parametrów (łącznie z interceptem): {int(full_res.df_model) + 1}")
    
    
    # 2. LR dla każdej zmiennej: full vs full-bez-danej-zmiennej
    LR_scores = {}
    
    print("\n=== Liczę LR dla każdej zmiennej (to chwilę potrwa) ===")
    for col in woe_feature_names:
        cols_reduced = [c for c in woe_feature_names if c != col]
        res_red = fit_logit_sm(X_train_woe_df[cols_reduced], y_train)
        ll_red = res_red.llf
        LR_j = 2 * (ll_full - ll_red)
        LR_scores[col] = LR_j
    
    # 3. Porządkowanie zmiennych malejąco po LR (największy wkład w logL na początku)
    ordered_vars = sorted(LR_scores.keys(), key=lambda c: LR_scores[c], reverse=True)
    
    print("\nTop 15 zmiennych wg LR (wkład do log-likelihood w obecności pozostałych):")
    for i, col in enumerate(ordered_vars[:15], start=1):
        print(f"{i:2d}. {col:30s}  LR = {LR_scores[col]:.4f}")
        
    # ============================================================
    #  FORWARD SELEKCJA PO LR + WARUNEK ZNAKU BETA (≤ 0) + WALIDACJA
    # ============================================================

    def predict_logit_sm(res, X_df_subset):
        """
        Predykcja prawdopodobieństw dla modelu statsmodels.Logit.
        X_df_subset – DataFrame z wybranymi kolumnami (bez const).
        """
        X_np = np.asarray(X_df_subset, dtype=float)
        X_const = sm.add_constant(X_np, has_constant="add")
        return res.predict(X_const)

    print("\n=== Forward selekcja zmiennych (warunek: wszystkie β ≤ 0) ===")

    selected_vars = []      # zaakceptowane zmienne
    excluded_vars = []      # zmienne odrzucone z powodu dodatniego beta
    forward_results = []    # wyniki po każdym kroku
    best_val_roc = -np.inf  # najlepszy jak dotąd ROC AUC na walidacji
    best_info = None        # info o najlepszym modelu

    # przechodzimy po zmiennych w kolejności od najważniejszej (po LR)
    for col in ordered_vars:
        if col in excluded_vars:
            continue

        candidate_vars = selected_vars + [col]

        # dopasowanie modelu na TRAIN dla kandydackiego zestawu cech
        res_cand = fit_logit_sm(X_train_woe_df[candidate_vars], y_train)

        # współczynniki: [β0, β1, ..., β_k]; pomijamy intercept (β0)
        beta = np.asarray(res_cand.params[1:], dtype=float)

        # jeśli którykolwiek β > 0 → odrzucamy NOWĄ cechę
        if np.any(beta > 0.0):
            print(f"⚠️ Zmienna {col} odrzucona – dodatni współczynnik beta w modelu.")
            excluded_vars.append(col)
            continue

        # jeśli wszystkie β ≤ 0 → akceptujemy nowy zestaw
        selected_vars = candidate_vars
        k = len(selected_vars)

        # predykcje na TRAIN/VAL/TEST
        y_train_proba = predict_logit_sm(res_cand, X_train_woe_df[selected_vars])
        y_val_proba   = predict_logit_sm(res_cand, X_val_woe_df[selected_vars])
        y_test_proba  = predict_logit_sm(res_cand, X_test_woe_df[selected_vars])

        # metryki
        train_m = compute_metrics(y_train, y_train_proba)
        val_m   = compute_metrics(y_val,   y_val_proba)
        test_m  = compute_metrics(y_test,  y_test_proba)

        forward_results.append({
            "k": k,
            "added_var": col,
            "vars": list(selected_vars),

            "ROC_AUC_VAL":  val_m["ROC AUC"],
            "PR_AUC_VAL":   val_m["PR AUC"],
            "Gini_VAL":     val_m["Gini"],
            "KS_VAL":       val_m["KS"],
            "Brier_VAL":    val_m["Brier score"],

            "ROC_AUC_TEST": test_m["ROC AUC"],
            "PR_AUC_TEST":  test_m["PR AUC"],
            "Gini_TEST":    test_m["Gini"],
            "KS_TEST":      test_m["KS"],
            "Brier_TEST":   test_m["Brier score"],
        })

        print(f"\n===== FORWARD – k={k}, dodana zmienna: {col} =====")
        print("Aktualny zestaw zmiennych:")
        for v in selected_vars:
            print("  -", v)
        print_metrics("TRAIN", train_m)
        print_metrics("VAL",   val_m)
        print_metrics("TEST",  test_m)

        # aktualizacja najlepszego modelu wg ROC AUC na walidacji
        if val_m["ROC AUC"] > best_val_roc:
            best_val_roc = val_m["ROC AUC"]
            best_info = {
                "k": k,
                "vars": list(selected_vars),
                "train_metrics": train_m,
                "val_metrics": val_m,
                "test_metrics": test_m,
            }

    # podsumowanie wszystkich kroków
    print("\n=== PODSUMOWANIE FORWARD SELECTION (wg ROC AUC na VAL) ===")
    for r in forward_results:
        print(
            f"k={r['k']:2d} | added={r['added_var']:30s} | "
            f"ROC_AUC_VAL={r['ROC_AUC_VAL']:.4f} | Gini_VAL={r['Gini_VAL']:.4f} | "
            f"KS_VAL={r['KS_VAL']:.4f} | Brier_VAL={r['Brier_VAL']:.4f} | "
            f"ROC_AUC_TEST={r['ROC_AUC_TEST']:.4f} | Gini_TEST={r['Gini_TEST']:.4f} | "
            f"KS_TEST={r['KS_TEST']:.4f} | Brier_TEST={r['Brier_TEST']:.4f}"
        )

    # najlepszy model wg walidacji – pełne metryki + lista cech
    if best_info is not None:
        print("\n=== NAJLEPSZY MODEL wg ROC AUC na WALIDACJI ===")
        print(f"k = {best_info['k']}")
        print("Zmienne:")
        for v in best_info["vars"]:
            print("  -", v)
        print_metrics("TRAIN – best", best_info["train_metrics"])
        print_metrics("VAL   – best", best_info["val_metrics"])
        print_metrics("TEST  – best", best_info["test_metrics"])
    
    # ============================================================
    #   ZAPIS KOLUMN DO WYRZUCENIA DLA NAJLEPSZEGO MODELU FORWARD
    # ============================================================
    
    if best_info is not None:
        # wszystkie cechy po WoE (tak jak wcześniej)
        all_features_after_woe = list(woe_feature_names)
    
        # cechy użyte w najlepszym modelu
        best_vars = best_info["vars"]
    
        # kolumny do drop: wszystko, czego NIE ma w best_vars
        drop_cols_best = [col for col in all_features_after_woe if col not in best_vars]
    
        print("\n=== LISTA KOLUMN DO USUNIĘCIA (BEST FORWARD MODEL) ===")
        print(f"Liczba wszystkich cech po WoE : {len(all_features_after_woe)}")
        print(f"Liczba cech w najlepszym modelu: {len(best_vars)}")
        print(f"Liczba kolumn do usunięcia     : {len(drop_cols_best)}")
    
        # katalog jak wcześniej przy BIC
        drop_dir = os.path.join(BASE_DIR, "interpretowalnosc_logit")
        os.makedirs(drop_dir, exist_ok=True)
    
        drop_forward_path = os.path.join(
            drop_dir,
            "drop_columns_forward_best.csv"  # inna nazwa niż przy BIC
        )
    
        # zapis w formacie dla DropColumnsTransformer (kolumna 'feature')
        df_drop_best = pd.DataFrame({"feature": drop_cols_best})
        df_drop_best.to_csv(drop_forward_path, index=False)
    
        print(f"\n✅ Zapisano listę kolumn do usunięcia dla najlepszego modelu "
              f"forward do pliku:\n{drop_forward_path}")
    else:
        print("\n⚠️ Nie udało się znaleźć najlepszego modelu (best_info is None). "
              "Nic nie zapisano.")
    
        
    # ============================================================
    #       BIC NA TRAIN
    # ============================================================
    
    # 4. Budowanie zagnieżdżonej rodziny modeli: 1,2,...,p najlepszych zmiennych
    print("\n=== Buduję zagnieżdżoną rodzinę modeli i liczę BIC na TRAIN ===")
    n_train = len(y_train)
    bic_list = []
    models_nested = {}  # trzymamy wyniki statsmodels dla ewentualnego wglądu
    
    for k in range(1, len(ordered_vars) + 1):
        cols_k = ordered_vars[:k]
        res_k = fit_logit_sm(X_train_woe_df[cols_k], y_train)
        ll_k = res_k.llf
        k_params = k + 1  # k beta + intercept
        bic_k = -2 * ll_k + k_params * np.log(n_train)
        bic_list.append((k, bic_k))
        models_nested[k] = res_k
    
    # 5. Wybór najlepszego k po BIC
    k_best, bic_best = min(bic_list, key=lambda t: t[1])
    print(f"\nNajlepszy model wg BIC ma k = {k_best} zmiennych, BIC = {bic_best:.2f}")
    
    print("\nTabela BIC dla pierwszych 15 modeli:")
    for k, bic_k in bic_list[:15]:
        delta = bic_k - bic_best
        print(f"k = {k:2d} | BIC = {bic_k:8.2f} | ΔBIC = {delta:6.2f}")


    
    best_vars = ordered_vars[:k_best]
    print("\nWybrane zmienne (wg BIC):")
    for i, col in enumerate(best_vars, start=1):
        print(f"{i:2d}. {col}")
    
    
    # ============================================================
    #    NOWY MODEL PREDYKCYJNY (sklearn) NA WYBRANYCH ZMIENNYCH przez BIC
    # ============================================================
    
    # 6. Przygotowanie macierzy tylko z wybranymi zmiennymi
    X_train_sel = X_train_woe_df[best_vars].values
    X_val_sel   = X_val_woe_df[best_vars].values
    X_test_sel  = X_test_woe_df[best_vars].values
    
    # 7. Trening nowego modelu LogisticRegression z tymi samymi hiperparametrami
    logit_reduced = LogisticRegression(
        C=0.01,
        penalty="l2",
        solver="newton-cg",
        max_iter=1000,
        n_jobs=-1,
        random_state=42
    )
    
    print("\n=== Trenuję nowy model sklearn.LogisticRegression na wybranych zmiennych ===")
    logit_reduced.fit(X_train_sel, y_train)
    
    # 8. Ocena jakości na TRAIN/VAL/TEST
    y_train_proba_red = logit_reduced.predict_proba(X_train_sel)[:, 1]
    y_val_proba_red   = logit_reduced.predict_proba(X_val_sel)[:, 1]
    y_test_proba_red  = logit_reduced.predict_proba(X_test_sel)[:, 1]
    
    train_metrics_red = compute_metrics(y_train, y_train_proba_red)
    val_metrics_red   = compute_metrics(y_val,   y_val_proba_red)
    test_metrics_red  = compute_metrics(y_test,  y_test_proba_red)
    
    print_metrics("TRAIN – logit REDUCED (BIC)", train_metrics_red)
    print_metrics("VAL   – logit REDUCED (BIC)", val_metrics_red)
    print_metrics("TEST  – logit REDUCED (BIC)", test_metrics_red)
    
    # ============================================================
    #               WYKRES BIC vs liczba zmiennych
    # ============================================================
    
    import matplotlib.pyplot as plt
    
    # bic_list = [(k, bic_value), ...]
    ks = [t[0] for t in bic_list]
    bic_vals = [t[1] for t in bic_list]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ks, bic_vals, marker="o", linestyle="-")
    plt.axvline(k_best, color="red", linestyle="--", label=f"Minimum BIC = {k_best}")
    
    plt.title("Krzywa BIC dla zagnieżdżonej rodziny modeli logistycznych")
    plt.xlabel("Liczba zmiennych (k)")
    plt.ylabel("BIC")
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(MODELS_DIR, "bic_curve.png")
    plt.savefig(output_path, dpi=200)
    plt.close()
    
    print(f"\nWykres BIC zapisany jako: {output_path}")

    # ============================================================
    #   PORÓWNANIE MODELI DLA RÓŻNYCH k (np. 3, 5, 8, 10)
    #   (używamy statsmodels.Logit – bez regularyzacji)
    # ============================================================
    
    def evaluate_k_list(k_list, ordered_vars,
                        X_train_df, X_val_df, X_test_df,
                        y_train, y_val, y_test):
        """
        Dla każdego k:
        - bierze pierwsze k zmiennych z ordered_vars,
        - dopasowuje klasyczny Logit (statsmodels) na TRAIN,
        - liczy metryki na TRAIN / VAL / TEST,
        - wypisuje wyniki + zwraca małe podsumowanie.
        """
        results = []
    
        # na wszelki wypadek rzutujemy y na float/numpy
        y_train_np = np.asarray(y_train, dtype=float)
        y_val_np   = np.asarray(y_val,   dtype=float)
        y_test_np  = np.asarray(y_test,  dtype=float)
    
        for k in k_list:
            cols_k = ordered_vars[:k]
    
            # podzbiór cech + rzutowanie na float
            X_train_k_np = np.asarray(X_train_df[cols_k], dtype=float)
            X_val_k_np   = np.asarray(X_val_df[cols_k],   dtype=float)
            X_test_k_np  = np.asarray(X_test_df[cols_k],  dtype=float)
    
            # dopisujemy wyraz wolny
            X_train_const = sm.add_constant(X_train_k_np, has_constant="add")
            X_val_const   = sm.add_constant(X_val_k_np,   has_constant="add")
            X_test_const  = sm.add_constant(X_test_k_np,  has_constant="add")
    
            # klasyczna regresja logistyczna (MLE, bez regularyzacji)
            model_k = sm.Logit(y_train_np, X_train_const)
            res_k = model_k.fit(disp=False, maxiter=200)
    
            # predykcje prawdopodobieństw
            y_train_proba_k = res_k.predict(X_train_const)
            y_val_proba_k   = res_k.predict(X_val_const)
            y_test_proba_k  = res_k.predict(X_test_const)
    
            # metryki
            train_m = compute_metrics(y_train_np, y_train_proba_k)
            val_m   = compute_metrics(y_val_np,   y_val_proba_k)
            test_m  = compute_metrics(y_test_np,  y_test_proba_k)
    
            # zapis do wyników (dla późniejszej tabelki)
            results.append({
                "k": k,
                "vars": cols_k,
    
                "ROC_AUC_TRAIN": train_m["ROC AUC"],
                "Gini_TRAIN":    train_m["Gini"],
                "KS_TRAIN":      train_m["KS"],
                "Brier_TRAIN":   train_m["Brier score"],
                "PR_AUC_TRAIN":  train_m["PR AUC"],
    
                "ROC_AUC_VAL":   val_m["ROC AUC"],
                "Gini_VAL":      val_m["Gini"],
                "KS_VAL":        val_m["KS"],
                "Brier_VAL":     val_m["Brier score"],
                "PR_AUC_VAL":    val_m["PR AUC"],
    
                "ROC_AUC_TEST":  test_m["ROC AUC"],
                "Gini_TEST":     test_m["Gini"],
                "KS_TEST":       test_m["KS"],
                "Brier_TEST":    test_m["Brier score"],
                "PR_AUC_TEST":   test_m["PR AUC"],
            })
    
            # szczegółowy print dla danego k
            print(f"\n===== MODEL z k = {k} zmiennymi (statsmodels.Logit) =====")
            print("Zmienne:")
            for v in cols_k:
                print("  -", v)
            print_metrics(f"TRAIN – k={k}", train_m)
            print_metrics(f"VAL   – k={k}", val_m)
            print_metrics(f"TEST  – k={k}", test_m)
    
        # małe podsumowanie – wszystkie zbiory
        print("\nPodsumowanie (TRAIN / VAL / TEST):")
        for r in results:
            print(
                f"k={r['k']:2d} | "
                f"ROC_T={r['ROC_AUC_TRAIN']:.4f}, ROC_V={r['ROC_AUC_VAL']:.4f}, ROC_E={r['ROC_AUC_TEST']:.4f} | "
                f"Gini_T={r['Gini_TRAIN']:.4f}, Gini_V={r['Gini_VAL']:.4f}, Gini_E={r['Gini_TEST']:.4f} | "
                f"KS_T={r['KS_TRAIN']:.4f}, KS_V={r['KS_VAL']:.4f}, KS_E={r['KS_TEST']:.4f} | "
                f"Brier_T={r['Brier_TRAIN']:.4f}, Brier_V={r['Brier_VAL']:.4f}, Brier_E={r['Brier_TEST']:.4f} | "
                f"PR_T={r['PR_AUC_TRAIN']:.4f}, PR_V={r['PR_AUC_VAL']:.4f}, PR_E={r['PR_AUC_TEST']:.4f}"
            )
    
    
    # >>> tutaj wybierasz, jakie k sprawdzić <<<
    k_candidates = [3, 5, 8, 10]
    
    evaluate_k_list(
        k_candidates,
        ordered_vars,
        X_train_woe_df, X_val_woe_df, X_test_woe_df,
        y_train, y_val, y_test
    )
    
    # ============================================================
    #   ZAPIS KOLUMN DO WYRZUCENIA (MODEL k = 5) DO CSV
    # ============================================================
    
    # 1. Zmiennie wybrane do modelu (k=5)
    k_selected = 10
    selected_vars_k5 = ordered_vars[:k_selected]
    
    print("\nWybrane zmienne do modelu k=5:")
    for v in selected_vars_k5:
        print("  -", v)
    
    # 2. Wszystkie cechy po WoE (tak jak używaliśmy w analizie)
    #    woe_feature_names pochodziło z preproc_logit.get_feature_names_out()
    all_features_after_woe = list(woe_feature_names)
    
    
    # 3. Kolumny do DROP: wszystko, co NIE jest w top-5
    drop_cols_k5 = [col for col in all_features_after_woe if col not in selected_vars_k5]
    print(drop_cols_k5)
    drop_cols_k5.append(np.str_('Zysk_netto'))
    print(drop_cols_k5)

    print(f"\nLiczba wszystkich cech po WoE: {len(all_features_after_woe)}")
    print(f"Liczba kolumn do usunięcia (k=5): {len(drop_cols_k5)}")
    
    # 4. Zapis do CSV w formacie dla DropColumnsTransformer
    #    (kolumna 'feature' z nazwami cech)
    drop_dir = os.path.join(BASE_DIR, "interpretowalnosc_logit")
    os.makedirs(drop_dir, exist_ok=True)
    
    drop_k5_path = os.path.join(drop_dir, "drop_columns_k5.csv")
    
    df_drop_k5 = pd.DataFrame({"feature": drop_cols_k5})
    df_drop_k5.to_csv(drop_k5_path, index=False)
    
    print(f"\n✅ Zapisano listę kolumn do usunięcia (k=5) do pliku:\n{drop_k5_path}")
    
    
    
