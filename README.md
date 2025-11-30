# 📄 MODEL CARD – Model oceny ryzyka kredytowego (Logit WoE + XGBoost)

## 1. **Nazwa modelu**
**Model oceny ryzyka kredytowego dla firm (Logit WoE + XGBoost)**  
Projekt w ramach kursu *IWUM – Interpretowalność i Wyjaśnialność Uczenia Maszynowego*.

---

## 2. 🎯 **Cel modelu**

Model został stworzony w celu:

- przewidywania **prawdopodobieństwa defaultu (PD)** dla klientów firmowych,  
- wspierania decyzji kredytowych w oparciu o dane,  
- automatycznego nadawania ratingów (AAA → CCC),  
- wyznaczania optymalnych progów decyzyjnych na podstawie cost curves.

System wykorzystuje dwa modele:

1. **Logistic Regression + Weight of Evidence (interpretowalny)**  
2. **XGBoost (black-box)**  

---

## 3. 📊 **Dane wejściowe**

### Źródło danych
Zbiór dostarczony w projekcie (plik: `zbiór_7.csv`).

### Charakterystyka:
- Typ danych: **firmy (SME)**  
- Zmienna celu: `default ∈ {0,1}`  
- Zmienne wejściowe: dane finansowe i opisowe przedsiębiorstw  
- Podział:
  - 60% train  
  - 20% val  
  - 20% test  
  - podział stratyfikowany  

### Przetwarzanie:
- WoE + binning (monotoniczny)  
- Scaling / preprocessing dla XGBoost  
- Odrzucenie zmiennych z dużą liczbą braków  

---

## 4. 📉 **Metody modelowania**

### 🔷 Logistic Regression (interpretable)
- WoE zapewnia monotoniczność cech  
- Prostota walidacji biznesowej  
- Łatwa interpretacja wpływu zmiennych  

### 🔶 XGBoost (black-box)
- Boosting drzew → wysoka jakość predykcji  
- Wyjaśnienia uzyskane przy użyciu SHAP  

---

## 5. 🧪 **Ocena i walidacja**

### Metryki:
- ROC AUC  
- KS  
- Brier Score  
- Calibration curve  

### Kalibracja PD
Model został skalibrowany, tak aby średnie PD wynosiło **ok. 4%**, zgodnie z historycznym poziomem strat.

---

## 6. 🔍 **Wyjaśnialność modelu**

### Wyjaśnienia globalne:
- Feature importance  
- SHAP Summary Plot  
- Heatmapy korelacji  
- Stabilność cech WoE  

### Wyjaśnienia lokalne:
- SHAP force plot dla pojedynczego klienta  
- Lista cech podwyższających/obniżających PD  

---

## 7. ⚠️ **Ograniczenia modelu**

### Dane:
- Zbiór może nie być w pełni reprezentatywny dla realnej populacji  
- Część cech posiada braki  
- Brak zmiennych makroekonomicznych

### Metody:
- Logit jest liniowy na log-odds  
- XGBoost może się przeuczać bez monitoringu  

### Zastosowanie:
- Model nie powinien podejmować decyzji automatycznie  
- Wymaga eksperckiej kontroli  

---

## 8. ⚡ **Ryzyka modelu**

### 1. **Ryzyko błędnej klasyfikacji**
- FP → udzielenie kredytu złemu klientowi (strata)  
- FN → odrzucenie dobrego klienta (utrata zysku)  

### 2. **Data drift**
- Zmiana zachowania firm  
- Zmiany makroekonomiczne  

### 3. **Ryzyko etyczne**
- Możliwa korelacja z cechami pośrednio wrażliwymi  

---

## 9. 🧭 **Ratingi i progi decyzyjne**

### Ratingi:
Rating = kwantyl PD z danych treningowych.  
Skala: **AAA, AA, A, BBB, BB, B, CCC**

### Progi decyzyjne:
Wybrane na podstawie:

- tabelek decyzyjnych  
- krzywych zysku (cost curves)  
- maksymalizacji oczekiwanego zysku portfela  

Optymalny próg PD znajduje się ok. **0.14–0.17**.

---

## 10. ⏱️ **Plan monitoringu modelu**

Monitorować co **miesiąc**, pełen przegląd co **kwartał**.

### Monitorowane elementy:

#### Dane:
- Rozkłady cech  
- Braki danych  
- PSI (Population Stability Index)

#### Model:
- AUC, KS  
- Brier score  
- Kalibracja PD  

#### Decyzje:
- Realny zysk/strata vs. cost curve  
- Stabilność progu decyzyjnego  

### Kiedy retrain?
- PSI > 0.25  
- Spadek AUC o > 5 p.p.  
- Zmiana default rate > 50%  

---

## 11. ✔️ **Podsumowanie**

Model łączy interpretowalność (Logit WoE) z wysoką jakością (XGBoost).  
Może wspierać proces kredytowy, ale wymaga regularnego monitoringu, walidacji i nadzoru analityka.

---

