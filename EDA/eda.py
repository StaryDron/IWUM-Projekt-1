import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------
# KONFIG
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # katalog z tym plikiem
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "zbiór_7.csv")

# EDA_wyniki w BASE (np. .../EDA/EDA_wyniki)
SAVE_DIR = os.path.join(BASE_DIR, "EDA_wyniki")
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------------------------------------------
# WCZYTANIE DANYCH
# -------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\n=== Podstawowe informacje ===")
print(df.info())
print(df.describe())

# ===================================================================
# 1) BAR PLOT: liczba defaultów
# ===================================================================

plt.figure(figsize=(6, 4))
df["default"].value_counts().sort_index().plot(kind="bar", color=["green", "red"])
plt.title("Rozkład defaultów")
plt.xlabel("Default (0 = good, 1 = bad)")
plt.ylabel("Liczba obserwacji")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "default_distribution.png"), dpi=150)
plt.close()

# ===================================================================
# 2) Podział train/val/test + bar plot
# ===================================================================

X = df.drop(columns=["default"])
y = df["default"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

sizes = {
    "Train": len(X_train),
    "Validation": len(X_val),
    "Test": len(X_test),
}

plt.figure(figsize=(6, 4))
plt.bar(sizes.keys(), sizes.values(), color=["blue", "orange", "green"])
plt.title("Liczność zbiorów: train / val / test")
plt.ylabel("Liczba obserwacji")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "train_val_test_counts.png"), dpi=150)
plt.close()

# ===================================================================
# 3) Braki danych
# ===================================================================

missing = df.isna().sum()
missing = missing[missing > 0]

print("\n=== Kolumny z brakami ===")
print(missing)

missing.to_csv(os.path.join(SAVE_DIR, "missing_values.csv"))

# ===================================================================
# 4) Korelacje – heatmap dla 10 zmiennych NAJBARDZIEJ skorelowanych z defaultem
# ===================================================================

# bierzemy tylko kolumny numeryczne
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "default"]

# korelacje z defaultem
corr_full = df[numeric_cols + ["default"]].corr()
corr_with_target = corr_full["default"].drop("default")

top10_vars = corr_with_target.abs().sort_values(ascending=False).head(10).index.tolist()

print("\nTop 10 zmiennych najbardziej skorelowanych z defaultem:")
print(top10_vars)

corr_top10 = df[top10_vars + ["default"]].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_top10, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap korelacji – top 10 zmiennych vs default")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "correlation_heatmap_top10.png"), dpi=150)
plt.close()

# ===================================================================
# 5) Scatter ploty dla najmocniejszych zmiennych
#     – wybieramy 3 najmocniej skorelowane z defaultem
# ===================================================================

target_corr_top = corr_top10["default"].abs().sort_values(ascending=False)
strong_vars = target_corr_top.index[1:4]  # pomijamy 'default' na pozycji 0

print("\nNajsilniej skorelowane zmienne z defaultem (do scatterów):")
print(strong_vars)

# Parowe scatter ploty
for i in range(len(strong_vars)):
    for j in range(i + 1, len(strong_vars)):
        v1 = strong_vars[i]
        v2 = strong_vars[j]

        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            data=df,
            x=v1,
            y=v2,
            hue="default",
            palette={0: "green", 1: "red"},
            alpha=0.6,
        )
        plt.title(f"Scatter: {v1} vs {v2} (kolor=default)")
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"scatter_{v1}_{v2}.png"), dpi=150)
        plt.close()

print("\nEDA zapisane do folderu:", SAVE_DIR)
