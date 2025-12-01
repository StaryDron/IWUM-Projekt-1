import os
import subprocess
import sys

# ---------------------------------------------------------
# Funkcja pomocnicza – bezpieczne odpalanie skryptów
# ---------------------------------------------------------
def run_script(path):
    print(f"\n=== Uruchamianie: {path} ===")
    if not os.path.exists(path):
        print(f"❌ Błąd: plik nie istnieje → {path}")
        sys.exit(1)

    result = subprocess.run([sys.executable, path])
    if result.returncode != 0:
        print(f"❌ Błąd podczas wykonywania {path}")
        sys.exit(result.returncode)

    print(f"✅ Zakończono: {path}")


# ---------------------------------------------------------
# Ścieżki bazowe
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EDA_DIR = os.path.join(BASE_DIR, "EDA")
INTERPRET_DIR = os.path.join(BASE_DIR, "Modele_interpretowalne")
BLACKBOX_DIR = os.path.join(BASE_DIR, "Modele_nieinterpretowalne")
KALIBR_DIR = os.path.join(BASE_DIR, "Kalibracja")
RATING_DIR = os.path.join(BASE_DIR, "Ratingi")

# ---------------------------------------------------------
# Lista skryptów w kolejności wykonania
# ---------------------------------------------------------

SCRIPTS = [
    # --- EDA ---
    os.path.join(EDA_DIR, "eda.py"),
    os.path.join(EDA_DIR, "eda_transformers.py"),
    os.path.join(EDA_DIR, "dopasowanie_pipeline.py"),

    # --- Modele interpretowalne ---
    os.path.join(INTERPRET_DIR, "model_interpretowalny.py"),
    os.path.join(INTERPRET_DIR, "ocena_jakosci_modelow_wykresy.py"),
    os.path.join(INTERPRET_DIR, "interpretowalnosc_regresja_logistyczna.py"),

    # --- Modele nieinterpretowalne ---
    os.path.join(BLACKBOX_DIR, "model_nieinterpretowalny.py"),

    # --- Kalibracja ---
    os.path.join(KALIBR_DIR, "kalibracja.py"),

    # --- Ratingi i progi ---
    os.path.join(RATING_DIR, "ratingi.py"),
]

# ---------------------------------------------------------
# Główna pętla
# ---------------------------------------------------------
if __name__ == "__main__":

    print("\n==============================================")
    print("   URUCHAMIANIE PEŁNEGO PIPELINE PROJEKTU")
    print("==============================================")

    for script in SCRIPTS:
        run_script(script)

    print("\n==============================================")
    print("   🎉 ZAKOŃCZONO cały pipeline bez błędów! 🎉")
    print("==============================================")