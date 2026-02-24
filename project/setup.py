"""
Configuration des chemins - Fichier 01
"""
import os
from pathlib import Path

# Racine du projet
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Fichiers
RAW_CSV = RAW_DATA_DIR / "watches.csv"
CLEANED_CSV = PROCESSED_DATA_DIR / "watches_cleaned.csv"

# Créer les dossiers
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"✅ Setup OK - ROOT: {ROOT}")