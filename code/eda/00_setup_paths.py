import os
import pandas as pd

# Dossiers fixés par rapport à ce fichier : code/eda/...
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))     # racine/
DATA_DIR = os.path.join(ROOT, "data")                       # racine/data
FIG_DIR  = os.path.join(DATA_DIR, "figures")                # racine/data/figures

RAW_CSV   = os.path.join(DATA_DIR, "watches.csv")
CLEAN_CSV = os.path.join(DATA_DIR, "watches_clean_v1.csv")
REPORT_MD = os.path.join(ROOT, "report_week1.md")

def load_data(path: str | None = None) -> pd.DataFrame:
    p = path or RAW_CSV
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Introuvable: {p}")
    df = pd.read_csv(p)
    df.columns = [c.strip().replace("\u00a0", " ").replace("\n", " ").strip() for c in df.columns]
    return df

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(how="all").copy()
    if "Price" in out.columns:
        out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
        out = out[out["Price"].notna() & (out["Price"] > 0)]
    for c in out.select_dtypes(include="object").columns:
        out[c] = out[c].astype(str).str.strip()
    return out