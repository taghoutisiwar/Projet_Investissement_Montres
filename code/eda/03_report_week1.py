import os, importlib.util
from datetime import datetime
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location("setup_paths", os.path.join(HERE, "00_setup_paths.py"))
setup_paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_paths)

FIGS_ORDER = [
    "01_price_hist.png", "02_price_hist_log.png", "03_top_brands.png",
    "04_top_brand_medians.png", "05_movement.png", "06_case_material.png",
    "07_condition.png", "08_year_hist.png", "09_price_vs_year.png",
    "10_price_by_condition_box.png"
]

def build_report():
    # stats (recalcul rapide)
    df_raw = setup_paths.load_data(setup_paths.RAW_CSV)
    if os.path.exists(setup_paths.CLEAN_CSV):
        df_clean = pd.read_csv(setup_paths.CLEAN_CSV)
        df_clean.columns = [c.strip() for c in df_clean.columns]
    else:
        df_clean = setup_paths.basic_cleaning(df_raw)

    raw_shape = df_raw.shape
    clean_shape = df_clean.shape
    price_desc = None
    if "Price" in df_clean.columns:
        price_desc = df_clean["Price"].describe(percentiles=[.25,.5,.75,.9,.95,.99]).to_dict()

    lines = []
    lines.append("# Week 1 – Setup, Cleaning & EDA") 
    lines.append(f"_Généré le {datetime.now():%Y-%m-%d %H:%M}_\n")

    lines.append("## Données")
    lines.append(f"- Brut     : **{raw_shape[0]}** lignes × **{raw_shape[1]}** colonnes")
    lines.append(f"- Nettoyé  : **{clean_shape[0]}** lignes × **{clean_shape[1]}** colonnes\n")

    if price_desc:
        lines.append("## Prix (statistiques rapides)")
        lines.append(f"- Médiane ≈ **{price_desc.get('50%', 0):.0f} €**")
        lines.append(f"- P95 ≈ **{price_desc.get('95%', 0):.0f} €**")
        lines.append(f"- Max ≈ **{price_desc.get('max', 0):.0f} €**\n")

    lines.append("## Figures")
    for f in FIGS_ORDER:
        fp = os.path.join(setup_paths.FIG_DIR, f)
        if os.path.exists(fp):
            # on écrit un chemin relatif propre depuis la racine
            lines.append(f"- {os.path.relpath(fp, setup_paths.ROOT)}")

    # Écriture (pas de création automatique de dossier)
    with open(setup_paths.REPORT_MD, "w", encoding="utf-8") as out:
        out.write("\n".join(lines))

    print("Rapport écrit ->", setup_paths.REPORT_MD)

if __name__ == "__main__":
    build_report()