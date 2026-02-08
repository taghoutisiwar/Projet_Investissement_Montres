import os, importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.abspath(os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location("setup_paths", os.path.join(HERE, "00_setup_paths.py"))
setup_paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup_paths)

sns.set(style="whitegrid")

def _save(fig_dir, name):
    # ❌ aucune création automatique
    if not os.path.isdir(fig_dir):
        raise FileNotFoundError(
            f"Dossier introuvable: {fig_dir}. Crée-le manuellement (ex: data/figures/)."
        )
    plt.tight_layout()
    out = os.path.join(fig_dir, name)
    plt.savefig(out, dpi=150)
    plt.close()
    print("Figure écrite :", out)

def plot_price_hist(df, fig_dir):
    if "Price" not in df.columns or df["Price"].dropna().empty: return
    plt.figure(figsize=(8,5))
    sns.histplot(df["Price"], bins=100, kde=True, color="#2E86AB")
    plt.title("Distribution du prix (€)")
    plt.xlabel("Prix (€)")
    _save(fig_dir, "01_price_hist.png")

def plot_price_hist_log(df, fig_dir):
    if "Price" not in df.columns or df["Price"].dropna().empty: return
    plt.figure(figsize=(8,5))
    sns.histplot(np.log1p(df["Price"]), bins=100, kde=True, color="#7D3C98")
    plt.title("Distribution du log(1+prix)")
    plt.xlabel("log(1+Prix)")
    _save(fig_dir, "02_price_hist_log.png")

def plot_top_brands(df, fig_dir, top_n=15):
    if "Brand" not in df.columns: return
    vc = df["Brand"].value_counts().head(top_n)
    if vc.empty: return
    plt.figure(figsize=(10,6))
    sns.barplot(x=vc.values, y=vc.index, color="#1ABC9C")
    plt.title(f"Top {top_n} marques (volume)")
    plt.xlabel("Nombre d'annonces")
    plt.ylabel("Marque")
    _save(fig_dir, "03_top_brands.png")

def plot_brand_median_price(df, fig_dir, min_count=5, top_n=15):
    req = {"Brand","Price"}
    if not req.issubset(df.columns): return
    g = (df[list(req)]
         .dropna()
         .query("Price > 0")  # <-- corrigé (&gt; -> >)
         .groupby("Brand")
         .agg(count=("Price","count"), median_price=("Price","median"))
         .query("count >= @min_count")  # <-- corrigé (&gt;= -> >=)
         .sort_values("median_price", ascending=False)
         .head(top_n))
    if g.empty: return
    plt.figure(figsize=(10,6))
    sns.barplot(x=g["median_price"], y=g.index, color="#F39C12")
    plt.title(f"Top {top_n} marques par médiane de prix (≥ {min_count} annonces)")
    plt.xlabel("Médiane de prix (€)")
    plt.ylabel("Marque")
    _save(fig_dir, "04_top_brand_medians.png")

def plot_movement(df, fig_dir, top_n=10):
    if "Movement" not in df.columns: return
    vc = df["Movement"].value_counts().head(top_n)
    if vc.empty: return
    plt.figure(figsize=(8,5))
    sns.barplot(x=vc.index, y=vc.values, color="#E74C3C")
    plt.xticks(rotation=45, ha="right")
    plt.title("Répartition des mouvements (top)")
    plt.ylabel("Nombre")
    _save(fig_dir, "05_movement.png")

def plot_case_material(df, fig_dir, top_n=10):
    if "Case material" not in df.columns: return
    vc = df["Case material"].value_counts().head(top_n)
    if vc.empty: return
    plt.figure(figsize=(10,5))
    sns.barplot(x=vc.index, y=vc.values, color="#16A085")
    plt.xticks(rotation=45, ha="right")
    plt.title("Matériau du boîtier (top)")
    plt.ylabel("Nombre")
    _save(fig_dir, "06_case_material.png")

def plot_condition(df, fig_dir, top_n=10):
    if "Condition" not in df.columns: return
    vc = df["Condition"].value_counts().head(top_n)
    if vc.empty: return
    plt.figure(figsize=(8,5))
    sns.barplot(x=vc.index, y=vc.values, color="#2C3E50")
    plt.xticks(rotation=45, ha="right")
    plt.title("État (top)")
    plt.ylabel("Nombre")
    _save(fig_dir, "07_condition.png")

def plot_year_hist(df, fig_dir):
    if "Year of production" not in df.columns: return
    s = pd.to_numeric(df["Year of production"], errors="coerce").dropna()
    if s.empty: return
    plt.figure(figsize=(10,5))
    sns.histplot(s, bins=50, kde=False, color="#27AE60")
    plt.title("Distribution des années de production")
    plt.xlabel("Année")
    _save(fig_dir, "08_year_hist.png")

def plot_price_vs_year(df, fig_dir):
    req = {"Price","Year of production"}
    if not req.issubset(df.columns): return
    tmp = df[list(req)].dropna()
    if tmp.empty: return
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=tmp, x="Year of production", y="Price", alpha=0.3)
    plt.title("Prix vs Année de production")
    _save(fig_dir, "09_price_vs_year.png")

def plot_price_by_condition_box(df, fig_dir, top_n_cond=6):
    req = {"Price","Condition"}
    if not req.issubset(df.columns): return
    cond_top = df["Condition"].value_counts().head(top_n_cond).index.tolist()
    subset = df[df["Condition"].isin(cond_top)]
    if subset.empty: return
    plt.figure(figsize=(10,6))
    sns.boxplot(data=subset, x="Condition", y="Price")
    plt.xticks(rotation=30, ha="right")
    plt.yscale("log")
    plt.title("Distribution du prix par état (échelle log)")
    _save(fig_dir, "10_price_by_condition_box.png")

def run_all_plots(df, fig_dir):
    plot_price_hist(df, fig_dir)
    plot_price_hist_log(df, fig_dir)
    plot_top_brands(df, fig_dir)
    plot_brand_median_price(df, fig_dir)
    plot_movement(df, fig_dir)
    plot_case_material(df, fig_dir)
    plot_condition(df, fig_dir)
    plot_year_hist(df, fig_dir)
    plot_price_vs_year(df, fig_dir)
    plot_price_by_condition_box(df, fig_dir)

def main():
    # Charge le clean si dispo, sinon le brut puis nettoie
    if os.path.exists(setup_paths.CLEAN_CSV):
        df = pd.read_csv(setup_paths.CLEAN_CSV)
        df.columns = [c.strip() for c in df.columns]
    else:
        df_raw = setup_paths.load_data(setup_paths.RAW_CSV)
        df = setup_paths.basic_cleaning(df_raw)

    run_all_plots(df, setup_paths.FIG_DIR)
    print("Toutes les figures sont dans:", setup_paths.FIG_DIR)

if __name__ == "__main__":
    main()