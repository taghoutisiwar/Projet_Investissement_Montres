"""
Exploration EDA - Fichier 02
Charge watches.csv, fait les visualisations et analyses statistiques
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import importlib.util

# Charger setup.py
setup_path = Path(__file__).parent / "setup.py"
spec = importlib.util.spec_from_file_location("setup", setup_path)
setup_module = importlib.util.module_from_spec(spec)
sys.modules["setup"] = setup_module
spec.loader.exec_module(setup_module)

RAW_CSV = setup_module.RAW_CSV
PROCESSED_DATA_DIR = setup_module.PROCESSED_DATA_DIR

sns.set(style="whitegrid")
FIG_DIR = PROCESSED_DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

def detect_separator(path):
    with open(path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        return ';' if ';' in first_line else ','

def load_data():
    """Charge les données brutes pour exploration"""
    sep = detect_separator(RAW_CSV)
    df = pd.read_csv(RAW_CSV, sep=sep)
    df.columns = [c.strip().replace("\u00a0", " ").strip() for c in df.columns]
    if "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df

# ============================================================================
# NOUVEAU : Analyses statistiques descriptives
# ============================================================================

def statistical_summary(df):
    """Affiche un résumé statistique complet"""
    print("\n" + "="*70)
    print("RÉSUMÉ STATISTIQUE DES DONNÉES")
    print("="*70)
    
    # Describe pour toutes les colonnes numériques
    print("\n📊 Statistiques descriptives (numériques):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe().round(2))
    
    # Pour les colonnes catégorielles
    print("\n📊 Top 5 valeurs par colonne catégorielle:")
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head(5))
    
    # Statistiques spécifiques au prix
    if "Price" in df.columns:
        print("\n" + "="*50)
        print("ANALYSE SPÉCIFIQUE AU PRIX")
        print("="*50)
        price = df["Price"].dropna()
        print(f"Nombre de montres avec prix: {len(price)}")
        print(f"Prix moyen: {price.mean():,.0f}€")
        print(f"Prix médian: {price.median():,.0f}€")
        print(f"Écart-type: {price.std():,.0f}€")
        print(f"Skewness: {price.skew():.2f} (asymétrie)")
        print(f"Kurtosis: {price.kurtosis():.2f} (aplatissement)")
        print(f"\nPercentiles:")
        for p in [5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"  {p}ème: {np.percentile(price, p):,.0f}€")

def correlation_matrix(df):
    """Crée et sauvegarde la matrice de corrélation"""
    print("\n" + "="*70)
    print("MATRICE DE CORRÉLATION")
    print("="*70)
    
    # Sélectionner uniquement les colonnes numériques
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("⚠️  Pas de colonnes numériques pour la corrélation")
        return
    
    # Calculer la corrélation
    corr = numeric_df.corr()
    
    # Afficher les corrélations les plus fortes avec Price
    if "Price" in corr.columns:
        print("\n📊 Corrélations avec Price (valeurs absolues > 0.1):")
        price_corr = corr["Price"].drop("Price").abs().sort_values(ascending=False)
        price_corr = price_corr[price_corr > 0.1]
        for feature, correlation in price_corr.items():
            actual_corr = corr["Price"][feature]
            print(f"  {feature}: {actual_corr:.3f}")
    
    # Créer la heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Masquer le triangle supérieur
    sns.heatmap(corr, 
                mask=mask,
                annot=True, 
                fmt=".2f", 
                cmap="RdBu_r",
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title("Matrice de Corrélation des Features Numériques", fontsize=14, pad=20)
    plt.tight_layout()
    
    out = FIG_DIR / "10_correlation_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Matrice de corrélation sauvegardée: {out}")

# ============================================================================
# Visualisations existantes (inchangées)
# ============================================================================

def save_fig(name):
    plt.tight_layout()
    out = FIG_DIR / name
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Figure: {out}")

def plot_price_distribution(df):
    if "Price" not in df.columns or df["Price"].dropna().empty:
        return
    plt.figure(figsize=(10, 5))
    sns.histplot(df["Price"], bins=100, kde=True, color="#2E86AB")
    plt.title("Distribution du prix")
    plt.xlabel("Prix (€)")
    save_fig("01_price_distribution.png")

def plot_price_log(df):
    if "Price" not in df.columns:
        return
    plt.figure(figsize=(10, 5))
    sns.histplot(np.log1p(df["Price"].dropna()), bins=100, kde=True, color="#7D3C98")
    plt.title("Distribution log(1+prix)")
    save_fig("02_price_log.png")

def plot_top_brands(df, n=15):
    if "Brand" not in df.columns:
        return
    top = df["Brand"].value_counts().head(n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top.values, y=top.index, color="#1ABC9C")
    plt.title(f"Top {n} marques")
    save_fig("03_top_brands.png")

def plot_brand_price_median(df, n=15):
    if "Brand" not in df.columns or "Price" not in df.columns:
        return
    g = df.groupby("Brand")["Price"].agg(['count', 'median']).query("count >= 5")
    g = g.sort_values("median", ascending=False).head(n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=g["median"], y=g.index, color="#F39C12")
    plt.title(f"Top {n} marques par médiane de prix")
    save_fig("04_brand_median_price.png")

def plot_movement(df):
    if "Movement" not in df.columns:
        return
    plt.figure(figsize=(10, 5))
    df["Movement"].value_counts().head(10).plot(kind='bar', color="#E74C3C")
    plt.title("Répartition des mouvements")
    plt.xticks(rotation=45)
    save_fig("05_movement.png")

def plot_case_material(df):
    if "Case material" not in df.columns:
        return
    plt.figure(figsize=(10, 5))
    df["Case material"].value_counts().head(10).plot(kind='bar', color="#16A085")
    plt.title("Matériaux du boîtier")
    plt.xticks(rotation=45)
    save_fig("06_case_material.png")

def plot_condition(df):
    if "Condition" not in df.columns:
        return
    plt.figure(figsize=(10, 5))
    df["Condition"].value_counts().plot(kind='bar', color="#2C3E50")
    plt.title("État des montres")
    plt.xticks(rotation=45)
    save_fig("07_condition.png")

def plot_year_distribution(df):
    if "Year of production" not in df.columns:
        return
    years = pd.to_numeric(df["Year of production"], errors="coerce").dropna()
    plt.figure(figsize=(10, 5))
    sns.histplot(years, bins=50, color="#27AE60")
    plt.title("Distribution des années")
    save_fig("08_year_distribution.png")

def plot_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return
    plt.figure(figsize=(12, 6))
    missing_pct = (missing / len(df) * 100)
    sns.barplot(x=missing_pct.values, y=missing_pct.index, palette="Reds_r")
    plt.title("Valeurs manquantes par colonne (%)")
    plt.axvline(x=40, color='red', linestyle='--', label='40%')
    save_fig("09_missing_values.png")

# ============================================================================
# MAIN
# ============================================================================

def run_eda():
    print("="*70)
    print("EXPLORATION DES DONNÉES (EDA)")
    print("="*70)
    
    df = load_data()
    print(f"\n📊 Dataset: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"\nColonnes: {list(df.columns)}")
    
    # NOUVEAU : Résumé statistique
    statistical_summary(df)
    
    # NOUVEAU : Matrice de corrélation
    correlation_matrix(df)
    
    # Résumé rapide
    print(f"\n--- Résumé ---")
    print(f"Price: min={df['Price'].min():.0f}, max={df['Price'].max():.0f}, null={df['Price'].isnull().sum()}")
    print(f"Duplications: {df.duplicated().sum()}")
    
    # Visualisations
    print(f"\n--- Génération des figures ---")
    plot_price_distribution(df)
    plot_price_log(df)
    plot_top_brands(df)
    plot_brand_price_median(df)
    plot_movement(df)
    plot_case_material(df)
    plot_condition(df)
    plot_year_distribution(df)
    plot_missing_values(df)
    
    print(f"\n✅ EDA terminé. Figures dans: {FIG_DIR}")
    return df

if __name__ == "__main__":
    run_eda()