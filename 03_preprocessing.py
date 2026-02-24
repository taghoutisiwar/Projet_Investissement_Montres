"""
Preprocessing et Feature Engineering - Fichier 03
Entrée: watches.csv
Sortie: watches_cleaned.csv (prêt pour le modeling)
Avec validation croisée des imputations
"""
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.model_selection import KFold

from .setup import RAW_CSV, CLEANED_CSV, PROCESSED_DATA_DIR

CURRENT_YEAR = 2026
RANDOM_STATE = 42

def detect_separator(path):
    with open(path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        return ';' if ';' in first_line else ','

def load_raw():
    """Charge les données brutes"""
    sep = detect_separator(RAW_CSV)
    df = pd.read_csv(RAW_CSV, sep=sep)
    df.columns = [c.strip().replace("\u00a0", " ").strip() for c in df.columns]
    return df

# ============================================================================
# ÉTAPE 1: NETTOYAGE
# ============================================================================

def remove_duplicates(df):
    """Supprime les lignes dupliquées"""
    before = len(df)
    df = df.drop_duplicates()
    print(f"✅ Duplications: {before - len(df)} lignes supprimées")
    return df

def clean_price(df):
    """Nettoie et filtre la colonne Price"""
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df[df["Price"].notna() & (df["Price"] > 0)]
    print(f"✅ Price nettoyé: {len(df)} lignes avec prix valide")
    return df

def remove_high_missing_columns(df, threshold=40):
    """Supprime colonnes avec >threshold% de manquants"""
    missing_pct = (df.isnull().sum() / len(df) * 100)
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    
    if 'Year of production' in cols_to_drop:
        cols_to_drop.remove('Year of production')
        print(f"⚠️ 'Year of production' conservée malgré {missing_pct['Year of production']:.1f}% manquants")
    
    df = df.drop(columns=cols_to_drop)
    print(f"✅ Colonnes supprimées (>40%): {cols_to_drop}")
    return df

# ============================================================================
# ÉTAPE 2: IMPUTATION AVEC VALIDATION CROISÉE
# ============================================================================

def validate_imputation(df, column, impute_func, n_splits=5):
    """
    Valide la qualité d'une imputation par validation croisée
    """
    print(f"\n🔍 Validation croisée de l'imputation pour '{column}'")
    
    # Séparer les lignes avec et sans valeur
    known = df[df[column].notna()].copy()
    missing = df[df[column].isna()].copy()
    
    if len(known) < n_splits * 2:
        print(f"   ⚠️  Trop peu de données connues pour la validation croisée")
        return None
    
    # Créer des folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    errors = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(known), 1):
        # Simuler des valeurs manquantes
        train_data = known.iloc[train_idx].copy()
        val_data = known.iloc[val_idx].copy()
        
        # Valeurs réelles à prédire
        true_values = val_data[column].copy()
        
        # Masquer les valeurs de validation
        val_data_copy = val_data.copy()
        val_data_copy[column] = np.nan
        
        # Combiner train et val pour l'imputation
        temp_df = pd.concat([train_data, val_data_copy])
        
        # Appliquer l'imputation
        temp_df = impute_func(temp_df)
        
        # Récupérer les valeurs imputées
        imputed_values = temp_df.loc[val_data.index, column]
        
        # Calculer l'erreur
        mae = np.mean(np.abs(imputed_values - true_values))
        errors.append(mae)
        print(f"   Fold {fold}: MAE = {mae:.2f}")
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    print(f"   📊 MAE moyenne: {mean_error:.2f} ± {std_error:.2f}")
    
    return {
        'mean_mae': mean_error,
        'std_mae': std_error,
        'fold_errors': errors
    }

def impute_year_of_production(df, validate=True):
    """Impute Year of production par médiane de Brand avec validation"""
    
    def impute_func(temp_df):
        brand_median = temp_df.groupby('Brand')['Year of production'].median()
        
        def impute_row(row):
            if pd.isna(row['Year of production']):
                brand = row['Brand']
                if pd.notna(brand) and brand in brand_median.index:
                    return brand_median[brand]
                return 2020
            return row['Year of production']
        
        temp_df['Year of production'] = temp_df.apply(impute_row, axis=1)
        return temp_df
    
    # Validation croisée
    if validate and 'Year of production' in df.columns:
        validation_result = validate_imputation(df, 'Year of production', impute_func)
    
    # Application réelle
    df = impute_func(df)
    print(f"✅ Year of production imputée (médiane par Brand)")
    if validate and validation_result:
        print(f"   Qualité estimée: MAE = {validation_result['mean_mae']:.2f}")
    return df

def impute_categorical(df, validate=True):
    """Impute les colonnes catégorielles par mode avec validation"""
    
    cat_cols = df.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            
            # Validation simple (pas de CV pour catégoriel, juste fréquence)
            missing_before = df[col].isnull().sum()
            df[col] = df[col].fillna(mode_val)
            print(f"✅ {col} imputée par mode: '{mode_val}' ({missing_before} valeurs)")
    
    return df

def impute_numerical(df, validate=True):
    """Impute les numériques (hors Price, Year) par médiane avec validation"""
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    exclude = ['Price', 'Year of production']
    
    for col in num_cols:
        if col not in exclude and df[col].isnull().sum() > 0:
            
            def impute_func(temp_df):
                median_val = temp_df[col].median()
                temp_df[col] = temp_df[col].fillna(median_val)
                return temp_df
            
            # Validation croisée
            if validate:
                validation_result = validate_imputation(df, col, impute_func)
            
            # Application
            median_val = df[col].median()
            missing_before = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            print(f"✅ {col} imputée par médiane: {median_val:.2f} ({missing_before} valeurs)")
            if validate and validation_result:
                print(f"   Qualité estimée: MAE = {validation_result['mean_mae']:.2f}")
    
    return df

# ============================================================================
# ÉTAPE 3: GESTION DES ANOMALIES
# ============================================================================

def fix_year_anomalies(df):
    """Corrige les années aberrantes"""
    mask = (df['Year of production'] < 1900) | (df['Year of production'] > CURRENT_YEAR)
    anomalies = mask.sum()
    
    brand_median = df.groupby('Brand')['Year of production'].median()
    
    df.loc[mask, 'Year of production'] = df.loc[mask, 'Brand'].map(
        lambda x: brand_median.get(x, 2020)
    )
    print(f"✅ Années aberrantes corrigées: {anomalies}")
    return df

def winsorize_face_area(df):
    """Winsorise Face Area au 99ème percentile"""
    if 'Face Area' in df.columns:
        p99 = df['Face Area'].quantile(0.99)
        df['Face Area'] = df['Face Area'].clip(upper=p99)
        print(f"✅ Face Area winsorisé (p99={p99:.2f})")
    return df

# ============================================================================
# ÉTAPE 4: FEATURE ENGINEERING
# ============================================================================

def create_features(df):
    """Crée toutes les nouvelles features"""
    print(f"\n{'='*50}")
    print("FEATURE ENGINEERING")
    print(f"{'='*50}")
    
    # Age
    df['age'] = CURRENT_YEAR - df['Year of production']
    print(f"✅ 'age' créée (moy: {df['age'].mean():.1f} ans)")
    
    # Is modern
    df['is_modern'] = (df['Year of production'] >= 2000).astype(int)
    print(f"✅ 'is_modern' créée ({df['is_modern'].sum()} montres)")
    
    # Log price (cible pour la régression)
    df['price_log'] = np.log1p(df['Price'])
    print(f"✅ 'price_log' créée")
    
    # Seller reputation score
    seller_cols = ['Fast Shipper', 'Trusted Seller', 'Punctuality']
    available = [c for c in seller_cols if c in df.columns]
    if available:
        df['seller_reputation_score'] = df[available].mean(axis=1)
        print(f"✅ 'seller_reputation_score' créée")
    
    # Scope score
    scope_map = {
        'Original box, original papers': 3,
        'Original box, no original papers': 2,
        'Original papers, no original box': 1,
        'No original box, no original papers': 0
    }
    if 'Scope of delivery' in df.columns:
        df['scope_score'] = df['Scope of delivery'].map(scope_map).fillna(0)
        print(f"✅ 'scope_score' créée")
    
    # Flags d'anomalies
    df['price_anomaly_low'] = (df['Price'] < 50).astype(int)
    df['price_anomaly_high'] = (df['Price'] > 1000000).astype(int)
    print(f"✅ Flags d'anomalies créés")
    
    # Consistance vendeur
    if all(c in df.columns for c in ['Watches Sold by the Seller', 'Active listing of the seller']):
        df['seller_consistent'] = (
            df['Watches Sold by the Seller'] >= df['Active listing of the seller']
        ).astype(int)
        print(f"✅ 'seller_consistent' créée")
    
    return df

# ============================================================================
# PIPELINE COMPLET
# ============================================================================

def run_preprocessing(validate_imputations=True):
    """Exécute tout le preprocessing"""
    print(f"\n{'='*70}")
    print("PREPROCESSING ET FEATURE ENGINEERING")
    print(f"{'='*70}")
    print(f"Validation des imputations: {'Activée' if validate_imputations else 'Désactivée'}")
    
    # Chargement
    df = load_raw()
    print(f"\n📥 Entrée: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    
    # Nettoyage
    print(f"\n{'='*50}")
    print("NETTOYAGE")
    print(f"{'='*50}")
    df = remove_duplicates(df)
    df = clean_price(df)
    df = remove_high_missing_columns(df)
    
    # Imputation avec validation
    print(f"\n{'='*50}")
    print("IMPUTATION (avec validation croisée)")
    print(f"{'='*50}")
    df = impute_year_of_production(df, validate=validate_imputations)
    df = impute_categorical(df, validate=validate_imputations)
    df = impute_numerical(df, validate=validate_imputations)
    
    # Anomalies
    print(f"\n{'='*50}")
    print("CORRECTION ANOMALIES")
    print(f"{'='*50}")
    df = fix_year_anomalies(df)
    df = winsorize_face_area(df)
    
    # Feature Engineering
    df = create_features(df)
    
    # Vérification finale
    print(f"\n{'='*50}")
    print("VÉRIFICATION FINALE")
    print(f"{'='*50}")
    print(f"Dimensions: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"Valeurs manquantes: {df.isnull().sum().sum()}")
    print(f"Colonnes: {list(df.columns)}")
    
    # Sauvegarde
    df.to_csv(CLEANED_CSV, index=False, sep=';')
    print(f"\n💾 Sauvegardé: {CLEANED_CSV}")
    
    return df

if __name__ == "__main__":
    from pathlib import Path
    # Par défaut, validation activée
    df_cleaned = run_preprocessing(validate_imputations=True)