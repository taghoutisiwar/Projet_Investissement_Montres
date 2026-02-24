"""
Modeling avec Multi-Modèles et MLflow - Fichier 04
Entrée: watches_cleaned.csv
Supporte: RandomForest, XGBoost, GradientBoosting
"""
import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer  # ← AJOUTÉ pour GradientBoosting
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# Configuration MLflow - LOCAL (résout WinError 5)
BASE_DIR = Path(__file__).resolve().parent.parent
MLRUNS_DIR = BASE_DIR / "mlruns"
MLRUNS_DIR.mkdir(exist_ok=True)

# Utiliser le répertoire local au lieu de SQLite
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
mlflow.set_experiment("watch_price_prediction")

print(f"📁 MLflow tracking: {MLRUNS_DIR}")
print(f"💡 Commande pour l'interface: mlflow ui --backend-store-uri file:///{MLRUNS_DIR.as_posix()}")
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from .setup import CLEANED_CSV, PROCESSED_DATA_DIR


RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = [
    "Year of production", "age", "Face Area",
    "Watches Sold by the Seller", "Active listing of the seller",
    "Seller Reviews", "seller_reputation_score", "scope_score",
    "Fast Shipper", "Trusted Seller", "Punctuality",
    "is_modern", "seller_consistent", "price_anomaly_low", "price_anomaly_high"
]

CATEGORICAL_FEATURES = [
    "Brand", "Movement", "Case material", "Bracelet material",
    "Condition", "Scope of delivery", "Gender", "Availability",
    "Shape", "Crystal", "Dial", "Bracelet color"
]

# ============================================================================
# CHARGEMENT ET PRÉPARATION (Partagé)
# ============================================================================

def load_cleaned_data():
    """Charge watches_cleaned.csv"""
    with open(CLEANED_CSV, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        sep = ';' if ';' in first_line else ','
    df = pd.read_csv(CLEANED_CSV, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    print(f"✅ Chargé: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df

def prepare_features(df):
    """Prépare X et y avec uniquement les features disponibles"""
    if 'price_log' not in df.columns:
        if 'Price' in df.columns:
            df['price_log'] = np.log1p(df['Price'])
        else:
            raise KeyError("Aucune colonne de prix trouvée")
    
    available_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    
    print(f"\n📊 Features numériques: {len(available_num)}")
    print(f"📊 Features catégorielles: {len(available_cat)}")
    
    missing_num = set(NUMERIC_FEATURES) - set(available_num)
    missing_cat = set(CATEGORICAL_FEATURES) - set(available_cat)
    if missing_num:
        print(f"⚠️  Numériques manquantes: {missing_num}")
    if missing_cat:
        print(f"⚠️  Catégorielles manquantes: {missing_cat}")
    
    X = df[available_num + available_cat].copy()
    y = df['price_log'].copy()
    
    # Vérifier les NaN dans X
    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        print(f"⚠️  {nan_count} valeurs manquantes détectées dans X")
        print(f"   Par colonne: \n{X.isnull().sum()[X.isnull().sum() > 0]}")
    
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    print(f"\n✅ Dataset final: {X.shape[0]} lignes × {X.shape[1]} features")
    return X, y, available_num, available_cat

# ============================================================================
# FACTORY DE MODÈLES (GradientBoosting)
# ============================================================================

def get_model_config(model_name):
    """
    Factory : retourne la configuration du modèle demandé
    """
    configs = {
        'random_forest': {
            'name': 'RandomForest',
            'model': RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'params': {
                'n_estimators': 300,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            },
            'needs_imputer': False  # ← Gère les NaN nativement
        },
        'gradient_boosting': {
            'name': 'GradientBoosting',
            'model': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=RANDOM_STATE
            ),
            'params': {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.1
            },
            'needs_imputer': True  # ← Nécessite un imputer
        },
        'xgboost': {
            'name': 'XGBoost',
            'model': xgb.XGBRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ) if XGBOOST_AVAILABLE else None,
            'params': {
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.05
            },
            'needs_imputer': False  # ← Gère les NaN nativement
        }
    }
    
    return configs.get(model_name.lower())

def create_pipeline(model_config, numeric_features, categorical_features):
    """
    Crée le pipeline avec le modèle choisi
    CORRECTION: Ajoute un imputer pour GradientBoosting
    """
    
    # Pour GradientBoosting: ajouter un imputer dans le pipeline numérique
    if model_config.get('needs_imputer', False):
        print("   🔧 Ajout d'un imputer pour les valeurs manquantes (médiane)")
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # ← Impute les NaN
            ('scaler', StandardScaler())
        ])
    else:
        numeric_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", model_config['model'])
    ])
    
    return pipeline

# ============================================================================
# ÉVALUATION (Partagé)
# ============================================================================

def evaluate_model(pipeline, X_test, y_test):
    """Évalue le modèle et retourne les métriques"""
    y_pred_log = pipeline.predict(X_test)
    
    # Métriques log
    mae_log = mean_absolute_error(y_test, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
    r2 = r2_score(y_test, y_pred_log)
    
    # Métriques réel
    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred_log)
    mae_real = mean_absolute_error(y_test_real, y_pred_real)
    rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mape = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100
    
    metrics = {
        'MAE_log': mae_log,
        'RMSE_log': rmse_log,
        'R2': r2,
        'MAE_real': mae_real,
        'RMSE_real': rmse_real,
        'MAPE_percent': mape
    }
    
    return metrics, y_pred_real, y_test_real

def analyze_by_segment(y_test_real, y_pred_real):
    """Analyse par segment de prix"""
    print("\n" + "="*60)
    print("ANALYSE PAR SEGMENT DE PRIX")
    print("="*60)
    
    segments = [
        (0, 5000, 'Entry (0-5K€)'),
        (5000, 50000, 'Mid (5K-50K€)'),
        (50000, 500000, 'High (50K-500K€)'),
        (500000, float('inf'), 'Luxury (500K€+)')
    ]
    
    results = []
    for low, high, name in segments:
        mask = (y_test_real >= low) & (y_test_real < high)
        count = mask.sum()
        
        if count == 0:
            continue
            
        mae = mean_absolute_error(y_test_real[mask], y_pred_real[mask])
        mape_seg = np.mean(np.abs((y_test_real[mask] - y_pred_real[mask]) / y_test_real[mask])) * 100
        mean_price = y_test_real[mask].mean()
        
        print(f"\n📊 {name}: {count} montres | Prix moy: {mean_price:,.0f}€ | MAE: {mae:,.0f}€ ({mae/mean_price*100:.1f}%) | MAPE: {mape_seg:.1f}%")
        
        results.append({
            'segment': name,
            'count': count,
            'mae': mae,
            'mape': mape_seg
        })
    
    return results

# ============================================================================
# ENTRAÎNEMENT UNIQUE 
# ============================================================================

def train_single_model(model_name, X_train, y_train, X_test, y_test, num_features, cat_features):
    """Entraîne et évalue un seul modèle"""
    
    config = get_model_config(model_name)
    if config is None:
        raise ValueError(f"Modèle '{model_name}' non reconnu")
    
    if config['model'] is None:
        print(f"⚠️  {config['name']} non disponible (bibliothèque manquante)")
        return None, None
    
    print(f"\n{'='*70}")
    print(f"MODÈLE: {config['name']}")
    print(f"{'='*70}")
    
    # Vérifier les NaN avant
    nan_train = X_train.isnull().sum().sum()
    nan_test = X_test.isnull().sum().sum()
    if nan_train > 0 or nan_test > 0:
        print(f"   ⚠️  NaN détectés: Train={nan_train}, Test={nan_test}")
        if not config.get('needs_imputer', False):
            print(f"   ✅ Le modèle gère les NaN nativement")
    
    # Pipeline
    pipeline = create_pipeline(config, num_features, cat_features)
    
    # MLflow
    ##mlflow.set_experiment("Luxury_Watch_Price_Prediction")
    
    with mlflow.start_run(run_name=f"{config['name']}_Price_Prediction"):
        
        # Entraînement
        print(f"\n🚀 Entraînement {config['name']}...")
        try:
            pipeline.fit(X_train, y_train)
            print("✅ Entraînement terminé")
        except Exception as e:
            print(f"❌ Erreur entraînement: {e}")
            raise
        
        # Évaluation
        metrics, y_pred_real, y_test_real = evaluate_model(pipeline, X_test, y_test)
        analyze_by_segment(y_test_real, y_pred_real)
        
        # Affichage
        print(f"\n{'='*50}")
        print("RÉSULTATS")
        print(f"{'='*50}")
        print(f"R²            : {metrics['R2']:.4f}")
        print(f"MAE (log)     : {metrics['MAE_log']:.4f}")
        print(f"MAE réel (€)  : {metrics['MAE_real']:,.0f}")
        print(f"MAPE          : {metrics['MAPE_percent']:.2f}%")
        
        # Log MLflow
        for key, value in config['params'].items():
            mlflow.log_param(key, value)
        mlflow.log_param('model_type', config['name'])
        mlflow.log_param('imputer_used', config.get('needs_imputer', False))
        
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        mlflow.sklearn.log_model(pipeline, f"price_prediction_{config['name'].lower()}")
        
        print(f"\n✅ {config['name']} loggé dans MLflow")
    
    return pipeline, metrics

# ============================================================================
# COMPARAISON MULTI-MODÈLES
# ============================================================================

def compare_models(model_list=['random_forest', 'gradient_boosting', 'xgboost']):
    """Entraîne et compare plusieurs modèles"""
    
    print("="*70)
    print("COMPARAISON MULTI-MODÈLES - RÉGRESSION")
    print("="*70)
    
    # Chargement
    df = load_cleaned_data()
    X, y, num_features, cat_features = prepare_features(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n✅ Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Entraîner tous les modèles
    results = {}
    for model_name in model_list:
        try:
            pipeline, metrics = train_single_model(
                model_name, X_train, y_train, X_test, y_test, 
                num_features, cat_features
            )
            if pipeline is not None:
                results[model_name] = {
                    'pipeline': pipeline,
                    'metrics': metrics
                }
        except Exception as e:
            print(f"❌ Erreur {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Tableau comparatif
    print(f"\n{'='*70}")
    print("TABLEAU COMPARATIF")
    print(f"{'='*70}")
    print(f"{'Modèle':<20} {'R²':>8} {'MAE_log':>10} {'MAE_real':>12} {'MAPE':>8}")
    print("-" * 70)
    
    for name, data in results.items():
        m = data['metrics']
        print(f"{name:<20} {m['R2']:>8.4f} {m['MAE_log']:>10.4f} {m['MAE_real']:>12,.0f} {m['MAPE_percent']:>7.2f}%")
    
    # Meilleur modèle
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['metrics']['R2'])
        print(f"\n🏆 Meilleur modèle: {best_model[0]} (R² = {best_model[1]['metrics']['R2']:.4f})")
        
        # Sauvegarder le meilleur
        import joblib
        joblib.dump(best_model[1]['pipeline'], PROCESSED_DATA_DIR / "best_price_model.pkl")
        print(f"💾 Meilleur modèle sauvegardé: {PROCESSED_DATA_DIR / 'best_price_model.pkl'}")
    
    return results

# ============================================================================
# MAIN
# ============================================================================

def run_modeling(model_name='random_forest'):
    """Entraîne un seul modèle"""
    return compare_models([model_name])

if __name__ == "__main__":
    # Comparer tous les modèles
    compare_models(['random_forest', 'gradient_boosting', 'xgboost'])