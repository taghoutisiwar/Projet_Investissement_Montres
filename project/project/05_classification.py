"""
Classification Multi-Modèles - Fichier 05
Classes: Bon investissement / Moyen / Risqué
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
import joblib
from pathlib import Path
import sys

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))
from setup import CLEANED_CSV, PROCESSED_DATA_DIR

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

LABEL_MAPPING = {
    "Risqué": 0,
    "Moyen": 1,
    "Bon investissement": 2
}

REVERSE_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}


def load_cleaned_data():
    with open(CLEANED_CSV, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        sep = ';' if ';' in first_line else ','
    df = pd.read_csv(CLEANED_CSV, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    print(f"Chargé: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df


def prepare_features(df):
    if 'price_log' not in df.columns:
        if 'Price' in df.columns:
            df['price_log'] = np.log1p(df['Price'])
        else:
            raise KeyError("Aucune colonne de prix trouvée")
    
    available_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    
    print(f"\nFeatures numériques: {len(available_num)}")
    print(f"Features catégorielles: {len(available_cat)}")
    
    X = df[available_num + available_cat].copy()
    y = df['price_log'].copy()
    
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    print(f"Dataset final: {X.shape[0]} lignes × {X.shape[1]} features")
    return X, y, available_num, available_cat


def load_price_model():
    model_path = PROCESSED_DATA_DIR / "best_price_model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"\nModèle de régression non trouvé: {model_path}\n"
            f"Veuillez d'abord exécuter: python -m project.04_modeling"
        )
    
    price_model = joblib.load(model_path)
    print(f"\nModèle de régression chargé: {model_path}")
    return price_model


def predict_future_prices(df, price_model):
    print("\n" + "="*50)
    print("PRÉDICTION DES PRIX FUTURS")
    print("="*50)
    
    X, _, num_features, cat_features = prepare_features(df)
    
    print("Prédiction en cours...")
    price_log_pred = price_model.predict(X)
    price_future = np.expm1(price_log_pred)
    
    print(f"Prix futurs: min={price_future.min():.0f}€, max={price_future.max():.0f}€, mean={price_future.mean():.0f}€")
    
    return price_future, X


def calculate_roi_and_labels(df, price_model):
    print("\n" + "="*50)
    print("CALCUL ROI ET LABELS")
    print("="*50)
    
    price_future, X = predict_future_prices(df, price_model)
    price_current = df['Price'].values
    
    roi = ((price_future - price_current) / price_current) * 100
    
    def label_roi(r):
        if r > 25:
            return "Bon investissement"
        elif r > 10:
            return "Moyen"
        else:
            return "Risqué"
    
    y_class = pd.Series(roi).apply(label_roi)
    
    print(f"\nDistribution:")
    print(y_class.value_counts())
    print(f"\nROI: min={roi.min():.1f}%, max={roi.max():.1f}%, mean={roi.mean():.1f}%")
    
    return roi, y_class, X


def encode_labels(y_class):
    """Encode les labels texte en nombres pour XGBoost"""
    return np.array([LABEL_MAPPING[label] for label in y_class])


def decode_labels(y_encoded):
    """Décode les nombres en labels texte"""
    return np.array([REVERSE_MAPPING[label] for label in y_encoded])


def get_classifier_config(model_name):
    configs = {
        'random_forest': {
            'name': 'RandomForest',
            'model': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'params': {'n_estimators': 200, 'max_depth': 15, 'class_weight': 'balanced'},
            'needs_imputer': False,
            'needs_label_encoder': False
        },
        'gradient_boosting': {
            'name': 'GradientBoosting',
            'model': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                random_state=RANDOM_STATE
            ),
            'params': {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.1},
            'needs_imputer': True,
            'needs_label_encoder': False
        },
        'logistic_regression': {
            'name': 'LogisticRegression',
            'model': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'params': {'max_iter': 1000, 'class_weight': 'balanced'},
            'needs_imputer': True,
            'needs_label_encoder': False
        }
    }
    
    if XGBOOST_AVAILABLE:
        configs['xgboost'] = {
            'name': 'XGBoost',
            'model': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'params': {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1},
            'needs_imputer': False,
            'needs_label_encoder': True  # XGBoost a besoin d'labels encodés
        }
    
    return configs.get(model_name.lower())


def create_classifier_pipeline(config, numeric_features, categorical_features):
    if config.get('needs_imputer', False):
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
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
    
    return Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", config['model'])
    ])


def evaluate_classifier(pipeline, X_test, y_test, roi_test, config):
    y_pred = pipeline.predict(X_test)
    
    # Décoder si nécessaire (pour XGBoost)
    if config.get('needs_label_encoder', False):
        y_pred = decode_labels(y_pred)
        y_test_display = decode_labels(y_test) if isinstance(y_test, np.ndarray) else y_test
    else:
        y_test_display = y_test
    
    print("\n" + "="*50)
    print("RAPPORT DE CLASSIFICATION")
    print("="*50)
    print(classification_report(y_test_display, y_pred))
    
    acc = accuracy_score(y_test_display, y_pred)
    f1 = f1_score(y_test_display, y_pred, average='weighted')
    
    print(f"\nAccuracy: {acc:.3f} | F1: {f1:.3f}")
    
    print(f"\nROI par classe prédite:")
    results_df = pd.DataFrame({
        'true': y_test_display,
        'pred': y_pred,
        'roi': roi_test
    })
    print(results_df.groupby('pred')['roi'].agg(['mean', 'count']))
    
    return {'accuracy': acc, 'f1_weighted': f1}


def train_single_classifier(model_name, X_train, y_train, X_test, y_test, roi_test, num_features, cat_features):
    config = get_classifier_config(model_name)
    if config is None or config['model'] is None:
        print(f"{model_name} non disponible")
        return None, None
    
    print(f"\n{'='*70}")
    print(f"CLASSIFIEUR: {config['name']}")
    print(f"{'='*70}")
    
    # Encoder les labels si nécessaire (XGBoost)
    if config.get('needs_label_encoder', False):
        print("Encodage des labels pour XGBoost...")
        y_train_enc = encode_labels(y_train)
        y_test_enc = encode_labels(y_test)
    else:
        y_train_enc = y_train
        y_test_enc = y_test
    
    pipeline = create_classifier_pipeline(config, num_features, cat_features)
    
    mlflow.set_experiment("Luxury_Watch_Investment_Classification")
    
    with mlflow.start_run(run_name=f"{config['name']}_Classifier"):
        
        print("Entraînement...")
        pipeline.fit(X_train, y_train_enc)
        print("Terminé")
        
        metrics = evaluate_classifier(pipeline, X_test, y_test_enc, roi_test, config)
        
        for k, v in config['params'].items():
            mlflow.log_param(k, v)
        mlflow.log_param('model_type', config['name'])
        mlflow.log_metric('accuracy', metrics['accuracy'])
        mlflow.log_metric('f1_weighted', metrics['f1_weighted'])
        mlflow.sklearn.log_model(pipeline, f"classifier_{config['name'].lower()}")
        
        print(f"\n{config['name']} loggé")
    
    return pipeline, metrics


def compare_classifiers(model_list=None):
    if model_list is None:
        model_list = ['random_forest', 'gradient_boosting', 'logistic_regression']
        if XGBOOST_AVAILABLE:
            model_list.append('xgboost')
    
    print("="*70)
    print("COMPARAISON CLASSIFIEURS")
    print("="*70)
    
    df = load_cleaned_data()
    price_model = load_price_model()
    roi, y_class, X = calculate_roi_and_labels(df, price_model)
    
    num_features = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]
    cat_features = [c for c in X.columns if X[c].dtype == 'object']
    
    X_train, X_test, y_train, y_test, roi_train, roi_test = train_test_split(
        X, y_class, roi, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_class
    )
    
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    
    results = {}
    for model_name in model_list:
        try:
            pipeline, metrics = train_single_classifier(
                model_name, X_train, y_train, X_test, y_test, roi_test,
                num_features, cat_features
            )
            if pipeline:
                results[model_name] = {'pipeline': pipeline, 'metrics': metrics}
        except Exception as e:
            print(f"Erreur {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if results:
        print(f"\n{'='*70}")
        print("TABLEAU COMPARATIF")
        print(f"{'Modèle':<25} {'Accuracy':>10} {'F1-Score':>10}")
        print("-" * 50)
        
        for name, data in results.items():
            m = data['metrics']
            print(f"{name:<25} {m['accuracy']:>10.3f} {m['f1_weighted']:>10.3f}")
        
        best = max(results.items(), key=lambda x: x[1]['metrics']['f1_weighted'])
        print(f"\nMeilleur: {best[0]} (F1 = {best[1]['metrics']['f1_weighted']:.3f})")
        
        joblib.dump(best[1]['pipeline'], PROCESSED_DATA_DIR / "best_classifier.pkl")
        print(f"Sauvegarde: best_classifier.pkl")
    
    return results


def run_classification(model_name='random_forest'):
    return compare_classifiers([model_name])


if __name__ == "__main__":
    compare_classifiers(['random_forest', 'gradient_boosting', 'logistic_regression', 'xgboost'])