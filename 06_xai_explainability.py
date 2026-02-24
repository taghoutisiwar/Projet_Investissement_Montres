"""
Explicabilité XAI - Fichier 06 (CORRIGÉ)
SHAP & LIME pour l'explication des prédictions
Entrée: best_price_model.pkl, best_classifier.pkl, watches_cleaned.csv
Sortie: Explications SHAP/LIME sauvegardées + visualisations

CORRECTIONS APPORTÉES:
1. Mapping correct des noms de features après preprocessing (OneHotEncoder)
2. Décodage des valeurs catégorielles pour LIME (labels lisibles)
3. Waterfall plots avec vrais noms de features
4. Validation des features à fort impact
5. Export JSON des mappings pour réutilisation API
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Suppression des warnings
warnings.filterwarnings('ignore')

# Import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP non installé. Installez avec: pip install shap")
    SHAP_AVAILABLE = False

# Import LIME
try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    print("⚠️  LIME non installé. Installez avec: pip install lime")
    LIME_AVAILABLE = False

# ============================================================================
# CONFIGURATION DES CHEMINS (CORRIGÉE)
# ============================================================================

# Détection automatique du chemin selon l'exécution
if __name__ == "__main__":
    # Exécution directe: python -m project.06_xai_explainability
    # __file__ est dans project/06_xai_explainability.py
    # On remonte de 2 niveaux pour avoir la racine du projet
    ROOT = Path(__file__).resolve().parent.parent
else:
    # Import comme module
    ROOT = Path(__file__).resolve().parent.parent

# Chemins des données et sorties
PROCESSED_DATA_DIR = ROOT / "project" / "data" / "processed"
FIG_DIR = PROCESSED_DATA_DIR / "xai_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)  # Crée aussi les parents si besoin

# Fichiers modèles
PRICE_MODEL_PATH = PROCESSED_DATA_DIR / "best_price_model.pkl"
CLASSIFIER_MODEL_PATH = PROCESSED_DATA_DIR / "best_classifier.pkl"
DATA_PATH = PROCESSED_DATA_DIR / "watches_cleaned.csv"
MAPPING_PATH = PROCESSED_DATA_DIR / "feature_mappings.json"

print(f"📁 ROOT: {ROOT}")
print(f"📁 PROCESSED_DATA_DIR: {PROCESSED_DATA_DIR}")
print(f"📁 FIG_DIR: {FIG_DIR}")
print(f"📄 PRICE_MODEL_PATH: {PRICE_MODEL_PATH} (exists: {PRICE_MODEL_PATH.exists()})")
print(f"📄 CLASSIFIER_MODEL_PATH: {CLASSIFIER_MODEL_PATH} (exists: {CLASSIFIER_MODEL_PATH.exists()})")
print(f"📄 DATA_PATH: {DATA_PATH} (exists: {DATA_PATH.exists()})")

# Features utilisées (doivent correspondre au modèle)
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
# CHARGEMENT DES DONNÉES ET MODÈLES
# ============================================================================

def load_cleaned_data():
    """Charge le dataset nettoyé"""
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        sep = ';' if ';' in first_line else ','
    df = pd.read_csv(DATA_PATH, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    print(f"✅ Données chargées: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df

def load_models():
    """Charge les modèles entraînés"""
    models = {}

    if PRICE_MODEL_PATH.exists():
        models['price'] = joblib.load(PRICE_MODEL_PATH)
        print(f"✅ Modèle de prix chargé: {PRICE_MODEL_PATH}")
    else:
        print(f"❌ Modèle de prix non trouvé: {PRICE_MODEL_PATH}")

    if CLASSIFIER_MODEL_PATH.exists():
        models['classifier'] = joblib.load(CLASSIFIER_MODEL_PATH)
        print(f"✅ Classifieur chargé: {CLASSIFIER_MODEL_PATH}")
    else:
        print(f"❌ Classifieur non trouvé: {CLASSIFIER_MODEL_PATH}")

    return models

def prepare_features_for_explainer(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Prépare les features pour les explainers
    Retourne X (features), numeric_features, categorical_features disponibles
    """
    available_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    # Sélectionner uniquement les features disponibles
    feature_cols = available_num + available_cat
    X = df[feature_cols].copy()

    print(f"📊 Features utilisées:")
    print(f"   Numériques: {len(available_num)}")
    print(f"   Catégorielles: {len(available_cat)}")

    return X, available_num, available_cat

# ============================================================================
# SHAP EXPLAINER (CORRIGÉ)
# ============================================================================

class ShapExplainer:
    """Classe pour gérer les explications SHAP avec mapping correct des features"""

    def __init__(self, model, X_background: pd.DataFrame, feature_names: List[str]):
        """
        Initialise l'explainer SHAP

        Args:
            model: Pipeline sklearn entraîné
            X_background: Données de background pour SHAP (échantillon représentatif)
            feature_names: Noms des features originaux (avant preprocessing)
        """
        self.model = model
        self.feature_names = feature_names
        self.X_background = X_background
        self.explainer = None
        self.expected_value = None
        self.processed_feature_names = None  # Noms après preprocessing
        self.preprocessor = None
        self.final_model = None

    def _extract_pipeline_components(self):
        """Extrait le preprocessor et le modèle final du pipeline"""
        if hasattr(self.model, 'named_steps'):
            self.preprocessor = self.model.named_steps.get('preprocessing')
            self.final_model = (self.model.named_steps.get('model') or 
                               self.model.named_steps.get('regressor') or 
                               self.model.named_steps.get('classifier'))
        else:
            self.preprocessor = None
            self.final_model = self.model
        
        model_type = type(self.final_model).__name__
        print(f"   Type de modèle détecté: {model_type}")
        return model_type

    def _get_processed_feature_names(self) -> List[str]:
        """
        Récupère les noms de features APRÈS preprocessing (OneHotEncoder inclus)
        """
        if self.preprocessor is None:
            return self.feature_names
        
        # Récupérer les transformateurs
        transformers = self.preprocessor.named_transformers_
        
        processed_names = []
        
        # Traiter chaque transformateur dans l'ordre du ColumnTransformer
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'num':
                # Features numériques : noms inchangés
                processed_names.extend(columns)
            elif name == 'cat':
                # Features catégorielles : récupérer les noms après OneHotEncoder
                if hasattr(transformer, 'get_feature_names_out'):
                    cat_names = transformer.get_feature_names_out(columns)
                    processed_names.extend(cat_names)
                else:
                    # Fallback si pas de get_feature_names_out
                    processed_names.extend([f"{col}_encoded" for col in columns])
            elif name == 'remainder':
                # Features non transformées (si 'passthrough')
                if transformer == 'passthrough':
                    processed_names.extend(columns)
        
        print(f"   Features après preprocessing: {len(processed_names)}")
        print(f"   Exemples: {processed_names[:5]}...")
        
        return processed_names

    def create_explainer(self, sample_size: int = 100):
        """Crée l'explainer SHAP adapté au type de modèle"""
        print("\n" + "="*60)
        print("CRÉATION DU EXPLAINER SHAP")
        print("="*60)

        # Extraire les composants du pipeline
        model_type = self._extract_pipeline_components()
        
        # Récupérer les noms de features après preprocessing
        self.processed_feature_names = self._get_processed_feature_names()

        # Échantillon de background
        if len(self.X_background) > sample_size:
            X_sample = self.X_background.sample(n=sample_size, random_state=42)
        else:
            X_sample = self.X_background

        # Appliquer le preprocessing si présent
        if self.preprocessor:
            X_processed = self.preprocessor.transform(X_sample)
            if hasattr(X_processed, 'toarray'):
                X_processed = X_processed.toarray()
            if hasattr(X_processed, 'values'):
                X_processed = X_processed.values
        else:
            X_processed = X_sample.values

        # Créer l'explainer selon le type de modèle
        if any(x in model_type for x in ['Tree', 'Forest', 'Boosting', 'XGB']):
            # TreeSHAP pour les modèles basés sur des arbres
            self.explainer = shap.TreeExplainer(self.final_model)
            print("✅ TreeExplainer créé (optimisé pour les arbres)")
        else:
            # KernelSHAP pour les autres modèles
            def predict_fn(X_transformed):
                return self.final_model.predict(X_transformed)
            
            X_shap_background = shap.sample(X_processed, min(50, len(X_processed)))
            self.explainer = shap.KernelExplainer(predict_fn, X_shap_background)
            print("✅ KernelExplainer créé")

        self.expected_value = self.explainer.expected_value
        if isinstance(self.expected_value, np.ndarray):
            print(f"   Valeur attendue: {self.expected_value[0]:.4f}")
        else:
            print(f"   Valeur attendue: {self.expected_value:.4f}")

        return self.explainer

    def _transform_data(self, X: pd.DataFrame) -> np.ndarray:
        """Transforme les données avec le preprocessor"""
        if self.preprocessor:
            X_proc = self.preprocessor.transform(X)
            if hasattr(X_proc, 'toarray'):
                X_proc = X_proc.toarray()
            if hasattr(X_proc, 'values'):
                X_proc = X_proc.values
        else:
            X_proc = X.values if hasattr(X, 'values') else X
        return X_proc

    def explain_instance(self, X_instance: pd.DataFrame) -> Dict:
        """
        Explique une prédiction individuelle avec les vrais noms de features
        """
        if self.explainer is None:
            raise ValueError("Explainer non créé. Appelez create_explainer() d'abord.")

        # Transformer les données
        X_proc = self._transform_data(X_instance)

        # Calcul SHAP values
        shap_values = self.explainer.shap_values(X_proc)

        # Pour la classification, shap_values est une liste (une par classe)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # S'assurer que c'est un array 1D pour une seule instance
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]

        # Créer le mapping feature -> impact
        feature_importance = []
        for i, (name, value) in enumerate(zip(self.processed_feature_names, shap_values)):
            feature_importance.append({
                'index': i,
                'name': name,
                'shap_value': float(value),
                'abs_impact': abs(float(value))
            })

        # Trier par impact absolu
        feature_importance.sort(key=lambda x: x['abs_impact'], reverse=True)

        # Valeur attendue
        base_value = (self.expected_value[0] if isinstance(self.expected_value, np.ndarray) 
                     else self.expected_value)

        return {
            'shap_values': shap_values,
            'base_value': base_value,
            'prediction': float(self.model.predict(X_instance)[0]),
            'feature_importance': feature_importance,
            'top_positive': [f for f in feature_importance if f['shap_value'] > 0][:5],
            'top_negative': [f for f in feature_importance if f['shap_value'] < 0][:5]
        }

    def explain_batch(self, X_batch: pd.DataFrame) -> np.ndarray:
        """Calcule les SHAP values pour un batch"""
        if self.explainer is None:
            raise ValueError("Explainer non créé.")

        X_proc = self._transform_data(X_batch)
        return self.explainer.shap_values(X_proc)

    def plot_summary(self, X: pd.DataFrame, max_display: int = 15, save: bool = True):
        """Plot l'importance globale des features avec vrais noms"""
        print("\n📊 Génération du summary plot SHAP...")

        X_proc = self._transform_data(X)
        shap_values = self.explainer.shap_values(X_proc)

        # Pour classification
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[0]
        else:
            shap_values_plot = shap_values

        # Créer le summary plot avec les vrais noms
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_plot,
            X_proc,
            feature_names=self.processed_feature_names,  # ← VRAIS NOMS
            max_display=max_display,
            show=False
        )
        plt.title("Importance Globale des Features (SHAP)", fontsize=14, pad=20)
        plt.tight_layout()

        if save:
            out_path = FIG_DIR / "shap_summary_global.png"
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"✅ Summary plot sauvegardé: {out_path}")
        plt.close()

    def plot_waterfall(self, X_instance: pd.DataFrame, instance_idx: int = 0, 
                      save: bool = True, max_display: int = 10):
        """Plot waterfall pour une instance spécifique avec vrais noms"""
        print(f"\n📊 Génération du waterfall plot (instance {instance_idx})...")

        # Transformer les données
        X_proc = self._transform_data(X_instance)
        
        # Recréer un TreeExplainer sur le modèle brut pour avoir les vraies valeurs
        if hasattr(self, 'final_model'):
            explainer_raw = shap.TreeExplainer(self.final_model)
            shap_values_raw = explainer_raw.shap_values(X_proc)
            
            if isinstance(shap_values_raw, np.ndarray) and len(shap_values_raw.shape) == 2:
                shap_values_plot = shap_values_raw[0]
            else:
                shap_values_plot = shap_values_raw
                
            base_value = (explainer_raw.expected_value[0] 
                         if isinstance(explainer_raw.expected_value, np.ndarray) 
                         else explainer_raw.expected_value)
        else:
            explanation = self.explain_instance(X_instance)
            shap_values_plot = explanation['shap_values']
            base_value = explanation['base_value']

        # Limiter aux top features pour la lisibilité
        if len(shap_values_plot) > max_display:
            # Garder les plus importantes
            importance_idx = np.argsort(np.abs(shap_values_plot))[::-1][:max_display]
            mask = np.zeros(len(shap_values_plot), dtype=bool)
            mask[importance_idx] = True
            
            # Somme des autres features
            other_value = np.sum(shap_values_plot[~mask])
            
            shap_values_display = np.concatenate([shap_values_plot[mask], [other_value]])
            feature_names_display = ([self.processed_feature_names[i] for i in importance_idx] + 
                                    [f"{len(shap_values_plot) - max_display} other features"])
            data_display = np.concatenate([X_proc[0][mask], [0]])
        else:
            shap_values_display = shap_values_plot
            feature_names_display = self.processed_feature_names
            data_display = X_proc[0]

        # Créer le waterfall plot
        plt.figure(figsize=(14, 10))
        
        explanation = shap.Explanation(
            values=shap_values_display,
            base_values=base_value,
            data=data_display,
            feature_names=feature_names_display
        )
        
        shap.plots.waterfall(explanation, max_display=len(shap_values_display), show=False)
        
        plt.title(f"Explication SHAP - Instance {instance_idx}\n" + 
                 f"Prédiction: {base_value + np.sum(shap_values_display):.3f}", 
                 fontsize=14)
        plt.tight_layout()

        if save:
            out_path = FIG_DIR / f"shap_waterfall_instance_{instance_idx}.png"
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"✅ Waterfall plot sauvegardé: {out_path}")
        plt.close()

    def validate_extreme_impacts(self, X: pd.DataFrame, threshold: float = 1.0):
        """Valide les features avec impact extrême (potentielles anomalies)"""
        print(f"\n🔍 Validation des features à fort impact (seuil: {threshold})...")
        
        explanations = []
        for idx in range(min(100, len(X))):  # Échantillon de 100
            X_inst = X.iloc[idx:idx+1]
            exp = self.explain_instance(X_inst)
            explanations.extend(exp['feature_importance'])
        
        # Trouver les features avec impact extrême
        extreme_features = [f for f in explanations if f['abs_impact'] > threshold]
        
        if extreme_features:
            print(f"⚠️  {len(extreme_features)} features avec impact > {threshold} détectées:")
            for f in extreme_features[:10]:  # Limiter l'affichage
                print(f"   - {f['name']}: {f['shap_value']:+.3f}")
        else:
            print(f"✅ Aucune feature avec impact extrême détectée")

    def get_feature_importance(self, X: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Retourne l'importance moyenne des features avec vrais noms"""
        shap_values = self.explain_batch(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Importance moyenne absolue
        importance = np.abs(shap_values).mean(axis=0)
        
        df_importance = pd.DataFrame({
            'feature': self.processed_feature_names,
            'shap_importance': importance
        }).sort_values('shap_importance', ascending=False).head(top_n)

        return df_importance

    def save_explainer(self, path: Path = None):
        """Sauvegarde l'explainer et les mappings"""
        if path is None:
            path = PROCESSED_DATA_DIR / "shap_explainer.pkl"

        joblib.dump({
            'explainer': self.explainer,
            'model': self.model,
            'feature_names': self.feature_names,
            'processed_feature_names': self.processed_feature_names,
            'expected_value': self.expected_value,
            'X_background': self.X_background,
            'preprocessor': self.preprocessor,
            'final_model': self.final_model
        }, path)
        
        # Sauvegarder aussi les mappings en JSON pour l'API
        mapping_data = {
            'original_features': self.feature_names,
            'processed_features': self.processed_feature_names,
            'numeric_features': [f for f in self.feature_names if f in NUMERIC_FEATURES],
            'categorical_features': [f for f in self.feature_names if f in CATEGORICAL_FEATURES]
        }
        with open(MAPPING_PATH, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Explainer SHAP sauvegardé: {path}")
        print(f"✅ Mappings sauvegardés: {MAPPING_PATH}")

    @staticmethod
    def load_explainer(path: Path = None):
        """Charge l'explainer sauvegardé"""
        if path is None:
            path = PROCESSED_DATA_DIR / "shap_explainer.pkl"
        
        data = joblib.load(path)
        shap_exp = ShapExplainer(data['model'], data['X_background'], data['feature_names'])
        shap_exp.explainer = data['explainer']
        shap_exp.expected_value = data['expected_value']
        shap_exp.processed_feature_names = data.get('processed_feature_names')
        shap_exp.preprocessor = data.get('preprocessor')
        shap_exp.final_model = data.get('final_model')
        
        print(f"✅ Explainer SHAP chargé: {path}")
        return shap_exp

# ============================================================================
# LIME EXPLAINER (CORRIGÉ)
# ============================================================================

class LimeExplainer:
    """Classe pour gérer les explications LIME avec labels lisibles"""

    def __init__(self, model, X_train: pd.DataFrame, feature_names: List[str],
                 categorical_features: List[str] = None, class_names: List[str] = None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.class_names = class_names or ['Risqué', 'Moyen', 'Bon investissement']
        self.explainer = None
        self.categorical_indices = None
        self.label_encoders = {}
        self.value_mappings = {}  # Nouveau: stocker les mappings valeur originale -> encodée

    def create_explainer(self):
        """Crée l'explainer LIME avec encodage des catégorielles"""
        print("\n" + "="*60)
        print("CRÉATION DU EXPLAINER LIME")
        print("="*60)

        # Encoder les données catégorielles
        X_train_encoded = self.X_train.copy()
        
        for col in self.categorical_features:
            if col in X_train_encoded.columns:
                le = LabelEncoder()
                # Convertir en string pour éviter les problèmes de type
                X_train_encoded[col] = X_train_encoded[col].astype(str)
                X_train_encoded[col] = le.fit_transform(X_train_encoded[col])
                self.label_encoders[col] = le
                
                # Créer le mapping inverse (encodé -> original)
                self.value_mappings[col] = {
                    i: label for i, label in enumerate(le.classes_)
                }
                print(f"   {col}: {len(le.classes_)} valeurs uniques")
        
        # Recalculer les indices catégoriels
        self.categorical_indices = [
            i for i, col in enumerate(self.feature_names)
            if col in self.categorical_features
        ]
        
        print(f"   Indices catégoriels: {self.categorical_indices}")

        # Convertir en numpy array
        X_train_numeric = X_train_encoded.values
        
        # === SOLUTION : Désactiver la discretisation ===
        self.explainer = LimeTabularExplainer(
            training_data=X_train_numeric,
            feature_names=self.feature_names,
            categorical_features=self.categorical_indices,
            class_names=self.class_names,
            mode='classification',
            discretize_continuous=False,
            sample_around_instance=True,
            random_state=42
        )

        print(f"✅ LimeTabularExplainer créé ({len(self.feature_names)} features)")
        return self.explainer

    def _encode_instance(self, X_instance: pd.DataFrame) -> np.ndarray:
        """Encode une instance pour LIME"""
        X_encoded = X_instance.copy()
        
        for col, le in self.label_encoders.items():
            if col in X_encoded.columns:
                val = str(X_encoded[col].iloc[0])
                if val in le.classes_:
                    X_encoded[col] = le.transform([val])[0]
                else:
                    # Valeur inconnue, utiliser la plus fréquente
                    X_encoded[col] = 0
                    print(f"   ⚠️  Valeur inconnue pour {col}: {val}")
        
        return X_encoded.values[0]

    def _decode_feature_label(self, feature_str: str) -> str:
        """
        Décode un label de feature LIME en label lisible
        Ex: "Brand=191" -> "Brand=Rolex"
        """
        if '=' not in feature_str:
            return feature_str
        
        parts = feature_str.split('=', 1)
        feat_name = parts[0]
        feat_val = parts[1]
        
        if feat_name in self.value_mappings:
            try:
                val_int = int(float(feat_val))
                real_val = self.value_mappings[feat_name].get(val_int, feat_val)
                return f"{feat_name}={real_val}"
            except (ValueError, TypeError):
                return feature_str
        
        return feature_str

    def explain_instance(self, X_instance: pd.DataFrame, num_features: int = 10) -> Dict:
        """Explique une instance avec LIME et labels lisibles"""
        if self.explainer is None:
            raise ValueError("Explainer non créé. Appelez create_explainer() d'abord.")

        # Encoder l'instance
        instance_array = self._encode_instance(X_instance)

        # Fonction de prédiction avec décodage
        def predict_fn(x):
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            
            X_df = pd.DataFrame(x, columns=self.feature_names)
            
            # Décoder pour le modèle sklearn
            for col, le in self.label_encoders.items():
                if col in X_df.columns:
                    vals = X_df[col].astype(int)
                    decoded = []
                    for v in vals:
                        if v in self.value_mappings[col]:
                            decoded.append(self.value_mappings[col][v])
                        else:
                            decoded.append(le.classes_[0])
                    X_df[col] = decoded
            
            pred = self.model.predict_proba(X_df)
            return pred

        # Générer l'explication
        explanation = self.explainer.explain_instance(
            data_row=instance_array,
            predict_fn=predict_fn,
            num_features=num_features,
            top_labels=1
        )

        top_label = explanation.available_labels()[0]
        
        # Décoder les explications pour les rendre lisibles
        raw_exp = explanation.as_list(label=top_label)
        readable_exp = [(self._decode_feature_label(f), w) for f, w in raw_exp]
        
        # Créer aussi un dictionnaire structuré
        structured_exp = []
        for feature_str, weight in readable_exp:
            if '=' in feature_str:
                feat_name, feat_val = feature_str.split('=', 1)
                structured_exp.append({
                    'feature': feat_name,
                    'value': feat_val,
                    'impact': float(weight),
                    'direction': 'positive' if weight > 0 else 'negative'
                })
            else:
                structured_exp.append({
                    'feature': feature_str,
                    'value': None,
                    'impact': float(weight),
                    'direction': 'positive' if weight > 0 else 'negative'
                })

        return {
            'explanation': explanation,
            'top_label': top_label,
            'top_label_name': self.class_names[top_label] if top_label < len(self.class_names) else 'Unknown',
            'score': explanation.score,
            'local_exp': raw_exp,
            'local_exp_readable': readable_exp,
            'structured_exp': structured_exp
        }

    def plot_explanation(self, X_instance: pd.DataFrame, instance_idx: int = 0, 
                    num_features: int = 10, save: bool = True):
        """Visualise l'explication LIME avec labels lisibles"""
        print(f"\n📊 Génération de l'explication LIME (instance {instance_idx})...")

        explanation_result = self.explain_instance(X_instance, num_features)
        explanation = explanation_result['explanation']
        top_label = explanation_result['top_label']

        # Modifier les labels dans l'objet explanation pour l'affichage
        # (Créer une copie modifiée pour le plot)
        fig = explanation.as_pyplot_figure(label=top_label)
        
        # Modifier les labels de l'axe Y pour les rendre lisibles
        ax = plt.gca()
        y_labels = [self._decode_feature_label(label.get_text()) 
                   for label in ax.get_yticklabels()]
        ax.set_yticklabels(y_labels)
        
        plt.title(f"Explication LIME - Instance {instance_idx}\n" +
                 f"Classe: {explanation_result['top_label_name']} " +
                 f"(confiance: {explanation_result['score']:.1%})", 
                 fontsize=12)
        plt.tight_layout()

        if save:
            out_path = FIG_DIR / f"lime_explanation_instance_{instance_idx}.png"
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"✅ Explication LIME sauvegardée: {out_path}")
        plt.close()

        return explanation_result

    def get_explanation_text(self, X_instance: pd.DataFrame, num_features: int = 5) -> str:
        """Retourne une explication textuelle détaillée"""
        explanation = self.explain_instance(X_instance, num_features)
        
        text_parts = []
        text_parts.append("="*50)
        text_parts.append(f"PRÉDICTION: {explanation['top_label_name']}")
        text_parts.append(f"CONFIANCE: {explanation['score']:.1%}")
        text_parts.append("="*50)
        text_parts.append("\n🔍 RAISONS PRINCIPALES:\n")
        
        for i, exp in enumerate(explanation['structured_exp'][:num_features], 1):
            direction = "📈 Augmente" if exp['direction'] == 'positive' else "📉 Diminue"
            if exp['value']:
                text_parts.append(f"{i}. {direction} la probabilité:")
                text_parts.append(f"   Feature: {exp['feature']}")
                text_parts.append(f"   Valeur: {exp['value']}")
                text_parts.append(f"   Impact: {abs(exp['impact']):.3f}")
            else:
                text_parts.append(f"{i}. {direction} la probabilité:")
                text_parts.append(f"   Feature: {exp['feature']}")
                text_parts.append(f"   Impact: {abs(exp['impact']):.3f}")
            text_parts.append("")

        return "\n".join(text_parts)

    def save_explainer(self, path: Path = None):
        """Sauvegarde la configuration LIME"""
        if path is None:
            path = PROCESSED_DATA_DIR / "lime_explainer.pkl"

        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'class_names': self.class_names,
            'X_train': self.X_train,
            'label_encoders': self.label_encoders,
            'value_mappings': self.value_mappings,
            'categorical_indices': self.categorical_indices
        }, path)
        print(f"✅ Configuration LIME sauvegardée: {path}")

    @staticmethod
    def load_explainer(path: Path = None, model=None):
        """Charge et recrée l'explainer LIME"""
        if path is None:
            path = PROCESSED_DATA_DIR / "lime_explainer.pkl"
        
        data = joblib.load(path)
        
        lime_exp = LimeExplainer(
            model=model or data['model'],
            X_train=data['X_train'],
            feature_names=data['feature_names'],
            categorical_features=data['categorical_features'],
            class_names=data['class_names']
        )
        
        lime_exp.label_encoders = data.get('label_encoders', {})
        lime_exp.value_mappings = data.get('value_mappings', {})
        lime_exp.categorical_indices = data.get('categorical_indices', [])
        
        lime_exp.create_explainer()
        
        print(f"✅ Explainer LIME recréé et prêt")
        return lime_exp

# ============================================================================
# FONCTIONS UTILITAIRES POUR L'API (CORRIGÉES)
# ============================================================================

def explain_watch(watch_features: Dict, models: Dict, X_background: pd.DataFrame,
                  feature_names: List[str], method: str = 'both',
                  shap_explainer: ShapExplainer = None,
                  lime_explainer: LimeExplainer = None) -> Dict:
    """
    Fonction principale pour expliquer une montre (version corrigée)
    
    Args:
        watch_features: Dict avec les caractéristiques de la montre
        models: Dict avec les modèles chargés
        X_background: DataFrame de background
        feature_names: Noms des features
        method: 'shap', 'lime', ou 'both'
        shap_explainer: Instance réutilisable de ShapExplainer (optionnel)
        lime_explainer: Instance réutilisable de LimeExplainer (optionnel)
    """
    # Convertir en DataFrame
    X_instance = pd.DataFrame([watch_features])

    # S'assurer que toutes les colonnes sont présentes
    for col in feature_names:
        if col not in X_instance.columns:
            X_instance[col] = 0

    X_instance = X_instance[feature_names]

    results = {
        'watch_features': watch_features,
        'predictions': {}
    }

    # Prédiction prix
    if 'price' in models:
        price_log_pred = models['price'].predict(X_instance)[0]
        price_pred = np.expm1(price_log_pred)
        results['predictions']['price_future'] = float(price_pred)
        results['predictions']['price_log'] = float(price_log_pred)

    # Prédiction classe
    if 'classifier' in models:
        class_pred = models['classifier'].predict(X_instance)[0]
        class_proba = models['classifier'].predict_proba(X_instance)[0]
        results['predictions']['investment_class'] = str(class_pred)
        results['predictions']['class_probabilities'] = {
            'Risqué': float(class_proba[0]),
            'Moyen': float(class_proba[1]),
            'Bon investissement': float(class_proba[2])
        }

    # Explications SHAP
    if method in ['shap', 'both'] and SHAP_AVAILABLE and 'price' in models:
        print("\n🔍 Génération explication SHAP...")
        
        # Réutiliser l'explainer existant ou en créer un nouveau
        if shap_explainer is None:
            shap_exp = ShapExplainer(models['price'], X_background, feature_names)
            shap_exp.create_explainer(sample_size=100)
        else:
            shap_exp = shap_explainer
        
        shap_result = shap_exp.explain_instance(X_instance)

        results['shap_explanation'] = {
            'base_value': shap_result['base_value'],
            'prediction': shap_result['prediction'],
            'top_positive_features': shap_result['top_positive'],
            'top_negative_features': shap_result['top_negative'],
            'all_features': shap_result['feature_importance'][:10]
        }

    # Explications LIME
    if method in ['lime', 'both'] and LIME_AVAILABLE and 'classifier' in models:
        print("\n🔍 Génération explication LIME...")
        
        # Réutiliser l'explainer existant ou en créer un nouveau
        if lime_explainer is None:
            available_cat = [c for c in CATEGORICAL_FEATURES if c in feature_names]
            lime_exp = LimeExplainer(
                models['classifier'], 
                X_background, 
                feature_names,
                categorical_features=available_cat
            )
            lime_exp.create_explainer()
        else:
            lime_exp = lime_explainer
        
        lime_result = lime_exp.explain_instance(X_instance)

        results['lime_explanation'] = {
            'predicted_class': lime_result['top_label_name'],
            'confidence': lime_result['score'],
            'key_factors': lime_result['structured_exp'][:5]
        }

    return results

# ============================================================================
# PIPELINE PRINCIPAL (CORRIGÉ)
# ============================================================================

def run_xai_analysis():
    """Exécute l'analyse XAI complète corrigée"""
    print("="*70)
    print("ANALYSE XAI - SHAP & LIME (VERSION CORRIGÉE)")
    print("="*70)

    # 1. Chargement
    print("\n📥 Chargement des données et modèles...")
    df = load_cleaned_data()
    models = load_models()

    if not models:
        print("❌ Aucun modèle chargé. Arrêt.")
        return

    # 2. Préparation features
    X, available_num, available_cat = prepare_features_for_explainer(df)
    feature_names = X.columns.tolist()

    # 3. Échantillon de background
    X_background = X.sample(n=min(1000, len(X)), random_state=42)

    # 4. Sélection d'instances à expliquer (exemples variés)
    # Sélectionner des instances avec des prix différents pour la diversité
    sample_indices = []
    for q in [0.1, 0.5, 0.9]:  # 10%, 50%, 90% des prix
        idx = df['Price'].quantile(q)
        closest_idx = (df['Price'] - idx).abs().idxmin()
        sample_indices.append(closest_idx)
    
    sample_instances = X.loc[sample_indices].reset_index(drop=True)
    print(f"\n📊 {len(sample_instances)} instances sélectionnées (variété de prix)")

    # 5. SHAP Analysis
    shap_exp = None
    if SHAP_AVAILABLE and 'price' in models:
        print("\n" + "="*70)
        print("ANALYSE SHAP - MODÈLE DE PRIX")
        print("="*70)

        shap_exp = ShapExplainer(models['price'], X_background, feature_names)
        shap_exp.create_explainer(sample_size=200)

        # Summary plot global
        shap_exp.plot_summary(X_background, max_display=15, save=True)

        # Waterfall pour chaque instance d'exemple
        for idx, (_, row) in enumerate(sample_instances.iterrows()):
            X_inst = pd.DataFrame([row.values], columns=feature_names)
            shap_exp.plot_waterfall(X_inst, instance_idx=idx, save=True, max_display=10)

        # Validation des impacts extrêmes
        shap_exp.validate_extreme_impacts(X_background, threshold=1.0)

        # Importance des features
        importance_df = shap_exp.get_feature_importance(X_background, top_n=15)
        print("\n📊 Top 15 features (SHAP):")
        print(importance_df.to_string(index=False))

        # Sauvegarde
        shap_exp.save_explainer()

    # 6. LIME Analysis
    lime_exp = None
    if LIME_AVAILABLE and 'classifier' in models:
        print("\n" + "="*70)
        print("ANALYSE LIME - CLASSIFIEUR")
        print("="*70)

        lime_exp = LimeExplainer(
            models['classifier'],
            X_background,
            feature_names,
            categorical_features=available_cat,
            class_names=['Risqué', 'Moyen', 'Bon investissement']
        )
        lime_exp.create_explainer()

        # Explications pour chaque instance
        for idx, (_, row) in enumerate(sample_instances.iterrows()):
            X_inst = pd.DataFrame([row.values], columns=feature_names)
            explanation = lime_exp.plot_explanation(X_inst, instance_idx=idx, save=True)

            # Afficher texte explicatif
            print(f"\n📋 Explication textuelle (Instance {idx}):")
            print(lime_exp.get_explanation_text(X_inst, num_features=5))

        # Sauvegarde
        lime_exp.save_explainer()

    # 7. Test de la fonction explain_watch avec expliquers réutilisables
    print("\n" + "="*70)
    print("TEST FONCTION EXPLAIN_WATCH (AVEC EXPLAINERS RÉUTILISABLES)")
    print("="*70)

    test_watch = {
        'Brand': 'Rolex',
        'Movement': 'Automatic',
        'Case material': 'Steel',
        'Bracelet material': 'Steel',
        'Year of production': 2020,
        'Condition': 'Used (Very good)',
        'Scope of delivery': 'Original box, original papers',
        'Gender': "Men's watch/Unisex",
        'Availability': 'Item is in stock',
        'Shape': 'Circular',
        'Face Area': 650.0,
        'Crystal': 'Sapphire crystal',
        'Dial': 'Black',
        'Bracelet color': 'Steel',
        'Watches Sold by the Seller': 500,
        'Active listing of the seller': 100,
        'Fast Shipper': 1,
        'Trusted Seller': 1,
        'Punctuality': 1,
        'Seller Reviews': 400,
        'age': 6,
        'is_modern': 1,
        'seller_reputation_score': 1.0,
        'scope_score': 3,
        'price_anomaly_low': 0,
        'price_anomaly_high': 0,
        'seller_consistent': 1
    }

    result = explain_watch(
        test_watch, 
        models, 
        X_background, 
        feature_names, 
        method='both',
        shap_explainer=shap_exp,  # Réutilisation
        lime_explainer=lime_exp   # Réutilisation
    )

    print("\n📊 Résultats pour la montre test:")
    if 'price_future' in result['predictions']:
        print(f"💰 Prix futur estimé: {result['predictions']['price_future']:,.0f}€")
    print(f"🏷️  Classe: {result['predictions'].get('investment_class', 'N/A')}")

    if 'shap_explanation' in result:
        print(f"\n🔍 SHAP - Top features positives:")
        for feat in result['shap_explanation']['top_positive_features'][:3]:
            print(f"   + {feat['name']}: {feat['shap_value']:+.3f}")

    if 'lime_explanation' in result:
        print(f"\n🔍 LIME - Confiance: {result['lime_explanation']['confidence']:.1%}")
        print(f"   Classe prédite: {result['lime_explanation']['predicted_class']}")

    # Sauvegarde du résultat test
    with open(PROCESSED_DATA_DIR / "xai_example_result.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✅ Résultat exemple sauvegardé: {PROCESSED_DATA_DIR / 'xai_example_result.json'}")

    print("\n" + "="*70)
    print("✅ ANALYSE XAI TERMINÉE AVEC SUCCÈS")
    print(f"📁 Figures sauvegardées dans: {FIG_DIR}")
    print(f"📁 Mappings sauvegardés dans: {MAPPING_PATH}")
    print("="*70)

if __name__ == "__main__":  # ✅ 4 underscores: __main__
    run_xai_analysis()