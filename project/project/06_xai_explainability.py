"""
Explicabilité XAI - Fichier 06
SHAP & LIME pour l'explication des prédictions
Entrée: best_price_model.pkl, best_classifier.pkl, watches_cleaned.csv
Sortie: Explications SHAP/LIME sauvegardées + visualisations
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
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

# Configuration des chemins
ROOT = Path(__file__).resolve().parent
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
FIG_DIR = PROCESSED_DATA_DIR / "xai_figures"
FIG_DIR.mkdir(exist_ok=True)

# Fichiers modèles
PRICE_MODEL_PATH = PROCESSED_DATA_DIR / "best_price_model.pkl"
CLASSIFIER_MODEL_PATH = PROCESSED_DATA_DIR / "best_classifier.pkl"
DATA_PATH = PROCESSED_DATA_DIR / "watches_cleaned.csv"

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
# SHAP EXPLAINER
# ============================================================================

class ShapExplainer:
    """Classe pour gérer les explications SHAP"""

    def __init__(self, model, X_background: pd.DataFrame, feature_names: List[str]):
        """
        Initialise l'explainer SHAP

        Args:
            model: Pipeline sklearn entraîné
            X_background: Données de background pour SHAP (échantillon représentatif)
            feature_names: Noms des features
        """
        self.model = model
        self.feature_names = feature_names
        self.X_background = X_background
        self.explainer = None
        self.expected_value = None

    def create_explainer(self, sample_size: int = 100):
        """Crée l'explainer SHAP adapté au type de modèle"""
        print("" + "="*60)
        print("CRÉATION DU EXPLAINER SHAP")
        print("="*60)

        # Extraire le modèle final du pipeline
        if hasattr(self.model, 'named_steps'):
            preprocessor = self.model.named_steps.get('preprocessing')
            final_model = self.model.named_steps.get('model') or self.model.named_steps.get('regressor') or self.model.named_steps.get('classifier')
        else:
            preprocessor = None
            final_model = self.model

        # Échantillon de background
        if len(self.X_background) > sample_size:
            X_sample = self.X_background.sample(n=sample_size, random_state=42)
        else:
            X_sample = self.X_background

        # Appliquer le preprocessing si présent
        if preprocessor:
            X_processed = preprocessor.transform(X_sample)
            # Convertir en DataFrame si c'est un array
            if hasattr(X_processed, 'toarray'):
                X_processed = X_processed.toarray()
            # S'assurer que c'est un numpy array
            if hasattr(X_processed, 'values'):
                X_processed = X_processed.values
        else:
            X_processed = X_sample.values

        # Créer l'explainer selon le type de modèle
        model_type = type(final_model).__name__
        print(f"Type de modèle détecté: {model_type}")

        if 'Tree' in model_type or 'Forest' in model_type or 'Boosting' in model_type or 'XGB' in model_type:
            # TreeSHAP pour les modèles basés sur des arbres (incluant XGBoost)
            # IMPORTANT: Passer les données BRUTES (non transformées) à TreeExplainer
            # car il va appliquer le preprocessing lui-même via le pipeline
            self.explainer = shap.TreeExplainer(final_model)
            print("✅ TreeExplainer créé (optimisé pour les arbres)")
            # Pour TreeExplainer, on garde X_processed pour les shap_values
            self.X_processed_background = X_processed
        else:
            # KernelSHAP pour les autres modèles
            # Pour KernelExplainer, on doit passer les données DÉJÀ transformées
            # et la fonction de prédiction doit prendre les données transformées
            def predict_fn(X_transformed):
                # X_transformed est déjà pré-traité, on passe directement au modèle
                return final_model.predict(X_transformed)
            
            # Utiliser seulement un petit échantillon pour KernelExplainer (lent)
            X_shap_background = shap.sample(X_processed, min(50, len(X_processed)))
            
            self.explainer = shap.KernelExplainer(predict_fn, X_shap_background)
            print("✅ KernelExplainer créé")
            self.X_processed_background = X_processed

        self.expected_value = self.explainer.expected_value
        print(f"Valeur attendue (expected value): {self.expected_value}")

        return self.explainer

    def _predict_processed(self, X, preprocessor, model):
        """Helper pour prédire avec preprocessing"""
        if preprocessor:
            X_proc = preprocessor.transform(X)
            if hasattr(X_proc, 'toarray'):
                X_proc = X_proc.toarray()
        else:
            X_proc = X
        return model.predict(X_proc)

    def explain_instance(self, X_instance: pd.DataFrame) -> Dict:
        """
        Explique une prédiction individuelle

        Returns:
            Dict avec shap_values, base_value, feature_importance
        """
        if self.explainer is None:
            raise ValueError("Explainer non créé. Appelez create_explainer() d'abord.")

        # Preprocessing
        if hasattr(self.model, 'named_steps') and 'preprocessing' in self.model.named_steps:
            X_proc = self.model.named_steps['preprocessing'].transform(X_instance)
            if hasattr(X_proc, 'toarray'):
                X_proc = X_proc.toarray()
        else:
            X_proc = X_instance.values

        # Calcul SHAP values
        shap_values = self.explainer.shap_values(X_proc)

        # Pour la classification, shap_values est une liste (une par classe)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Prendre la première classe

        return {
            'shap_values': shap_values,
            'base_value': self.expected_value if not isinstance(self.expected_value, list) else self.expected_value[0],
            'prediction': self.model.predict(X_instance)[0],
            'feature_names': self.feature_names
        }

    def explain_batch(self, X_batch: pd.DataFrame) -> np.ndarray:
        """Calcule les SHAP values pour un batch"""
        if self.explainer is None:
            raise ValueError("Explainer non créé.")

        if hasattr(self.model, 'named_steps') and 'preprocessing' in self.model.named_steps:
            X_proc = self.model.named_steps['preprocessing'].transform(X_batch)
            if hasattr(X_proc, 'toarray'):
                X_proc = X_proc.toarray()
        else:
            X_proc = X_batch.values

        return self.explainer.shap_values(X_proc)

    def plot_summary(self, X: pd.DataFrame, max_display: int = 15, save: bool = True):
        """Plot l'importance globale des features"""
        print("📊 Génération du summary plot SHAP...")

        # Utiliser les données transformées pour le calcul des SHAP values
        if hasattr(self.model, 'named_steps') and 'preprocessing' in self.model.named_steps:
            X_proc = self.model.named_steps['preprocessing'].transform(X)
            if hasattr(X_proc, 'toarray'):
                X_proc = X_proc.toarray()
        else:
            X_proc = X.values if hasattr(X, 'values') else X

        shap_values = self.explainer.shap_values(X_proc)

        # Pour classification
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[0]
        else:
            shap_values_plot = shap_values

        # Créer le summary plot sans spécifier feature_names
        # SHAP utilisera les indices par défaut
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_plot,
            X_proc,
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

    def plot_waterfall(self, X_instance: pd.DataFrame, instance_idx: int = 0, save: bool = True):
        """Plot waterfall pour une instance spécifique"""
        print(f"📊 Génération du waterfall plot (instance {instance_idx})...")

        explanation = self.explain_instance(X_instance)
        shap_values = explanation['shap_values']
        
        # Pour TreeExplainer avec XGBoost, récupérer les valeurs brutes du modèle
        # au lieu des valeurs transformées
        if hasattr(self.model, 'named_steps'):
            preprocessor = self.model.named_steps.get('preprocessing')
            final_model = self.model.named_steps.get('model') or self.model.named_steps.get('regressor')
            
            # Créer un TreeExplainer directement sur le modèle brut
            # et calculer les SHAP values sur les données brutes
            explainer_raw = shap.TreeExplainer(final_model)
            
            # Transformer les données pour le modèle
            X_proc = preprocessor.transform(X_instance)
            if hasattr(X_proc, 'toarray'):
                X_proc = X_proc.toarray()
            
            # Obtenir les SHAP values sur données transformées
            shap_values_raw = explainer_raw.shap_values(X_proc)
            
            # Pour régression, shap_values_raw est un array 2D
            if isinstance(shap_values_raw, np.ndarray) and len(shap_values_raw.shape) == 2:
                shap_values_plot = shap_values_raw[0]
            else:
                shap_values_plot = shap_values_raw
                
            base_value = explainer_raw.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0]
        else:
            shap_values_plot = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            base_value = explanation['base_value']

        # Créer le waterfall plot avec les données transformées
        # Mais on ne peut pas afficher les noms de features facilement
        # Donc on utilise un beeswarm ou summary plot individuel à la place
        
        plt.figure(figsize=(14, 8))
        
        # Utiliser un force plot ou summary plot horizontal à la place
        # car waterfall nécessite des correspondances exactes feature_names/valeurs
        
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values_plot,
                base_values=base_value,
                data=X_proc[0] if 'X_proc' in locals() else X_instance.iloc[0].values,
                feature_names=[f"f_{i}" for i in range(len(shap_values_plot))]
            ),
            max_display=15,
            show=False
        )
        
        plt.title(f"Explication SHAP - Instance {instance_idx}", fontsize=14)
        plt.tight_layout()

        if save:
            out_path = FIG_DIR / f"shap_waterfall_instance_{instance_idx}.png"
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"✅ Waterfall plot sauvegardé: {out_path}")
        plt.close()

    def get_feature_importance(self, X: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Retourne l'importance moyenne des features"""
        shap_values = self.explain_batch(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Importance moyenne absolue
        importance = np.abs(shap_values).mean(axis=0)
        
        # === CORRECTION SIMPLE : Créer des noms génériques basés sur la taille réelle ===
        actual_feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        df_importance = pd.DataFrame({
            'feature': actual_feature_names,
            'shap_importance': importance
        }).sort_values('shap_importance', ascending=False).head(top_n)

        return df_importance

    def save_explainer(self, path: Path = None):
        """Sauvegarde l'explainer pour utilisation ultérieure"""
        if path is None:
            path = PROCESSED_DATA_DIR / "shap_explainer.pkl"

        joblib.dump({
            'explainer': self.explainer,
            'model': self.model,
            'feature_names': self.feature_names,
            'expected_value': self.expected_value,
            'X_background': self.X_background
        }, path)
        print(f"✅ Explainer SHAP sauvegardé: {path}")

# ============================================================================
# LIME EXPLAINER
# ============================================================================

class LimeExplainer:
    """Classe pour gérer les explications LIME"""

    def __init__(self, model, X_train: pd.DataFrame, feature_names: List[str],
                 categorical_features: List[str] = None, class_names: List[str] = None):
        """
        Initialise l'explainer LIME

        Args:
            model: Pipeline sklearn
            X_train: Données d'entraînement (pour définir les distributions)
            feature_names: Noms des features
            categorical_features: Liste des features catégorielles
            class_names: Noms des classes (pour classification)
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.class_names = class_names or ['Risqué', 'Moyen', 'Bon investissement']
        self.explainer = None
        self.categorical_indices = None

    def _remove_constant_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Supprime les features avec variance nulle"""
        # Ne traiter que les colonnes numériques (déjà encodées)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return X
            
        stds = X[numeric_cols].std()
        constant_cols = stds[stds < 0.001].index.tolist()
        
        if constant_cols:
            print(f"   ⚠️  Features constantes supprimées: {constant_cols}")
            X = X.drop(columns=constant_cols)
            
            # Mettre à jour les listes de features (retirer les colonnes supprimées)
            removed_indices = []
            new_feature_names = []
            for i, f in enumerate(self.feature_names):
                if f in constant_cols:
                    removed_indices.append(i)
                else:
                    new_feature_names.append(f)
            
            self.feature_names = new_feature_names
            
            # Mettre à jour la liste des catégorielles
            self.categorical_features = [
                f for f in self.categorical_features 
                if f not in constant_cols
            ]
            
            print(f"   Features restantes: {len(self.feature_names)}")
        
        return X

    def create_explainer(self):
        """Crée l'explainer LIME tabulaire avec données encodées"""
        print("" + "="*60)
        print("CRÉATION DU EXPLAINER LIME")
        print("="*60)

        # Encoder les données catégorielles
        X_train_encoded = self.X_train.copy()
        
        self.label_encoders = {}
        for col in self.categorical_features:
            if col in X_train_encoded.columns:
                le = LabelEncoder()
                X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        # Recalculer les indices catégoriels sur toutes les features originales
        if self.categorical_features:
            self.categorical_indices = [
                i for i, col in enumerate(self.feature_names)
                if col in self.categorical_features
            ]
        
        print(f"   Indices catégoriels: {self.categorical_indices}")

        # Convertir en numpy array
        X_train_numeric = X_train_encoded.values
        
        print(f"   Données encodées: {X_train_numeric.shape}")

        # === SOLUTION : Désactiver la discretisation qui cause l'erreur truncnorm ===
        self.explainer = LimeTabularExplainer(
            training_data=X_train_numeric,
            feature_names=self.feature_names,
            categorical_features=self.categorical_indices,
            class_names=self.class_names,
            mode='classification',
            discretize_continuous=False,  # ← CHANGEMENT CLÉ : désactiver la discretisation
            sample_around_instance=True,
            random_state=42
        )

        print(f"✅ LimeTabularExplainer créé")
        print(f"   Mode: classification")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Discretisation: désactivée (pour éviter l'erreur truncnorm)")

        return self.explainer

    def explain_instance(self, X_instance: pd.DataFrame, num_features: int = 10) -> Dict:
        """
        Explique une instance avec LIME
        """
        if self.explainer is None:
            raise ValueError("Explainer non créé. Appelez create_explainer() d'abord.")

        # === IMPORTANT : Préparer deux versions des données ===
        # 1. X_instance_encoded : pour LIME (tout numérique)
        # 2. X_instance_raw : pour le modèle sklearn (strings pour catégorielles)

        X_instance_encoded = X_instance.copy()
        X_instance_raw = X_instance.copy()
        
        # Encoder pour LIME (numérique)
        for col, le in self.label_encoders.items():
            if col in X_instance_encoded.columns:
                val = X_instance_encoded[col].iloc[0]
                if val in le.classes_:
                    X_instance_encoded[col] = le.transform([val])[0]
                else:
                    X_instance_encoded[col] = 0
        
        # Garder les valeurs brutes pour le modèle sklearn (pas d'encodage)
        # X_instance_raw garde les strings originales

        instance_array = X_instance_encoded.values[0]

        # Fonction de prédiction - utilise les données BRUTES pour le modèle sklearn
        def predict_fn(x):
            # x vient de LIME et est encodé (numérique), il faut décoder pour le modèle
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            
            # Créer DataFrame avec les noms de features
            X_df = pd.DataFrame(x, columns=self.feature_names)
            
            # === Décoder les catégorielles pour le modèle sklearn ===
            for col, le in self.label_encoders.items():
                if col in X_df.columns:
                    # Convertir les entiers en strings (classes)
                    vals = X_df[col].astype(int)
                    # Gérer les valeurs hors limites
                    decoded = []
                    for v in vals:
                        if 0 <= v < len(le.classes_):
                            decoded.append(le.classes_[v])
                        else:
                            decoded.append(le.classes_[0])  # Valeur par défaut
                    X_df[col] = decoded
            
            # Maintenant X_df a les bonnes types pour le modèle sklearn
            pred = self.model.predict_proba(X_df) if hasattr(self.model, 'predict_proba') else self.model.predict(X_df)
            
            if len(pred.shape) == 1:
                pred = np.column_stack([1 - pred, pred])
            return pred

        # Générer l'explication avec LIME (données encodées)
        explanation = self.explainer.explain_instance(
            data_row=instance_array,
            predict_fn=predict_fn,
            num_features=num_features,
            top_labels=1
        )

        return {
            'explanation': explanation,
            'top_label': explanation.available_labels()[0],
            'score': explanation.score,
            'local_exp': explanation.as_list(label=explanation.available_labels()[0])
        }

    def plot_explanation(self, X_instance: pd.DataFrame, instance_idx: int = 0, 
                    num_features: int = 10, save: bool = True):
        """Visualise l'explication LIME"""
        print(f"📊 Génération de l'explication LIME (instance {instance_idx})...")

        explanation_result = self.explain_instance(X_instance, num_features)
        explanation = explanation_result['explanation']
        top_label = explanation_result['top_label']

        # Sauvegarder la figure
        fig = explanation.as_pyplot_figure(label=top_label)
        plt.title(f"Explication LIME - Instance {instance_idx}", fontsize=14)
        plt.tight_layout()

        if save:
            out_path = FIG_DIR / f"lime_explanation_instance_{instance_idx}.png"
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"✅ Explication LIME sauvegardée: {out_path}")
        plt.close()

        return explanation_result

    def get_explanation_text(self, X_instance: pd.DataFrame, num_features: int = 5) -> str:
        """Retourne une explication textuelle"""
        explanation = self.explain_instance(X_instance, num_features)
        top_label = explanation['top_label']

        text_parts = []
        text_parts.append(f"Prédiction: {self.class_names[top_label] if top_label < len(self.class_names) else 'Inconnue'}")
        text_parts.append(f"Confiance: {explanation['score']:.2%}")
        text_parts.append("\nRaisons principales:")

        for feature, weight in explanation['local_exp'][:num_features]:
            direction = "augmente" if weight > 0 else "diminue"
            text_parts.append(f"  • {feature}: {direction} la probabilité (impact: {abs(weight):.3f})")

        return "\n".join(text_parts)

    def save_explainer(self, path: Path = None):
        """Sauvegarde l'explainer LIME (sans l'objet LIME lui-même qui n'est pas picklable)"""
        if path is None:
            path = PROCESSED_DATA_DIR / "lime_explainer.pkl"

        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'class_names': self.class_names,
            'X_train': self.X_train,
            'label_encoders': self.label_encoders,
            'categorical_indices': self.categorical_indices
        }, path)
        print(f"✅ Configuration LIME sauvegardée: {path}")

    @staticmethod
    def load_explainer(path: Path = None, model=None):
        """
        Charge la configuration LIME et recrée l'explainer fonctionnel
        
        Usage dans votre API:
            lime_exp = LimeExplainer.load_explainer(path, model)
            explanation = lime_exp.explain_instance(X_instance)
        """
        if path is None:
            path = PROCESSED_DATA_DIR / "lime_explainer.pkl"
        
        print(f"📥 Chargement configuration LIME: {path}")
        data = joblib.load(path)
        
        # Créer une nouvelle instance
        lime_exp = LimeExplainer(
            model=model or data['model'],
            X_train=data['X_train'],
            feature_names=data['feature_names'],
            categorical_features=data['categorical_features'],
            class_names=data['class_names']
        )
        
        # Restaurer les attributs
        lime_exp.label_encoders = data.get('label_encoders', {})
        lime_exp.categorical_indices = data.get('categorical_indices', [])
        
        # Recréer l'explainer LIME fonctionnel
        lime_exp.create_explainer()
        
        print(f"✅ Explainer LIME recréé et prêt à l'emploi")
        return lime_exp

# ============================================================================
# FONCTIONS UTILITAIRES POUR L'API
# ============================================================================

def explain_watch(watch_features: Dict, models: Dict, X_background: pd.DataFrame,
                  feature_names: List[str], method: str = 'both') -> Dict:
    """
    Fonction principale pour expliquer une montre

    Args:
        watch_features: Dict avec les caractéristiques de la montre
        models: Dict avec les modèles chargés
        X_background: DataFrame de background
        feature_names: Noms des features
        method: 'shap', 'lime', ou 'both'

    Returns:
        Dict avec les explications
    """
    # Convertir en DataFrame
    X_instance = pd.DataFrame([watch_features])

    # S'assurer que toutes les colonnes sont présentes
    for col in feature_names:
        if col not in X_instance.columns:
            X_instance[col] = 0  # Valeur par défaut

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
        shap_exp = ShapExplainer(models['price'], X_background, feature_names)
        shap_exp.create_explainer(sample_size=100)
        shap_result = shap_exp.explain_instance(X_instance)

        # Top features influentes
        shap_values = shap_result['shap_values'][0] if len(shap_result['shap_values'].shape) > 1 else shap_result['shap_values']
        feature_importance = list(zip(feature_names, shap_values))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        results['shap_explanation'] = {
            'base_value': float(shap_result['base_value']),
            'top_positive_features': [
                {'feature': f, 'impact': float(v)} 
                for f, v in feature_importance[:5] if v > 0
            ],
            'top_negative_features': [
                {'feature': f, 'impact': float(v)} 
                for f, v in feature_importance[:5] if v < 0
            ]
        }

    # Explications LIME
    if method in ['lime', 'both'] and LIME_AVAILABLE and 'classifier' in models:
        print("\n🔍 Génération explication LIME...")
        lime_exp = LimeExplainer(
            models['classifier'], 
            X_background, 
            feature_names,
            categorical_features=[c for c in CATEGORICAL_FEATURES if c in feature_names]
        )
        lime_exp.create_explainer()
        lime_result = lime_exp.explain_instance(X_instance)

        results['lime_explanation'] = {
            'top_label': int(lime_result['top_label']),
            'confidence': float(lime_result['score']),
            'key_factors': [
                {'feature': f, 'impact': float(w)} 
                for f, w in lime_result['local_exp'][:5]
            ]
        }

    return results

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def run_xai_analysis():
    """Exécute l'analyse XAI complète"""
    print("="*70)
    print("ANALYSE XAI - SHAP & LIME")
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

    # 4. Sélection d'instances à expliquer (exemples)
    sample_instances = X.sample(n=3, random_state=42)

    # 5. SHAP Analysis
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
            shap_exp.plot_waterfall(X_inst, instance_idx=idx, save=True)

        # Importance des features
        importance_df = shap_exp.get_feature_importance(X_background, top_n=10)
        print("\n📊 Top 10 features (SHAP):")
        print(importance_df.to_string(index=False))

        # Sauvegarde
        shap_exp.save_explainer()

    # 6. LIME Analysis
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

    # 7. Test de la fonction explain_watch
    print("\n" + "="*70)
    print("TEST FONCTION EXPLAIN_WATCH")
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
        method='both'
    )

    print("\n📊 Résultats pour la montre test:")
    print(f"Prix futur estimé: {result['predictions'].get('price_future', 'N/A'):,.0f}€" if 'price_future' in result['predictions'] else "Prix: N/A")
    print(f"Classe: {result['predictions'].get('investment_class', 'N/A')}")

    if 'shap_explanation' in result:
        print(f"\nTop feature positive SHAP: {result['shap_explanation']['top_positive_features'][0] if result['shap_explanation']['top_positive_features'] else 'N/A'}")

    if 'lime_explanation' in result:
        print(f"Confiance LIME: {result['lime_explanation']['confidence']:.2%}")

    # Sauvegarde du résultat test
    with open(PROCESSED_DATA_DIR / "xai_example_result.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✅ Résultat exemple sauvegardé: {PROCESSED_DATA_DIR / 'xai_example_result.json'}")

    print("\n" + "="*70)
    print("✅ ANALYSE XAI TERMINÉE")
    print(f"📁 Figures sauvegardées dans: {FIG_DIR}")
    print("="*70)

if __name__ == "__main__":
    run_xai_analysis()