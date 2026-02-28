"""
API FastAPI - Système d'Investissement Montres de Luxe
VERSION STRUCTURÉE PAR ONGLETS - Correspondance exacte avec le Frontend
Endpoints: /general, /materials, /seller, /analyze (tout combiné)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional, Dict, Any, Literal
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from enum import Enum

# Imports XAI
try:
    import shap
    SHAP_OK = True
except:
    SHAP_OK = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    from sklearn.preprocessing import LabelEncoder
    LIME_OK = True
except:
    LIME_OK = False

from setup import PROCESSED_DATA_DIR, CLEANED_CSV

# ============================================================
# CONFIGURATION
# ============================================================

class InvestmentThreshold:
    ROI_EXCELLENT = 25.0
    ROI_GOOD = 15.0
    ROI_ACCEPTABLE = 5.0

app = FastAPI(
    title="API Investissement Montres de Luxe - Par Onglets",
    description="API organisée selon les 3 onglets du frontend: Général, Matériaux, Vendeur",
    version="3.0-structured"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ENUMS - Listes de valeurs pour les dropdowns
# ============================================================

class BrandEnum(str, Enum):
    ROLEX = "Rolex"
    PATEK_PHILIPPE = "Patek Philippe"
    AUDEMARS_PIGUET = "Audemars Piguet"
    OMEGA = "Omega"
    CARTIER = "Cartier"
    TUDOR = "Tudor"
    TAG_HEUER = "TAG Heuer"
    BREITLING = "Breitling"
    IWC = "IWC"
    JAEGER_LECOULTRE = "Jaeger-LeCoultre"
    PANERAI = "Panerai"
    HUBLOT = "Hublot"
    ZENITH = "Zenith"
    SEIKO = "Seiko"
    CASIO = "Casio"
    CITIZEN = "Citizen"
    TISSOT = "Tissot"
    LONGINES = "Longines"
    RADO = "Rado"
    HAMILTON = "Hamilton"

class ConditionEnum(str, Enum):
    NEW = "New"
    UNWORN = "Unworn"
    LIKE_NEW = "Like new & unworn"
    USED_MINT = "Used (Mint)"
    USED_VERY_GOOD = "Used (Very good)"
    USED_GOOD = "Used (Good)"
    USED_FAIR = "Used (Fair)"
    USED_POOR = "Used (Poor)"

class MovementEnum(str, Enum):
    AUTOMATIC = "Automatic"
    MANUAL = "Manual winding"
    QUARTZ = "Quartz"

class GenderEnum(str, Enum):
    MEN = "Men's watch"
    WOMEN = "Women's watch"
    UNISEX = "Unisex"

class ShapeEnum(str, Enum):
    ROUND = "Round"
    SQUARE = "Square"
    RECTANGULAR = "Rectangular"
    TONNEAU = "Tonneau"
    CUSHION = "Cushion"
    OCTAGONAL = "Octagonal"

class MaterialEnum(str, Enum):
    STEEL = "Steel"
    GOLD = "Gold"
    ROSE_GOLD = "Rose gold"
    WHITE_GOLD = "White gold"
    PLATINUM = "Platinum"
    TITANIUM = "Titanium"
    CERAMIC = "Ceramic"
    CARBON = "Carbon"

class CrystalEnum(str, Enum):
    SAPPHIRE = "Sapphire crystal"
    MINERAL = "Mineral crystal"
    PLEXIGLASS = "Plexiglass"

class ColorEnum(str, Enum):
    BLACK = "Black"
    SILVER = "Silver"
    WHITE = "White"
    BLUE = "Blue"
    GREEN = "Green"
    BROWN = "Brown"
    CHAMPAGNE = "Champagne"
    GREY = "Grey"

class AvailabilityEnum(str, Enum):
    IN_STOCK = "Item is in stock"
    PARTNER = "Item is in stock at a partner boutique"
    SERVICING = "Item is being serviced"
    HOLD = "Item is on hold"

class DeliveryEnum(str, Enum):
    BOX_PAPERS = "Original box, original papers"
    BOX_ONLY = "Original box, no original papers"
    PAPERS_ONLY = "Original papers, no original box"
    NOTHING = "No original box, no original papers"

# ============================================================
# MODÈLES PAR ONGLET
# ============================================================

class GeneralTab(BaseModel):
    """Onglet GÉNÉRAL - 8 champs"""
    model_config = ConfigDict(populate_by_name=True, extra='ignore')

    prix_achat: float = Field(
        default=10000,
        alias="Prix d'achat ($)",
        description="Prix d'achat en dollars",
        ge=1000,
        le=10000000
    )
    marque: BrandEnum = Field(
        default=BrandEnum.ROLEX,
        alias="Marque"
    )
    etat: ConditionEnum = Field(
        default=ConditionEnum.LIKE_NEW,
        alias="État"
    )
    mouvement: MovementEnum = Field(
        default=MovementEnum.AUTOMATIC,
        alias="Mouvement"
    )
    genre: GenderEnum = Field(
        default=GenderEnum.MEN,
        alias="Genre"
    )
    annee_production: int = Field(
        default=2020,
        alias="Année de production",
        ge=1900,
        le=2030
    )
    forme: ShapeEnum = Field(
        default=ShapeEnum.ROUND,
        alias="Forme"
    )
    surface_cadran: float = Field(
        default=40.0,
        alias="Surface du cadran (mm²)",
        gt=0
    )
    contenu_livraison: DeliveryEnum = Field(
        default=DeliveryEnum.BOX_PAPERS,
        alias="Contenu de la livraison"
    )

    horizon_annees: int = Field(default=3, ge=1, le=10)

class MaterialsTab(BaseModel):
    """Onglet MATÉRIAUX - 6 champs"""
    model_config = ConfigDict(populate_by_name=True, extra='ignore')

    materiau_boitier: MaterialEnum = Field(
        default=MaterialEnum.STEEL,
        alias="Matériau du boîtier"
    )
    materiau_bracelet: MaterialEnum = Field(
        default=MaterialEnum.STEEL,
        alias="Matériau du bracelet"
    )
    verre: CrystalEnum = Field(
        default=CrystalEnum.SAPPHIRE,
        alias="Verre"
    )
    cadran: ColorEnum = Field(
        default=ColorEnum.BLACK,
        alias="Cadran"
    )
    couleur_bracelet: ColorEnum = Field(
        default=ColorEnum.SILVER,
        alias="Couleur du bracelet"
    )
    disponibilite: AvailabilityEnum = Field(
        default=AvailabilityEnum.IN_STOCK,
        alias="Disponibilité"
    )

class SellerTab(BaseModel):
    """Onglet VENDEUR - 6 champs (3 numériques + 3 toggles)"""
    model_config = ConfigDict(populate_by_name=True, extra='ignore')

    montres_vendues: int = Field(
        default=50,
        alias="Montres vendues par le vendeur",
        ge=0
    )
    annonces_actives: int = Field(
        default=10,
        alias="Annonces actives du vendeur",
        ge=0
    )
    avis_vendeur: int = Field(
        default=100,
        alias="Avis du vendeur",
        ge=0
    )
    expedition_rapide: int = Field(
        default=1,
        alias="Expédition rapide",
        description="Le vendeur expédie rapidement (toggle)",
        ge=0,
        le=1
    )
    vendeur_confiance: int = Field(
        default=1,
        alias="Vendeur de confiance",
        description="Le vendeur est vérifié (toggle)",
        ge=0,
        le=1
    )
    ponctualite: int = Field(
        default=1,
        alias="Ponctualité",
        description="Le vendeur est ponctuel (toggle)",
        ge=0,
        le=1
    )

class CompleteWatchInput(BaseModel):
    """Tous les onglets combinés"""
    model_config = ConfigDict(populate_by_name=True, extra='ignore')

    # Général
    prix_achat: float = Field(default=10000, alias="Prix d'achat ($)", gt=0 , ge=1000)
    marque: BrandEnum = Field(default=BrandEnum.ROLEX, alias="Marque")
    etat: ConditionEnum = Field(default=ConditionEnum.LIKE_NEW, alias="État")
    mouvement: MovementEnum = Field(default=MovementEnum.AUTOMATIC, alias="Mouvement")
    genre: GenderEnum = Field(default=GenderEnum.MEN, alias="Genre")
    annee_production: int = Field(default=2020, alias="Année de production", ge=1900, le=2030)
    forme: ShapeEnum = Field(default=ShapeEnum.ROUND, alias="Forme")
    surface_cadran: float = Field(default=40.0, alias="Surface du cadran (mm²)", gt=0)
    contenu_livraison: DeliveryEnum = Field(default=DeliveryEnum.BOX_PAPERS, alias="Contenu de la livraison")

    # Matériaux
    materiau_boitier: MaterialEnum = Field(default=MaterialEnum.STEEL, alias="Matériau du boîtier")
    materiau_bracelet: MaterialEnum = Field(default=MaterialEnum.STEEL, alias="Matériau du bracelet")
    verre: CrystalEnum = Field(default=CrystalEnum.SAPPHIRE, alias="Verre")
    cadran: ColorEnum = Field(default=ColorEnum.BLACK, alias="Cadran")
    couleur_bracelet: ColorEnum = Field(default=ColorEnum.SILVER, alias="Couleur du bracelet")
    disponibilite: AvailabilityEnum = Field(default=AvailabilityEnum.IN_STOCK, alias="Disponibilité")

    # Vendeur
    montres_vendues: int = Field(default=50, alias="Montres vendues par le vendeur", ge=0)
    annonces_actives: int = Field(default=10, alias="Annonces actives du vendeur", ge=0)
    avis_vendeur: int = Field(default=100, alias="Avis du vendeur", ge=0)
    expedition_rapide: int = Field(default=1, alias="Expédition rapide", ge=0, le=1)
    vendeur_confiance: int = Field(default=1, alias="Vendeur de confiance", ge=0, le=1)
    ponctualite: int = Field(default=1, alias="Ponctualité", ge=0, le=1)

    horizon_annees: int = Field(default=3, ge=1, le=10)

class InvestmentResult(BaseModel):
    """Résultat d'évaluation"""
    prix_achat: float
    prix_futur_estime: float
    plus_value: float
    roi_percent: float
    roi_annualise: float
    horizon_annees: int
    recommandation: str
    confiance: str
    risque: str
    evaluation_simple: Literal["Bon", "Moyen", "Risqué"]
    details: Dict[str, Any]

class TabValidationResponse(BaseModel):
    """Réponse de validation d'un onglet"""
    onglet: str
    valide: bool
    erreurs: List[str]
    donnees_recues: Dict[str, Any]

# ============================================================
# CHARGEMENT MODÈLES
# ============================================================

print("📦 Chargement des modèles...")

try:
    price_model = joblib.load(PROCESSED_DATA_DIR / "best_price_model.pkl")
    print("✅ Modèle prix chargé")
except Exception as e:
    price_model = None
    print(f"❌ Erreur modèle prix: {e}")

try:
    clf_model = joblib.load(PROCESSED_DATA_DIR / "best_classifier.pkl")
    print("✅ Classifieur chargé")
except Exception as e:
    clf_model = None
    print(f"❌ Erreur classifieur: {e}")

try:
    with open(CLEANED_CSV, 'r', encoding='utf-8') as f:
        sep = ';' if ';' in f.readline() else ','
    data_sample = pd.read_csv(CLEANED_CSV, sep=sep, nrows=500)
    print(f"✅ Données chargées: {len(data_sample)} lignes")
except Exception as e:
    data_sample = None
    print(f"❌ Erreur données: {e}")

MODELS_READY = price_model is not None and clf_model is not None

FEATURES = [
    "Year of production", "age", "Face Area",
    "Watches Sold by the Seller", "Active listing of the seller",
    "Seller Reviews", "seller_reputation_score", "scope_score",
    "Fast Shipper", "Trusted Seller", "Punctuality",
    "is_modern", "seller_consistent", "price_anomaly_low", "price_anomaly_high",
    "current_price_estimate",
    "Brand", "Movement", "Case material", "Bracelet material",
    "Condition", "Scope of delivery", "Gender", "Availability",
    "Shape", "Crystal", "Dial", "Bracelet color"
]

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def combine_tabs(general: GeneralTab, materials: MaterialsTab, seller: SellerTab) -> CompleteWatchInput:
    """Combine les 3 onglets en un seul objet"""
    return CompleteWatchInput(
        prix_achat=general.prix_achat,
        marque=general.marque,
        etat=general.etat,
        mouvement=general.mouvement,
        genre=general.genre,
        annee_production=general.annee_production,
        forme=general.forme,
        surface_cadran=general.surface_cadran,
        contenu_livraison=general.contenu_livraison,
        materiau_boitier=materials.materiau_boitier,
        materiau_bracelet=materials.materiau_bracelet,
        verre=materials.verre,
        cadran=materials.cadran,
        couleur_bracelet=materials.couleur_bracelet,
        disponibilite=materials.disponibilite,
        montres_vendues=seller.montres_vendues,
        annonces_actives=seller.annonces_actives,
        avis_vendeur=seller.avis_vendeur,
        expedition_rapide=seller.expedition_rapide,
        vendeur_confiance=seller.vendeur_confiance,
        ponctualite=seller.ponctualite,
        horizon_annees=general.horizon_annees
    )

def prepare_data(watch: CompleteWatchInput) -> pd.DataFrame:
    """Prépare les features pour le modèle"""
    current_year = 2026
    age = current_year - watch.annee_production

    # Mapping des valeurs
    data = {
        "Year of production": watch.annee_production,
        "age": age,
        "Face Area": watch.surface_cadran,
        "Watches Sold by the Seller": watch.montres_vendues,
        "Active listing of the seller": watch.annonces_actives,
        "Seller Reviews": watch.avis_vendeur,
        "Fast Shipper": watch.expedition_rapide,
        "Trusted Seller": watch.vendeur_confiance,
        "Punctuality": watch.ponctualite,
        "is_modern": 1 if watch.annee_production >= 2000 else 0,
        "seller_reputation_score": np.mean([
            watch.expedition_rapide, 
            watch.vendeur_confiance, 
            watch.ponctualite
        ]),
        "scope_score": {
            'Original box, original papers': 3,
            'Original box, no original papers': 2,
            'Original papers, no original box': 1,
            'No original box, no original papers': 0
        }.get(watch.contenu_livraison.value, 0),
        "seller_consistent": 1 if watch.montres_vendues >= watch.annonces_actives else 0,
        "price_anomaly_low": 0,
        "price_anomaly_high": 0,
        "current_price_estimate": watch.prix_achat,
        "Brand": watch.marque.value,
        "Movement": watch.mouvement.value,
        "Case material": watch.materiau_boitier.value,
        "Bracelet material": watch.materiau_bracelet.value,
        "Condition": watch.etat.value,
        "Scope of delivery": watch.contenu_livraison.value,
        "Gender": watch.genre.value,
        "Availability": watch.disponibilite.value,
        "Shape": watch.forme.value,
        "Crystal": watch.verre.value,
        "Dial": watch.cadran.value,
        "Bracelet color": watch.couleur_bracelet.value
    }

    df = pd.DataFrame([data])
    return df

def estimate_market_price(brand: str, age: int, condition: str) -> float:
    """Estime le prix du marché actuel"""
    base_prices = {
        "Rolex": 12000, "Patek Philippe": 40000, "Audemars Piguet": 35000,
        "Omega": 6000, "Cartier": 8000, "Tudor": 4000, "TAG Heuer": 3000,
        "Breitling": 5000, "IWC": 7000, "Jaeger-LeCoultre": 8000,
        "Panerai": 7000, "Hublot": 15000, "Zenith": 6000,
        "Seiko": 500, "Casio": 200, "Citizen": 300, "Tissot": 400,
        "Longines": 1500, "Rado": 2000, "Hamilton": 800
    }
    
    base = base_prices.get(brand, 1000)
    
    if age <= 1:
        age_factor = 0.85
    elif age <= 5:
        age_factor = 0.75 - (age - 1) * 0.03
    elif age <= 15:
        age_factor = 0.60 - (age - 5) * 0.02
    elif age <= 30:
        age_factor = 0.40 - (age - 15) * 0.01
    else:
        age_factor = 0.25
    
    condition_factors = {
        "New": 1.0, "Unworn": 0.95, "Like new & unworn": 0.92,
        "Used (Mint)": 0.88, "Used (Very good)": 0.80,
        "Used (Good)": 0.65, "Used (Fair)": 0.50, "Used (Poor)": 0.35
    }
    condition_factor = condition_factors.get(condition, 0.70)
    
    brand_premium = 1.2 if brand in ["Rolex", "Patek Philippe", "Audemars Piguet"] else \
                    0.7 if brand in ["Seiko", "Casio", "Citizen"] else 1.0
    
    return max(100, base * age_factor * condition_factor * brand_premium)

def evaluate_investment(watch: CompleteWatchInput) -> InvestmentResult:
    """Évalue l'investissement"""
    if not MODELS_READY:
        raise HTTPException(503, "Modèles non chargés")

    if watch.prix_achat < 100:
        raise HTTPException(status_code=400, detail="Prix d'achat minimum: 100$")

    X = prepare_data(watch)

    # Prédiction prix
    price_log = price_model.predict(X)[0]
    prix_futur = float(np.expm1(price_log))

    # Calculs ROI
    plus_value = prix_futur - watch.prix_achat
    roi_total = (plus_value / watch.prix_achat) * 100
    roi_annualise = roi_total / watch.horizon_annees if watch.horizon_annees > 0 else 0

    # Classification risque
    proba = clf_model.predict_proba(X)[0]
    risque_score = proba[0]

    # Recommandation
    if roi_annualise >= InvestmentThreshold.ROI_EXCELLENT and risque_score < 0.4:
        recommandation = "EXCELLENT INVESTISSEMENT"
        confiance = "Élevée"
        risque = "Faible"
        evaluation_simple = "Bon"
    elif roi_annualise >= InvestmentThreshold.ROI_GOOD and risque_score < 0.5:
        recommandation = "BON INVESTISSEMENT"
        confiance = "Moyenne à Élevée"
        risque = "Modéré"
        evaluation_simple = "Bon"
    elif roi_annualise >= InvestmentThreshold.ROI_ACCEPTABLE:
        recommandation = "INVESTISSEMENT ACCEPTABLE"
        confiance = "Moyenne"
        risque = "Moyen"
        evaluation_simple = "Moyen"
    elif plus_value > 0:
        recommandation = "INVESTISSEMENT MARGINAL"
        confiance = "Faible"
        risque = "Élevé"
        evaluation_simple = "Risqué"
    else:
        recommandation = "MAUVAIS INVESTISSEMENT - ÉVITER"
        confiance = "Élevée"
        risque = "Très Élevé"
        evaluation_simple = "Risqué"

    # Estimation prix marché
    age = int(X["age"].iloc[0])
    prix_marche = estimate_market_price(watch.marque.value, age, watch.etat.value)
    
    if watch.prix_achat < prix_marche * 0.9:
        deal_quality = "Excellente affaire (sous le marché)"
    elif watch.prix_achat < prix_marche * 1.05:
        deal_quality = "Prix du marché"
    else:
        deal_quality = "Cher (au-dessus du marché)"

    return InvestmentResult(
        prix_achat=watch.prix_achat,
        prix_futur_estime=round(prix_futur, 2),
        plus_value=round(plus_value, 2),
        roi_percent=round(roi_total, 2),
        roi_annualise=round(roi_annualise, 2),
        horizon_annees=watch.horizon_annees,
        recommandation=recommandation,
        confiance=confiance,
        risque=risque,
        evaluation_simple=evaluation_simple,
        details={
            "prix_marche_actuel": round(prix_marche, 2),
            "qualite_affaire": deal_quality,
            "difference_avec_marche": round(watch.prix_achat - prix_marche, 2),
            "probabilites": {
                "risque": round(float(proba[0]), 3),
                "moyen": round(float(proba[1]), 3),
                "bon": round(float(proba[2]), 3)
            }
        }
    )

# ============================================================
# MODÈLES DE RÉPONSE XAI
# ============================================================

class SHAPFeature(BaseModel):
    name: str
    value: float  # Valeur SHAP
    impact: str  # "positive" ou "negative"
    description: str

class LIMEContribution(BaseModel):
    feature: str
    value: str
    contribution: float
    impact: str  # "favorable" ou "defavorable"
    description: str

class XAIResponse(BaseModel):
    shap: Dict[str, Any]
    lime: Dict[str, Any]
    top_features: List[Dict[str, Any]]

# ============================================================
# FONCTIONS XAI
# ============================================================

def generate_shap_explanation(X: pd.DataFrame, model) -> Dict[str, Any]:
    """Génère l'explication SHAP"""
    if not SHAP_OK:
        return {"error": "SHAP non disponible", "available": False}
    
    try:
        if model is None:
            return {"error": "Modèle non chargé", "available": False}
        
        # Extraire le vrai modèle si c'est un Pipeline
        actual_model = model
        if hasattr(model, 'named_steps'):
            # C'est un Pipeline, prendre le dernier step
            last_step_name = list(model.named_steps.keys())[-1]
            actual_model = model.named_steps[last_step_name]
        
        # Mapping noms lisibles (défini au début)
        feature_names_fr = {
            "Year of production": "Année",
            "age": "Âge",
            "Face Area": "Surface",
            "Watches Sold by the Seller": "Ventes",
            "Active listing of the seller": "Annonces",
            "Seller Reviews": "Avis",
            "seller_reputation_score": "Réputation",
            "scope_score": "Livraison",
            "Fast Shipper": "Expédition",
            "Trusted Seller": "Confiance",
            "Punctuality": "Ponctualité",
            "is_modern": "Moderne",
            "seller_consistent": "Cohérence",
            "price_anomaly_low": "Anomalie -",
            "price_anomaly_high": "Anomalie +",
            "current_price_estimate": "Prix actuel",
            "Brand": "Marque",
            "Movement": "Mouvement",
            "Case material": "Matériau",
            "Bracelet material": "Bracelet",
            "Condition": "État",
            "Scope of delivery": "Box/Papiers",
            "Gender": "Genre",
            "Availability": "Stock",
            "Shape": "Forme",
            "Crystal": "Verre",
            "Dial": "Cadran",
            "Bracelet color": "Couleur"
        }
        
        # Convertir en numérique
        X_numeric = X.copy()
        for col in X_numeric.columns:
            if X_numeric[col].dtype == 'object':
                le = LabelEncoder()
                X_numeric[col] = le.fit_transform(X_numeric[col].astype(str))
        
        # TreeExplainer
        try:
            explainer = shap.TreeExplainer(actual_model)
            shap_values = explainer.shap_values(X_numeric)
        except Exception as tree_error:
            # Fallback sur KernelExplainer
            print(f"TreeExplainer failed: {tree_error}, using KernelExplainer")
            explainer = shap.KernelExplainer(actual_model.predict, X_numeric.iloc[:5])
            shap_values = explainer.shap_values(X_numeric.iloc[0:1])
        
        # Gérer shape des shap_values
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        if len(shap_values.shape) > 1:
            shap_values_instance = shap_values[0]
        else:
            shap_values_instance = shap_values
        
        # Créer features
        features = []
        for i, col in enumerate(X_numeric.columns):
            if i < len(shap_values_instance):
                val = float(shap_values_instance[i])
                features.append({
                    "name": feature_names_fr.get(col, col),
                    "value": round(val, 4),
                    "impact": "positive" if val > 0 else "negative"
                })
        
        # Trier
        features.sort(key=lambda x: abs(x["value"]), reverse=True)
        
        # Chart data
        chart_data = []
        for f in features[:8]:
            chart_data.append({
                "feature": f["name"],
                "value": f["value"],
                "color": "#22c55e" if f["impact"] == "positive" else "#ef4444"
            })
        
        return {
            "available": True,
            "chart_type": "bar",
            "title": "Importance des caractéristiques (SHAP)",
            "description": "Impact de chaque caractéristique sur le prix prédit. Vert = augmente, Rouge = diminue.",
            "data": chart_data,
            "top_features": [
                {"rank": i+1, "name": f["name"]} 
                for i, f in enumerate(features[:5])
            ]
        }
            
    except Exception as e:
        import traceback
        print(f"SHAP Error: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e), "available": False}

def generate_lime_explanation(X: pd.DataFrame, model, feature_names: List[str]) -> Dict[str, Any]:
    """Génère l'explication LIME"""
    if not LIME_OK:
        return {"error": "LIME non disponible", "available": False}
    
    try:
        if model is None or not hasattr(model, 'predict_proba'):
            return {"error": "Modèle incompatible", "available": False}
        
        # Mapping noms (défini AU DÉBUT)
        feature_names_fr = {
            "Year of production": "Année",
            "Brand": "Marque",
            "Condition": "État",
            "current_price_estimate": "Prix",
            "Trusted Seller": "Vendeur",
            "Seller Reviews": "Avis",
            "Movement": "Mouvement",
            "Case material": "Matériau",
            "age": "Âge",
            "Face Area": "Surface"
        }
        
        # Convertir en numérique
        X_numeric = X.copy()
        categorical_features = []
        categorical_names = {}
        
        for i, col in enumerate(X_numeric.columns):
            if X_numeric[col].dtype == 'object':
                categorical_features.append(i)
                le = LabelEncoder()
                X_numeric[col] = le.fit_transform(X_numeric[col].astype(str))
                categorical_names[i] = list(le.classes_)
        
        # Background data
        background_data = X_numeric.values
        
        # Noms de features pour LIME
        lime_feature_names = [feature_names_fr.get(f, f) for f in X_numeric.columns]
        
        # Créer explainer
        explainer = LimeTabularExplainer(
            training_data=background_data,
            feature_names=lime_feature_names,
            class_names=['Mauvais', 'Moyen', 'Bon'],
            mode='classification',
            discretize_continuous=True,
            categorical_features=categorical_features if categorical_features else None,
            categorical_names=categorical_names if categorical_names else None,
            verbose=False
        )
        
        # Explication
        def predict_proba_wrapper(x):
            """Wrapper pour gérer les types"""
            if isinstance(x, list):
                x = np.array(x)
            return model.predict_proba(x)
        
        exp = explainer.explain_instance(
            X_numeric.values[0], 
            predict_proba_wrapper, 
            num_features=6,
            top_labels=1
        )
        
        # Récupérer explications
        label = exp.available_labels()[0]
        explanations = exp.as_list(label=label)
        
        # Formater contributions
        contributions = []
        
        for feature, weight in explanations:
            feature_str = str(feature)
            weight_float = float(weight)
            
            # Parser
            if '=' in feature_str:
                parts = feature_str.split('=', 1)
                feat_name = parts[0].strip()
                feat_val = parts[1].strip()
            else:
                feat_name = feature_str
                feat_val = ""
            
            is_positive = weight_float > 0
            
            # Descriptions
            descriptions = {
                "Marque": "Excellente réputation" if is_positive else "Moins reconnue",
                "État": "Excellent état" if is_positive else "État moyen",
                "Prix": "Bon prix" if is_positive else "Prix élevé",
                "Vendeur": "Vendeur fiable" if is_positive else "Risque vendeur",
                "Année": "Année recherchée" if is_positive else "Moins collectible",
                "Avis": "Avis positifs" if is_positive else "Peu d'avis"
            }
            
            contributions.append({
                "feature": feat_name,
                "value": feat_val,
                "contribution": round(abs(weight_float), 4),
                "impact": "favorable" if is_positive else "defavorable",
                "description": descriptions.get(feat_name, "Impact sur prédiction")
            })
        
        # Trier
        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        
        # Chart data
        chart_data = []
        for c in contributions:
            val = c["contribution"] if c["impact"] == "favorable" else -c["contribution"]
            display_name = f"{c['feature']}: {c['value']}" if c['value'] else c['feature']
            chart_data.append({
                "feature": display_name,
                "value": val,
                "color": "#22c55e" if c["impact"] == "favorable" else "#ef4444"
            })
        
        return {
            "available": True,
            "chart_type": "bar_horizontal",
            "title": "Explication locale (LIME)",
            "description": "Contributions des caractéristiques pour cette montre spécifique.",
            "data": chart_data,
            "contributions": contributions
        }
            
    except Exception as e:
        import traceback
        print(f"LIME Error: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e), "available": False}
# ============================================================
# ENDPOINTS PAR ONGLET
# ============================================================

@app.get("/health")
def health():
    """Vérifie l'état de l'API"""
    return {
        "status": "ok" if MODELS_READY else "error",
        "version": "3.0-structured",
        "models": {
            "price": price_model is not None, 
            "classifier": clf_model is not None
        },
        "xai": {
            "shap": {
                "available": SHAP_OK,
                "version": shap.__version__ if SHAP_OK else None
            },
            "lime": {
                "available": LIME_OK,
                "version": None  # LIME n'a pas souvent d'attribut __version__ direct
            }
        },
        "endpoints": ["/general", "/materials", "/seller", "/analyze", "/explain"]
    }

@app.get("/options")
def get_form_options():
    """
    Retourne toutes les options pour les dropdowns du frontend
    """
    return {
        "marques": [b.value for b in BrandEnum],
        "etats": [c.value for c in ConditionEnum],
        "mouvements": [m.value for m in MovementEnum],
        "genres": [g.value for g in GenderEnum],
        "formes": [s.value for s in ShapeEnum],
        "materiaux": [m.value for m in MaterialEnum],
        "verres": [c.value for c in CrystalEnum],
        "couleurs": [c.value for c in ColorEnum],
        "disponibilites": [a.value for a in AvailabilityEnum],
        "box": [d.value for d in DeliveryEnum]
    }
# ============================================================
# ENDPOINTS INDÉPENDANTS AVEC PRÉDICTION
# ============================================================

@app.post("/general")
def analyze_general(data: GeneralTab):
    """
    Étape 1: Analyse basée UNIQUEMENT sur l'onglet Général
    Matériaux et vendeur = valeurs par défaut standards
    """
    if not MODELS_READY:
        raise HTTPException(503, "Modèles non chargés")
    
    # Crée objet complet avec défauts pour matériaux et vendeur
    complete = CompleteWatchInput(
        # Général (depuis input)
        prix_achat=data.prix_achat,
        marque=data.marque,
        etat=data.etat,
        mouvement=data.mouvement,
        genre=data.genre,
        annee_production=data.annee_production,
        forme=data.forme,
        surface_cadran=data.surface_cadran,
        contenu_livraison=data.contenu_livraison,
        horizon_annees=data.horizon_annees,
        # Matériaux (défauts neutres)
        materiau_boitier=MaterialEnum.STEEL,
        materiau_bracelet=MaterialEnum.STEEL,
        verre=CrystalEnum.SAPPHIRE,
        cadran=ColorEnum.BLACK,
        couleur_bracelet=ColorEnum.SILVER,
        disponibilite=AvailabilityEnum.IN_STOCK,
        # Vendeur (défauts moyens)
        montres_vendues=50,
        annonces_actives=10,
        avis_vendeur=100,
        expedition_rapide=1,
        vendeur_confiance=1,
        ponctualite=1
    )
    
    result = evaluate_investment(complete)
    
    return {
        "onglet": "general",
        "analyse_basee_sur": "Caractéristiques générales uniquement",
        "precision": "Estimation grossière",
        "avertissement": "Matériaux et vendeur utilisent des valeurs par défaut",
        "prediction": {
            "prix_achat": result.prix_achat,
            "prix_futur_estime": result.prix_futur_estime,
            "plus_value": result.plus_value,
            "roi_percent": result.roi_percent,
            "roi_annualise": result.roi_annualise,
            "evaluation": result.evaluation_simple,
            "recommandation": result.recommandation,
            "confiance": "Faible - complétez les onglets Matériaux et Vendeur"
        },
        "hypotheses_utilisees": {
            "materiaux": {
                "boitier": "Steel (défaut)",
                "bracelet": "Steel (défaut)",
                "verre": "Sapphire crystal (défaut)"
            },
            "vendeur": {
                "reputation": "Moyenne (défaut)",
                "avis": 100,
                "confiance": "Standard"
            }
        },
        "prochaine_etape": "Rendez-vous dans l'onglet Matériaux pour affiner la prédiction"
    }

@app.post("/materials")
def analyze_materials(data: MaterialsTab):
    """
    Étape 2: Analyse basée UNIQUEMENT sur l'onglet Matériaux
    Général et vendeur = valeurs par défaut
    """
    if not MODELS_READY:
        raise HTTPException(503, "Modèles non chargés")
    
    # Valeurs par défaut pour Général (montre "standard" milieu de gamme)
    complete = CompleteWatchInput(
        # Général (défauts)
        prix_achat=15000,  # Prix moyen par défaut
        marque=BrandEnum.OMEGA,  # Marque mid-tier par défaut
        etat=ConditionEnum.LIKE_NEW,
        mouvement=MovementEnum.AUTOMATIC,
        genre=GenderEnum.MEN,
        annee_production=2020,
        forme=ShapeEnum.ROUND,
        surface_cadran=40.0,
        contenu_livraison=DeliveryEnum.BOX_PAPERS,
        horizon_annees=3,
        # Matériaux (depuis input)
        materiau_boitier=data.materiau_boitier,
        materiau_bracelet=data.materiau_bracelet,
        verre=data.verre,
        cadran=data.cadran,
        couleur_bracelet=data.couleur_bracelet,
        disponibilite=data.disponibilite,
        # Vendeur (défauts)
        montres_vendues=50,
        annonces_actives=10,
        avis_vendeur=100,
        expedition_rapide=1,
        vendeur_confiance=1,
        ponctualite=1
    )
    
    result = evaluate_investment(complete)
    
    # Analyse spécifique matériaux
    premium_materials = []
    if data.materiau_boitier in [MaterialEnum.GOLD, MaterialEnum.PLATINUM, MaterialEnum.ROSE_GOLD, MaterialEnum.WHITE_GOLD]:
        premium_materials.append(f"Boîtier {data.materiau_boitier.value} (+valeur)")
    if data.materiau_bracelet in [MaterialEnum.GOLD, MaterialEnum.PLATINUM, MaterialEnum.ROSE_GOLD, MaterialEnum.WHITE_GOLD]:
        premium_materials.append(f"Bracelet {data.materiau_bracelet.value} (+valeur)")
    if data.verre == CrystalEnum.SAPPHIRE:
        premium_materials.append("Verre saphir (standard qualité)")
    
    return {
        "onglet": "materials",
        "analyse_basee_sur": "Matériaux uniquement",
        "precision": "Estimation matériaux isolés",
        "avertissement": "Caractéristiques générales et vendeur utilisent des valeurs par défaut",
        "prediction": {
            "prix_achat": result.prix_achat,
            "prix_futur_estime": result.prix_futur_estime,
            "plus_value": result.plus_value,
            "roi_percent": result.roi_percent,
            "roi_annualise": result.roi_annualise,
            "evaluation": result.evaluation_simple,
            "recommandation": result.recommandation,
            "confiance": "Moyenne - complétez l'onglet Vendeur pour plus de précision"
        },
        "analyse_materiaux": {
            "composition": {
                "boitier": data.materiau_boitier.value,
                "bracelet": data.materiau_bracelet.value,
                "verre": data.verre.value,
                "cadran": data.cadran.value
            },
            "qualite_percue": "Premium" if len(premium_materials) >= 2 else "Standard",
            "elements_premium": premium_materials if premium_materials else ["Acier standard"],
            "impact_estime": "Élevé" if len(premium_materials) >= 2 else "Modéré"
        },
        "hypotheses_utilisees": {
            "general": {
                "marque": "Omega (défaut)",
                "prix_achat": "15 000$ (défaut)",
                "annee": 2020
            },
            "vendeur": "Standard (défaut)"
        },
        "prochaine_etape": "Rendez-vous dans l'onglet Vendeur pour la prédiction finale"
    }

@app.post("/seller")
def analyze_seller(data: SellerTab):
    """
    Étape 3: Analyse basée UNIQUEMENT sur l'onglet Vendeur
    Général et matériaux = valeurs par défaut
    """
    if not MODELS_READY:
        raise HTTPException(503, "Modèles non chargés")
    
    # Valeurs par défaut pour Général et Matériaux
    complete = CompleteWatchInput(
        # Général (défauts)
        prix_achat=15000,
        marque=BrandEnum.OMEGA,
        etat=ConditionEnum.LIKE_NEW,
        mouvement=MovementEnum.AUTOMATIC,
        genre=GenderEnum.MEN,
        annee_production=2020,
        forme=ShapeEnum.ROUND,
        surface_cadran=40.0,
        contenu_livraison=DeliveryEnum.BOX_PAPERS,
        horizon_annees=3,
        # Matériaux (défauts)
        materiau_boitier=MaterialEnum.STEEL,
        materiau_bracelet=MaterialEnum.STEEL,
        verre=CrystalEnum.SAPPHIRE,
        cadran=ColorEnum.BLACK,
        couleur_bracelet=ColorEnum.SILVER,
        disponibilite=AvailabilityEnum.IN_STOCK,
        # Vendeur (depuis input)
        montres_vendues=data.montres_vendues,
        annonces_actives=data.annonces_actives,
        avis_vendeur=data.avis_vendeur,
        expedition_rapide=data.expedition_rapide,
        vendeur_confiance=data.vendeur_confiance,
        ponctualite=data.ponctualite
    )
    
    result = evaluate_investment(complete)
    
    # Analyse spécifique vendeur
    reputation_score = (data.expedition_rapide + data.vendeur_confiance + data.ponctualite) / 3
    activite_ratio = data.montres_vendues / max(data.annonces_actives, 1)
    
    qualite_vendeur = "Excellente" if reputation_score >= 0.9 and data.avis_vendeur > 200 else \
                     "Bonne" if reputation_score >= 0.7 and data.avis_vendeur > 50 else \
                     "Moyenne" if reputation_score >= 0.5 else "À vérifier"
    
    return {
        "onglet": "seller",
        "analyse_basee_sur": "Profil vendeur uniquement",
        "precision": "Analyse risque transactionnel",
        "avertissement": "Caractéristiques montre et matériaux utilisent des valeurs par défaut",
        "prediction": {
            "prix_achat": result.prix_achat,
            "prix_futur_estime": result.prix_futur_estime,
            "plus_value": result.plus_value,
            "roi_percent": result.roi_percent,
            "roi_annualise": result.roi_annualise,
            "evaluation": result.evaluation_simple,
            "recommandation": result.recommandation,
            "confiance": "Variable selon qualité vendeur"
        },
        "analyse_vendeur": {
            "profil": {
                "qualite": qualite_vendeur,
                "reputation_score": round(reputation_score * 100, 1),
                "nb_avis": data.avis_vendeur,
                "experience": "Confirmé" if data.montres_vendues > 100 else "Débutant"
            },
            "fiabilite": {
                "expedition_rapide": bool(data.expedition_rapide),
                "vendeur_verifie": bool(data.vendeur_confiance),
                "ponctuel": bool(data.ponctualite)
            },
            "activite": {
                "montres_vendues": data.montres_vendues,
                "annonces_actives": data.annonces_actives,
                "ratio_rotation": round(activite_ratio, 2)
            },
            "impact_transaction": "Faible risque" if qualite_vendeur == "Excellente" else "Risque modéré" if qualite_vendeur == "Bonne" else "Attention requise"
        },
        "hypotheses_utilisees": {
            "general": "Montre Omega 15k$ (défaut)",
            "materiaux": "Acier standard (défaut)"
        },
        "note_finale": "Pour une prédiction précise, utilisez /analyze avec toutes les données réelles"
    }

@app.post("/analyze", response_model=InvestmentResult)
def analyze_complete(data: CompleteWatchInput):
    """
    Analyse COMPLÈTE avec TOUS les onglets
    Prédiction finale précise - utilisez celui-ci pour le résultat définitif
    """
    try:
        result = evaluate_investment(data)
        return result
    except Exception as e:
        raise HTTPException(500, f"Erreur analyse: {str(e)}")
    

@app.post("/explain")
def explain_prediction(data: CompleteWatchInput):
    """
    Explications XAI complètes (SHAP + LIME) pour la prédiction
    """
    if not MODELS_READY:
        raise HTTPException(503, "Modèles non chargés")
    
    # Préparer les données
    X = prepare_data(data)
    
    # Convertir les types si nécessaire
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype(str)
    
    result = {
        "evaluation": {},
        "shap": {},
        "lime": {}
    }
    
    # Évaluation
    try:
        eval_result = evaluate_investment(data)
        result["evaluation"] = {
            "prix_achat": eval_result.prix_achat,
            "prix_futur": eval_result.prix_futur_estime,
            "roi": eval_result.roi_annualise,
            "recommandation": eval_result.recommandation,
            "evaluation_simple": eval_result.evaluation_simple
        }
    except Exception as e:
        result["evaluation_error"] = str(e)
    
    # SHAP avec le modèle de prix
    result["shap"] = generate_shap_explanation(X, price_model)
    
    # LIME avec le classifieur
    result["lime"] = generate_lime_explanation(X, clf_model, list(X.columns))
    
    return result
# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 API Montres de Luxe - Structure par Onglets v3.0")
    print("="*60)
    print("📚 Documentation: http://localhost:8000/docs")
    print("🔍 Health Check:  http://localhost:8000/health")
    print("\nEndpoints:")
    print("   POST /general    → Analyse onglet Général")
    print("   POST /materials  → Analyse onglet Matériaux (nécessite GeneralTab)")
    print("   POST /seller     → Analyse onglet Vendeur (nécessite GeneralTab)")
    print("   POST /analyze    → Analyse complète (tous onglets)")
    print("   GET  /options    → Options dropdowns")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)