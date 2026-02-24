"""
API FastAPI - Système d'Investissement Montres de Luxe
VERSION INVESTISSEUR - Prix d'achat connu
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Imports optionnels XAI
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

# Charger config
from setup import PROCESSED_DATA_DIR, CLEANED_CSV

# ============================================================
# CONFIGURATION
# ============================================================

class InvestmentThreshold:
    """Seuils pour la classification d'investissement"""
    ROI_EXCELLENT = 25.0  # %
    ROI_GOOD = 15.0       # %
    ROI_ACCEPTABLE = 5.0  # %
    HOLD_YEARS = 3        # Horizon d'investissement par défaut

# ============================================================
# APP FASTAPI
# ============================================================

app = FastAPI(
    title="API Investissement Montres de Luxe",
    description="Évalue si un achat à prix connu est un bon investissement",
    version="2.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MODÈLES DE DONNÉES
# ============================================================

class WatchInput(BaseModel):
    """Données de la montre avec prix d'achat connu"""
    model_config = ConfigDict(
        populate_by_name=True,
        extra='ignore',
        validate_assignment=True
    )
    
    # Prix d'achat (OBLIGATOIRE)
    prix_achat: float = Field(
        ..., 
        description="Prix d'achat connu (€)", 
        gt=0,
        json_schema_extra={"example": 8500.0}
    )
    horizon_annees: int = Field(
        default=3, 
        description="Horizon de revente en années", 
        ge=1, 
        le=10
    )
    
    # Caractéristiques de la montre
    Brand: str = Field(default="Rolex", description="Marque de la montre")
    Movement: str = Field(default="Automatic", description="Mouvement")
    Case_material: Optional[str] = Field(default="Steel", alias="Case material")
    Bracelet_material: Optional[str] = Field(default="Steel", alias="Bracelet material")
    Year_of_production: int = Field(default=2020, alias="Year of production", ge=1900, le=2030)
    Condition: Optional[str] = Field(default="Used (Very good)")
    Scope_of_delivery: Optional[str] = Field(default="Original box, original papers", alias="Scope of delivery")
    Gender: Optional[str] = Field(default="Men's watch/Unisex")
    Availability: Optional[str] = Field(default="Item is in stock")
    Shape: Optional[str] = Field(default="Circular")
    Face_Area: Optional[float] = Field(default=650.0, alias="Face Area")
    Crystal: Optional[str] = Field(default="Sapphire crystal")
    Dial: Optional[str] = Field(default="Black")
    Bracelet_color: Optional[str] = Field(default="Steel", alias="Bracelet color")
    
    # Info vendeur
    Watches_Sold_by_the_Seller: Optional[int] = Field(default=500, alias="Watches Sold by the Seller")
    Active_listing_of_the_seller: Optional[int] = Field(default=100, alias="Active listing of the seller")
    Fast_Shipper: Optional[int] = Field(default=1, alias="Fast Shipper")
    Trusted_Seller: Optional[int] = Field(default=1, alias="Trusted Seller")
    Punctuality: Optional[int] = Field(default=1)
    Seller_Reviews: Optional[int] = Field(default=400, alias="Seller Reviews")

    @field_validator('prix_achat', mode='before')
    @classmethod
    def validate_prix_achat(cls, v):
        """Validation stricte du prix d'achat"""
        if v is None:
            raise ValueError("Le prix d'achat est obligatoire")
        try:
            val = float(v)
            if val <= 0:
                raise ValueError(f"Le prix d'achat doit être > 0, reçu: {val}")
            if val < 100:
                print(f"⚠️ Attention: prix d'achat très bas détecté: {val}")
            return val
        except (TypeError, ValueError) as e:
            raise ValueError(f"Prix d'achat invalide: {v} ({type(v).__name__})")

class InvestmentEvaluation(BaseModel):
    """Résultat d'évaluation d'investissement"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "prix_achat": 8500.0,
        "prix_futur_estime": 12000.0,
        "plus_value": 3500.0,
        "roi_percent": 41.18,
        "roi_annualise": 13.73,
        "horizon_annees": 3,
        "recommandation": "BON INVESTISSEMENT",
        "confiance": "Moyenne à Élevée",
        "risque": "Modéré",
        "evaluation_simple": "Bon",
        "details": {}
    }})
    
    prix_achat: float
    prix_futur_estime: float
    plus_value: float
    roi_percent: float
    roi_annualise: float
    horizon_annees: int
    recommandation: str
    confiance: str
    risque: str
    evaluation_simple: str
    details: Dict[str, Any]

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

# Features utilisées
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

def estimate_current_market_price(brand: str, age: int, condition: str = "Used (Very good)") -> float:
    """Estime le prix actuel du marché"""
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
        "New": 1.0, "Unworn": 0.95, "Used (Mint)": 0.90,
        "Used (Very good)": 0.80, "Used (Good)": 0.65,
        "Used (Fair)": 0.50, "Used (Poor)": 0.35
    }
    condition_factor = condition_factors.get(condition, 0.70)
    
    brand_premium = 1.2 if brand in ["Rolex", "Patek Philippe", "Audemars Piguet"] else \
                    0.7 if brand in ["Seiko", "Casio", "Citizen"] else 1.0
    
    return max(100, base * age_factor * condition_factor * brand_premium)

def prepare_data(watch: WatchInput) -> pd.DataFrame:
    """Prépare les features pour le modèle"""
    data = watch.model_dump(by_alias=True)
    
    current_year = 2026
    data["age"] = current_year - data["Year of production"]
    data["is_modern"] = 1 if data["Year of production"] >= 2000 else 0
    
    price_market = estimate_current_market_price(
        watch.Brand, 
        data["age"], 
        watch.Condition or "Used (Very good)"
    )
    data["current_price_estimate"] = price_market

    scores = [
        data.get("Fast Shipper", 0) or 0, 
        data.get("Trusted Seller", 0) or 0, 
        data.get("Punctuality", 0) or 0
    ]
    data["seller_reputation_score"] = np.mean(scores)
    
    scope_map = {
        'Original box, original papers': 3,
        'Original box, no original papers': 2,
        'Original papers, no original box': 1,
        'No original box, no original papers': 0
    }
    data["scope_score"] = scope_map.get(data.get("Scope of delivery"), 0)
    
    sold = data.get("Watches Sold by the Seller", 0) or 0
    active = data.get("Active listing of the seller", 0) or 0
    data["seller_consistent"] = 1 if sold >= active else 0
    data["price_anomaly_low"] = 0
    data["price_anomaly_high"] = 0
    
    df = pd.DataFrame([data])
    
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0
    
    return df[FEATURES]

def evaluate_investment(watch: WatchInput) -> InvestmentEvaluation:
    """Évalue si l'achat à prix_achat est un bon investissement"""
    if not MODELS_READY:
        raise HTTPException(503, "Modèles non chargés")
    
    prix_achat = watch.prix_achat
    
    if prix_achat is None or not isinstance(prix_achat, (int, float)):
        raise HTTPException(422, f"Prix d'achat invalide (type: {type(prix_achat)})")
    
    if prix_achat <= 0:
        raise HTTPException(422, f"Prix d'achat doit être > 0, reçu: {prix_achat}")
    
    # Prédiction du prix futur
    X = prepare_data(watch)
    price_log = price_model.predict(X)[0]
    prix_futur = float(np.expm1(price_log))
    
    # Calculs financiers
    plus_value = prix_futur - prix_achat
    roi_total = ((prix_futur - prix_achat) / prix_achat) * 100 if prix_achat > 0 else 0
    roi_annualise = roi_total / watch.horizon_annees if watch.horizon_annees > 0 else 0
    
    # Classification du risque
    proba = clf_model.predict_proba(X)[0]
    risque_score = proba[0]
    
    # Détermination de l'évaluation
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
    
    prix_marche = estimate_current_market_price(
        watch.Brand, 
        int(X["age"].iloc[0]), 
        watch.Condition or "Used (Very good)"
    )
    
    if prix_achat < prix_marche * 0.9:
        deal_quality = "Excellente affaire (sous le marché)"
    elif prix_achat < prix_marche * 1.05:
        deal_quality = "Prix du marché"
    else:
        deal_quality = "Cher (au-dessus du marché)"
    
    details = {
        "prix_marche_actuel": round(prix_marche, 2),
        "difference_avec_marche": round(prix_achat - prix_marche, 2),
        "qualite_affaire": deal_quality,
        "plus_value_brute": round(plus_value, 2),
        "rentabilite_annualisee": round(roi_annualise, 2),
        "risque_modele": round(float(risque_score), 3),
        "probabilites_classe": {
            "risque": round(float(proba[0]), 3),
            "moyen": round(float(proba[1]), 3),
            "bon": round(float(proba[2]), 3)
        },
        "seuils_utilises": {
            "excellent": InvestmentThreshold.ROI_EXCELLENT,
            "bon": InvestmentThreshold.ROI_GOOD,
            "acceptable": InvestmentThreshold.ROI_ACCEPTABLE
        }
    }
    
    return InvestmentEvaluation(
        prix_achat=float(prix_achat),
        prix_futur_estime=round(prix_futur, 2),
        plus_value=round(plus_value, 2),
        roi_percent=round(roi_total, 2),
        roi_annualise=round(roi_annualise, 2),
        horizon_annees=watch.horizon_annees,
        recommandation=recommandation,
        confiance=confiance,
        risque=risque,
        evaluation_simple=evaluation_simple,
        details=details
    )

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    """Vérifie que l'API fonctionne"""
    return {
        "status": "ok" if MODELS_READY else "error",
        "version": "2.2-investisseur",
        "models": {
            "price": price_model is not None,
            "classifier": clf_model is not None,
            "shap": SHAP_OK,
            "lime": LIME_OK
        },
        "time": datetime.now().isoformat()
    }

@app.post("/evaluate", response_model=InvestmentEvaluation)
def evaluate_endpoint(watch: WatchInput):
    """
    Endpoint principal : Évalue un investissement avec prix d'achat connu
    """
    try:
        result = evaluate_investment(watch)
        return result
    except Exception as e:
        print(f"❌ ERREUR dans /evaluate: {str(e)}")
        raise HTTPException(500, f"Erreur interne: {str(e)}")

@app.post("/compare")
def compare_scenarios(watch: WatchInput, prix_alternatifs: List[float] = Query(..., description="Liste de prix d'achat à comparer")):
    """Compare plusieurs scénarios de prix d'achat pour la même montre"""
    if not prix_alternatifs:
        raise HTTPException(422, "Fournir au moins un prix alternatif")
    
    scenarios = []
    main_eval = evaluate_investment(watch)
    scenarios.append({
        "scenario": "Votre prix",
        "prix_achat": main_eval.prix_achat,
        "prix_futur": main_eval.prix_futur_estime,
        "roi_annualise": main_eval.roi_annualise,
        "evaluation": main_eval.evaluation_simple,
        "recommandation": main_eval.recommandation
    })
    
    for prix in prix_alternatifs:
        watch_alt = watch.model_copy(update={"prix_achat": prix})
        eval_alt = evaluate_investment(watch_alt)
        scenarios.append({
            "scenario": f"Alternative {prix}€",
            "prix_achat": prix,
            "prix_futur": eval_alt.prix_futur_estime,
            "roi_annualise": eval_alt.roi_annualise,
            "evaluation": eval_alt.evaluation_simple,
            "recommandation": eval_alt.recommandation
        })
    
    scenarios.sort(key=lambda x: x["roi_annualise"], reverse=True)
    
    return {
        "meilleur_scenario": scenarios[0],
        "comparaison": scenarios,
        "economie_potentielle": round(scenarios[0]["prix_achat"] - main_eval.prix_achat, 2) if scenarios[0]["prix_achat"] != main_eval.prix_achat else 0
    }

@app.post("/investment_report")
def detailed_report(watch: WatchInput):
    """Rapport d'investissement complet avec explications"""
    evaluation = evaluate_investment(watch)
    X = prepare_data(watch)
    
    report = {
        "evaluation": {
            "prix_achat": evaluation.prix_achat,
            "prix_futur_estime": evaluation.prix_futur_estime,
            "roi_annualise": evaluation.roi_annualise,
            "evaluation_simple": evaluation.evaluation_simple,
            "recommandation": evaluation.recommandation,
            "risque": evaluation.risque
        },
        "analyse": {
            "resume": f"En achetant cette montre {watch.Brand} à {evaluation.prix_achat}€, "
                     f"vous pouvez espérer la revendre {evaluation.prix_futur_estime}€ "
                     f"dans {evaluation.horizon_annees} ans.",
            "rentabilite": f"ROI total: {evaluation.roi_percent}% | "
                          f"ROI annualisé: {evaluation.roi_annualise}%",
            "verdict": evaluation.evaluation_simple,
            "conseil": ""
        },
        "facteurs_cles": []
    }
    
    if evaluation.roi_annualise >= InvestmentThreshold.ROI_EXCELLENT:
        report["analyse"]["conseil"] = "C'est une opportunité à saisir rapidement."
    elif evaluation.roi_annualise >= InvestmentThreshold.ROI_GOOD:
        report["analyse"]["conseil"] = "C'est un achat solide, procédez si vous avez les fonds."
    elif evaluation.roi_annualise > 0:
        report["analyse"]["conseil"] = "Rentabilité faible, envisagez de négocier le prix."
    else:
        report["analyse"]["conseil"] = "Ne procédez pas à cet achat ou négociez fortement."
    
    age = int(X["age"].iloc[0])
    report["facteurs_cles"] = [
        {"facteur": "Âge de la montre", "valeur": f"{age} ans", "impact": "Négatif" if age > 20 else "Positif"},
        {"facteur": "Marque", "valeur": watch.Brand, "impact": "Très Positif" if watch.Brand in ["Rolex", "Patek Philippe", "Audemars Piguet"] else "Neutre"},
        {"facteur": "État", "valeur": watch.Condition or "Non spécifié", "impact": "Positif" if "Good" in (watch.Condition or "") else "Variable"},
        {"facteur": "Boîte/Papiers", "valeur": watch.Scope_of_delivery or "Non spécifié", "impact": "Très Positif" if "box, original papers" in (watch.Scope_of_delivery or "") else "Négatif"}
    ]
    
    return report

@app.post("/explain")
def explain(watch: WatchInput):
    """Explique la prédiction avec SHAP et LIME"""
    if not MODELS_READY:
        raise HTTPException(503, "Modèles non chargés")
    
    evaluation = evaluate_investment(watch)
    X = prepare_data(watch)
    
    result = {
        "evaluation": {
            "prix_achat": evaluation.prix_achat,
            "prix_futur_estime": evaluation.prix_futur_estime,
            "roi_annualise": evaluation.roi_annualise,
            "evaluation_simple": evaluation.evaluation_simple,
            "recommandation": evaluation.recommandation
        },
        "input_features": X.iloc[0].to_dict(),
        "explanations": {}
    }
    
    # SHAP pour le prix futur
    if SHAP_OK and price_model:
        try:
            if hasattr(price_model, 'named_steps'):
                preprocessor = None
                model = None
                for step_name in ['preprocessing', 'preprocessor', 'prep', 'transform']:
                    if step_name in price_model.named_steps:
                        preprocessor = price_model.named_steps[step_name]
                        break
                
                for step_name in ['model', 'regressor', 'estimator', 'clf', 'rf', 'xgb', 'lgb']:
                    if step_name in price_model.named_steps:
                        model = price_model.named_steps[step_name]
                        break
                
                if preprocessor and model:
                    X_proc = preprocessor.transform(X)
                    X_proc_dense = X_proc.toarray() if hasattr(X_proc, 'toarray') else np.array(X_proc)
                elif model:
                    X_proc_dense = X.values
                    model = price_model
                else:
                    raise ValueError("Impossible d'extraire le modèle du pipeline")
            else:
                preprocessor = None
                model = price_model
                X_proc_dense = X.values
            
            if hasattr(model, 'tree_') or "Tree" in str(type(model)) or "XGB" in str(type(model)) or "LGBM" in str(type(model)):
                explainer = shap.TreeExplainer(model)
            else:
                if data_sample is not None and len(data_sample) > 10:
                    bg = data_sample.sample(min(100, len(data_sample)))[FEATURES]
                    explainer = shap.KernelExplainer(model.predict, bg.values)
                else:
                    explainer = shap.KernelExplainer(model.predict, X_proc_dense[:5])
            
            shap_vals = explainer.shap_values(X_proc_dense)
            
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            
            vals = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
            
            feature_importance = []
            feature_names = FEATURES[:len(vals)] if len(vals) <= len(FEATURES) else FEATURES
            
            for i, feat in enumerate(feature_names):
                if i < len(vals):
                    feature_importance.append({
                        "feature": feat,
                        "impact_prix": round(float(vals[i]), 2),
                        "direction": "augmente" if vals[i] > 0 else "diminue"
                    })
            
            feature_importance.sort(key=lambda x: abs(x["impact_prix"]), reverse=True)
            result["explanations"]["shap_top_features"] = feature_importance[:10]
            result["explanations"]["shap_base_value"] = float(explainer.expected_value) if hasattr(explainer, 'expected_value') else None
            
        except Exception as e:
            result["explanations"]["shap_error"] = str(e)
            result["explanations"]["shap_available"] = False
    else:
        result["explanations"]["shap_available"] = SHAP_OK
        if not SHAP_OK:
            result["explanations"]["shap_error"] = "Bibliothèque SHAP non installée"
    
    # LIME pour la classification
    if LIME_OK and clf_model and data_sample is not None:
        try:
            bg = data_sample[FEATURES].copy()
            
            cat_cols = bg.select_dtypes(include=['object']).columns
            encoders = {}
            for col in cat_cols:
                le = LabelEncoder()
                bg[col] = le.fit_transform(bg[col].astype(str))
                encoders[col] = le
            
            explainer = LimeTabularExplainer(
                bg.values,
                feature_names=FEATURES,
                class_names=['Risqué', 'Moyen', 'Bon investissement'],
                mode='classification',
                discretize_continuous=True,
                sample_around_instance=True
            )
            
            X_enc = X.copy()
            for col, le in encoders.items():
                if col in X_enc.columns:
                    val = X_enc[col].iloc[0]
                    try:
                        if str(val) in le.classes_:
                            X_enc[col] = le.transform([str(val)])[0]
                        else:
                            X_enc[col] = 0
                    except:
                        X_enc[col] = 0
            
            def predict_fn(x):
                df = pd.DataFrame(x, columns=FEATURES)
                for col, le in encoders.items():
                    if col in df.columns:
                        try:
                            vals = df[col].astype(int).clip(0, len(le.classes_)-1)
                            df[col] = le.inverse_transform(vals)
                        except:
                            df[col] = "Unknown"
                return clf_model.predict_proba(df)
            
            exp = explainer.explain_instance(
                X_enc.values[0], 
                predict_fn, 
                num_features=6,
                top_labels=1
            )
            
            label = exp.available_labels()[0]
            class_names = ['Risqué', 'Moyen', 'Bon investissement']
            
            result["explanations"]["lime"] = {
                "disponible": True,
                "classe_predite": class_names[label] if label < len(class_names) else str(label),
                "confiance": round(float(exp.score), 3),
                "facteurs_decisifs": [
                    {
                        "facteur": f,
                        "contribution": round(float(w), 4),
                        "sens": "favorise l'investissement" if w > 0 else "déconseille l'investissement"
                    } for f, w in exp.as_list(label)
                ]
            }
            
        except Exception as e:
            result["explanations"]["lime"] = {
                "disponible": False,
                "error": str(e)
            }
    else:
        result["explanations"]["lime"] = {
            "disponible": False,
            "reason": "LIME non disponible" if not LIME_OK else "Données d'entraînement manquantes"
        }
    
    return result

# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 API Investissement Montres de Luxe v2.2")
    print("="*60)
    print("📚 Documentation: http://localhost:8000/docs")
    print("🔍 Health Check:  http://localhost:8000/health")
    print("💡 Endpoints principaux:")
    print("   POST /evaluate       - Évaluer un investissement")
    print("   POST /compare        - Comparer plusieurs prix")
    print("   POST /explain        - Explications SHAP + LIME")
    print("   POST /investment_report - Rapport complet")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)