# Projet_Investissement_Montres

## PROJET : SYSTÈME D’AIDE À L’INVESTISSEMENT DANS LES MONTRES DE LUXE
**Groupe :** Siwar Taghouti et Maryem Jbeli  
**Technologies :** Python, Machine Learning, MLflow, FastAPI, React, Docker, XAI (LIME & SHAP)

---
# Columns description:
**Brand**: The manufacturer or brand name of the watch (e.g., Alpina, Audemars Piguet)

**Movement**: The mechanism type used in the watch (e.g., Quartz, Automatic).

**Case Material**: The material of the watch's casing (e.g., Steel, Titanium).

**Bracelet Material**: The material of the bracelet or strap (e.g., Leather, Rubber).

**Year of Production**: The year the watch was produced (numeric).

**Condition**: The state of the watch (e.g., New, Used, Like new & unworn).

**Scope of Delivery**: Details about the delivery package (e.g., "Original box, original papers").

**Gender**: Target audience for the watch (e.g., Men's watch, Women's watch, Unisex).

**Price**: The price of the watch in $.

**Availability**: Availability status (e.g., "Item is in stock").

**Shape**: Shape of the watch's case (e.g., Round, Rectangular).

**Face Area**: Numeric area of the watch face in square units.

**Water Resistance**: Water resistance rating in meters.

**Crystal**: Material of the watch crystal (e.g., Sapphire crystal).

**Dial**: Color or design of the watch dial (e.g., Black, Grey, Champagne).

**Bracelet Color**: The color of the bracelet (e.g., Black, Brown).

**Clasp**: The type of clasp used in the bracelet (e.g., Buckle, Fold clasp).

**Watches Sold by the Seller**: The total number of watches sold by the seller.

**Active Listing of the Seller**: The number of watches currently listed by the seller.

**Fast Shipper**: Indicator if the seller ships quickly (binary: 1 = Yes, 0 = No).

**Trusted Seller**: Indicator if the seller is marked as trusted (binary: 1 = Yes, 0 = No).

**Punctuality**: Punctuality score of the seller (binary: 1 = Yes, 0 = No).

**Seller Reviews**: Total number of reviews for the seller.


## IDÉE GLOBALE DU PROJET
Les montres de luxe (Rolex, Omega, Patek Philippe…) sont aujourd’hui considérées comme des **actifs d’investissement alternatifs**.

**Objectif :** Développer une application intelligente qui aide un utilisateur à décider s’il doit acheter une montre comme investissement, en s’appuyant sur :  

- Des données historiques (`watches.csv`)  
- Des modèles de Machine Learning  
- Un suivi et une gestion des expériences ML avec **MLflow**  
- Des méthodes d’explicabilité (XAI) à l’aide de **SHAP** et **LIME**

---

## CÔTÉ UTILISATEUR

### Cas d’usage principal
> Je veux acheter une montre de luxe et savoir si c’est un bon investissement.

### Interaction utilisateur
1. L’utilisateur accède à l’application web.  
2. Il saisit les caractéristiques de la montre :  
   - Marque (Rolex, Omega…)  
   - Modèle  
   - Année de production  
   - Prix d’achat  
   - État  
3. Il clique sur le bouton **« Analyser »**.

### Résultats affichés
- **Prix futur estimé** (Régression)  
- **ROI (%)**  
- **Classe d’investissement** (Classification) définie à partir du ROI prédit :  
  - Bon investissement  
  - Moyen  
  - Risqué  
- **Explication de la prédiction (XAI)** :  
  - **SHAP** : identification des variables les plus influentes sur les prédictions globales  
  - **LIME** : justification de la décision pour une montre spécifique analysée par l’utilisateur

---
