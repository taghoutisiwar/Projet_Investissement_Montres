# Projet_Investissement_Montres

## PROJET : SYSTÈME D’AIDE À L’INVESTISSEMENT DANS LES MONTRES DE LUXE
**Groupe :** Siwar Taghouti et Maryem Jbeli  
**Technologies :** Python, Machine Learning, MLflow, FastAPI, React, Docker, XAI (LIME & SHAP)

---

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
