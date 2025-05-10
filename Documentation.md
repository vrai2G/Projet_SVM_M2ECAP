
# Source des données

Cette étude repose sur les données du **Behavioral Risk Factor Surveillance System (BRFSS) 2015**, une enquête annuelle téléphonique menée par les **Centers for Disease Control and Prevention (CDC)**. Le BRFSS collecte plus de 400 000 réponses par an sur les comportements de santé, les maladies chroniques et l’accès aux soins.

* **Documentation** : *CDC BRFSS 2015 Codebook*
* **Jeu de données** : *Diabetes Health Indicators Dataset* sur Kaggle

---

## Contexte de l'étude

Le diabète est un problème de santé publique majeur aux États-Unis, avec des millions de personnes concernées et des coûts économiques importants. En 2018, 34,2 millions d’Américains étaient diabétiques et 88 millions prédiabétiques. Un grand nombre ignorent leur condition, ce qui complique la prévention. Des modèles prédictifs peuvent aider à cibler les populations à risque et à intervenir plus tôt.

---

## Données utilisées

Le fichier initial (`diabetes_binary_health_indicators_BRFSS2015.csv`) contient 253 680 observations mais a été remplacé pour des raisons techniques par une version prétraitée et équilibrée :
`diabetes_binary_5050split_health_indicators_BRFSS2015.csv`, avec une répartition équitable entre diabétiques/prédiabétiques (classe 1) et non-diabétiques (classe 0).

---

## Variables du modèle

### Variable cible

* **Diabetes\_binary** : 1 = diabétique ou pré-diabétique, 0 = non-diabétique

### Antécédents médicaux

* **HighBP** : Hypertension
* **HighChol** : Cholestérol élevé
* **Stroke** : Antécédent d'AVC
* **HeartDiseaseorAttack** : Antécédent de maladie cardiaque

### Comportements et habitudes de vie

* **Smoker** : A fumé ≥ 100 cigarettes
* **PhysActivity** : Activité physique (hors travail)
* **HvyAlcoholConsump** : Consommation excessive d'alcool
* **Fruits** : Consommation quotidienne de fruits
* **Veggies** : Consommation quotidienne de légumes

### Accès aux soins

* **AnyHealthcare** : Couverture santé
* **NoDocbcCost** : Renoncement aux soins pour raisons financières
* **CholCheck** : Test de cholestérol dans les 5 dernières années

### État de santé

* **GenHlth** : État de santé auto-évalué (1 = excellent à 5 = mauvais)
* **MentHlth** : Jours de mauvaise santé mentale
* **PhysHlth** : Jours de mauvaise santé physique
* **DiffWalk** : Difficulté à marcher

### Démographie

* **Sex** : Genre
* **Age** : Groupe d'âge
* **Education** : Niveau d'éducation
* **Income** : Revenu annuel
* **BMI** : Indice de masse corporelle




# Transformations : 

# Méthodologie et analyse des données

## Analyse exploratoire des données (EDA)

Notre projet a débuté par une phase d'analyse exploratoire approfondie pour comprendre la structure et les caractéristiques de notre jeu de données.

### Analyse préliminaire
Nous avons d'abord généré un rapport statistique complet grâce au package `ydata-profiling`, offrant une vision globale des propriétés de nos données : distribution des variables, corrélations, valeurs aberrantes et autres métriques descriptives.

### Examen de la qualité des données
L'analyse de la qualité des données a comporté plusieurs étapes :
- Vérification des valeurs manquantes
- Validation des types de données
- Suppression des observations dupliquées
- Identification des variables binaires (dummies) et non binaires (BMI, GenHlth, MentHlth, PhysHlth, Age, Education et Income)

### Analyse des distributions
Nous avons examiné attentivement :
- La distribution de chaque variable
- L'équilibre des variables binaires
- La distribution de notre target (Diabetes_binary)
- Les interactions entre la target et les variables explicatives via des analyses de corrélation et des analyses bivariées

## Transformation des données

Suite à notre EDA, nous avons identifié des déséquilibres significatifs dans certaines variables non binaires, ce qui nous a amenés à les recoder pour améliorer leur distribution et leur interprétabilité :

### Recodage de l'IMC (BMI)
La variable BMI présentait une distribution étalée avec des valeurs aberrantes significatives. Nous l'avons catégorisée en 6 niveaux selon les standards de l'OMS, avec un plafonnement des valeurs extrêmes :
```python
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 1    # Sous-poids
    elif bmi < 25:
        return 2    # Normal
    elif bmi < 30:
        return 3    # Surpoids
    elif bmi < 35:
        return 4    # Obésité classe I
    elif bmi < 40:
        return 5    # Obésité classe II
    else:
        return 6    # Obésité classe III
```

### Recodage des variables de santé mentale et physique
Les variables MentHlth et PhysHlth (nombre de jours de mauvaise santé au cours du dernier mois) présentaient des distributions très asymétriques. Nous les avons recodées en 4 catégories plus informatives :
```python
def categorize_health_days(days):
    if days == 0:
        return 0    # Aucun jour
    elif days <= 5:
        return 1    # Problèmes occasionnels
    elif days <= 15:
        return 2    # Problèmes fréquents
    else:
        return 3    # Problèmes chroniques
```

## Modélisation

Pour prédire le risque de diabète, nous avons opté pour deux algorithmes d'apprentissage automatique reconnus pour leur performance dans les tâches de classification :

### Modèles sélectionnés
- **Random Forest** : Robuste aux données bruitées et capable de capturer des relations non linéaires
- **XGBoost** : Réputé pour ses performances élevées dans les compétitions de machine learning

### Optimisation des hyperparamètres
Nous avons employé la méthode GridSearchCV pour identifier les paramètres optimaux de nos modèles. Cette approche, bien que computationnellement intensive, permet d'explorer systématiquement l'espace des hyperparamètres. Les contraintes matérielles ont toutefois limité l'étendue de notre recherche, en particulier compte tenu du nombre important de variables dans notre jeu de données.

### Modèles alternatifs
Nous avons également évalué des modèles de régression logistique et SVM (Support Vector Classification), mais leurs performances se sont avérées inférieures aux modèles retenus. Par ailleurs, nous avons testé nos modèles sur les données brutes (sans transformation), ce qui a confirmé l'utilité de nos transformations préalables.

### Sélection du modèle final
Bien que les performances des modèles Random Forest et XGBoost soient relativement similaires, nous avons finalement retenu le Random Forest comme modèle optimal. Ce choix a été motivé principalement par sa capacité à fournir des interprétations claires de l'importance des variables, facilitant ainsi l'identification des facteurs de risque les plus significatifs pour le diabète.

Des analyses plus détaillées et des interprétations approfondies sont disponibles dans les commentaires du notebook associé à ce projet.

Valentin GOTTARDINI & SINEAU Angel - M2 ECAP
