Source des données
Cette étude s'appuie sur les données du Behavioral Risk Factor Surveillance System (BRFSS) de 2015, une enquête téléphonique annuelle sur la santé menée par les Centers for Disease Control and Prevention (CDC). Le BRFSS collecte chaque année plus de 400 000 réponses d'Américains concernant leurs comportements à risque liés à la santé, leurs conditions médicales chroniques et leur utilisation des services préventifs. Cette enquête est réalisée depuis 1984, constituant ainsi une source fiable et longitudinale de données de santé publique.
•	Documentation officielle : CDC BRFSS 2015 Codebook
•	Jeu de données : Diabetes Health Indicators Dataset sur Kaggle
Contexte de l'étude

Le diabète constitue un enjeu majeur de santé publique aux États-Unis, touchant des millions d'Américains et générant un impact économique considérable. Cette maladie chronique, caractérisée par une incapacité à réguler correctement le taux de glucose sanguin, peut entraîner de graves complications : maladies cardiovasculaires, pertes de vision, amputations et insuffisances rénales.

Selon les Centers for Disease Control and Prevention (CDC), 34,2 millions d'Américains souffraient de diabète en 2018, et 88 millions présentaient un prédiabète. Plus alarmant encore, environ 20% des personnes diabétiques et 80% des personnes prédiabétiques ignorent leur condition. Le diabète de type II, forme la plus répandue, présente une prévalence variable selon différents déterminants sociaux de santé (âge, éducation, revenu, localisation géographique et origine ethnique). Les coûts associés sont estimés à 327 milliards de dollars pour les cas diagnostiqués, approchant 400 milliards si l'on inclut les cas non diagnostiqués et prédiabétiques.

Un diagnostic précoce permettant d'initier des changements de mode de vie et des traitements adaptés reste essentiel pour limiter les conséquences de cette maladie. C'est pourquoi le développement de modèles prédictifs du risque diabétique constitue un outil précieux pour les acteurs de santé publique.

Pour cette étude, nous avons initialement souhaité utiliser le jeu de données complet (diabetes_binary_health_indicators_BRFSS2015.csv) contenant 253 680 observations, malgré son déséquilibre de classes, en prévoyant de le rééquilibrer via des algorithmes comme SMOTE (Synthetic Minority Over-sampling Technique). Cependant, les contraintes techniques liées au volume important de données nous ont conduits à utiliser la version prétraitée et équilibrée (diabetes_binary_5050split_health_indicators_BRFSS2015.csv) , offrant une répartition égale entre individus diabétiques/prédiabétiques et non-diabétiques.

Description du jeu de données :
La variable cible, Diabetes_binary, comporte deux classes : 0 pour l'absence de diabète, et 1 pour la présence de prédiabète ou de diabète. Ce jeu de données comprend 21 variables explicatives et présente l'avantage d'être équilibré, facilitant ainsi l'entraînement des modèles de classification.

Variables du modèle
. Ces variables peuvent être regroupées en plusieurs catégories :
Variable cible
•	Diabetes_binary : Indique si la personne est diabétique ou pré-diabétique (1) ou non (0)
Antécédents médicaux
•	HighBP : Présence d'hypertension artérielle
•	HighChol : Taux de cholestérol élevé
•	Stroke : Antécédent d'accident vasculaire cérébral (AVC)
•	HeartDiseaseorAttack : Antécédent de maladie cardiaque ou d'infarctus
Comportements et habitudes de vie
•	Smoker : Personne ayant fumé au moins 100 cigarettes au cours de sa vie
•	PhysActivity : Pratique d'une activité physique au cours des 30 derniers jours (hors activité professionnelle)
•	HvyAlcoholConsump : Consommation excessive d'alcool
•	Fruits : Consommation quotidienne de fruits
•	Veggies : Consommation quotidienne de légumes
Accès aux soins de santé
•	AnyHealthcare : Bénéficie d'une couverture d'assurance maladie
•	NoDocbcCost : Renonciation à une consultation médicale pour des raisons financières
•	CholCheck : Réalisation d'un test de cholestérol au cours des 5 dernières années
État de santé général
•	GenHlth : Auto-évaluation de l'état de santé général (échelle de 1 à 5, où 1=excellent et 5=mauvais)
•	MentHlth : Nombre de jours de mauvaise santé mentale au cours des 30 derniers jours
•	PhysHlth : Nombre de jours de mauvaise santé physique au cours des 30 derniers jours
•	DiffWalk : Difficulté à marcher ou à monter des escaliers
Caractéristiques démographiques
•	Sex : Genre du répondant
•	Age : Groupe d'âge (de 18-24 ans à 80 ans et plus)
•	Education : Niveau d'éducation atteint
•	Income : Tranche de revenu annuel du ménage
•	BMI : Indice de masse corporelle
Ce jeu de données complet nous permet d'examiner l'influence de divers facteurs médicaux, comportementaux, socio-économiques et démographiques sur le risque de développer un diabète ou un prédiabète.


Transformations : 
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
