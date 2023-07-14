# Regression pour le jeu de données Prostate
  Le code en R présenté ici implémente différentes méthodes de régression pour analyser le jeu de données Prostate. Ce jeu de données contient des informations sur plusieurs variables prédictives liées au cancer de la prostate, ainsi qu'une variable cible représentant le logarithme de l'antigène spécifique de la prostate. 
  
  Les méthodes de régression mises en œuvre dans ce code comprennent :
  
  1. Méthode des moindres carrés ordinaires (MCO) : Cette méthode cherche à ajuster les coefficients de régression de manière à minimiser la somme des carrés des écarts entre les valeurs prédites et les valeurs réelles de la variable cible.
  2. Régression Ridge : Cette méthode ajoute une pénalité à la fonction de coût du MCO en utilisant une régularisation L2 (norme euclidienne). Cela permet de réduire les coefficients de régression et d'éviter le surajustement (overfitting) du modèle.
  3. Régression Lasso : Contrairement à la régression Ridge, la régression Lasso utilise une régularisation L1 (norme de Manhattan) pour pénaliser les coefficients de régression. Cela conduit à une sélection de variables plus robuste et à une possible réduction de certaines variables prédictives.
  4. Elastic Net : L'Elastic Net est une combinaison de la régression Ridge et de la régression Lasso. Elle utilise à la fois une régularisation L1 et L2 pour obtenir un équilibre entre la sélection de variables et la stabilité des coefficients de régression.
  5. Régression en composantes principales (CP) : Cette méthode réduit la dimensionnalité des données en projetant les variables prédictives sur un espace de variables orthogonales appelé "composantes principales". Ensuite, une régression ordinaire est effectuée sur ces composantes principales pour obtenir les coefficients de régression.

## Description du jeu de données
Le jeu de données Prostate contient les variables prédictives suivantes :

  - lcavol : log(volume du cancer)
  - lweight : log(poids de la prostate)
  - age
  - lbph : log(quantité d'hyperplasie bénigne de la prostate)
  - svi : invasion des vésicules séminales
  - lcp : log(pénétration capsulaire)
  - gleason : score de Gleason
  - pgg45 : pourcentage score de Gleason 4 ou 5

La variable cible est :

  - lpsa : log(antigène spécifique de la prostate)

## Installation
  1. Assure-toi d'avoir R et RStudio installés sur ton ordinateur.
  2. Télécharge le fichier contenant le code et le jeu de données Prostate.
  3. Ouvre le fichier dans RStudio.

## Utilisation
Le code est organisé en différentes sections pour chaque méthode de régression implémentée : MCO (Méthode des moindres carrés ordinaires), Ridge, Lasso, Elastic Net et Régression en composantes principales (CP).

Chaque section du code contient les étapes suivantes :

  1. Chargement du jeu de données Prostate.
  2. Division du jeu de données en ensemble d'entraînement (train set) et ensemble de test (test set).
  3. Préparation des données d'entraînement et de test en extrayant les variables prédictives appropriées.
  4. Application de la méthode de régression spécifique aux données d'entraînement pour obtenir les coefficients de régression.
  5. Calcul de l'erreur quadratique moyenne de prédiction sur les données de test.
   
Assure-toi d'exécuter chaque section du code séparément pour obtenir les résultats correspondants à chaque méthode de régression.
