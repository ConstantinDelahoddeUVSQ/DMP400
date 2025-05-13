<h1><center>Guide</center></h1>


Notre objectif est de coder chaque partie du SIMS indépendamment l'une de l'autre. Dans cette mesure le projet est divisé en 2 parties (ou plus) indépendantes (les 2 parties du SIMS que l'on peut retrouver dans le diagramme : Violette (déviation électrique) et Verte (déviation magnétique). Nous pourrons également faire en option le reste des parties privées de la partie rose), et les utilisateurs pourront aborder le projet dans l'ordre souhaité.

On retrouvera ici la structure [structure.ipynb](./structure.ipynb) et ici les objectifs du projet [objectifs.ipynb](./objectifs.ipynb)<br><br>

## [SIMS](./SIMS)
 - ### [main](./SIMS/main.py)
    - Ce fichier est le fichier principal du projet. Il constitue l'interface principale permettant de naviguer à travers les différentes fonctionalités du projet.
 - ### [deviation_magnetique](./SIMS/deviation_magnetique)
    - #### [Equations](./SIMS/deviation_magnetique/Equations)
        - On y retrouve 2 fichiers : <br>
            - [Schéma_partie_verte](./SIMS/deviation_magnetique/Equations/Schéma_partie_verte.png) : Schéma de la partie de déviation magnétique<br>
            - [Trajectoire_champ_magnetique](./SIMS/deviation_magnetique/Equations/Trajectoire_champ_magnétique.ipynb) : Ce fichier nous guide à travers le raisonnement qui nous a mené jusqu'à l'équation de la trajectoire d'une particule qui traversait le champ magnétique de la partie verte du SIMS.<br><br><br>
    - #### [Code](./SIMS/deviation_magnetique/Code)
        - On y retrouve 1 fichier :<br>
            - [partie_electroaimant](./SIMS/deviation_magnetique/Code/partie_electroaimant.py) : Ce fichier est celui sur lequel on retrouve tout le code nécessaire pour remplir le premier objectif. <br><br><br>

 - ### [deviation_electrique](./SIMS/deviation_electrique)
    - #### [Equations](./SIMS/deviation_electrique/Equations)
    - On y retrouve 4 fichiers : <br>
        - [Schéma_partie_bleue](./SIMS/deviation_electrique/Equations/schema_partie_bleue.png) : Schéma de la partie de déviation électrique (on considère que la partie bleue est prolongée jusqu'au contact de l'échantillon)<br>
        - [Schéma_racine](./SIMS/deviation_electrique/Equations/schema_racine.png) : Schéma justifiant une décision lors d'une équation<br>
        - [Calcul_angle_incident](./SIMS/deviation_electrique/Equations/Calcul_angle_incident.ipynb) : Ce fichier nous guide à travers le raisonnement qui nous a mené jusqu'à l'équation nous permettant de calculer l'angle incident.<br>
        - [Calcul_champ_électrique](./SIMS/deviation_electrique/Equations/Calcul_champ_électrique.ipynb) : Ce fichier nous guide à travers le raisonnement qui nous a mené jusqu'à l'élaboration des équations d'une particule qui traverse le champ électrique de la partie violette du SIMS.<br><br><br>
    - #### [Code](./SIMS/deviation_electrique/Code)
        - On y retrouve 2 fichiers : <br>
            - [deviation](./SIMS/deviation_electrique/Code/deviation.py) : Ce fichier est celui sur lequel on retrouve le code nécessaire pour remplir le second objectif. <br>
            - [incertitude](./SIMS/deviation_electrique/Code/incertitude.py) : Ce fichier est celui sur lequel on retrouve le code nécéssaire pour calculer l'incertitude sur le point de contact (Objectif Bonus)<br><br><br>


## [Vérifications_Calculs](./Vérifications_Calculs)<br>
 - On y retrouve 3 fichiers : <br>
    - [Vérification_delta_xs](./Vérifications_Calculs/Vérification_delta_xs.ipynb) : Ce fichier est celui où nous verifions manuellement que le delta xs trouvé dans l'interface est correct <br>
    - [Vérification_xs](./Vérifications_Calculs/Vérification_xs.py) : Ce fichier est celui où nous verifions manuellement que le xs trouvé dans la partie déviation est correct <br>
    - [Vérification_champ_magnetique](./Vérifications_Calculs/Vérification_champ_magnetique.py) : Ce fichier est celui où nous verifions manuellement que la trajectoire trouvée dans la partie magnétique est correcte<br><br><br>

## [Procédures_test](./Procédures_test)<br>
 - On y retrouve 4 fichiers : <br>
    - [Procédures_test_Main](./Procédures_test/Procédures_test_Main.ipynb) : Ce fichier est celui où nous donnons les procédures de test du fichier [main.py](./SIMS/main.py) <br>
    - [Procédures_test_partie_electroaimant](./Procédures_test/Procédures_test_partie_electroaimant.ipynb) : Ce fichier est celui où nous donnons les procédures de test du fichier [partie_electroaimant.py](./SIMS/deviation_magnetique/Code/partie_electroaimant.py) <br>
    - [Procédures_test_déviation](./Procédures_test/Procédures_test_déviation.ipynb) : Ce fichier est celui où nous donnons les procédures de test du fichier [deviation.py](./SIMS/deviation_electrique/Code/deviation.py) <br>
    - [Procédures_test_incertitudes](./Procédures_test/Procédures_test_incertitudes.ipynb) : Ce fichier est celui où nous donnons les procédures de test du fichier [incertitude.py](./SIMS/deviation_electrique/Code/incertitude.py) <br><br><br><br>

On peut aussi trouver un fichier [SIMS diagram](./SIMS%20diagram.png) qui illustre le SIMS que l'on essaye de reproduire.
[Guide](./Guide.ipynb) est un fichier identique à celui-ci permettant une meilleure lecture dans un éditeur