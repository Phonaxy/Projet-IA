Le MLP est un réseau à propagation avant classique où chaque neurone d'une couche est connecté à tous les neurones de la couche suivante.

Structure technique
Flatten : Conversion de l'image 2D (28x28) en un vecteur plat de 784 pixels.

Dense (Hidden) : 512 neurones.

Dropout : Utilisé pour prévenir le surapprentissage.

Paramètres totaux : 407 050 (1.55 MB).

Résultats d'entraînement
Précision (Test) : 98,27%

Vitesse : ~6-7 ms/step.

Temps total : Entre 1m 21s et 1m 44s.

Observation : Très rapide à entraîner, mais atteint un "plafond" de précision plus bas que le CNN.






Le CNN utilise des filtres (couches de convolution) pour extraire des motifs spatiaux (bords, boucles, traits) avant la classification.

Structure technique
Conv2D & MaxPooling : Deux blocs d'extraction de caractéristiques.

Flatten : Entrée de 3136 neurones vers la partie dense.

Dense (Hidden) : 512 neurones.

Paramètres totaux : 1 630 090 (6.22 MB).

Résultats d'entraînement
Précision (Test) : 99,31%

Vitesse : ~22-24 ms/step.

Temps total : Entre 3m 16s et 4m 20s.

Observation : Environ 3 fois plus lent que le MLP, mais dépasse la barre symbolique des 99% de précision.




Métrique,                       MLP,                CNN
Précision Finale,             "98,27%"           "99,31%"
Complexité (Params),            407k               1,63M
Rapidité d'exécution,          Élevée             Modérée
Efficacité Spatiale,           Faible            Excellente


Le CNN est le grand gagnant en termes de performance pure, cependant, le MLP reste une option très pertinente si les ressources de calcul sont limitées (CNN prend plus de temps et plus lourd), offrant un excellent rapport précision/temps.