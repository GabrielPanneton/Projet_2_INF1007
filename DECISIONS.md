# Fiche de décisions

| Choix | Options envisagées | Choix retenu et justification | Validé par l'auxiliaire (date) |
|---|---|---|---|
| Domaine | Aucun, nous avions un sujet en tête et avons continuer avec celui-ci. | Jeux vidéo / e-sport (League of Legends). Nous aimons les jeu-vidéo et plus précisement celui-ci. Le site permettra à l'utilisateur de prédire s'il gagnera sa partie après 10 minutes de jeu selon les statistiques de la partie. | Samuelle St-Onge (2026-10-23) |
| Jeu de données | Jeu sur les statistiques e-sport, jeu sur des parties complètes. | League of Legends Diamond Ranked Games (10 min), Kaggle (bobbyscience). Il nous semblait plus approprié et contenait des variables auxquelles on se serait entendu afin de prédire qui gagnera la partie si nous avions du les trouver nous-même. Il contenait assez de colones et il avait au-moins une ligne avec une variable catégoricielle. | Samuelle St-Onge (2026-10-23) |
| Variable catégorielle | Ajouter une autre colonne, changer de jeu | Nous avons décidé de prendre les valeur binaire de la colonne comme Oui/Non | (2026-10-23) |
| Variable cible et métrique | Nous savions que nous voulions travailler sur la prédiction de la victiore d'une équipe alors nous n'avons pas envisager autre chose, et pour les métriques, plusieurs exemples étaient fournit tel que l'exactitude, la précision et autre | Cible : `blueWins`. Métrique : Exactitude, selon nos recherche, c'est une bonne métrique à utiliser dans notre contexte ou notre variable est environs 50/50 | *(...)* |
| Modèle | régression logistique ou forêt aléatoire | Régession logisitique, selon nos recherches, c'était ce qui était le plus adapter a notre situation | *(...)* |
| Technologie d'interface | *(ex. : Tkinter, Streamlit, Gradio)* | *(à compléter, avec justification)* | *(...)* |
| Nettoyage des données | `gameId`| retirer parce que c'est un identifiant de partie, pas une caractéristique | pas à validé|
| Nettoyage des données | `redKills`, `redDeaths`, `redFirstBlood`, `redGoldDiff`, `redExperienceDiff` | Retirées parce que ce sont des copies ou des inverse d'autres colonnes qui elles sont conservées | pas à validé |
| Nettoyage des données | `blueGoldPerMin`, `redGoldPerMin`, `blueCSPerMin`, `redCSPerMin`| Retirées parce que ce sont des totaux déjà conservée diviser par 10. ne donne pas d'inforamtion dupplémetaire. | pas à validé |
| Nettoyage des données | `blueWardsPlaced`, `redWardsPlaced`| Om suspect un bug lors de la prise de donnée, car plus d'un pourcent des donnée semble aberrante. De plus, un équivalent similaire 'blueWardsDestroyed' donne un information differente, mais similaire. et puis selon nos connaissance, les wards dont parmis les critere les moins influent sur le taux de victoire. | pas à validé |


