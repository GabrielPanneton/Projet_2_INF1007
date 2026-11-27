# Projet_2_INF1007
Lien du repertoire github : https://github.com/GabrielPanneton/Projet_2_INF1007
## Description
Ce projet prédit la probabilité que l’équipe bleue gagne une partie de League of Legends à partir des statistiques des 10 premières minutes. Le modèle est une régression logistique entraînée sur 9 879 parties. L’interface Streamlit permet de saisir une partie, d’obtenir une probabilité, de voir les contributions des variables et de consulter l’historique des prédictions.

## Résultats du modèle
- Exactitude du modèle : 70,7 % sur 1 976 parties de test.
- Exactitude de la référence, classe majoritaire : 50,1 %.
- Matrice de confusion : [[703, 287], [292, 694]].

## Structure du code
- `src/donnees.py` : chargement, nettoyage, redondances et augmentation des données.
- `src/modele.py` : séparation train/test, entraînement, évaluation, sauvegarde du modèle et des métadonnées.
- `src/interface.py` : interface Streamlit, validation des saisies, graphiques et historique.
- `src/exploration.py` : exploration des données et graphique d’exploration.
- `src/main.py` : point d’entrée pour entraîner et sauvegarder le modèle.
- `tests/` : tests de la logique, hors interface.
- `donnees/high_diamond_ranked_10min.csv` : jeu de données utilisé.
- `modeles/modele.joblib` et `modeles/metadonnees.json` : modèle entraîné et métadonnées lues par l’interface.

## Installation et lancement : 
    pip install -r requirements.txt
    python src/main.py              # entraîne, évalue et sauvegarde le modèle
    streamlit run src/interface.py  # lance l'interface (volet 3)
    python -m pytest                # lance les tests

## Modèle : 
    Variable cible : "blueWins"
    Problème : Classification binaire
    Métrique : Exactitude
    Modèle : Régression logistique

## Technologie : Streamlit
    Pour lancer : streamlit run src/interface.py
    Contenu :
        - Champs à remplir
        - Lorsqu'un champ est laissé vide, remplissage automatique avec la moyenne d'entrainement, ce qui n'influence pas le résultat.
        - Lorsqu'une valeur est hors de l'intervalle observé lors de l'entrainement, avertissement, et lorsque la variable est impossible, échec du test.
        - Résultat selon le modèle en pourcentage + conclusion du modèle + pourcentage de précision
        - Graphique des coefficients des variables
        - Historique des prédictions

## Limites :
    - Lorsque les parties sont serrées, le modèle revient à presque un pile ou face, puisque dans ce cas il faudrait d'autres variables dans le csv, comme les champions, ou même le winrate des joueurs
    - le facteur humain, c'est-à-dire qu'à tout moment un joueur peut décider d'arrêter de jouer et de perdre, ce qui fausse les résultats.

## Tests et couverture
- 24 tests logiques dans `tests/test_donnees.py` et `tests/test_modele.py`.
- Lancer les tests : `python -m pytest`
- Couverture : 100 % sur la logique testée (`src/donnees.py`, `src/modele.py`, `src/exceptions.py`), mesurée avec `coverage run -m pytest` puis `coverage report --include="src/donnees.py,src/modele.py,src/exceptions.py"`.

## Outil du chapitre 7 : 
    Un décorateur utilisé dans interface.py, soit @abstractmethod, qui permet à chaque classe enfant de définir sa propre version de la fonction qui suit.
    Le tri des contributions utilise une fonction lambda comme clé de tri : `contributions.sort(key=lambda item: item[1], reverse=True)`.

## Source du Dataset : 
    Y. L. Ma, « League of Legends Diamond Ranked Games (10 min), » Kaggle, 2019. [En ligne]. Disponible : https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min. [Consulté le : 23 oct. 2026].

    Licence du jeu de données : CC0: Public Domain.

## Patchnote : 
    P. Perscheid, « Patch 9.23 notes, » League of Legends, Riot Games, nov. 2019. [En ligne]. Disponible : https://www.leagueoflegends.com/en-gb/news/game-updates/patch-9-23-notes/. [Consulté le : 23 oct. 2026].

## Documentation:
    pandas : The pandas development team, « pandas documentation, » pandas. [En ligne]. Disponible : https://pandas.pydata.org/docs/. [Consulté le : 30 octobre 2026 ].
    Scikit-learn : Scikit-learn developers, « scikit-learn: machine learning in Python, » scikit-learn. [En ligne]. Disponible : https://scikit-learn.org/stable/. [Consulté le : 6 novembre 2026 ].
    Streamlit : Streamlit, « Streamlit documentation, » Streamlit. [En ligne]. Disponible : https://docs.streamlit.io/. [Consulté le : 13 novembre 2026 ].
