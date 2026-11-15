# Projet_2_INF1007

Installation et lancement : 
    pip install -r requirements.txt
    python src/main.py              # entraîne, évalue et sauvegarde le modèle
    streamlit run src/interface.py  # lance l'interface (volet 3)
    python -m pytest                # lance les tests

Modèle : 
    Variable cible : "blueWins"
    Problème : Classification binaire
    Métrique : Exactitude
    Modèle : Régression logistique

Tecnhologie : Streamlit
    Pour lancer : streamlit run src/interface.py
    Contenu :
        - Champs à remplir
        - Lorsqu'un champ est laissé vide, remplissage automatique avec la moyenne d'entrainement, ce qui n'influence pas le résultat.
        - Lorsqu'une valeur est hor de l'intervalle pbservé lors de l'entrainement, avertissement, et lorsque la variable est impossible, échec du test.
        - Résultat selon le modèle en pourcentage + conclusion du modèle + pourcentage de précision
        - Graphique des coefficient des variables
        - Historique des prédictions

Limites :
    - Lorsque les partie sont serrés, le modèle revient à presqu'un pile ou face, puique dans ce cas il faudrait d'autres variables dans le csv, commes les champions, ou même le winrate des joueurs
    - le facteur humain, c'est à dire qu'à tou moment un joueur peut décider d'arrêter de jouer et de perdre, ce qui fausse les résultats.

Test :
    - Les tests sont principalement sur les fonctions du modèle ainsi que sur les fonctions sur l'exploration des données.
    - mesurée avec coverage run -m pytest.
    - Lancer les tests : python -m pytest

Outil du chapitre 7 : 
    Un décorateur utilisé dans interface.py, soit @abstractmethod, qui permet a chaque classe enfant de définir sa propre version de la fonction qui suit.

Source du Dataset : 
    Y. L. Ma, « League of Legends Diamond Ranked Games (10 min), » Kaggle, 2019. [En ligne]. Disponible : https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min. [Consulté le : 23 oct. 2026].

Patchnote : 
    P. Perscheid, « Patch 9.23 notes, » League of Legends, Riot Games, nov. 2019. [En ligne]. Disponible : https://www.leagueoflegends.com/en-gb/news/game-updates/patch-9-23-notes/. [Consulté le : 23 oct. 2026].

Documentation:
    pandas : The pandas development team, « pandas documentation, » pandas. [En ligne]. Disponible : https://pandas.pydata.org/docs/. [Consulté le : 30 octobre 2026 ].
    Sickitlearn : Scikit-learn developers, « scikit-learn: machine learning in Python, » scikit-learn. [En ligne]. Disponible : https://scikit-learn.org/stable/. [Consulté le : 6 novembre 2026 ].
    Streamlit : Streamlit, « Streamlit documentation, » Streamlit. [En ligne]. Disponible : https://docs.streamlit.io/. [Consulté le : 13 novembre 2026 ].
