import os
import matplotlib.pyplot as plt
from donnees import charger_donnees
import numpy as np
import pandas as pd

racine = os.path.join(os.path.dirname(__file__), "..")
df = charger_donnees(os.path.join(racine, "donnees", "high_diamond_ranked_10min.csv"))
# adaptez le nom du fichier au vôtre

#print(df)                      #toutes les stats dans un genre de tableau

print(df.shape)                # lignes et colonnes, (9879, 40)

print(df.dtypes)                # types de toutes les valeurs (soit int64 ou float64)

#print(df.isna())                # Pour chaque stats donne True si donnée manquante else False
print(df.isna().sum())          # Pour chaque categorie donne le nb de valeur manquante, 0   
print(df.isna().sum().sum())    # total des valeurs manquantes, 0 


#print(df.duplicated())          # Passe chaque ligne et regarde si elle a deja été passé si oui True else False
print(df.duplicated().sum())    # fait la somme de toutes les lignes dupliqué, 0

#print(df["gameId"].duplicated())        # Meme chose qu'au dessus, mais juste pour les gameId
print(df["gameId"].duplicated().sum())   # Somme de toute les partie dupliquer, 0 

#print(df["blueWins"])                   # La collone blueWins associer a leur ligne.  blueWins, 9879, int64
print(df["blueWins"].value_counts(normalize=True))   # proportion de win/loss, 0/1, float64

print(df.describe())        #Count, mean, std, min , max, 25%, 50%, 75% pour chaque colonnes
print(df.describe().T)          # la trasposée de lautes en haut. 


print((df["blueKills"] == df["redDeaths"]).all())       #regarde si les blusKills dont tjr egals 
print((df["blueGoldDiff"] == -df["redGoldDiff"]).all())     #regarde si les diffgoldred egal -diffgoldblue

