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


# 1. Combien de valeurs extrêmes de wards ?
print(df[["blueWardsPlaced", "redWardsPlaced"]].quantile([0.5, 0.9, 0.99, 0.999]))
print((df["blueWardsPlaced"] > 100).sum(), (df["redWardsPlaced"] > 100).sum())

# 2. Valeurs hors limites (règle de l'écart interquartile) pour quelques colonnes
for col in ["blueWardsPlaced", "blueWardsDestroyed",
            "blueTotalJungleMinionsKilled", "blueAvgLevel"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    bas, haut = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = ((df[col] < bas) | (df[col] > haut)).sum()
    print(col, "limites :", bas, haut, "|", n, "valeurs hors limites")

# 3. Colonnes redondantes

print(np.allclose(df["blueCSPerMin"] * 10, df["blueTotalMinionsKilled"]))
print(np.allclose(df["blueGoldPerMin"] * 10, df["blueTotalGold"]))
print((df["blueEliteMonsters"] == df["blueDragons"] + df["blueHeralds"]).all())


print(df[["blueWardsPlaced", "redWardsPlaced", "blueWins"]].corr()["blueWins"])

tranches = pd.cut(df["blueWardsPlaced"], [0, 10, 15, 20, 30, 50, 100, 300])
print(df.groupby(tranches, observed=True)["blueWins"].agg(["mean", "count"]))