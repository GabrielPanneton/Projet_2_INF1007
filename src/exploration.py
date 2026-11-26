import os
import matplotlib.pyplot as plt
from donnees import charger_donnees, trouver_redondances, ajouter_ecarts_relatifs
import numpy as np
import pandas as pd

racine = os.path.join(os.path.dirname(__file__), "..")
df = charger_donnees(os.path.join(racine, "donnees", "high_diamond_ranked_10min.csv"))
# adaptez le nom du fichier au vôtre

#print(df)                      #toutes les stats dans un genre de tableau

print(df.shape)                # lignes et colonnes, (9879, 40)

#print(df.dtypes)                # types de toutes les valeurs (soit int64 ou float64)

#print(df.isna())                # Pour chaque stats donne True si donnée manquante else False
#print(df.isna().sum())          # Pour chaque categorie donne le nb de valeur manquante, 0   
#print(df.isna().sum().sum())    # total des valeurs manquantes, 0 


#print(df.duplicated())          # Passe chaque ligne et regarde si elle a deja été passé si oui True else False
#print(df.duplicated().sum())    # fait la somme de toutes les lignes dupliqué, 0

#print(df["gameId"].duplicated())        # Meme chose qu'au dessus, mais juste pour les gameId
#print(df["gameId"].duplicated().sum())   # Somme de toute les partie dupliquer, 0 

#print(df["blueWins"])                   # La collone blueWins associer a leur ligne.  blueWins, 9879, int64
print(df["blueWins"].value_counts(normalize=True))   # proportion de win/loss, 0/1, float64

#print(df.describe())        #Count, mean, std, min , max, 25%, 50%, 75% pour chaque colonnes
print(df.describe().T)          # la trasposée de lautes en haut. 


#print((df["blueKills"] == df["redDeaths"]).all())       #regarde si les blusKills dont tjr egals 
#print((df["blueGoldDiff"] == -df["redGoldDiff"]).all())     #regarde si les diffgoldred egal -diffgoldblue





##########################################Correlation############################################




print((df["redEliteMonsters"] == df["redDragons"] + df["redHeralds"]).all())
print((df["blueEliteMonsters"] == df["blueDragons"] + df["blueHeralds"]).all())
print(np.allclose(df["blueGoldDiff"], df["blueTotalGold"] - df["redTotalGold"]))
print(np.allclose(df["blueExperienceDiff"], df["blueTotalExperience"] - df["redTotalExperience"]))

redondances = trouver_redondances(df, seuil=0.95)
print(redondances.to_string())





################# Nettoyage #############

df = ajouter_ecarts_relatifs(df)

#print(df)
print(df.shape)  

redondances = trouver_redondances(df, seuil=0.95)
print(redondances.to_string())

########################  Tableau


os.makedirs(os.path.join(racine, "figures"), exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Répartition de la cible
df["blueWins"].value_counts().sort_index().plot(kind="bar", ax=axes[0])
axes[0].set_title("Répartition de la victoire de l'équipe bleue")
axes[0].set_xlabel("blueWins (0 = défaite, 1 = victoire)")
axes[0].set_ylabel("Nombre de parties")

# Écart d'or selon le résultat
df.boxplot(column="ecartOrRelatif", by="blueWins", ax=axes[1])
axes[1].set_title("Écart d'or à 10 min selon le résultat")
axes[1].set_xlabel("blueWins")
axes[1].set_ylabel("Écart d'or relatif")

plt.suptitle("")
plt.tight_layout()
plt.savefig(os.path.join(racine, "figures", "exploration.png"))
plt.show()