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




# ==========================================
# ANALYSE DES WARDS ET DE LA VICTOIRE
# ==========================================

print("\n" + "="*50)
print("ANALYSE DES WARDS")
print("="*50)

# Corrélation simple
corr_blue = df["blueWardsPlaced"].corr(df["blueWins"])
corr_red = df["redWardsPlaced"].corr(df["blueWins"])

print(f"\nCorrélation blueWardsPlaced / blueWins : {corr_blue:.4f}")
print(f"Corrélation redWardsPlaced / blueWins  : {corr_red:.4f}")

# Différence de wards
df["wardDiff"] = df["blueWardsPlaced"] - df["redWardsPlaced"]

corr_diff = df["wardDiff"].corr(df["blueWins"])

print(f"\nCorrélation wardDiff / blueWins : {corr_diff:.4f}")

# Moyennes selon victoire ou défaite
print("\nMoyennes de wards placées :")
print(
    df.groupby("blueWins")[
        ["blueWardsPlaced", "redWardsPlaced", "wardDiff"]
    ].mean()
)

# ==========================================
# ANALYSE DES OUTLIERS
# ==========================================

q1 = df["blueWardsPlaced"].quantile(0.25)
q3 = df["blueWardsPlaced"].quantile(0.75)

iqr = q3 - q1

borne_inf = q1 - 1.5 * iqr
borne_sup = q3 + 1.5 * iqr

df_sans_outliers = df[
    (df["blueWardsPlaced"] >= borne_inf)
    & (df["blueWardsPlaced"] <= borne_sup)
]

print("\nNombre de parties originales :", len(df))
print("Nombre de parties sans outliers :", len(df_sans_outliers))
print("Parties retirées :", len(df) - len(df_sans_outliers))

# Corrélation avant/après
corr_avant = df["blueWardsPlaced"].corr(df["blueWins"])

corr_apres = (
    df_sans_outliers["blueWardsPlaced"]
    .corr(df_sans_outliers["blueWins"])
)

print(f"\nCorrélation avant nettoyage : {corr_avant:.4f}")
print(f"Corrélation après nettoyage : {corr_apres:.4f}")

# ==========================================
# RETRAIT DU TOP 1%
# ==========================================

seuil_99 = df["blueWardsPlaced"].quantile(0.99)

df_99 = df[df["blueWardsPlaced"] <= seuil_99]

corr_99 = (
    df_99["blueWardsPlaced"]
    .corr(df_99["blueWins"])
)

print(f"\nSeuil du 99e percentile : {seuil_99:.2f}")
print(f"Corrélation sans le top 1% : {corr_99:.4f}")

# ==========================================
# TAUX DE VICTOIRE PAR GROUPE DE WARDS
# ==========================================

print("\nTaux de victoire selon le nombre de wards :")

tranches = pd.qcut(
    df["blueWardsPlaced"],
    q=10,
    duplicates="drop"
)

resultat = (
    df.groupby(tranches, observed=True)["blueWins"]
    .agg(["mean", "count"])
)

resultat["mean"] *= 100

print(resultat)

# ==========================================
# BOXPLOT
# ==========================================

plt.figure(figsize=(8, 5))

df.boxplot(
    column="blueWardsPlaced",
    by="blueWins"
)

plt.title("Blue Wards Placed selon la victoire")
plt.suptitle("")
plt.xlabel("blueWins (0 = défaite, 1 = victoire)")
plt.ylabel("Blue Wards Placed")

plt.show()