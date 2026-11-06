import os
import pandas as pd
import numpy as np
import exceptions

def charger_donnees(chemin: str) -> pd.DataFrame:
    """Charge le CSV des parties.

    Reçoit le chemin du fichier CSV.
    Retourne un DataFrame avec une ligne par partie.
    Échoue avec FileNotFoundError si le fichier n'existe pas.
    """
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    return pd.read_csv(chemin)

###########################

COLONNES_A_RETIRER: list[str] = [
    "gameId",
    "redFirstBlood", "redDeaths", "redKills",
    "blueGoldPerMin", "blueCSPerMin", "redGoldPerMin", "redCSPerMin",
    "redGoldDiff", "redExperienceDiff",
    "blueWardsPlaced", "redWardsPlaced",
]


def nettoyer_donnees(
    df: pd.DataFrame,
    colonnes_a_retirer: list[str] | None = None,
) -> pd.DataFrame:
    """Retire les colonnes inutiles ou redondantes.

    Reçoit :
        df : le DataFrame brut.
        colonnes_a_retirer : colonnes à enlever (COLONNES_A_RETIRER par défaut).

    Renvoie :
        Un nouveau DataFrame nettoyé (l'original n'est pas modifié).

    Lève :
        DonneeInvalideError : si une colonne à retirer n'existe pas.
    """
    if colonnes_a_retirer is None:
        colonnes_a_retirer = COLONNES_A_RETIRER
    absentes = [c for c in colonnes_a_retirer if c not in df.columns]
    if absentes:
        raise exceptions.DonneeInvalideError(f"Colonnes introuvables : {absentes}")
    return df.drop(columns=colonnes_a_retirer)

#######################################

def trouver_redondances(
    df: pd.DataFrame,
    seuil: float = 0.95,
    exclure: tuple[str, ...] = ("gameId",),
) -> pd.DataFrame:
    """Repère les paires de colonnes numériques redondantes.

    Reçoit :
        df : le DataFrame à examiner.
        seuil : corrélation absolue à partir de laquelle une paire est signalée.
        exclure : colonnes à ignorer (identifiants, par exemple).

    Renvoie :
        Un DataFrame avec une ligne par paire signalée (colonnes
        'colonne_a', 'colonne_b', 'correlation', 'relation'), trié de la
        corrélation absolue la plus forte à la plus faible.
        'relation' vaut 'identique', 'opposée', 'proportionnelle'
        ou 'très corrélée'.
    """
    numeriques = df.select_dtypes(include="number").drop(
        columns=list(exclure), errors="ignore"
    )
    noms = numeriques.columns
    corr = numeriques.corr().to_numpy()

    # Indices de toutes les paires (i, j) avec i < j, sans doublon
    indices_i, indices_j = np.triu_indices(len(noms), k=1)

    resultats = []
    for i, j in zip(indices_i, indices_j):
        r = corr[i, j]
        if np.isnan(r) or abs(r) < seuil:
            continue
        a = numeriques[noms[i]].to_numpy()
        b = numeriques[noms[j]].to_numpy()
        if np.allclose(a, b):
            relation = "identique"
        elif np.allclose(a, -b):
            relation = "opposée"
        elif abs(r) > 0.9999:
            relation = "proportionnelle"
        else:
            relation = "très corrélée"
        resultats.append(
            {
                "colonne_a": noms[i],
                "colonne_b": noms[j],
                "correlation": round(float(r), 4),
                "relation": relation,
            }
        )

    colonnes = ["colonne_a", "colonne_b", "correlation", "relation"]
    resultat = pd.DataFrame(resultats, columns=colonnes)
    return resultat.reindex(
        resultat["correlation"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def ajouter_ecarts_relatifs(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute l'avantage relatif en or et en XP de l'équipe bleue.

    Reçoit un DataFrame avec les totaux d'or et d'XP des deux équipes.
    Retourne une copie avec deux colonnes de plus, comprises entre -1 et 1.
    """
    resultat = df.copy()
    or_b, or_r = df["blueTotalGold"], df["redTotalGold"]
    xp_b, xp_r = df["blueTotalExperience"], df["redTotalExperience"]
    resultat["ecartOrRelatif"] = (or_b - or_r) / (or_b + or_r)
    resultat["ecartXpRelatif"] = (xp_b - xp_r) / (xp_b + xp_r)
    return resultat