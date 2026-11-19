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
    "blueWardsPlaced", "redWardsPlaced", "blueEliteMonsters", "redEliteMonsters",
    "blueGoldDiff", "blueExperienceDiff",
    # Modèle simple (2026-11-14) : l'or et l'XP sont des résumés de
    # tout le reste (kills, assists, farm, objectifs) et ils sont
    # très corrélés aux autres variables (kills <-> or bleu : 0,889).
    # Les garder fausse les coefficients : par exemple blueKills
    # devenait négatif. On les retire donc du modèle.
    "blueTotalGold", "redTotalGold",
    "blueTotalExperience", "redTotalExperience",
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
    return resultat.reindex(resultat["correlation"].abs().sort_values(ascending=False).index).reset_index(drop=True)

############################

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


#####################

CHEMIN_CSV: str = os.path.join(
    os.path.dirname(__file__), "..", "donnees", "high_diamond_ranked_10min.csv"
)


def charger_donnees_propres(chemin: str = CHEMIN_CSV) -> pd.DataFrame:
    """Charge le CSV puis le nettoie.

    Reçoit : le chemin du CSV (celui du projet par défaut).
    Renvoie : le DataFrame nettoyé (19 variables + la cible blueWins).
    Lève : DonneeInvalideError si le fichier ou une colonne à retirer est absent.

    Modèle simple (2026-11-14) : on n'ajoute plus les écarts relatifs
    d'or et d'XP, car ils sont calculés à partir de l'or et de l'XP,
    qui ne font plus partie du modèle. La fonction
    ajouter_ecarts_relatifs reste disponible dans ce fichier pour
    l'exploration, mais elle n'est plus appelée ici.
    """
    return nettoyer_donnees(charger_donnees(chemin))

COLONNES_ECHANGE = {
    "blueKills": "blueDeaths", "blueDeaths": "blueKills",
    "blueAssists": "redAssists", "redAssists": "blueAssists",
    "blueDragons": "redDragons", "redDragons": "blueDragons",
    "blueHeralds": "redHeralds", "redHeralds": "blueHeralds",
    "blueTowersDestroyed": "redTowersDestroyed",
    "redTowersDestroyed": "blueTowersDestroyed",
    "blueAvgLevel": "redAvgLevel", "redAvgLevel": "blueAvgLevel",
    "blueTotalMinionsKilled": "redTotalMinionsKilled",
    "redTotalMinionsKilled": "blueTotalMinionsKilled",
    "blueTotalJungleMinionsKilled": "redTotalJungleMinionsKilled",
    "redTotalJungleMinionsKilled": "blueTotalJungleMinionsKilled",
    "blueWardsDestroyed": "redWardsDestroyed",
    "redWardsDestroyed": "blueWardsDestroyed",
}


def echanger_equipes(donnees):
    """Copie des parties avec bleu et rouge échangés.

    Le premier sang n'a pas d'équivalent rouge : il s'inverse
    (1 - valeur). On lit toujours dans `donnees` (l'original),
    donc l'échange ne se mélange pas avec lui-même.
    """
    echangees = donnees.copy()
    for colonne, equivalente in COLONNES_ECHANGE.items():
        echangees[colonne] = donnees[equivalente]
    echangees["blueFirstBlood"] = 1 - donnees["blueFirstBlood"]
    return echangees


def augmenter_donnees(x_train, y_train):
    """Double l'entraînement : chaque partie + sa version échangée.

    À appeler APRÈS separer_donnees, sur le train seulement.
    """
    x_echange = echanger_equipes(x_train)
    x_aug = pd.concat([x_train, x_echange], ignore_index=True)
    y_aug = pd.concat([y_train, 1 - y_train], ignore_index=True)
    return x_aug, y_aug
