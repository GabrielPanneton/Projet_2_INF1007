import os

import numpy as np
import pandas as pd
import pytest
import exceptions
from donnees import (
    ajouter_ecarts_relatifs,
    augmenter_donnees,
    charger_donnees,
    charger_donnees_propres,
    echanger_equipes,
    nettoyer_donnees,
    trouver_redondances,
)

CHEMIN_CSV = os.path.join(
    os.path.dirname(__file__), "..", "donnees", "high_diamond_ranked_10min.csv"
)
a_le_csv = os.path.exists(CHEMIN_CSV)


def test_csv_present_dans_le_depot():
    assert os.path.exists(CHEMIN_CSV), f"CSV introuvable : {CHEMIN_CSV}"
    assert os.path.getsize(CHEMIN_CSV) > 0, "Le CSV est vide"


def test_charger_fichier_absent():
    with pytest.raises(FileNotFoundError):
        charger_donnees("inexistant.csv")

# ---------- Tests du détecteur (données fabriquées, rapides) ----------

def faire_df_avec_redondances() -> pd.DataFrame:
    """DataFrame de test : une paire identique, une paire opposée, le reste indépendant."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    return pd.DataFrame(
        {
            "gameId": np.arange(200),
            "blueKills": x,
            "redDeaths": x,          # identique à blueKills
            "blueGoldDiff": y,
            "redGoldDiff": -y,       # opposée à blueGoldDiff
            "blueWardsPlaced": rng.normal(size=200),
        }
    )


def test_trouver_redondances_detecte_les_paires_connues():
    resultat = trouver_redondances(faire_df_avec_redondances())

    paires = {frozenset((r.colonne_a, r.colonne_b)) for r in resultat.itertuples()}
    assert paires == {
        frozenset(("blueKills", "redDeaths")),
        frozenset(("blueGoldDiff", "redGoldDiff")),
    }
    relations = {frozenset((r.colonne_a, r.colonne_b)): r.relation
                 for r in resultat.itertuples()}
    assert relations[frozenset(("blueKills", "redDeaths"))] == "identique"
    assert relations[frozenset(("blueGoldDiff", "redGoldDiff"))] == "opposée"


def test_trouver_redondances_vide_si_colonnes_independantes():
    df = faire_df_avec_redondances().drop(columns=["redDeaths", "redGoldDiff"])

    resultat = trouver_redondances(df)

    assert resultat.empty


def test_trouver_redondances_proportionnelle_et_tres_correllee():
    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    df = pd.DataFrame(
        {
            "a": x,
            "b": 2 * x,                                # proportionnelle à a
            "c": x + rng.normal(scale=0.1, size=200),  # très corrélée à a
        }
    )

    resultat = trouver_redondances(df)

    relations = {frozenset((r.colonne_a, r.colonne_b)): r.relation
                 for r in resultat.itertuples()}
    assert relations[frozenset(("a", "b"))] == "proportionnelle"
    assert relations[frozenset(("a", "c"))] == "très corrélée"


# ---------- Tests de l'augmentation (échange des équipes) ----------

def faire_petite_partie() -> pd.DataFrame:
    """Une partie avec les 19 colonnes du modèle, valeurs distinctes."""
    return pd.DataFrame(
        {
            "blueWardsDestroyed": [2, 3],
            "blueFirstBlood": [0, 1],
            "blueKills": [1, 2],
            "blueDeaths": [3, 4],
            "blueAssists": [5, 6],
            "blueDragons": [0, 1],
            "blueHeralds": [0, 0],
            "blueTowersDestroyed": [1, 2],
            "blueAvgLevel": [6.0, 7.0],
            "blueTotalMinionsKilled": [100, 110],
            "blueTotalJungleMinionsKilled": [40, 45],
            "redWardsDestroyed": [1, 4],
            "redAssists": [7, 8],
            "redDragons": [1, 0],
            "redHeralds": [1, 1],
            "redTowersDestroyed": [3, 4],
            "redAvgLevel": [5.0, 6.0],
            "redTotalMinionsKilled": [90, 95],
            "redTotalJungleMinionsKilled": [30, 35],
        }
    )


def test_echanger_equipes_inverse_bleu_rouge():
    df = faire_petite_partie()

    echangees = echanger_equipes(df)

    assert echangees["blueKills"].tolist() == [3, 4]
    assert echangees["blueDeaths"].tolist() == [1, 2]
    assert echangees["blueFirstBlood"].tolist() == [1, 0]
    assert echangees["redDragons"].tolist() == [0, 1]
    assert echangees["blueAvgLevel"].tolist() == [5.0, 6.0]
    assert df["blueKills"].tolist() == [1, 2]  # l'original n'est pas modifié


def test_augmenter_donnees_double_et_inverse_cible():
    X_train = faire_petite_partie()
    y_train = pd.Series([0, 1])

    X_aug, y_aug = augmenter_donnees(X_train, y_train)

    assert len(X_aug) == 4
    assert y_aug.tolist() == [0, 1, 1, 0]
    assert X_aug.iloc[2]["blueKills"] == 3  # kills échangés = morts d'origine
    assert X_aug.iloc[3]["blueFirstBlood"] == 0  # premier sang inversé


# ---------- Tests des écarts relatifs et du chargement propre ----------

def test_ajouter_ecarts_relatifs():
    df = pd.DataFrame(
        {
            "blueTotalGold": [100, 150],
            "redTotalGold": [50, 150],
            "blueTotalExperience": [200, 100],
            "redTotalExperience": [100, 100],
        }
    )

    resultat = ajouter_ecarts_relatifs(df)

    assert resultat["ecartOrRelatif"].tolist() == pytest.approx([1 / 3, 0.0])
    assert resultat["ecartXpRelatif"].tolist() == pytest.approx([1 / 3, 0.0])
    assert "ecartOrRelatif" not in df.columns  # l'original n'est pas modifié


@pytest.mark.skipif(not a_le_csv, reason="CSV absent du dépôt")
def test_charger_donnees_propres_colonnes_et_lignes():
    df = charger_donnees_propres()

    assert len(df) == 9879
    assert len(df.columns) == 20  # 19 variables + la cible
    assert "blueWins" in df.columns
    assert "blueTotalGold" not in df.columns


# ---------- Tests du nettoyage (données réelles) ----------

@pytest.fixture(scope="module")
def df_brut() -> pd.DataFrame:
    return charger_donnees(CHEMIN_CSV)


@pytest.mark.skipif(not a_le_csv, reason="CSV absent du dépôt")
def test_apres_nettoyage_plus_de_colonnes_redondantes(df_brut):
    df_propre = nettoyer_donnees(df_brut)

    redondances = trouver_redondances(df_propre, seuil=0.95)

    assert redondances.empty, (
        "Colonnes encore redondantes :\n" + redondances.to_string()
    )


@pytest.mark.skipif(not a_le_csv, reason="CSV absent du dépôt")
def test_nettoyage_retire_gameid_et_garde_la_cible(df_brut):
    df_propre = nettoyer_donnees(df_brut)


    assert "gameId" not in df_propre.columns
    assert "redFirstBlood" not in df_propre.columns
    assert "redDeaths" not in df_propre.columns
    assert "redKills" not in df_propre.columns
    assert "blueGoldPerMin" not in df_propre.columns
    assert "blueCSPerMin" not in df_propre.columns
    assert "redGoldPerMin" not in df_propre.columns
    assert "redCSPerMin" not in df_propre.columns
    assert "redGoldDiff" not in df_propre.columns
    assert "redExperienceDiff" not in df_propre.columns
    assert "blueWardsPlaced" not in df_propre.columns
    assert "redWardsPlaced" not in df_propre.columns
    assert "blueTotalGold" not in df_propre.columns
    assert "redTotalGold" not in df_propre.columns
    assert "blueTotalExperience" not in df_propre.columns
    assert "redTotalExperience" not in df_propre.columns
    assert "blueWins" in df_propre.columns
    assert len(df_propre) == len(df_brut)   # on retire des colonnes, pas des parties


@pytest.mark.skipif(not a_le_csv, reason="CSV absent du dépôt")
def test_nettoyage_ne_modifie_pas_le_dataframe_original(df_brut):
    colonnes_avant = list(df_brut.columns)

    nettoyer_donnees(df_brut)

    assert list(df_brut.columns) == colonnes_avant


@pytest.mark.skipif(not a_le_csv, reason="CSV absent du dépôt")
def test_nettoyer_colonne_absente_leve_une_erreur(df_brut):
    with pytest.raises(exceptions.DonneeInvalideError):
        nettoyer_donnees(df_brut, colonnes_a_retirer=["colonne_inexistante"])
