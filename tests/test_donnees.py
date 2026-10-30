import os

import numpy as np
import pandas as pd
import pytest

from donnees import charger_donnees, nettoyer_donnees, trouver_redondances

CHEMIN_CSV = os.path.join(
    os.path.dirname(__file__), "..", "donnees", "high_diamond_ranked_10min.csv"
)
a_le_csv = os.path.exists(CHEMIN_CSV)

def test_csv_present_dans_le_depot():
    assert os.path.exists(CHEMIN_CSV), f"CSV introuvable : {CHEMIN_CSV}"
    assert os.path.getsize(CHEMIN_CSV) > 0, "Le CSV est vide"

"""def test_initial() :
    assert 1 + 1 == 2"""

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
    assert "blueWins" in df_propre.columns
    assert len(df_propre) == len(df_brut)   # on retire des colonnes, pas des parties


@pytest.mark.skipif(not a_le_csv, reason="CSV absent du dépôt")
def test_nettoyage_ne_modifie_pas_le_dataframe_original(df_brut):
    colonnes_avant = list(df_brut.columns)

    nettoyer_donnees(df_brut)

    assert list(df_brut.columns) == colonnes_avant