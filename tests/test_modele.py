import pandas as pd
import pytest
import numpy as np
from sklearn.metrics import accuracy_score
from exceptions import DonneeInvalideError, ModeleIntrouvableError
from modele import (
    calculer_exactitude,
    calculer_matrice_confusion,
    calculer_reference,
    charger_metadonnees,
    charger_modele,
    construire_metadonnees,
    entrainer_modele,
    evaluer_modele,
    sauvegarder_metadonnees,
    sauvegarder_modele,
    separer_donnees,
)


def test_separer_donnees_tailles_et_cible():
    df = pd.DataFrame({"a": range(100), "blueWins": [0, 1] * 50})

    X_train, X_test, y_train, y_test = separer_donnees(df, taille_test=0.2)

    assert len(X_test) == 20
    assert len(X_train) == 80
    assert "blueWins" not in X_train.columns   # pas de fuite de la cible


def test_separer_donnees_reproductible():
    df = pd.DataFrame({"a": range(100), "blueWins": [0, 1] * 50})

    premier = separer_donnees(df)[1]
    second = separer_donnees(df)[1]

    assert premier.equals(second)


def test_separer_donnees_cible_absente():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(DonneeInvalideError):
        separer_donnees(df)


##########
#########

def test_calculer_exactitude_cas_simple():
    y_vrai = pd.Series([0, 1, 1, 0])
    y_predit = np.array([0, 1, 0, 0])      # 3 bonnes sur 4

    assert calculer_exactitude(y_vrai, y_predit) == pytest.approx(0.75)


def test_calculer_exactitude_coincide_avec_sklearn():
    rng = np.random.default_rng(0)
    y_vrai = rng.integers(0, 2, size=200)
    y_predit = rng.integers(0, 2, size=200)

    assert calculer_exactitude(y_vrai, y_predit) == pytest.approx(
        accuracy_score(y_vrai, y_predit)
    )


def test_calculer_reference_classe_majoritaire():
    X_train = pd.DataFrame({"a": [1, 2, 3, 4]})
    y_train = pd.Series([0, 0, 0, 1])
    X_test = pd.DataFrame({"a": [5, 6, 7]})
    y_test = pd.Series([0, 0, 1])

    assert calculer_reference(X_train, X_test, y_train, y_test) == pytest.approx(2 / 3)


def test_calculer_matrice_confusion_cas_separable():
    X = pd.DataFrame({"a": [0, 1, 2, 3, 10, 11, 12, 13]})
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    modele = entrainer_modele(X, y)

    matrice = calculer_matrice_confusion(modele, X, y)

    assert (matrice == np.array([[4, 0], [0, 4]])).all()


def test_entrainer_modele_apprend_un_cas_separable():
    X = pd.DataFrame({"a": [0, 1, 2, 3, 10, 11, 12, 13]})
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    modele = entrainer_modele(X, y)

    assert evaluer_modele(modele, X, y) == pytest.approx(1.0)


def test_construire_metadonnees_contenu():
    X_train = pd.DataFrame(
        {
            "blueKills": [0, 1, 2, 3],
            "blueFirstBlood": [0, 1, 0, 1],
            "blueDragons": [0, 0, 1, 1],
        }
    )

    meta = construire_metadonnees(X_train, 0.9, 0.5)

    assert meta["colonnes"] == ["blueKills", "blueFirstBlood", "blueDragons"]
    assert meta["metrique"] == "exactitude"
    assert meta["exactitude_modele"] == 0.9
    assert meta["exactitude_reference"] == 0.5
    variables = {v["nom"]: v for v in meta["variables"]}
    assert variables["blueKills"]["type"] == "numerique"
    assert variables["blueKills"]["min"] == 0.0
    assert variables["blueKills"]["max"] == 3.0
    assert variables["blueFirstBlood"]["type"] == "categorielle"
    assert variables["blueFirstBlood"]["valeurs"] == [0, 1]


def test_sauvegarder_charger_modele_et_metadonnees(tmp_path):
    X = pd.DataFrame({"a": [0, 1, 2, 3, 10, 11, 12, 13]})
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    modele = entrainer_modele(X, y)
    meta = construire_metadonnees(X, 1.0, 0.5)
    chemin_modele = str(tmp_path / "modele.joblib")
    chemin_meta = str(tmp_path / "metadonnees.json")

    sauvegarder_modele(modele, chemin_modele)
    sauvegarder_metadonnees(meta, chemin_meta)

    assert charger_metadonnees(chemin_meta) == meta
    assert evaluer_modele(charger_modele(chemin_modele), X, y) == pytest.approx(1.0)


def test_charger_modele_absent():
    with pytest.raises(ModeleIntrouvableError):
        charger_modele("inexistant.joblib")
    with pytest.raises(ModeleIntrouvableError):
        charger_metadonnees("inexistant.json")
