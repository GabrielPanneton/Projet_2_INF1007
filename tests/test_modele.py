import pandas as pd
import pytest
import numpy as np
from sklearn.metrics import accuracy_score
from exceptions import DonneeInvalideError
from modele import separer_donnees, calculer_exactitude, evaluer_modele, entrainer_modele


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


def test_entrainer_modele_apprend_un_cas_separable():
    X = pd.DataFrame({"a": [0, 1, 2, 3, 10, 11, 12, 13]})
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    modele = entrainer_modele(X, y)

    assert evaluer_modele(modele, X, y) == pytest.approx(1.0)