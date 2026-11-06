"""Logique du modèle : séparation des données (et, plus tard, entraînement et évaluation)."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from exceptions import DonneeInvalideError, ModeleIntrouvableError
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import json
import os
import joblib

GRAINE: int = 42
TAILLE_TEST: float = 0.2


def separer_donnees(
    df: pd.DataFrame,
    cible: str = "blueWins",
    taille_test: float = TAILLE_TEST,
    graine: int = GRAINE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Sépare les données en ensembles d'entraînement et de test.

    Reçoit :
        df : le DataFrame nettoyé, cible incluse.
        cible : nom de la colonne à prédire.
        taille_test : proportion des données réservée au test (entre 0 et 1).
        graine : graine aléatoire, pour des résultats reproductibles.

    Renvoie :
        (X_train, X_test, y_train, y_test)

    Lève :
        DonneeInvalideError : si la colonne cible est absente.
    """
    if cible not in df.columns:
        raise DonneeInvalideError(f"Colonne cible introuvable : {cible}")
    X = df.drop(columns=[cible])
    y = df[cible]
    return train_test_split(
        X, y, test_size=taille_test, random_state=graine, stratify=y
    )

################################################

def calculer_reference(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> float:
    """Calcule l'exactitude d'un modèle trivial (classe majoritaire).

    Reçoit :
        X_train, y_train : données d'entraînement.
        X_test, y_test : données de test.

    Renvoie :
        L'exactitude de la référence sur l'ensemble de test (entre 0 et 1).
    """
    reference = DummyClassifier(strategy="most_frequent")
    reference.fit(X_train, y_train)
    return float(reference.score(X_test, y_test))

################################

def entrainer_modele(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Entraîne une régression logistique (avec normalisation des colonnes).

    Reçoit : les entrées et la cible d'entraînement.
    Renvoie : le modèle entraîné (un Pipeline scikit-learn).
    """
    modele = Pipeline(
        [
            ("normalisation", StandardScaler()),
            ("regression", LogisticRegression(max_iter=1000)),
        ]
    )
    modele.fit(X_train, y_train)
    return modele


def calculer_exactitude(y_vrai: pd.Series, y_predit: np.ndarray) -> float:
    """Calcule l'exactitude avec NumPy (calcul vectorisé, sans boucle).

    Reçoit : les vraies valeurs et les valeurs prédites.
    Renvoie : la proportion de prédictions correctes (entre 0 et 1).
    """
    return float(np.mean(np.asarray(y_vrai) == np.asarray(y_predit)))


def evaluer_modele(
    modele: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """Calcule l'exactitude du modèle sur l'ensemble de test."""
    return calculer_exactitude(y_test, modele.predict(X_test))


def calculer_matrice_confusion(
    modele: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> np.ndarray:
    """Renvoie la matrice de confusion 2x2 sur l'ensemble de test."""
    return confusion_matrix(y_test, modele.predict(X_test))

###################

DOSSIER_MODELES: str = os.path.join(os.path.dirname(__file__), "..", "modeles")
CHEMIN_MODELE: str = os.path.join(DOSSIER_MODELES, "modele.joblib")
CHEMIN_METADONNEES: str = os.path.join(DOSSIER_MODELES, "metadonnees.json")

# Une colonne avec au plus ce nombre de valeurs distinctes est traitée comme catégorielle.
SEUIL_CATEGORIE: int = 2


def construire_metadonnees(
    X_train: pd.DataFrame,
    exactitude: float,
    reference: float,
    cible: str = "blueWins",
) -> dict:
    """Décrit les variables d'entrée et les résultats du modèle.

    Reçoit :
        X_train : les entrées d'entraînement (l'ordre des colonnes est conservé).
        exactitude : exactitude du modèle sur l'ensemble de test.
        reference : exactitude de la référence sur l'ensemble de test.
        cible : nom de la variable cible.

    Renvoie :
        Un dictionnaire sérialisable en JSON : colonnes, type de chaque variable
        (valeurs possibles ou intervalle) et métriques.
    """
    variables = []
    for nom in X_train.columns:
        valeurs = sorted(X_train[nom].unique().tolist())
        if len(valeurs) <= SEUIL_CATEGORIE:
            variables.append({"nom": nom, "type": "categorielle", "valeurs": valeurs})
        else:
            variables.append(
                {
                    "nom": nom,
                    "type": "numerique",
                    "min": float(X_train[nom].min()),
                    "max": float(X_train[nom].max()),
                }
            )
    return {
        "cible": cible,
        "colonnes": list(X_train.columns),
        "variables": variables,
        "metrique": "exactitude",
        "exactitude_modele": exactitude,
        "exactitude_reference": reference,
    }


def sauvegarder_modele(modele: Pipeline, chemin: str = CHEMIN_MODELE) -> None:
    """Enregistre le modèle entraîné avec joblib (crée le dossier au besoin)."""
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    joblib.dump(modele, chemin)


def sauvegarder_metadonnees(metadonnees: dict, chemin: str = CHEMIN_METADONNEES) -> None:
    """Écrit les métadonnées dans un fichier JSON."""
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as fichier:
        json.dump(metadonnees, fichier, ensure_ascii=False, indent=2)


def charger_modele(chemin: str = CHEMIN_MODELE) -> Pipeline:
    """Charge le modèle sauvegardé.

    Lève : ModeleIntrouvableError si le fichier n'existe pas.
    """
    if not os.path.exists(chemin):
        raise ModeleIntrouvableError(f"Modèle introuvable : {chemin}")
    return joblib.load(chemin)


def charger_metadonnees(chemin: str = CHEMIN_METADONNEES) -> dict:
    """Charge les métadonnées JSON.

    Lève : ModeleIntrouvableError si le fichier n'existe pas.
    """
    if not os.path.exists(chemin):
        raise ModeleIntrouvableError(f"Métadonnées introuvables : {chemin}")
    with open(chemin, encoding="utf-8") as fichier:
        return json.load(fichier)