import os
import matplotlib.pyplot as plt
from donnees import charger_donnees, nettoyer_donnees, trouver_redondances, ajouter_ecarts_relatifs, charger_donnees_propres
from modele import separer_donnees, calculer_reference, entrainer_modele, evaluer_modele, calculer_matrice_confusion
import numpy as np
import pandas as pd
import joblib

racine = os.path.join(os.path.dirname(__file__), "..")
df = charger_donnees(os.path.join(racine, "donnees", "high_diamond_ranked_10min.csv"))

##############################

df = charger_donnees_propres()
X_train, X_test, y_train, y_test = separer_donnees(df)

reference = calculer_reference(X_train, X_test, y_train, y_test)
print(f"Exactitude de la référence : {reference:.1%}")


##################################################


"""modele = entrainer_modele(X_train, y_train)
exactitude = evaluer_modele(modele, X_test, y_test)

print(f"Référence : {reference:.1%}")
print(f"Modèle    : {exactitude:.1%}")
print(calculer_matrice_confusion(modele, X_test, y_test))"""

############################

from donnees import charger_donnees_propres
from modele import (
    calculer_reference, construire_metadonnees, entrainer_modele,
    evaluer_modele, sauvegarder_metadonnees, sauvegarder_modele, separer_donnees,
)

df = charger_donnees_propres()
X_train, X_test, y_train, y_test = separer_donnees(df)

reference = calculer_reference(X_train, X_test, y_train, y_test)
modele = entrainer_modele(X_train, y_train)
exactitude = evaluer_modele(modele, X_test, y_test)

sauvegarder_modele(modele)
sauvegarder_metadonnees(construire_metadonnees(X_train, exactitude, reference))
print(f"Référence : {reference:.1%} | Modèle : {exactitude:.1%}")

#3#####

modele = joblib.load("modeles/modele.joblib")
print(modele)