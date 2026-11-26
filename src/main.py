"""Point d’entrée : entraîne, évalue et sauvegarde le modèle."""

from donnees import augmenter_donnees, charger_donnees_propres
from modele import (
    CHEMIN_MODELE,
    GRAINE,
    calculer_matrice_confusion,
    calculer_reference,
    charger_modele,
    construire_metadonnees,
    entrainer_modele,
    evaluer_modele,
    sauvegarder_metadonnees,
    sauvegarder_modele,
    separer_donnees,
)


def main() -> None:
    """Entraîne le modèle augmenté, l’évalue et le sauvegarde."""
    df = charger_donnees_propres()
    X_train, X_test, y_train, y_test = separer_donnees(df)
    X_train, y_train = augmenter_donnees(X_train, y_train)

    reference = calculer_reference(X_train, X_test, y_train, y_test)
    modele = entrainer_modele(X_train, y_train)
    exactitude = evaluer_modele(modele, X_test, y_test)
    matrice = calculer_matrice_confusion(modele, X_test, y_test)

    sauvegarder_modele(modele)
    sauvegarder_metadonnees(construire_metadonnees(X_train, exactitude, reference))

    print(f"Référence : {reference:.1%}")
    print(f"Modèle : {exactitude:.1%}")
    print(f"Matrice de confusion :\n{matrice}")

    modele_charge = charger_modele(CHEMIN_MODELE)
    une_partie = X_test.sample(1, random_state=GRAINE)
    prob_bleue = modele_charge.predict_proba(une_partie)[0][1]
    print(f"{prob_bleue:.0%} de chances de gagner")
    print("Vrai résultat :", y_test.loc[une_partie.index[0]])


if __name__ == "__main__":
    main()