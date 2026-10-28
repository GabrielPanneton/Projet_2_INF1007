import os
import pandas as pd


def charger_donnees(chemin: str) -> pd.DataFrame:
    """Charge le CSV des parties.

    Reçoit le chemin du fichier CSV.
    Retourne un DataFrame avec une ligne par partie.
    Échoue avec FileNotFoundError si le fichier n'existe pas.
    """
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    return pd.read_csv(chemin)