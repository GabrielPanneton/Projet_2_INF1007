"""Interface Streamlit — Projet 2 INF1007 : prédire la victoire bleue à 10 min.

Lancer depuis la racine du projet (après ``pip install streamlit``) :
    streamlit run src/interface.py

Fonctionnement :
- 19 champs à remplir : les 19 colonnes du modèle simple (sans or,
  sans XP et sans écarts relatifs — voir ``donnees.py``).
- Un champ laissé vide prend la valeur neutre : la moyenne de la
  variable sur le jeu d'entraînement. Une valeur neutre ne change pas
  le score du modèle (elle vaut 0 une fois normalisée), donc elle
  n'influence pas la prédiction.
- Valeur hors de l'intervalle observé (min/max de
  ``modeles/metadonnees.json``) : un avertissement est affiché, mais
  le calcul continue.
- Valeur impossible (règles du jeu, par exemple un niveau moyen
  supérieur à 18) : une erreur est affichée et le calcul n'a pas lieu.
  Une valeur neutre ne peut jamais bloquer le calcul : si elle
  contredit une valeur saisie, elle est ajustée à la valeur la plus
  proche qui respecte la règle (par exemple les assists neutres sont
  plafonnés à 4 fois les kills saisis), et l'ajustement est affiché.
- Le bouton « Prédire » affiche la probabilité pour cette partie,
  l'exactitude globale du modèle (71,1 %) à part, et ajoute la
  prédiction à l'historique (``historique_predictions.csv`` à la
  racine du projet). Si des
  champs sont laissés vides, un avertissement rappelle que le
  résultat peut être moins fiable, car il repose en partie sur les
  valeurs moyennes.
"""

import math
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st

from donnees import charger_donnees_propres
from exceptions import ModeleIntrouvableError
from modele import charger_metadonnees, charger_modele, separer_donnees

# Règles du jeu utilisées pour les erreurs bloquantes.
NIVEAU_MAX = 18        # niveau maximum dans League of Legends
TOURS_MAX = 11         # 11 tours par équipe
ASSISTS_PAR_KILL_MAX = 4  # un kill donne des assists à 4 coéquipiers au plus

# Libellés affichés dans l'interface, dans l'ordre des métadonnées.
LIBELLES = {
    "blueWardsDestroyed": "Wards détruites (bleu)",
    "blueFirstBlood": "Premier sang (bleu)",
    "blueKills": "Kills (bleu)",
    "blueDeaths": "Morts (bleu)",
    "blueAssists": "Assists (bleu)",
    "blueDragons": "Dragons (bleu)",
    "blueHeralds": "Hérauts (bleu)",
    "blueTowersDestroyed": "Tours détruites (bleu)",
    "blueAvgLevel": "Niveau moyen (bleu)",
    "blueTotalMinionsKilled": "Sbires tués (bleu)",
    "blueTotalJungleMinionsKilled": "Sbires de la jungle tués (bleu)",
    "redWardsDestroyed": "Wards détruites (rouge)",
    "redAssists": "Assists (rouge)",
    "redDragons": "Dragons (rouge)",
    "redHeralds": "Hérauts (rouge)",
    "redTowersDestroyed": "Tours détruites (rouge)",
    "redAvgLevel": "Niveau moyen (rouge)",
    "redTotalMinionsKilled": "Sbires tués (rouge)",
    "redTotalJungleMinionsKilled": "Sbires de la jungle tués (rouge)",
}

# Valeurs neutres de secours (moyennes du jeu d'entraînement, 7 903 parties,
# graine 42), utilisées seulement si le fichier CSV est introuvable.
MOYENNES_SECOURS = {
    "blueWardsDestroyed": 2.85, "blueFirstBlood": 0.50, "blueKills": 6.18,
    "blueDeaths": 6.14, "blueAssists": 6.64, "blueDragons": 0.36,
    "blueHeralds": 0.19, "blueTowersDestroyed": 0.05,
    "blueAvgLevel": 6.92,
    "blueTotalMinionsKilled": 216.69, "blueTotalJungleMinionsKilled": 50.46,
    "redWardsDestroyed": 2.73, "redAssists": 6.67, "redDragons": 0.41,
    "redHeralds": 0.16, "redTowersDestroyed": 0.05,
    "redAvgLevel": 6.93,
    "redTotalMinionsKilled": 217.40, "redTotalJungleMinionsKilled": 51.30,
}



class Variable(ABC):
    """Un champ de l'interface : sait s'afficher et se compléter.

    Chaque sous-classe redéfinit ``afficher`` (polymorphisme) : un champ
    numérique s'affiche comme une zone de saisie, un champ catégoriel
    comme une liste déroulante.
    """

    def __init__(self, nom, libelle, neutre, minimum, maximum):
        self.nom = nom
        self.libelle = libelle
        self.neutre = neutre
        self.minimum = minimum
        self.maximum = maximum

    @abstractmethod
    def afficher(self):
        """Affiche le widget Streamlit, retourne la valeur brute (None si vide)."""

    def completer(self, valeur):
        """Valeur neutre si le champ est vide, sinon la valeur entrée."""
        return self.neutre if valeur is None else valeur

    def __repr__(self):
        return f"{type(self).__name__}({self.nom})"


class VariableNumerique(Variable):
    """Un champ numérique : zone de saisie, vide = valeur neutre."""

    def __init__(self, nom, libelle, neutre, minimum, maximum, pas):
        super().__init__(nom, libelle, neutre, minimum, maximum)
        self.pas = pas

    def afficher(self):
        return st.number_input(
            f"{self.libelle} ({self.nom})",
            value=None,
            step=self.pas,
            help=f"Vide = valeur neutre ({self.neutre:.2f}). "
                 f"Intervalle observé : {self.minimum} à {self.maximum}.",
        )


class VariableCategorielle(Variable):
    """Un champ catégoriel : liste déroulante, « Non renseigné » = neutre."""

    CORRESPONDANCE = {"Oui": 1.0, "Non": 0.0, "0": 0.0, "1": 1.0}

    def __init__(self, nom, libelle, neutre, options):
        super().__init__(nom, libelle, neutre, 0, 1)
        self.options = options

    def afficher(self):
        choix = st.selectbox(f"{self.libelle} ({self.nom})", self.options)
        if choix == "Non renseigné":
            return None
        return self.CORRESPONDANCE[choix]


def _pas(nom):
    """Le pas des flèches de la zone de saisie, selon la variable."""
    if nom in ("blueAvgLevel", "redAvgLevel"):
        return 0.1
    return 1.0


def construire_variables(metadonnees, moyennes):
    """Les 19 champs, dans l'ordre des métadonnées."""
    infos = {v["nom"]: v for v in metadonnees["variables"]}
    variables = []
    for nom in metadonnees["colonnes"]:
        info = infos[nom]
        if info["type"] == "categorielle":
            if nom == "blueFirstBlood":
                options = ["Non renseigné", "Oui", "Non"]
            else:
                options = ["Non renseigné", "0", "1"]
            variables.append(
                VariableCategorielle(nom, LIBELLES[nom], moyennes[nom], options)
            )
        else:
            variables.append(
                VariableNumerique(
                    nom, LIBELLES[nom], moyennes[nom],
                    info["min"], info["max"], _pas(nom),
                )
            )
    return variables


@st.cache_data
def charger_statistiques_entrainement():
    """Moyennes neutres du jeu d'entraînement.

    Retourne (moyennes, ok) ; si le CSV est introuvable, ``ok`` vaut
    False et les valeurs de secours sont utilisées.
    """
    try:
        donnees = charger_donnees_propres()
        x_train, _, _, _ = separer_donnees(donnees)
        moyennes = {
            nom: float(x_train[nom].mean()) for nom in LIBELLES
        }
        return moyennes, True
    except Exception:
        return dict(MOYENNES_SECOURS), False


def reparer_valeurs_neutres(valeurs, saisies):
    """Ajuste les valeurs neutres pour respecter les règles du jeu.

    Une valeur neutre (moyenne d'entraînement) n'a aucun sens si elle
    contredit une valeur saisie : par exemple, 6,67 assists rouges
    neutres sont impossibles si l'équipe rouge n'a fait qu'1 kill
    (1 mort bleue). On ajuste alors la valeur neutre, jamais une
    valeur saisie : on prend la valeur la plus proche de la moyenne
    qui respecte la règle. Retourne (valeurs, ajustements), où
    ``ajustements`` liste les modifications faites, à afficher.
    """
    ajustements = []

    # Assists : 4 au maximum par kill (les kills rouges = les morts bleues).
    for kills, assists, libelle in (
        ("blueKills", "blueAssists", "Assists (bleu)"),
        ("blueDeaths", "redAssists", "Assists (rouge)"),
    ):
        if assists not in saisies:
            plafond = ASSISTS_PAR_KILL_MAX * valeurs[kills]
            if valeurs[assists] > plafond:
                valeurs[assists] = plafond
                ajustements.append(
                    f"{libelle} non renseignés : ajustés à {plafond:.0f} "
                    f"pour rester cohérents avec {valeurs[kills]:.0f} kill(s) "
                    f"({ASSISTS_PAR_KILL_MAX} assists au maximum par kill)."
                )
        elif kills not in saisies:
            plancher = math.ceil(valeurs[assists] / ASSISTS_PAR_KILL_MAX)
            if valeurs[kills] < plancher:
                valeurs[kills] = float(plancher)
                ajustements.append(
                    f"{LIBELLES[kills]} non renseignés : ajustés à "
                    f"{plancher} pour rester cohérents avec "
                    f"{valeurs[assists]:.0f} assists."
                )

    # Sbires de la jungle : un sous-ensemble des sbires totaux.
    for jungle, total in (
        ("blueTotalJungleMinionsKilled", "blueTotalMinionsKilled"),
        ("redTotalJungleMinionsKilled", "redTotalMinionsKilled"),
    ):
        if jungle not in saisies:
            if valeurs[jungle] > valeurs[total]:
                valeurs[jungle] = valeurs[total]
                ajustements.append(
                    f"{LIBELLES[jungle]} non renseignés : ajustés à "
                    f"{valeurs[total]:.0f}, comme les sbires totaux."
                )
        elif total not in saisies:
            if valeurs[total] < valeurs[jungle]:
                valeurs[total] = valeurs[jungle]
                ajustements.append(
                    f"{LIBELLES[total]} non renseignés : ajustés à "
                    f"{valeurs[jungle]:.0f}, comme les sbires de la jungle."
                )
    return valeurs, ajustements


def message_valeurs_manquantes(manquantes):
    """Message d'avertissement quand des champs sont vides.

    Retourne None si rien ne manque. ``manquantes`` est la liste des
    noms (clés) des champs non renseignés, complétés avec les valeurs
    moyennes d'entraînement : le résultat repose alors en partie sur
    ces valeurs par défaut, il peut donc être moins fiable.
    """
    nombre = len(manquantes)
    if nombre == 0:
        return None
    if nombre <= 5:
        noms = ", ".join(LIBELLES[nom] for nom in manquantes)
    else:
        premiers = ", ".join(LIBELLES[nom] for nom in manquantes[:5])
        reste = nombre - 5
        autres = "autre" if reste == 1 else "autres"
        noms = f"{premiers} et {reste} {autres}"
    mot = "champ non renseigné" if nombre == 1 else "champs non renseignés"
    accord = "complété" if nombre == 1 else "complétés"
    return (
        f"Attention : {nombre} {mot} ({noms}) {accord} avec les valeurs "
        f"moyennes d'entraînement. Le résultat peut donc être moins "
        f"fiable — renseignez plus de champs pour une prédiction plus "
        f"précise."
    )


def erreurs_bloquantes(valeurs):
    """Valeurs impossibles selon les règles du jeu : le calcul n'a pas lieu."""
    erreurs = []

    # Niveau moyen : entre 1 et 18 (niveau maximum dans League of Legends).
    for nom, libelle in (("blueAvgLevel", "Niveau moyen (bleu)"),
                         ("redAvgLevel", "Niveau moyen (rouge)")):
        if valeurs[nom] < 1 or valeurs[nom] > NIVEAU_MAX:
            erreurs.append(
                f"{libelle} : {valeurs[nom]} — impossible, "
                f"le niveau moyen est entre 1 et {NIVEAU_MAX}."
            )

    # Assists : 4 au maximum par kill (les 4 coéquipiers).
    # Les kills rouges sont les morts bleues (chaque mort bleue est un kill rouge).
    for kills, assists, libelle in (
        ("blueKills", "blueAssists", "Assists (bleu)"),
        ("blueDeaths", "redAssists", "Assists (rouge)"),
    ):
        if valeurs[assists] > ASSISTS_PAR_KILL_MAX * valeurs[kills]:
            erreurs.append(
                f"{libelle} : {valeurs[assists]} pour {valeurs[kills]} kills — "
                f"impossible, au maximum {ASSISTS_PAR_KILL_MAX} assists par kill."
            )

    # Sbires de la jungle : un sous-ensemble des sbires totaux.
    for jungle, total, libelle in (
        ("blueTotalJungleMinionsKilled", "blueTotalMinionsKilled",
         "Sbires de la jungle tués (bleu)"),
        ("redTotalJungleMinionsKilled", "redTotalMinionsKilled",
         "Sbires de la jungle tués (rouge)"),
    ):
        if valeurs[jungle] > valeurs[total]:
            erreurs.append(
                f"{libelle} : {valeurs[jungle]}, plus que les sbires totaux "
                f"({valeurs[total]}) — impossible."
            )

    # Tours détruites : 11 au maximum par équipe.
    for nom in ("blueTowersDestroyed", "redTowersDestroyed"):
        if valeurs[nom] > TOURS_MAX:
            erreurs.append(
                f"{LIBELLES[nom]} : {valeurs[nom]} — impossible, "
                f"il n'y a que {TOURS_MAX} tours par équipe."
            )

    # Tout le reste : aucune valeur négative.
    for nom in ("blueWardsDestroyed", "blueKills", "blueDeaths", "blueAssists",
                "blueTotalMinionsKilled", "blueTotalJungleMinionsKilled",
                "redWardsDestroyed", "redAssists",
                "redTotalMinionsKilled", "redTotalJungleMinionsKilled"):
        if valeurs[nom] < 0:
            erreurs.append(
                f"{LIBELLES[nom]} : {valeurs[nom]} — impossible, "
                "la valeur ne peut pas être négative."
            )
    return erreurs


def avertissements_hors_intervalle(variables, valeurs):
    """Valeurs hors de l'intervalle observé : avertissement, calcul maintenu."""
    avertissements = []
    for variable in variables:
        if isinstance(variable, VariableCategorielle):
            continue  # 0 ou 1, toujours dans l'intervalle
        valeur = valeurs[variable.nom]
        if valeur < variable.minimum or valeur > variable.maximum:
            avertissements.append(
                f"⚠ {variable.libelle} : {valeur} est en dehors de "
                f"l'intervalle observé à l'entraînement "
                f"({variable.minimum} à {variable.maximum}). "
                "Le calcul continue, mais la prédiction est moins fiable."
            )
    return avertissements


def construire_ligne(valeurs, colonnes):
    """Une ligne de 19 colonnes, dans l'ordre du modèle."""
    ligne = pd.DataFrame([valeurs])
    return ligne[colonnes]


def chemin_historique():
    """Le fichier CSV de l'historique, à la racine du projet."""
    racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(racine, "historique_predictions.csv")


def charger_historique():
    """L'historique des prédictions, ou un tableau vide."""
    chemin = chemin_historique()
    if os.path.exists(chemin):
        return pd.read_csv(chemin)
    return pd.DataFrame()


def ajouter_a_historique(valeurs, probabilite):
    """Ajoute une prédiction à l'historique (CSV). Retourne False si
    l'enregistrement échoue (par exemple dossier en lecture seule) :
    la prédiction reste affichée, seul l'historique est perdu."""
    ligne = {"horodatage": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    ligne.update(valeurs)
    ligne["probabilite_bleue"] = probabilite
    ligne["prediction"] = ("victoire bleue" if probabilite >= 0.5
                           else "défaite bleue")
    try:
        historique = charger_historique()
        historique = pd.concat([historique, pd.DataFrame([ligne])],
                               ignore_index=True)
        historique.to_csv(chemin_historique(), index=False)
    except OSError:
        return False
    return True


class Application:
    """L'application : le modèle, les métadonnées et les 19 champs."""

    def __init__(self):
        self.modele = charger_modele()
        self.metadonnees = charger_metadonnees()
        self.moyennes, self.stats_ok = charger_statistiques_entrainement()
        self.variables = construire_variables(self.metadonnees, self.moyennes)

    def executer(self):
        """Affiche l'interface et gère le bouton Prédire."""   
        
        st.image("assets/lol.png", use_container_width=True)

        st.title("League of Legends — Qui gagne à 10 minutes ?")

        st.write("Remplissez les statistiques de la partie à 10 minutes. "
                 "Un champ vide prend la valeur neutre (moyenne "
                 "d'entraînement) : elle n'influence pas la prédiction.")
        
        if not self.stats_ok:
            st.warning("Données d'entraînement introuvables : les valeurs "
                       "neutres utilisent les valeurs de secours enregistrées "
                       "dans le code.")

        valeurs_brutes = self._afficher_saisie()

        if st.button("Prédire"):
            self._predire(valeurs_brutes)

        self._afficher_historique()

    def _afficher_saisie(self):
        """Les 19 champs, équipe bleue à gauche, équipe rouge à droite."""
        valeurs_brutes = {}
        col_bleu, col_rouge = st.columns(2)
        with col_bleu:
            st.subheader("Équipe bleue")
            for variable in self.variables:
                if variable.nom.startswith("blue"):
                    valeurs_brutes[variable.nom] = variable.afficher()
        with col_rouge:
            st.subheader("Équipe rouge")
            for variable in self.variables:
                if variable.nom.startswith("red"):
                    valeurs_brutes[variable.nom] = variable.afficher()
        return valeurs_brutes

    def _predire(self, valeurs_brutes):
        """Valide, calcule les écarts, prédit, affiche et enregistre."""
        valeurs = {v.nom: v.completer(valeurs_brutes[v.nom])
                   for v in self.variables}
        saisies = {nom for nom, brut in valeurs_brutes.items()
                   if brut is not None}

        # Une valeur neutre ne bloque jamais le calcul : on l'ajuste
        # si elle contredit une valeur saisie, et on l'affiche.
        valeurs, ajustements = reparer_valeurs_neutres(valeurs, saisies)
        for ajustement in ajustements:
            st.info(ajustement)

        erreurs = erreurs_bloquantes(valeurs)
        if erreurs:
            for erreur in erreurs:
                st.error(erreur)
            st.error("Prédiction impossible : corrigez les valeurs ci-dessus.")
            return

        for avertissement in avertissements_hors_intervalle(self.variables,
                                                             valeurs):
            st.warning(avertissement)

        ligne = construire_ligne(valeurs, self.metadonnees["colonnes"])
        probabilite = float(self.modele.predict_proba(ligne)[0][1])

        message = message_valeurs_manquantes(
            [nom for nom, brut in valeurs_brutes.items() if brut is None])
        if message is not None:
            st.warning(message)

        st.metric("Chance que l'équipe bleue gagne", f"{probabilite:.0%}")
        st.write(f"L'équipe bleue a environ {probabilite:.0%} de chances "
                 "de gagner cette partie.")
        st.write("Prédiction du modèle : "
                 + ("victoire bleue" if probabilite >= 0.5
                    else "défaite bleue"))
        st.caption(
            f"Exactitude globale du modèle : "
            f"{self.metadonnees['exactitude_modele']:.1%} sur 1 976 parties "
            "de test — ce chiffre décrit le modèle en général, "
            "pas cette partie-ci."
        )
        if not ajouter_a_historique(valeurs, probabilite):
            st.warning("Prédiction calculée, mais l'historique n'a pas pu "
                       "être enregistré (dossier en lecture seule ?).")

    def _afficher_historique(self):
        """L'historique des prédictions, lu depuis le CSV."""
        st.subheader("Historique des prédictions")
        historique = charger_historique()
        if historique.empty:
            st.write("Aucune prédiction pour l'instant.")
            return
        st.write(f"{len(historique)} prédiction(s) enregistrée(s).")
        st.dataframe(historique.iloc[::-1], use_container_width=True)
        if st.button("Effacer l'historique"):
            os.remove(chemin_historique())
            st.rerun()


def main():
    """Point d'entrée de l'application Streamlit."""
    st.set_page_config(page_title="Prédire le gagnant à 10 minutes",
                       layout="wide")
    try:
        application = Application()
    except (ModeleIntrouvableError, FileNotFoundError) as erreur:
        st.error(f"Impossible de charger le modèle : {erreur}")
        return
    application.executer()


if __name__ == "__main__":
    main()
