"""Exceptions personnalisées du projet."""


class ErreurProjet(Exception):
    """Classe parente de toutes les erreurs du projet."""


class DonneeInvalideError(ErreurProjet):
    """Levée quand une donnée est invalide (colonne absente, saisie hors intervalle, etc.)."""


class ModeleIntrouvableError(ErreurProjet):
    """Levée quand le fichier du modèle ou des métadonnées est introuvable."""