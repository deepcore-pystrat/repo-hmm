from __future__ import annotations

from dataclasses import dataclass

from models.student_hmm.student_data import StudentTHMMConfig


@dataclass
class StudentTHSMMConfig(StudentTHMMConfig):
    """
    Configuration du Student-t HSMM.

    Les paramètres hérités de StudentTHMMConfig contrôlent
    les émissions Student-t.

    Les paramètres ci-dessous contrôlent uniquement les durées
    explicites des régimes.
    """

    # Première version du moteur :
    # les régimes peuvent durer entre 1 et max_duration observations.
    min_duration: int = 1
    max_duration: int = 252

    # Nombre minimal de blocs empiriques requis pour estimer
    # une distribution de durée spécifique à un état.
    min_blocks_per_state: int = 3

    # Séquence utilisée pour initialiser les durées :
    # "filter" = états causaux ;
    # "smooth" = états lissés sur le train uniquement.
    duration_state_path: str = "filter"

    def __post_init__(self) -> None:
        if self.min_duration != 1:
            raise ValueError(
                "La première version du filtre HSMM nécessite "
                "min_duration=1."
            )

        if self.max_duration < self.min_duration:
            raise ValueError(
                "max_duration doit être supérieur ou égal "
                "à min_duration."
            )

        if self.max_duration < 2:
            raise ValueError(
                "max_duration doit être supérieur ou égal à 2."
            )

        if self.min_blocks_per_state < 1:
            raise ValueError(
                "min_blocks_per_state doit être supérieur "
                "ou égal à 1."
            )

        allowed_paths = {
            "filter",
            "smooth",
        }

        if self.duration_state_path not in allowed_paths:
            raise ValueError(
                "duration_state_path doit être 'filter' ou 'smooth'."
            )