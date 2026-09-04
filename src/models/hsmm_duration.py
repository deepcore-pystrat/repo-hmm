from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln


@dataclass(frozen=True)
class NegativeBinomialDurationConfig:
    """
    Configuration d'une loi de durée binomiale négative décalée.

    La durée est définie par :

        D = min_duration + Y

    avec :

        Y ~ NegativeBinomial(r, p)

    où r > 0 et 0 < p < 1.
    """

    min_duration: int = 1
    max_duration: int = 252

    min_shape: float = 0.10
    max_shape: float = 1_000.0

    min_probability: float = 1e-6
    max_probability: float = 1.0 - 1e-6

    probability_floor: float = 1e-12

    def __post_init__(self) -> None:
        if self.min_duration < 1:
            raise ValueError(
                "min_duration doit être supérieur ou égal à 1."
            )

        if self.max_duration < self.min_duration:
            raise ValueError(
                "max_duration doit être supérieur ou égal "
                "à min_duration."
            )

        if self.min_shape <= 0:
            raise ValueError(
                "min_shape doit être strictement positif."
            )

        if self.max_shape <= self.min_shape:
            raise ValueError(
                "max_shape doit être supérieur à min_shape."
            )

        if not 0 < self.min_probability < self.max_probability < 1:
            raise ValueError(
                "Les bornes de probabilité doivent appartenir à ]0, 1[."
            )

        if self.probability_floor <= 0:
            raise ValueError(
                "probability_floor doit être strictement positif."
            )


class NegativeBinomialDuration:
    """
    Distribution de durée explicite pour un état HSMM.

    La distribution est tronquée sur :

        min_duration, ..., max_duration

    puis renormalisée afin que la somme soit égale à 1.
    """

    def __init__(
        self,
        shape: float,
        probability: float,
        config: NegativeBinomialDurationConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else NegativeBinomialDurationConfig()
        )

        self.shape = self._validate_shape(shape)
        self.probability = self._validate_probability(probability)

        self._pmf = self._build_truncated_pmf()
        self._survival = self._build_survival()
        self._hazard = self._build_hazard()

    def _validate_shape(self, value: float) -> float:
        value = float(value)

        if not np.isfinite(value):
            raise ValueError("shape doit être une valeur finie.")

        return float(
            np.clip(
                value,
                self.config.min_shape,
                self.config.max_shape,
            )
        )

    def _validate_probability(self, value: float) -> float:
        value = float(value)

        if not np.isfinite(value):
            raise ValueError(
                "probability doit être une valeur finie."
            )

        return float(
            np.clip(
                value,
                self.config.min_probability,
                self.config.max_probability,
            )
        )

    @property
    def support(self) -> np.ndarray:
        """
        Durées possibles :

            min_duration, ..., max_duration
        """
        return np.arange(
            self.config.min_duration,
            self.config.max_duration + 1,
            dtype=int,
        )

    def _log_pmf_untruncated(
        self,
        durations: np.ndarray,
    ) -> np.ndarray:
        """
        Log-PMF de D = min_duration + Y avec Y ~ NB(r, p).

        Convention utilisée :

            P(Y=y)
            =
            Gamma(y+r) / [Gamma(r) Gamma(y+1)]
            * p^r
            * (1-p)^y
        """
        durations = np.asarray(durations, dtype=int)

        y = durations - self.config.min_duration

        log_probability = np.full(
            shape=durations.shape,
            fill_value=-np.inf,
            dtype=float,
        )

        valid = y >= 0

        if not np.any(valid):
            return log_probability

        y_valid = y[valid].astype(float)

        r = self.shape
        p = self.probability

        log_probability[valid] = (
            gammaln(y_valid + r)
            - gammaln(r)
            - gammaln(y_valid + 1.0)
            + r * np.log(p)
            + y_valid * np.log1p(-p)
        )

        return log_probability

    def _build_truncated_pmf(self) -> np.ndarray:
        log_pmf = self._log_pmf_untruncated(self.support)

        max_log_pmf = np.max(log_pmf)

        if not np.isfinite(max_log_pmf):
            raise RuntimeError(
                "Impossible de construire la distribution de durée."
            )

        pmf = np.exp(log_pmf - max_log_pmf)

        pmf = np.maximum(
            pmf,
            self.config.probability_floor,
        )

        total = pmf.sum()

        if not np.isfinite(total) or total <= 0:
            raise RuntimeError(
                "La somme de la PMF de durée est invalide."
            )

        return pmf / total

    def _build_survival(self) -> np.ndarray:
        """
        S(d) = P(D >= d).

        L'indice 0 correspond à min_duration.
        """
        survival = np.cumsum(self._pmf[::-1])[::-1]

        return np.clip(
            survival,
            self.config.probability_floor,
            1.0,
        )

    def _build_hazard(self) -> np.ndarray:
        """
        h(d) = P(D=d | D>=d) = PMF(d) / S(d).
        """
        hazard = self._pmf / self._survival

        hazard = np.clip(
            hazard,
            0.0,
            1.0,
        )

        # À la durée maximale, le régime doit nécessairement terminer.
        hazard[-1] = 1.0

        return hazard

    def pmf(self) -> np.ndarray:
        return self._pmf.copy()

    def survival(self) -> np.ndarray:
        return self._survival.copy()

    def hazard(self) -> np.ndarray:
        return self._hazard.copy()

    def probability_of_duration(self, duration: int) -> float:
        index = self._duration_to_index(duration)
        return float(self._pmf[index])

    def survival_probability(self, duration: int) -> float:
        index = self._duration_to_index(duration)
        return float(self._survival[index])

    def exit_probability(self, age: int) -> float:
        """
        Probabilité conditionnelle que le régime se termine
        lorsqu'il a atteint l'âge donné.
        """
        index = self._duration_to_index(age)
        return float(self._hazard[index])

    def continuation_probability(self, age: int) -> float:
        return 1.0 - self.exit_probability(age)

    def expected_duration(self) -> float:
        return float(
            np.sum(
                self.support.astype(float)
                * self._pmf
            )
        )

    def variance_duration(self) -> float:
        mean = self.expected_duration()

        return float(
            np.sum(
                ((self.support - mean) ** 2)
                * self._pmf
            )
        )

    def expected_remaining_duration(self, age: int) -> float:
        """
        E[D - age | D >= age].

        À l'âge courant, le résultat indique le nombre attendu
        d'observations restantes avant la sortie du régime.
        """
        index = self._duration_to_index(age)

        valid_support = self.support[index:]
        conditional_pmf = self._pmf[index:]

        conditional_total = conditional_pmf.sum()

        if conditional_total <= 0:
            return 0.0

        conditional_pmf = conditional_pmf / conditional_total

        remaining = valid_support - age

        return float(
            np.sum(
                remaining.astype(float)
                * conditional_pmf
            )
        )

    def _duration_to_index(self, duration: int) -> int:
        duration = int(duration)

        if duration < self.config.min_duration:
            raise ValueError(
                f"La durée {duration} est inférieure à "
                f"min_duration={self.config.min_duration}."
            )

        if duration > self.config.max_duration:
            raise ValueError(
                f"La durée {duration} est supérieure à "
                f"max_duration={self.config.max_duration}."
            )

        return duration - self.config.min_duration

    @classmethod
    def fit_from_durations(
        cls,
        durations: np.ndarray,
        config: NegativeBinomialDurationConfig | None = None,
    ) -> "NegativeBinomialDuration":
        """
        Initialisation par méthode des moments.

        Pour Y = D - min_duration :

            E[Y]   = r(1-p)/p
            Var[Y] = r(1-p)/p²

        donc :

            p = E[Y] / Var[Y]
            r = E[Y]² / (Var[Y] - E[Y])

        Lorsque la variance empirique n'est pas supérieure
        à la moyenne, une approximation proche de Poisson
        est utilisée.
        """
        config = (
            config
            if config is not None
            else NegativeBinomialDurationConfig()
        )

        durations = np.asarray(
            durations,
            dtype=float,
        )

        durations = durations[np.isfinite(durations)]

        durations = durations[
            (durations >= config.min_duration)
            & (durations <= config.max_duration)
        ]

        if durations.size == 0:
            raise ValueError(
                "Aucune durée valide n'a été fournie."
            )

        y = durations - config.min_duration

        mean_y = float(np.mean(y))

        if y.size >= 2:
            variance_y = float(np.var(y, ddof=1))
        else:
            variance_y = mean_y

        if mean_y <= 1e-8:
            shape = config.max_shape
            probability = config.max_probability

        elif variance_y > mean_y + 1e-8:
            probability = mean_y / variance_y

            shape = (
                mean_y ** 2
                / (variance_y - mean_y)
            )

        else:
            # Cas sous-dispersé ou presque poissonnien.
            # Une grande valeur de shape donne une NB
            # proche d'une loi de Poisson.
            shape = config.max_shape

            probability = (
                shape
                / (shape + mean_y)
            )

        return cls(
            shape=shape,
            probability=probability,
            config=config,
        )

    def summary(self) -> dict[str, float]:
        return {
            "shape": self.shape,
            "probability": self.probability,
            "expected_duration": self.expected_duration(),
            "variance_duration": self.variance_duration(),
            "min_duration": float(self.config.min_duration),
            "max_duration": float(self.config.max_duration),
        }