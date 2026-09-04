from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .hsmm_duration import NegativeBinomialDuration


@dataclass(frozen=True)
class HSMMFilterContext:
    """
    Contexte causal à transmettre entre deux périodes.

    Exemple :
        fin du train -> début du test.

    age_state_posterior[k, a] représente :

        P(
            état courant = k,
            âge courant = a + 1
            | observations disponibles
        )
    """

    age_state_posterior: np.ndarray

    def __post_init__(self) -> None:
        posterior = np.asarray(
            self.age_state_posterior,
            dtype=float,
        )

        if posterior.ndim != 2:
            raise ValueError(
                "age_state_posterior doit avoir la forme "
                "(n_states, max_duration)."
            )

        if not np.all(np.isfinite(posterior)):
            raise ValueError(
                "age_state_posterior contient NaN ou Inf."
            )

        if np.any(posterior < 0):
            raise ValueError(
                "age_state_posterior contient des probabilités négatives."
            )

        total = posterior.sum()

        if not np.isfinite(total) or total <= 0:
            raise ValueError(
                "La somme du posterior état × âge doit être positive."
            )

        normalized = posterior / total

        object.__setattr__(
            self,
            "age_state_posterior",
            normalized,
        )


@dataclass(frozen=True)
class HSMMFilterResult:
    """
    Résultat complet du filtrage causal HSMM.
    """

    states: np.ndarray

    # P(S_t = k | X_1:t)
    state_probabilities: np.ndarray

    # P(S_t = k, Age_t = a | X_1:t)
    age_state_probabilities: np.ndarray

    # E[Age_t | X_1:t]
    expected_age: np.ndarray

    # Probabilité que le régime courant se termine après t
    exit_probability: np.ndarray

    # Durée restante attendue du régime courant
    expected_remaining_duration: np.ndarray

    log_likelihood: float

    last_context: HSMMFilterContext


def embedded_transition_matrix_from_hmm(
    hmm_transition_matrix: np.ndarray,
    atol: float = 1e-10,
) -> np.ndarray:
    """
    Transforme la matrice de transition d'un HMM en matrice
    de transition entre segments pour un HSMM.

    Dans un HMM :

        A[i, i]

    représente la persistance de l'état.

    Dans un HSMM explicite, cette persistance est gérée par la
    distribution de durée. La diagonale doit donc être nulle.

    On calcule :

        B[i, j]
        =
        A[i, j] / sum_{l != i} A[i, l]

    pour j != i.
    """

    matrix = np.asarray(
        hmm_transition_matrix,
        dtype=float,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "La matrice HMM doit être bidimensionnelle."
        )

    n_rows, n_columns = matrix.shape

    if n_rows != n_columns:
        raise ValueError(
            "La matrice HMM doit être carrée."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "La matrice HMM contient NaN ou Inf."
        )

    if np.any(matrix < 0):
        raise ValueError(
            "La matrice HMM contient des probabilités négatives."
        )

    row_sums = matrix.sum(axis=1)

    if not np.allclose(
        row_sums,
        1.0,
        atol=atol,
    ):
        raise ValueError(
            "Chaque ligne de la matrice HMM doit sommer à 1."
        )

    embedded = matrix.copy()

    np.fill_diagonal(
        embedded,
        0.0,
    )

    off_diagonal_sums = embedded.sum(axis=1)

    if np.any(off_diagonal_sums <= atol):
        bad_states = np.where(
            off_diagonal_sums <= atol
        )[0].tolist()

        raise ValueError(
            "Impossible de construire une transition HSMM pour "
            f"les états {bad_states} : aucune probabilité de sortie."
        )

    embedded = (
        embedded
        / off_diagonal_sums[:, None]
    )

    np.fill_diagonal(
        embedded,
        0.0,
    )

    return embedded


class ExplicitDurationHSMMFilter:
    """
    Filtre causal HSMM à durée explicite.

    Le filtre travaille sur les couples :

        (état courant, âge courant du régime)

    La partie émission n'est pas calculée ici.

    Le modèle appelant doit fournir :

        emission_log_likelihood[t, k]
        =
        log p(X_t | S_t = k)

    Cela permettra ensuite de réutiliser directement les émissions
    Student-t de votre modèle actuel.
    """

    def __init__(
        self,
        duration_models: list[NegativeBinomialDuration],
        transition_matrix: np.ndarray,
        initial_state_probabilities: np.ndarray | None = None,
    ) -> None:

        if len(duration_models) < 2:
            raise ValueError(
                "Le HSMM doit contenir au moins deux états."
            )

        self.duration_models = list(
            duration_models
        )

        self.n_states = len(
            self.duration_models
        )

        self.max_duration = self._validate_duration_models()

        self.transition_matrix = self._validate_transition_matrix(
            transition_matrix
        )

        self.initial_state_probabilities = (
            self._validate_initial_probabilities(
                initial_state_probabilities
            )
        )

        self.hazard_matrix = self._build_hazard_matrix()

        self.continuation_matrix = np.clip(
            1.0 - self.hazard_matrix,
            0.0,
            1.0,
        )

        # À la durée maximale, aucune continuation n'est autorisée.
        self.continuation_matrix[:, -1] = 0.0

        self.expected_remaining_matrix = (
            self._build_expected_remaining_matrix()
        )

    def _validate_duration_models(self) -> int:
        first_config = self.duration_models[0].config

        if first_config.min_duration != 1:
            raise ValueError(
                "Cette première version du filtre HSMM nécessite "
                "min_duration=1."
            )

        max_duration = first_config.max_duration

        for state, model in enumerate(
            self.duration_models
        ):
            if not isinstance(
                model,
                NegativeBinomialDuration,
            ):
                raise TypeError(
                    "Chaque modèle de durée doit être une instance de "
                    "NegativeBinomialDuration. "
                    f"État invalide : {state}."
                )

            if model.config.min_duration != 1:
                raise ValueError(
                    f"L'état {state} n'utilise pas min_duration=1."
                )

            if model.config.max_duration != max_duration:
                raise ValueError(
                    "Tous les états doivent avoir le même max_duration."
                )

            hazard = model.hazard()

            if hazard.shape != (max_duration,):
                raise ValueError(
                    f"Hazard invalide pour l'état {state}."
                )

        return int(max_duration)

    def _validate_transition_matrix(
        self,
        transition_matrix: np.ndarray,
    ) -> np.ndarray:

        matrix = np.asarray(
            transition_matrix,
            dtype=float,
        )

        expected_shape = (
            self.n_states,
            self.n_states,
        )

        if matrix.shape != expected_shape:
            raise ValueError(
                "transition_matrix doit avoir la forme "
                f"{expected_shape}, reçu {matrix.shape}."
            )

        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "transition_matrix contient NaN ou Inf."
            )

        if np.any(matrix < 0):
            raise ValueError(
                "transition_matrix contient des valeurs négatives."
            )

        if not np.allclose(
            np.diag(matrix),
            0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Dans un HSMM explicite, la diagonale de la matrice "
                "de transition doit être nulle."
            )

        row_sums = matrix.sum(axis=1)

        if np.any(row_sums <= 0):
            raise ValueError(
                "Chaque état doit pouvoir passer vers au moins "
                "un autre état."
            )

        matrix = (
            matrix
            / row_sums[:, None]
        )

        return matrix

    def _validate_initial_probabilities(
        self,
        probabilities: np.ndarray | None,
    ) -> np.ndarray:

        if probabilities is None:
            return np.full(
                self.n_states,
                1.0 / self.n_states,
                dtype=float,
            )

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        if probabilities.shape != (
            self.n_states,
        ):
            raise ValueError(
                "initial_state_probabilities doit avoir la forme "
                f"({self.n_states},)."
            )

        if not np.all(np.isfinite(probabilities)):
            raise ValueError(
                "initial_state_probabilities contient NaN ou Inf."
            )

        if np.any(probabilities < 0):
            raise ValueError(
                "initial_state_probabilities contient "
                "des probabilités négatives."
            )

        total = probabilities.sum()

        if total <= 0:
            raise ValueError(
                "La somme des probabilités initiales doit être positive."
            )

        return probabilities / total

    def _build_hazard_matrix(self) -> np.ndarray:
        """
        hazard_matrix[k, a] correspond à la probabilité de sortie
        de l'état k lorsque son âge vaut a + 1.
        """

        hazard = np.vstack(
            [
                model.hazard()
                for model in self.duration_models
            ]
        )

        if hazard.shape != (
            self.n_states,
            self.max_duration,
        ):
            raise RuntimeError(
                "Dimensions invalides pour la matrice de hazard."
            )

        if not np.all(np.isfinite(hazard)):
            raise RuntimeError(
                "La matrice de hazard contient NaN ou Inf."
            )

        hazard = np.clip(
            hazard,
            0.0,
            1.0,
        )

        hazard[:, -1] = 1.0

        return hazard

    def _build_expected_remaining_matrix(
        self,
    ) -> np.ndarray:
        """
        expected_remaining_matrix[k, a] contient :

            E[D_k - age | D_k >= age]

        avec age = a + 1.
        """

        matrix = np.zeros(
            (
                self.n_states,
                self.max_duration,
            ),
            dtype=float,
        )

        for state, model in enumerate(
            self.duration_models
        ):
            for age in range(
                1,
                self.max_duration + 1,
            ):
                matrix[state, age - 1] = (
                    model.expected_remaining_duration(
                        age=age
                    )
                )

        return matrix

    @staticmethod
    def _safe_log(
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """
        log(0) = -Inf sans produire d'avertissement.
        """

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        output = np.full(
            probabilities.shape,
            -np.inf,
            dtype=float,
        )

        positive = probabilities > 0

        output[positive] = np.log(
            probabilities[positive]
        )

        return output

    def _initial_log_prediction(
        self,
    ) -> np.ndarray:
        """
        Au début d'une nouvelle série, tous les états commencent
        à l'âge 1.
        """

        log_prediction = np.full(
            (
                self.n_states,
                self.max_duration,
            ),
            -np.inf,
            dtype=float,
        )

        log_prediction[:, 0] = self._safe_log(
            self.initial_state_probabilities
        )

        return log_prediction

    def _predict_from_previous_posterior(
        self,
        previous_posterior: np.ndarray,
    ) -> np.ndarray:
        """
        Étape prédictive :

            P(S_t, Age_t | X_1:t-1)

        à partir de :

            P(S_t-1, Age_t-1 | X_1:t-1)
        """

        previous_posterior = np.asarray(
            previous_posterior,
            dtype=float,
        )

        expected_shape = (
            self.n_states,
            self.max_duration,
        )

        if previous_posterior.shape != expected_shape:
            raise ValueError(
                "Le posterior précédent doit avoir la forme "
                f"{expected_shape}."
            )

        if not np.all(np.isfinite(previous_posterior)):
            raise ValueError(
                "Le posterior précédent contient NaN ou Inf."
            )

        if np.any(previous_posterior < 0):
            raise ValueError(
                "Le posterior précédent contient des valeurs négatives."
            )

        total = previous_posterior.sum()

        if total <= 0:
            raise ValueError(
                "La somme du posterior précédent doit être positive."
            )

        previous_posterior = (
            previous_posterior
            / total
        )

        log_previous = self._safe_log(
            previous_posterior
        )

        log_hazard = self._safe_log(
            self.hazard_matrix
        )

        log_continuation = self._safe_log(
            self.continuation_matrix
        )

        log_transition = self._safe_log(
            self.transition_matrix
        )

        log_prediction = np.full(
            expected_shape,
            -np.inf,
            dtype=float,
        )

        # ============================================================
        # 1. CONTINUATION DU MÊME RÉGIME
        # ============================================================
        #
        # état k, âge a
        #     ->
        # état k, âge a + 1
        #

        log_prediction[:, 1:] = (
            log_previous[:, :-1]
            + log_continuation[:, :-1]
        )

        # ============================================================
        # 2. FIN D'UN RÉGIME ET ENTRÉE DANS UN AUTRE
        # ============================================================
        #
        # Toutes les probabilités de sortie d'un état source
        # sont agrégées, puis distribuées vers les autres états.
        #

        log_exit_mass_by_source_state = logsumexp(
            log_previous + log_hazard,
            axis=1,
        )

        for destination_state in range(
            self.n_states
        ):
            transition_terms = (
                log_exit_mass_by_source_state
                + log_transition[
                    :,
                    destination_state,
                ]
            )

            log_prediction[
                destination_state,
                0,
            ] = logsumexp(
                transition_terms
            )

        return log_prediction

    def _validate_emission_log_likelihood(
        self,
        emission_log_likelihood: np.ndarray,
    ) -> np.ndarray:

        emissions = np.asarray(
            emission_log_likelihood,
            dtype=float,
        )

        if emissions.ndim != 2:
            raise ValueError(
                "emission_log_likelihood doit être une matrice "
                "(n_observations, n_states)."
            )

        if emissions.shape[1] != self.n_states:
            raise ValueError(
                "Le nombre de colonnes d'émission doit être égal "
                f"au nombre d'états : {self.n_states}."
            )

        if emissions.shape[0] == 0:
            raise ValueError(
                "Aucune observation d'émission n'a été fournie."
            )

        if np.any(np.isnan(emissions)):
            raise ValueError(
                "emission_log_likelihood contient NaN."
            )

        if np.any(np.isposinf(emissions)):
            raise ValueError(
                "emission_log_likelihood contient +Inf."
            )

        valid_rows = np.any(
            np.isfinite(emissions),
            axis=1,
        )

        if not np.all(valid_rows):
            bad_rows = np.where(
                ~valid_rows
            )[0].tolist()

            raise ValueError(
                "Toutes les émissions sont impossibles pour les "
                f"observations {bad_rows}."
            )

        return emissions

    def filter(
        self,
        emission_log_likelihood: np.ndarray,
        initial_context: HSMMFilterContext | None = None,
    ) -> HSMMFilterResult:
        """
        Filtrage causal du HSMM.

        Paramètres
        ----------
        emission_log_likelihood:
            Matrice de forme (T, K).

            Chaque ligne t contient :

                log p(X_t | S_t = k)

        initial_context:
            None :
                début d'une nouvelle série.

            HSMMFilterContext :
                continuation d'une série précédente, notamment
                passage du train vers le test.

        Returns
        -------
        HSMMFilterResult
        """

        emissions = self._validate_emission_log_likelihood(
            emission_log_likelihood
        )

        n_observations = emissions.shape[0]

        age_state_probabilities = np.zeros(
            (
                n_observations,
                self.n_states,
                self.max_duration,
            ),
            dtype=float,
        )

        total_log_likelihood = 0.0

        previous_posterior: np.ndarray | None

        if initial_context is None:
            previous_posterior = None
        else:
            previous_posterior = np.asarray(
                initial_context.age_state_posterior,
                dtype=float,
            )

            expected_shape = (
                self.n_states,
                self.max_duration,
            )

            if previous_posterior.shape != expected_shape:
                raise ValueError(
                    "Le contexte initial est incompatible avec "
                    f"le filtre. Forme attendue : {expected_shape}, "
                    f"reçue : {previous_posterior.shape}."
                )

        for time_index in range(
            n_observations
        ):
            if (
                time_index == 0
                and previous_posterior is None
            ):
                log_prediction = (
                    self._initial_log_prediction()
                )
            else:
                if time_index > 0:
                    previous_posterior = (
                        age_state_probabilities[
                            time_index - 1
                        ]
                    )

                if previous_posterior is None:
                    raise RuntimeError(
                        "Posterior précédent indisponible."
                    )

                log_prediction = (
                    self._predict_from_previous_posterior(
                        previous_posterior
                    )
                )

            # L'émission dépend uniquement de l'état,
            # pas directement de l'âge.
            log_unnormalized = (
                log_prediction
                + emissions[
                    time_index,
                    :,
                    None,
                ]
            )

            log_normalizer = logsumexp(
                log_unnormalized
            )

            if not np.isfinite(log_normalizer):
                raise RuntimeError(
                    "Impossible de normaliser le posterior HSMM "
                    f"à l'observation {time_index}."
                )

            log_posterior = (
                log_unnormalized
                - log_normalizer
            )

            posterior = np.exp(
                log_posterior
            )

            posterior_sum = posterior.sum()

            if (
                not np.isfinite(posterior_sum)
                or posterior_sum <= 0
            ):
                raise RuntimeError(
                    "Posterior HSMM invalide à l'observation "
                    f"{time_index}."
                )

            posterior = (
                posterior
                / posterior_sum
            )

            age_state_probabilities[
                time_index
            ] = posterior

            total_log_likelihood += float(
                log_normalizer
            )

        state_probabilities = (
            age_state_probabilities.sum(
                axis=2
            )
        )

        states = np.argmax(
            state_probabilities,
            axis=1,
        ).astype(int)

        age_values = np.arange(
            1,
            self.max_duration + 1,
            dtype=float,
        )

        expected_age = np.einsum(
            "tkd,d->t",
            age_state_probabilities,
            age_values,
        )

        exit_probability = np.einsum(
            "tkd,kd->t",
            age_state_probabilities,
            self.hazard_matrix,
        )

        expected_remaining_duration = np.einsum(
            "tkd,kd->t",
            age_state_probabilities,
            self.expected_remaining_matrix,
        )

        last_context = HSMMFilterContext(
            age_state_posterior=(
                age_state_probabilities[-1].copy()
            )
        )

        return HSMMFilterResult(
            states=states,
            state_probabilities=state_probabilities,
            age_state_probabilities=(
                age_state_probabilities
            ),
            expected_age=expected_age,
            exit_probability=exit_probability,
            expected_remaining_duration=(
                expected_remaining_duration
            ),
            log_likelihood=float(
                total_log_likelihood
            ),
            last_context=last_context,
        )