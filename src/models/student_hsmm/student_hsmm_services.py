from __future__ import annotations

import numpy as np

from models.basis.basis_data import HMMResult
from models.basis.basis_services import BaseHMM

from models.student_hmm.student_services import StudentTHMM

from models.hsmm_duration import (
    NegativeBinomialDuration,
    NegativeBinomialDurationConfig,
)

from models.hsmm_inference import (
    ExplicitDurationHSMMFilter,
    HSMMFilterContext,
    HSMMFilterResult,
    embedded_transition_matrix_from_hmm,
)

from models.student_hsmm.student_hsmm_data import (
    StudentTHSMMConfig,
)


class StudentTHSMM(StudentTHMM):
    """
    Student-t HSMM à durée explicite.

    Architecture
    ------------
    1. Les émissions Student-t sont entraînées avec le StudentTHMM.
    2. Une première séquence d'états est obtenue sur le train.
    3. Les durées empiriques des blocs sont extraites par état.
    4. Une loi binomiale négative est ajustée pour chaque état.
    5. Le filtrage causal est effectué sur les couples :

           (état courant, âge courant du régime)

    Important
    ---------
    Cette première version n'effectue pas encore un EM-HSMM complet.
    Les paramètres d'émission sont estimés par le HMM, puis réutilisés
    dans le moteur HSMM.
    """

    _label = "Student-t HSMM initialization"

    def __init__(
        self,
        config: StudentTHSMMConfig,
    ) -> None:
        super().__init__(config)

        self.cfg: StudentTHSMMConfig = config

        self.duration_models: list[
            NegativeBinomialDuration
        ] | None = None

        self.segment_transition_matrix: (
            np.ndarray | None
        ) = None

        self.hsmm_engine: (
            ExplicitDurationHSMMFilter | None
        ) = None

        self.duration_fit_summary: (
            list[dict[str, object]]
        ) = []

        self._last_hsmm_result: (
            HSMMFilterResult | None
        ) = None

    # ============================================================
    # ENTRAÎNEMENT
    # ============================================================

    def fit(
        self,
        X: np.ndarray,
    ) -> HMMResult:
        """
        Entraîne d'abord le Student-t HMM, puis initialise
        la couche de durée explicite du HSMM.

        Returns
        -------
        HMMResult
            Résultat de l'étape d'entraînement des émissions HMM.
            Les paramètres de durée sont conservés dans le modèle.
        """

        X = self._validate_X(X)

        # --------------------------------------------------------
        # 1. Entraînement des émissions Student-t par le HMM
        # --------------------------------------------------------

        hmm_result = super().fit(X)

        # --------------------------------------------------------
        # 2. Première séquence d'états pour estimer les durées
        # --------------------------------------------------------

        if self.cfg.duration_state_path == "filter":
            initial_states, _, _ = BaseHMM.filter(
                self,
                X,
            )

        elif self.cfg.duration_state_path == "smooth":
            initial_states, _, _ = BaseHMM.smooth(
                self,
                X,
            )

        else:
            raise RuntimeError(
                "duration_state_path non reconnu : "
                f"{self.cfg.duration_state_path}"
            )

        # --------------------------------------------------------
        # 3. Construction de la couche HSMM
        # --------------------------------------------------------

        self._initialize_hsmm_layer(
            states=initial_states,
        )

        return hmm_result

    # ============================================================
    # INITIALISATION DES DURÉES
    # ============================================================

    def _initialize_hsmm_layer(
        self,
        states: np.ndarray,
    ) -> None:
        """
        Initialise :

        - les distributions de durée par état ;
        - la matrice de transition entre segments ;
        - le filtre causal état × âge.
        """

        if self.A is None:
            raise RuntimeError(
                "La matrice HMM A n'est pas disponible."
            )

        if self.pi is None:
            raise RuntimeError(
                "Les probabilités initiales HMM ne sont pas disponibles."
            )

        complete_durations, all_durations, truncation_counts = (
            self._extract_state_durations(
                states=states,
                n_states=self.cfg.K,
                max_duration=self.cfg.max_duration,
            )
        )

        duration_config = NegativeBinomialDurationConfig(
            min_duration=self.cfg.min_duration,
            max_duration=self.cfg.max_duration,
        )

        duration_models: list[
            NegativeBinomialDuration
        ] = []

        duration_summary: list[
            dict[str, object]
        ] = []

        for state in range(self.cfg.K):
            complete_values = np.asarray(
                complete_durations[state],
                dtype=int,
            )

            all_values = np.asarray(
                all_durations[state],
                dtype=int,
            )

            # Les blocs internes sont privilégiés :
            # le premier et le dernier bloc peuvent être censurés.
            if (
                len(complete_values)
                >= self.cfg.min_blocks_per_state
            ):
                selected_durations = complete_values
                estimation_source = "complete_internal_blocks"

                duration_model = (
                    NegativeBinomialDuration.fit_from_durations(
                        durations=selected_durations,
                        config=duration_config,
                    )
                )

            elif (
                len(all_values)
                >= self.cfg.min_blocks_per_state
            ):
                selected_durations = all_values
                estimation_source = (
                    "all_blocks_including_boundaries"
                )

                duration_model = (
                    NegativeBinomialDuration.fit_from_durations(
                        durations=selected_durations,
                        config=duration_config,
                    )
                )

            else:
                # Pas assez de blocs empiriques :
                # initialisation géométrique cohérente avec A_kk.
                selected_durations = np.asarray(
                    [],
                    dtype=int,
                )

                estimation_source = (
                    "geometric_hmm_fallback"
                )

                continuation_probability = float(
                    np.clip(
                        self.A[state, state],
                        0.0,
                        1.0 - 1e-6,
                    )
                )

                exit_probability = (
                    1.0 - continuation_probability
                )

                # Avec shape=1 :
                #
                # D = 1 + NB(1, p)
                #
                # est une loi géométrique de moyenne 1/p.
                duration_model = NegativeBinomialDuration(
                    shape=1.0,
                    probability=exit_probability,
                    config=duration_config,
                )

            duration_models.append(
                duration_model
            )

            model_summary = duration_model.summary()

            duration_summary.append(
                {
                    "state": int(state),
                    "n_complete_blocks": int(
                        len(complete_values)
                    ),
                    "n_all_blocks": int(
                        len(all_values)
                    ),
                    "n_selected_blocks": int(
                        len(selected_durations)
                    ),
                    "truncated_blocks": int(
                        truncation_counts[state]
                    ),
                    "estimation_source": (
                        estimation_source
                    ),
                    "shape": float(
                        model_summary["shape"]
                    ),
                    "probability": float(
                        model_summary["probability"]
                    ),
                    "expected_duration": float(
                        model_summary[
                            "expected_duration"
                        ]
                    ),
                    "variance_duration": float(
                        model_summary[
                            "variance_duration"
                        ]
                    ),
                }
            )

        self.duration_models = duration_models
        self.duration_fit_summary = duration_summary

        self.segment_transition_matrix = (
            self._build_segment_transition_matrix()
        )

        self.hsmm_engine = ExplicitDurationHSMMFilter(
            duration_models=self.duration_models,
            transition_matrix=(
                self.segment_transition_matrix
            ),
            initial_state_probabilities=self.pi,
        )

        self._last_hsmm_result = None

    @staticmethod
    def _extract_state_durations(
        states: np.ndarray,
        n_states: int,
        max_duration: int,
    ) -> tuple[
        dict[int, list[int]],
        dict[int, list[int]],
        dict[int, int],
    ]:
        """
        Extrait les durées des blocs continus.

        Returns
        -------
        complete_durations:
            Blocs internes uniquement. Le premier et le dernier bloc
            sont exclus car potentiellement censurés.

        all_durations:
            Tous les blocs, y compris les frontières.

        truncation_counts:
            Nombre de blocs dépassant max_duration.
        """

        states = np.asarray(
            states,
            dtype=int,
        )

        if states.ndim != 1:
            raise ValueError(
                "states doit être un vecteur unidimensionnel."
            )

        if len(states) == 0:
            raise ValueError(
                "states ne peut pas être vide."
            )

        if np.any(states < 0) or np.any(
            states >= n_states
        ):
            raise ValueError(
                "states contient un identifiant d'état invalide."
            )

        complete_durations = {
            state: []
            for state in range(n_states)
        }

        all_durations = {
            state: []
            for state in range(n_states)
        }

        truncation_counts = {
            state: 0
            for state in range(n_states)
        }

        change_positions = (
            np.flatnonzero(
                states[1:] != states[:-1]
            )
            + 1
        )

        block_starts = np.concatenate(
            [
                np.array([0], dtype=int),
                change_positions,
            ]
        )

        block_ends = np.concatenate(
            [
                change_positions,
                np.array(
                    [len(states)],
                    dtype=int,
                ),
            ]
        )

        number_of_blocks = len(
            block_starts
        )

        for block_index, (
            start,
            end,
        ) in enumerate(
            zip(
                block_starts,
                block_ends,
            )
        ):
            state = int(
                states[start]
            )

            raw_duration = int(
                end - start
            )

            if raw_duration > max_duration:
                truncation_counts[state] += 1

            stored_duration = int(
                min(
                    raw_duration,
                    max_duration,
                )
            )

            all_durations[state].append(
                stored_duration
            )

            is_boundary_block = (
                block_index == 0
                or block_index
                == number_of_blocks - 1
            )

            if not is_boundary_block:
                complete_durations[state].append(
                    stored_duration
                )

        return (
            complete_durations,
            all_durations,
            truncation_counts,
        )

    # ============================================================
    # MATRICE DE TRANSITION ENTRE SEGMENTS
    # ============================================================

    def _build_segment_transition_matrix(
        self,
    ) -> np.ndarray:
        """
        Convertit la matrice HMM A vers une matrice HSMM B.

        Dans B :

            B[k, k] = 0

        car la persistance est contrôlée par la distribution
        explicite de durée.
        """

        if self.A is None:
            raise RuntimeError(
                "La matrice HMM A n'est pas disponible."
            )

        try:
            return embedded_transition_matrix_from_hmm(
                self.A
            )

        except ValueError:
            # Sécurité si une ligne HMM ne possède pratiquement
            # aucune probabilité hors diagonale.
            regularized = np.asarray(
                self.A,
                dtype=float,
            ).copy()

            n_states = regularized.shape[0]

            off_diagonal_mask = (
                np.ones_like(
                    regularized,
                    dtype=bool,
                )
            )

            np.fill_diagonal(
                off_diagonal_mask,
                False,
            )

            regularized[
                off_diagonal_mask
            ] += 1e-8

            regularized = (
                regularized
                / regularized.sum(
                    axis=1,
                    keepdims=True,
                )
            )

            return embedded_transition_matrix_from_hmm(
                regularized
            )

    # ============================================================
    # FILTRAGE CAUSAL HSMM
    # ============================================================

    def filter_full(
        self,
        X: np.ndarray,
        initial_context: (
            HSMMFilterContext | None
        ) = None,
    ) -> HSMMFilterResult:
        """
        Retourne tout le résultat causal HSMM.

        Cette méthode est recommandée pour conserver le contexte
        état × âge entre le train et le test.
        """

        self._require_hsmm_ready()

        X = self._validate_X(X)

        log_emissions = (
            self._compute_log_emissions(X)
        )

        result = self.hsmm_engine.filter(
            emission_log_likelihood=log_emissions,
            initial_context=initial_context,
        )

        self._last_hsmm_result = result

        return result

    def filter(
        self,
        X: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
    ]:
        """
        Interface compatible avec BaseHMM.filter().

        Commence une nouvelle séquence avec âge initial égal à 1.
        """

        result = self.filter_full(
            X=X,
            initial_context=None,
        )

        return (
            result.states,
            result.state_probabilities,
            result.log_likelihood,
        )

    def filter_with_context(
        self,
        X: np.ndarray,
        initial_context: HSMMFilterContext,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
    ]:
        """
        Filtrage causal avec continuité état × âge.

        C'est cette méthode qui doit être utilisée entre le train
        et le test.
        """

        result = self.filter_full(
            X=X,
            initial_context=initial_context,
        )

        return (
            result.states,
            result.state_probabilities,
            result.log_likelihood,
        )

    def filter_with_pi(
        self,
        X: np.ndarray,
        pi_override: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
    ]:
        """
        Compatibilité avec l'ancienne interface HMM.

        Attention
        ---------
        Cette méthode commence une nouvelle séquence à âge 1.
        Elle ne conserve donc pas l'âge du régime précédent.

        Pour une vraie continuité train/test, utiliser :

            filter_full(..., initial_context=...)
        """

        self._require_hsmm_ready()

        X = self._validate_X(X)

        pi_override = np.asarray(
            pi_override,
            dtype=float,
        )

        if pi_override.shape != (
            self.cfg.K,
        ):
            raise ValueError(
                "pi_override doit avoir la forme "
                f"({self.cfg.K},)."
            )

        if not np.all(
            np.isfinite(pi_override)
        ):
            raise ValueError(
                "pi_override contient NaN ou Inf."
            )

        if np.any(pi_override < 0):
            raise ValueError(
                "pi_override contient des valeurs négatives."
            )

        total = pi_override.sum()

        if total <= 0:
            raise ValueError(
                "La somme de pi_override doit être positive."
            )

        pi_override = (
            pi_override / total
        )

        temporary_engine = (
            ExplicitDurationHSMMFilter(
                duration_models=self.duration_models,
                transition_matrix=(
                    self.segment_transition_matrix
                ),
                initial_state_probabilities=(
                    pi_override
                ),
            )
        )

        log_emissions = (
            self._compute_log_emissions(X)
        )

        result = temporary_engine.filter(
            emission_log_likelihood=log_emissions
        )

        self._last_hsmm_result = result

        return (
            result.states,
            result.state_probabilities,
            result.log_likelihood,
        )

    # ============================================================
    # DIAGNOSTICS
    # ============================================================

    def get_last_filter_context(
        self,
    ) -> HSMMFilterContext:
        """
        Retourne le contexte état × âge de la dernière observation.
        """

        if self._last_hsmm_result is None:
            raise RuntimeError(
                "Aucun filtrage HSMM n'a encore été exécuté."
            )

        return (
            self._last_hsmm_result.last_context
        )

    def get_duration_summary(
        self,
    ) -> list[dict[str, object]]:
        """
        Résumé des lois de durée estimées.
        """

        self._require_hsmm_ready()

        return [
            dict(row)
            for row in self.duration_fit_summary
        ]

    def get_duration_pmf(
        self,
        state: int,
    ) -> np.ndarray:
        self._require_hsmm_ready()

        state = int(state)

        if state < 0 or state >= self.cfg.K:
            raise ValueError(
                f"État invalide : {state}."
            )

        return (
            self.duration_models[state].pmf()
        )

    def get_duration_hazard(
        self,
        state: int,
    ) -> np.ndarray:
        self._require_hsmm_ready()

        state = int(state)

        if state < 0 or state >= self.cfg.K:
            raise ValueError(
                f"État invalide : {state}."
            )

        return (
            self.duration_models[state].hazard()
        )

    def smooth(
        self,
        X: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
    ]:
        """
        Le lissage HSMM complet n'est pas encore implémenté.

        Le lissage HMM peut être utilisé uniquement pendant
        l'initialisation des durées via duration_state_path="smooth".
        """

        raise NotImplementedError(
            "Le lissage HSMM état × âge n'est pas encore "
            "implémenté. Utilisez filter() ou filter_full()."
        )

    # ============================================================
    # VALIDATIONS
    # ============================================================

    def _require_hsmm_ready(
        self,
    ) -> None:
        if self.duration_models is None:
            raise RuntimeError(
                "Les modèles de durée ne sont pas initialisés. "
                "Exécutez fit(X_train) avant le filtrage."
            )

        if self.segment_transition_matrix is None:
            raise RuntimeError(
                "La matrice de transition HSMM "
                "n'est pas initialisée."
            )

        if self.hsmm_engine is None:
            raise RuntimeError(
                "Le moteur HSMM n'est pas initialisé."
            )

    @staticmethod
    def _validate_X(
        X: np.ndarray,
    ) -> np.ndarray:
        X = np.asarray(
            X,
            dtype=float,
        )

        if X.ndim != 2:
            raise ValueError(
                "X doit être une matrice "
                "(n_observations, n_features)."
            )

        if X.shape[0] == 0:
            raise ValueError(
                "X ne peut pas être vide."
            )

        if X.shape[1] == 0:
            raise ValueError(
                "X doit contenir au moins une variable."
            )

        if not np.all(
            np.isfinite(X)
        ):
            raise ValueError(
                "X contient NaN ou Inf."
            )

        return X