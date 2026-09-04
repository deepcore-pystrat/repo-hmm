from pathlib import Path
import sys

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from models.student_hsmm import (
    StudentTHSMM,
    StudentTHSMMConfig,
)


rng = np.random.default_rng(42)

X_train = np.vstack(
    [
        rng.normal(-2.0, 0.5, size=(100, 3)),
        rng.normal(0.0, 1.0, size=(100, 3)),
        rng.normal(2.0, 0.7, size=(100, 3)),
    ]
)

X_test = np.vstack(
    [
        rng.normal(-2.0, 0.5, size=(20, 3)),
        rng.normal(2.0, 0.7, size=(20, 3)),
    ]
)

config = StudentTHSMMConfig(
    K=3,
    n_iter=10,
    seed=42,
    cov_type="diag",
    sticky=1.0,
    estimate_nu=True,
    min_duration=1,
    max_duration=60,
    min_blocks_per_state=2,
    duration_state_path="filter",
)

model = StudentTHSMM(config)

model.fit(X_train)

train_result = model.filter_full(X_train)

test_result = model.filter_full(
    X_test,
    initial_context=train_result.last_context,
)

print("\nDuration summary:")

for row in model.get_duration_summary():
    print(row)

print("\nTrain:")
print("states shape:", train_result.states.shape)
print("probabilities shape:", train_result.state_probabilities.shape)
print("posterior sum:", train_result.last_context.age_state_posterior.sum())

print("\nTest:")
print("states shape:", test_result.states.shape)
print("probabilities shape:", test_result.state_probabilities.shape)
print("posterior sum:", test_result.last_context.age_state_posterior.sum())
print("expected age:", test_result.expected_age[-5:])
print("exit probability:", test_result.exit_probability[-5:])