"""
Diagnostic tools to validate the diagonal covariance assumption.

After fitting an HMM with diagonal emissions, call:

    check_diagonal_assumption(X_train_z, states_train, feature_names)

to see if the assumption holds for the features you chose.
"""

from __future__ import annotations

import numpy as np


# ── Within-regime correlation analysis ────────────────────────────────

def check_diagonal_assumption(
    X: np.ndarray,
    states: np.ndarray,
    feature_names: list[str] | None = None,
    threshold: float = 0.30,
) -> dict:
    """Check whether the diagonal covariance assumption is reasonable.

    For each regime *k*, computes the correlation matrix of the
    observations assigned to that regime.  If large off-diagonal
    correlations exist, the diagonal assumption is violated.

    Parameters
    ----------
    X : (T, d) array — standardised features (same as used for fit/filter).
    states : (T,) array — state assignments (e.g. from filter or viterbi).
    feature_names : optional names for pretty-printing.
    threshold : absolute correlation above which a pair is flagged.

    Returns
    -------
    dict with keys:
        "per_state" : list of dicts per state with corr_matrix, max_offdiag, flagged_pairs
        "global_ok" : bool — True if *all* states pass the diagonal test
        "worst_pair": (state, feat_i, feat_j, corr) — single worst violation
    """
    X = np.asarray(X, dtype=float)
    states = np.asarray(states, dtype=int)
    T, d = X.shape

    if feature_names is None:
        feature_names = [f"feat_{j}" for j in range(d)]

    unique_states = np.unique(states)
    per_state: list[dict] = []
    global_ok = True
    worst = (0, 0, 0, 0.0)  # (state, i, j, corr)

    for k in unique_states:
        mask = states == k
        Xk = X[mask]
        n_k = mask.sum()

        if n_k < d + 2:
            per_state.append({
                "state": int(k), "n": n_k,
                "corr_matrix": None, "max_offdiag": np.nan,
                "flagged_pairs": [], "warning": "too few samples",
            })
            continue

        # Correlation matrix within this regime
        corr = np.corrcoef(Xk, rowvar=False)  # (d, d)

        # Off-diagonal max
        mask_offdiag = ~np.eye(d, dtype=bool)
        abs_offdiag = np.abs(corr[mask_offdiag])
        max_offdiag = float(abs_offdiag.max()) if abs_offdiag.size > 0 else 0.0

        # Flag pairs above threshold
        flagged = []
        for i in range(d):
            for j in range(i + 1, d):
                r_ij = corr[i, j]
                if abs(r_ij) > threshold:
                    flagged.append((feature_names[i], feature_names[j], float(r_ij)))
                    if abs(r_ij) > abs(worst[3]):
                        worst = (int(k), i, j, float(r_ij))

        if flagged:
            global_ok = False

        per_state.append({
            "state": int(k), "n": n_k,
            "corr_matrix": corr,
            "max_offdiag": max_offdiag,
            "flagged_pairs": flagged,
        })

    return {
        "per_state": per_state,
        "global_ok": global_ok,
        "worst_pair": worst,
    }


# ── Pretty-print ─────────────────────────────────────────────────────

def print_diagonal_diagnostic(
    X: np.ndarray,
    states: np.ndarray,
    feature_names: list[str] | None = None,
    threshold: float = 0.30,
) -> dict:
    """Run the diagnostic and print a human-readable report.

    Returns the raw diagnostic dict for programmatic use.
    """
    diag = check_diagonal_assumption(X, states, feature_names, threshold)

    print("\n" + "=" * 60)
    print("DIAGONAL COVARIANCE DIAGNOSTIC")
    print(f"  Threshold: |r| > {threshold}")
    print("=" * 60)

    for info in diag["per_state"]:
        k = info["state"]
        n = info["n"]
        print(f"\n  State {k}  (n={n})")

        if "warning" in info:
            print(f"    ⚠ {info['warning']}")
            continue

        max_r = info["max_offdiag"]
        status = "✓ OK" if not info["flagged_pairs"] else "✗ VIOLATED"
        print(f"    Max |off-diag corr|: {max_r:.3f}  → {status}")

        if info["flagged_pairs"]:
            for fi, fj, r_ij in info["flagged_pairs"]:
                print(f"      {fi}  ↔  {fj}  :  r = {r_ij:+.3f}")

        # Print full correlation matrix if small enough
        corr = info["corr_matrix"]
        if corr is not None and corr.shape[0] <= 8:
            names = feature_names or [f"f{j}" for j in range(corr.shape[0])]
            # Compact table
            max_name = max(len(n) for n in names)
            header = " " * (max_name + 4) + "  ".join(f"{n[:6]:>6s}" for n in names)
            print(f"\n    {header}")
            for i, name in enumerate(names):
                row = "  ".join(f"{corr[i, j]:+6.3f}" for j in range(len(names)))
                print(f"    {name:>{max_name}s}  {row}")

    print("\n" + "-" * 60)
    if diag["global_ok"]:
        print("  RESULT: Diagonal assumption looks VALID for all regimes.")
        print("  No feature pair exceeds the correlation threshold.")
    else:
        sk, i, j, r = diag["worst_pair"]
        names = feature_names or [f"feat_{j}" for j in range(X.shape[1])]
        print(f"  RESULT: Diagonal assumption VIOLATED in at least one regime.")
        print(f"  Worst: state {sk}, {names[i]} ↔ {names[j]}, r = {r:+.3f}")
        print(f"\n  Options:")
        print(f"    1. Remove one of the correlated features")
        print(f"    2. Replace with a decorrelated version (PCA, residualise)")
        print(f"    3. Switch to full-covariance HMM")
    print("-" * 60)

    return diag
