import numpy as np
import pandas as pd


def print_causality_report(weighted_results: pd.DataFrame) -> None:
    print("\nInterpretation by regime:")
    for _, row in weighted_results.iterrows():
        print(
            f"State {row['regime']} ({row.get('state_name', 'unlabeled')}) | "
            f"significant={row['spot_to_fut_significant']} | "
            f"direction={row.get('spot_effect_direction', 'N/A')} | "
            f"sum_coef={row.get('spot_sum_coef', np.nan)} | "
            f"sum_pvalue={row.get('spot_sum_pvalue', np.nan)}"
        )