from __future__ import annotations

from pathlib import Path
import pandas as pd


def save_saliency_report(
    ranking: pd.DataFrame,
    output_dir: str | Path = "outputs/feature_saliency",
    filename: str = "saliency_ranking.csv",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / filename
    ranking.to_csv(path, index=False)
    return path