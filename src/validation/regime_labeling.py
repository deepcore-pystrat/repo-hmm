import pandas as pd


def build_state_summary(regime_df: pd.DataFrame) -> pd.DataFrame:
    summary = regime_df.groupby("state_smooth").agg(
        mean_spot_ret=("dlog_spot", "mean"),
        vol_spot_ret=("dlog_spot", "std"),
        mean_fut_ret=("dlog_fut", "mean"),
        vol_fut_ret=("dlog_fut", "std"),
        mean_spread=("BA_SPREAD", "mean"),
        mean_mid=("MID", "mean"),
        count=("state_smooth", "size"),
    ).round(6)
    return summary


def label_states_from_summary(state_summary: pd.DataFrame) -> dict:
    summary = state_summary.copy()

    summary["crisis_score"] = (
        summary["vol_spot_ret"].rank(pct=True) +
        summary["vol_fut_ret"].rank(pct=True) +
        summary["mean_spread"].rank(pct=True)
    )

    crisis_state = summary["crisis_score"].idxmax()

    remaining = [s for s in summary.index if s != crisis_state]

    summary["trend_score"] = summary["mean_spot_ret"].abs()
    trend_state = summary.loc[remaining, "trend_score"].idxmax()

    calm_state = [s for s in remaining if s != trend_state][0]

    return {
        crisis_state: "crisis",
        trend_state: "trend",
        calm_state: "calm",
    }