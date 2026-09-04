"""
Walk-forward HMM regime-sizing strategy -- sugar markets.

Instead of a single 80/20 split the data is sliced into rolling windows
(train_bars / test_bars).  On each window the full pipeline runs:

    standardise → fit HMM → filter train → map exposures → filter test → backtest

Strategy: always LONG (buy & hold), HMM regimes control exposure level.
Calm regime → full exposure (1.0), turbulent → reduced (inverse vol ratio).
"""
from __future__ import annotations

import pandas as pd

from data_loader.loader_data import DataConfig
from data_loader.laoding_services import get_sugar_data, load_futures_sugar
from features.registry import build_features
from models import StudentTHMMConfig, GaussianHMMConfig
from validation.walk_forward_engine import walk_forward


# ── Data loading ──────────────────────────────────────────────────────

def load_sugar_data() -> pd.DataFrame:
    cfg_vhp  = DataConfig(data_path="data/data_spot_VHP.xlsx",     asset_name="VHP")
    cfg_thp  = DataConfig(data_path="data/data_spot_THP.xlsx",     asset_name="THP")
    cfg_sb11 = DataConfig(data_path="data/data_futures_sb11.xlsx", asset_name="SB11")

    df = pd.concat(
        [get_sugar_data(cfg_vhp), get_sugar_data(cfg_thp), load_futures_sugar(cfg_sb11)],
        axis=1,
    ).dropna()

    df["SPREAD_VHP_THP"] = df["MID_VHP"] - df["MID_THP"]

    # import matplotlib.pyplot as plt
    # fig, ax1 = plt.subplots(figsize=(10, 6))
    # ax2 = ax1.twinx()
    # for year in range(2016, 2017):
    #     chunk = df['SPREAD_VHP_THP'].loc[f'{year}-01-01':f'{year+10}-12-31']
    #     chunk_ = df['Close'].loc[f'{year}-01-01':f'{year+10}-12-31']

    #     if not chunk.empty:
    #         ax1.plot(range(len(chunk)), chunk.values, label=f"SPREAD_VHP_THP {year}", color="tab:blue")
    #         ax2.plot(range(len(chunk_)), chunk_.values, label=f"Close {year}", color="tab:orange")
    # ax1.set_xlabel("Jour de l'année")
    # ax1.set_ylabel("MID_VHP (spot)", color="tab:blue")
    # ax2.set_ylabel("Close (futures)", color="tab:orange")
    # ax1.legend(loc="upper left")
    # ax2.legend(loc="upper right")
    # plt.show()



    return df


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:

    # ── Configuration ─────────────────────────────────────────────────
    feature_specs = [
        # ("id", "Volume"),
        # ("binary_returns", "SPREAD_VHP_THP"),

        ("rvol",       "Close", {"window": 10}),            
        ("rvol_ratio", "Close", {"fast": 5, "slow": 21}),   
        ("atr",        "Close", {"window": 10}),            
        ("log_range",  "Close", {"window": 5}),             

        # ("rvol",       "SPREAD_VHP_THP", {"window": 10}),
        # ("log_range",  "SPREAD_VHP_THP", {"window": 5}),




        # ("rvol_ratio", "Close", {"fast": 5, "slow": 21}),
        # ("zscore", "SPREAD_VHP_THP", {"window": 42}),

        # ("diff_ma", "Close", {"period": 1, "window": 21}),

        # ("id", "Close"),                          # direction brute
        # ("rvol", "Close", {"window": 10}),             # vol courte (spike = crash)
        # ("drawdown", "Close", {"window": 63}),         # distance au plus haut (négatif = chute)
        # ("skew", "SPREAD_VHP_THP", {"window": 21}),             # skew négative = fat left tail

        # ("skew", "Close", {"window": 42}),
        # ("kurt", "Close", {"window": 42}), 
        # ("returns",    "MID_VHP"),
        # ("returns",    "MID_THP"),
        # ("id",    "MID_VHP"),
        # ("id",    "MID_THP"),
        # ("diff_ma", "MID_VHP", {"period": 1, "window": 21}),
        # ("diff_ma", "MID_THP", {"period": 1, "window": 21}),
        # ("diff_ma", "SPREAD_VHP_THP", {"period": 1, "window": 21}),

        # ("ma", "MID_THP", {"window": 63}),
        # ("ma", "MID_VHP", {"window": 63}),
        # ("ma", "SPREAD_VHP_THP", {"window": 63}),
        # ("delta", "SPREAD_VHP_THP"),
        # ("rvol", "SPREAD_VHP_THP", {"window": 21}),
        # ("log_returns", "SPREAD_VHP_THP"),
        # ("zscore", "SPREAD_VHP_THP", {"window": 20}),
        # ("kurt","SPREAD_VHP_THP", {"window": 20}),
        # ("skew","SPREAD_VHP_THP", {"window": 20}),
        # ("log_returns", "SPREAD_VHP_THP"),
    ]

    # hmm_cfg = StudentTHMMConfig(
    #     K=2, n_iter=30, seed=42,
    #     init_nu=5.0, estimate_nu=True, cov_type="full",
    # )
    # hmm_cfg = GaussianHMMConfig(K=2, n_iter=30, seed=42, cov_type="full")
    hmm_cfg = StudentTHMMConfig(K=2, n_iter=40, seed=42, init_nu=5.0, estimate_nu=True, cov_type="full")


    # train_bars = 756
    # test_bars  = 189

    # train_bars = 168
    # test_bars  = 42

    # train_bars = 504
    # test_bars  = 126

    train_bars = 1008
    test_bars  = 252

    # ── 1) Load data ──────────────────────────────────────────────────
    print("Loading data ...")
    data_df = load_sugar_data()
    print(f"  {data_df.shape}  |  {data_df.index[0]} → {data_df.index[-1]}")

    # ── 2) Build features ─────────────────────────────────────────────
    print("\nBuilding features ...")
    feat_df = build_features(data_df, feature_specs, shift_cols={"Close": 1, "Volume": 1})
    print(f"  {feat_df.shape}  |  {feat_df.index[0]} → {feat_df.index[-1]}")

    # ── 3) Walk-forward ───────────────────────────────────────────────
    walk_forward(
        data_df=data_df,
        feat_df=feat_df,
        hmm_cfg=hmm_cfg,
        train_bars=train_bars,
        test_bars=test_bars,
        bear_threshold=-0.05,   # Sharpe < -0.5 AND mean<0 → exit (expo=0)
        bull_threshold=0.05,    # Sharpe > +0.5 AND mean>0 → boost (expo=max)
        max_exposure=1.5,      # multiplier for bullish regimes
        vol_target=0.20,
        vol_lookback=42,
        max_leverage=2.0,
        K_range=[2],
    )


if __name__ == "__main__":
    main()
