from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import regime2_subhmm_3states as module

FEATURE_COLUMNS = module.FEATURE_COLUMNS
SubHMMConfig = module.SubHMMConfig
build_causal_futures_features = module.build_causal_futures_features
project_main_feed_to_futures = module.project_main_feed_to_futures
build_regime2_sequences = module.build_regime2_sequences


def test_features_are_causal():
    index = pd.date_range("2020-01-01", periods=120, freq="B")
    price = pd.Series(np.linspace(100.0, 120.0, len(index)), index=index)

    futures_a = pd.DataFrame({"FUTURES_CLOSE": price})
    futures_b = futures_a.copy()
    futures_b.loc[index[100]:, "FUTURES_CLOSE"] *= 2.0

    a = build_causal_futures_features(futures_a, price_column="FUTURES_CLOSE")
    b = build_causal_futures_features(futures_b, price_column="FUTURES_CLOSE")

    pd.testing.assert_frame_equal(a.loc[:index[99]], b.loc[:index[99]])


def test_regime2_blocks_are_separate_sequences():
    index = pd.date_range("2024-01-01", periods=12, freq="B")

    features = pd.DataFrame(
        {
            feature: np.arange(len(index), dtype=float) + offset
            for offset, feature in enumerate(FEATURE_COLUMNS)
        },
        index=index,
    )

    feed = pd.DataFrame(
        {
            "spot_state": [2, 2, 2, 0, 0, 2, 2, 2, 2, 1, 2, 2],
            "hmm_block_id": [10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 14, 14],
        },
        index=index,
    )

    frame = project_main_feed_to_futures(
        market_features=features,
        state_feed=feed,
        state_column="spot_state",
        main_block_column="hmm_block_id",
    )

    config = SubHMMConfig(min_train_sequence_length=1)
    arrays, indexes, catalog = build_regime2_sequences(
        frame,
        config=config,
        training=True,
    )

    assert [len(x) for x in arrays] == [3, 4, 2]
    assert [len(index) for index in indexes] == [3, 4, 2]
    assert catalog["main_regime_block_id"].tolist() == ["10.0", "12.0", "14.0"]


def test_short_train_blocks_are_catalogued_but_excluded():
    index = pd.date_range("2024-01-01", periods=8, freq="B")

    frame = pd.DataFrame(
        {
            feature: np.linspace(0.0, 1.0, len(index))
            for feature in FEATURE_COLUMNS
        },
        index=index,
    )
    frame["spot_state"] = 2
    frame["main_regime_block_id"] = ["A", "A", "B", "B", "B", "B", "B", "B"]

    config = SubHMMConfig(min_train_sequence_length=5)
    arrays, _, catalog = build_regime2_sequences(
        frame,
        config=config,
        training=True,
    )

    assert len(arrays) == 1
    assert arrays[0].shape == (6, 4)
    assert catalog["used"].tolist() == [False, True]