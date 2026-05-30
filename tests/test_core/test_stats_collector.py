"""Tests for cache_edit.core.stats_collector."""

import pandas as pd

from cache_edit.core.stats_collector import KeyTokenStatsCollector


class TestKeyTokenStatsCollector:
    def test_record_populates_records_and_stats(self):
        c = KeyTokenStatsCollector()
        c.record(step=0, layer_idx=0, count=10, group_key="cond")
        c.record(step=0, layer_idx=1, count=12, group_key="cond")
        c.record(step=1, layer_idx=0, count=8, group_key="cond")

        assert len(c.records) == 3
        assert c.stats["cond"][0] == [10, 12]
        assert c.stats["cond"][1] == [8]

    def test_disabled_collector_records_nothing(self):
        c = KeyTokenStatsCollector(enabled=False)
        c.record(step=0, layer_idx=0, count=5)
        assert len(c.records) == 0
        assert c.stats == {}

    def test_multiple_groups_isolated(self):
        c = KeyTokenStatsCollector()
        c.record(step=0, layer_idx=0, count=1, group_key="single")
        c.record(step=0, layer_idx=0, count=2, group_key="double")
        assert c.stats["single"][0] == [1]
        assert c.stats["double"][0] == [2]

    def test_to_dataframe_sorted(self):
        c = KeyTokenStatsCollector()
        c.record(step=1, layer_idx=0, count=5, group_key="b")
        c.record(step=0, layer_idx=0, count=3, group_key="a")
        df = c.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["step", "layer", "group", "count"]
        # Sort: group=a row first
        assert df.iloc[0]["group"] == "a"

    def test_to_dataframe_empty(self):
        c = KeyTokenStatsCollector()
        df = c.to_dataframe()
        assert df.empty

    def test_get_summary_when_empty(self):
        c = KeyTokenStatsCollector()
        s = c.get_summary()
        assert s["total_records"] == 0
        assert s["enabled"] is True

    def test_get_summary_with_data(self):
        c = KeyTokenStatsCollector()
        c.record(step=0, layer_idx=0, count=10)
        c.record(step=0, layer_idx=1, count=20)
        s = c.get_summary()
        assert s["total_records"] == 2
        assert s["max_count"] == 20
        assert s["min_count"] == 10
        assert s["avg_count"] == 15

    def test_reset_clears(self):
        c = KeyTokenStatsCollector()
        c.record(step=0, layer_idx=0, count=10)
        c.reset()
        assert len(c.records) == 0
        assert c.stats == {}

    def test_enable_disable_toggle(self):
        c = KeyTokenStatsCollector(enabled=False)
        assert c.enabled is False
        c.enable()
        assert c.enabled is True
        c.disable()
        assert c.enabled is False

    def test_len(self):
        c = KeyTokenStatsCollector()
        c.record(step=0, layer_idx=0, count=1)
        c.record(step=1, layer_idx=0, count=2)
        assert len(c) == 2

    def test_layer_gap_padded_with_zeros(self):
        c = KeyTokenStatsCollector()
        c.record(step=0, layer_idx=0, count=5)
        c.record(step=0, layer_idx=3, count=8)  # skip layers 1, 2
        assert c.stats["default"][0] == [5, 0, 0, 8]

    def test_report_does_not_error(self, capsys):
        c = KeyTokenStatsCollector()
        c.record(step=0, layer_idx=0, count=10)
        c.report()
        out = capsys.readouterr().out
        assert "Key Token Stats" in out

        c2 = KeyTokenStatsCollector()
        c2.report()  # empty path
        out = capsys.readouterr().out
        assert "No stats collected" in out
