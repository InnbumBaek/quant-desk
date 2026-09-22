from core.portfolio.center_book import net_orders


def test_offsetting_pods_net_to_zero_and_save_turnover():
    result = net_orders({"statarb": {"SPY": 0.04}, "trend": {"SPY": -0.04}})
    assert result.net_targets == {}
    assert result.gross_pod_turnover == 0.08
    assert result.netted_turnover == 0.0
    assert result.turnover_saved == 1.0


def test_same_direction_pods_add_up():
    result = net_orders({"a": {"IWM": 0.02}, "b": {"IWM": 0.01}})
    assert result.net_targets == {"IWM": 0.03}
    assert result.turnover_saved == 0.0


def test_overlay_trims_crowded_name_only_downwards():
    result = net_orders({"a": {"NVDA": 0.04}, "b": {"NVDA": 0.04}}, single_name_max=0.05)
    assert result.net_targets == {"NVDA": 0.05}
    assert round(result.trimmed["NVDA"], 10) == 0.03
    # The overlay never grows a position.
    grown = net_orders({"a": {"NVDA": 0.01}}, single_name_max=0.05)
    assert grown.net_targets == {"NVDA": 0.01} and grown.trimmed == {}
