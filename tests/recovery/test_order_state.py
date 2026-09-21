import pytest

from core.execution.orders import Order, State, blocking_orders, client_order_id


def test_client_id_is_idempotent():
    a = client_order_id("2026-09-21", "statarb-001", "SPY", 0)
    b = client_order_id("2026-09-21", "statarb-001", "SPY", 0)
    c = client_order_id("2026-09-21", "statarb-001", "SPY", 1)
    assert a == b and a != c


def test_illegal_transition_raises():
    order = Order("2026-09-21", "statarb-001", "SPY", 0, 100)
    order.transition(State.SUBMITTED)
    order.transition(State.PARTIAL, filled=40)
    order.transition(State.FILLED, filled=100)
    with pytest.raises(ValueError):
        order.transition(State.SUBMITTED)


def test_partial_fill_blocks_new_round():
    partial = Order("2026-09-21", "statarb-001", "SPY", 0, 100)
    partial.transition(State.SUBMITTED)
    partial.transition(State.PARTIAL, filled=40)
    done = Order("2026-09-21", "statarb-001", "IWM", 0, 50)
    done.transition(State.SUBMITTED)
    done.transition(State.FILLED, filled=50)
    assert [o.client_id for o in blocking_orders([partial, done])] == [partial.client_id]
