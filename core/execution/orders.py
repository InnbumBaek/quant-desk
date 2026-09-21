"""Order identity and state machine.

Two rules this module exists to enforce:
  1. Every order carries a deterministic client id, so a retry after a crash
     can never become a duplicate order.
  2. An order moves only along the declared transitions, and a run with any
     unresolved order must not submit new ones.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum


class State(StrEnum):
    INTENDED = "intended"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


TRANSITIONS: dict[State, set[State]] = {
    State.INTENDED: {State.SUBMITTED, State.CANCELLED},
    State.SUBMITTED: {State.PARTIAL, State.FILLED, State.REJECTED, State.CANCELLED},
    State.PARTIAL: {State.PARTIAL, State.FILLED, State.CANCELLED},
    State.FILLED: set(),
    State.REJECTED: set(),
    State.CANCELLED: set(),
}

TERMINAL = {State.FILLED, State.REJECTED, State.CANCELLED}


def client_order_id(trade_date: str, alpha_id: str, symbol: str, seq: int) -> str:
    """Idempotent id: the same intent always hashes to the same id."""
    raw = f"{trade_date}|{alpha_id}|{symbol}|{seq}"
    return "qd-" + hashlib.sha256(raw.encode()).hexdigest()[:20]


@dataclass
class Order:
    trade_date: str
    alpha_id: str
    symbol: str
    seq: int
    quantity: int
    state: State = State.INTENDED
    filled: int = 0
    history: list[State] = field(default_factory=list)

    @property
    def client_id(self) -> str:
        return client_order_id(self.trade_date, self.alpha_id, self.symbol, self.seq)

    def transition(self, new: State, filled: int | None = None) -> None:
        if new not in TRANSITIONS[self.state]:
            raise ValueError(f"illegal transition {self.state.value} -> {new.value}")
        self.history.append(self.state)
        self.state = new
        if filled is not None:
            self.filled = filled

    @property
    def unresolved(self) -> bool:
        return self.state not in TERMINAL


def blocking_orders(orders: list[Order]) -> list[Order]:
    """Orders that must be resolved before a new submission round."""
    return [o for o in orders if o.unresolved and o.state is not State.INTENDED]
