"""Shared test helpers."""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Sequence, Union

import pytest

from wingspan_rl.constants import MAX_BIRDS_PER_HABITAT, Habitat
from wingspan_rl.engine import GameConfig, WingspanGame

Pick = Union[int, Callable]


def make_game(num_players: int = 2, seed: int = 0, **kwargs) -> WingspanGame:
    kwargs.setdefault("keep_log", False)
    return WingspanGame(GameConfig(num_players=num_players, seed=seed, **kwargs))


def drive(flow, picks: Sequence[Pick] = ()) -> List[str]:
    """Run an engine sub-flow, answering each decision from ``picks``.

    A pick is either an option index or a callable taking the decision and
    returning the option to choose.  Missing picks default to option 0.
    """
    prompts: List[str] = []
    iterator = iter(picks)
    try:
        decision = flow.send(None)
        while True:
            prompts.append(decision.prompt)
            pick = next(iterator, 0)
            option = pick(decision) if callable(pick) else decision.options[pick]
            decision = flow.send(option.payload)
    except StopIteration:
        pass
    return prompts


def by_kind(kind: str):
    """Pick the first option of a given kind (falls back to the first option)."""
    def chooser(decision):
        for option in decision.options:
            if option.kind == kind:
                return option
        return decision.options[0]

    return chooser


def play_random(game: WingspanGame, rng: random.Random, check: Optional[Callable] = None,
                max_steps: int = 20000) -> int:
    steps = 0
    while not game.done:
        game.step(rng.choice(game.legal_action_ids()))
        steps += 1
        if check is not None:
            check(game)
        assert steps < max_steps, "game did not terminate"
    return steps


def assert_invariants(game: WingspanGame) -> None:
    state = game.state
    accounted = len(state.deck) + len(state.discard) + len(state.tray)
    for board in state.players:
        accounted += len(board.hand)
        for placed in board.all_birds():
            card = game.cards[placed.card_id]
            accounted += 1 + placed.tucked
            assert 0 <= placed.eggs <= card.egg_capacity
            assert all(c >= 0 for c in placed.cached)
        assert all(f >= 0 for f in board.food)
        for habitat in Habitat:
            assert len(board.habitats[habitat]) <= MAX_BIRDS_PER_HABITAT
    assert accounted == len(game.cards), "cards appeared or vanished"
    assert len(state.feeder.dice) <= 5


@pytest.fixture
def game() -> WingspanGame:
    return make_game()
