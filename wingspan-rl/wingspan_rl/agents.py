"""Baseline agents: a uniform random policy and a hand-written heuristic."""

from __future__ import annotations

import random
from typing import List, Optional, Protocol

from .constants import Food, Habitat, N_FOOD, Timing
from .decisions import Decision, Option


class Agent(Protocol):
    def act(self, game, player: int) -> int:  # pragma: no cover - interface
        ...


class RandomAgent:
    """Picks uniformly among the legal actions."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def act(self, game, player: int) -> int:
        return self.rng.choice(game.legal_action_ids())


class GreedyAgent:
    """A simple, deterministic-ish heuristic - the default training opponent.

    It is far from optimal, but it plays birds, keeps its engine running and
    lays eggs late, which makes it a much better yardstick than random play.
    """

    #: Tunable weights for the four main actions (see :meth:`_score_option`).
    W_PLAY = 5.0
    W_PLAY_CARD = 0.7
    W_FOOD = 0.4
    W_EGG_LATE = 2.6
    W_EGG_EARLY = 1.4
    W_DRAW = 0.4

    def __init__(self, seed: Optional[int] = None, noise: float = 0.15, **weights):
        self.rng = random.Random(seed)
        self.noise = noise
        for key, value in weights.items():
            if not hasattr(type(self), key.upper()):
                raise TypeError(f"unknown weight {key!r}")
            setattr(self, key.upper(), value)

    # -- helpers -----------------------------------------------------------
    def _food_need(self, game, player: int) -> List[float]:
        """How much each food type is wanted by the cards in hand."""
        board = game.state.board(player)
        need = [0.0] * N_FOOD
        for card_id in board.hand:
            cost = game.cards[card_id].costs[0]
            for i, n in enumerate(cost.food):
                need[i] += n
            if cost.wild:
                for i in range(N_FOOD):
                    need[i] += cost.wild / N_FOOD
        return [max(0.0, need[i] - board.food[i]) for i in range(N_FOOD)]

    def _card_value(self, game, card_id: int) -> float:
        card = game.cards[card_id]
        value = card.points + 0.8 * card.egg_capacity
        if card.power.timing is Timing.BROWN:
            value += 1.5
        elif card.power.timing is Timing.WHITE:
            value += 1.0
        return value - 0.8 * card.min_cost

    def _score_option(self, game, player: int, option: Option, decision: Decision) -> float:
        state = game.state
        board = state.board(player)
        rounds_left = 4 - state.round_index
        kind = option.kind

        if kind == "play_bird":
            card = game.cards[option.card_id]
            return (self.W_PLAY + self._card_value(game, card.id) * self.W_PLAY_CARD
                    + 0.6 * rounds_left)
        if kind == "gain_food":
            hungry = sum(self._food_need(game, player))
            return 2.0 + option.value * self.W_FOOD + min(hungry, 4.0) * 0.5
        if kind == "lay_eggs":
            space = board.egg_space(game.cards)
            if space == 0:
                return -1.0
            weight = self.W_EGG_LATE if rounds_left <= 2 else self.W_EGG_EARLY
            return 1.0 + option.value * weight
        if kind == "draw_cards":
            if len(board.hand) >= 8:
                return 0.5
            return 2.5 + option.value * self.W_DRAW - 0.3 * len(board.hand)

        if kind == "payment":
            # Spend what we have most of, keep scarce food for later.
            supply = board.food
            return sum(n * (supply[i] - n + 1) for i, n in enumerate(option.payload))
        if kind == "die":
            need = self._food_need(game, player)
            from .state import DIE_FACES

            foods = DIE_FACES[option.payload]
            return sum(1.0 + need[int(f)] for f in foods)
        if kind == "food":
            need = self._food_need(game, player)
            food = option.payload
            return need[int(food)] if isinstance(food, (Food, int)) else 0.0
        if kind == "habitat":
            habitat: Habitat = option.payload
            # Prefer the shortest row: it upgrades the weakest action.
            return 3.0 - board.row_size(habitat) - option.value * 1.2
        if kind == "bird":
            placed = option.payload
            card = game.cards[placed.card_id]
            if decision.kind in ("lay_egg",):
                return (card.egg_capacity - placed.eggs) + 0.2 * card.points
            if decision.kind in ("spend_egg", "discard_egg", "exchange_egg_card"):
                return placed.eggs - 0.2 * card.points
            if decision.kind == "repeat_power":
                return self._card_value(game, placed.card_id)
            return 1.0
        if kind == "tray_card":
            card = game.cards[option.card_id]
            return 1.0 + self._card_value(game, card.id) * 0.5
        if kind == "deck_card":
            return 1.4
        if kind in ("hand_card", "keep_card"):
            # These options discard/tuck a card: throw away the worst one.
            return 4.0 - self._card_value(game, option.card_id)
        if kind == "bonus_card":
            bonus = game.bonus_cards[option.payload]
            return 1.0 + 2.0 * bonus.score(board, game.cards) + 0.4 * bonus.count(
                board, game.cards
            )
        if kind == "exchange":
            return 0.5
        if kind == "pass":
            if decision.kind in ("setup_discard",):
                return 2.5
            if decision.kind in ("tuck", "discard_egg"):
                return 3.0
            return 1.5
        return 0.0

    # -- policy ------------------------------------------------------------
    def act(self, game, player: int) -> int:
        decision = game.pending
        best_id, best_score = None, -1e9
        for option in decision.options:
            score = self._score_option(game, player, option, decision)
            score += self.rng.uniform(0.0, self.noise)
            if score > best_score:
                best_id, best_score = option.action_id, score
        return int(best_id)


def make_agent(spec, seed: Optional[int] = None) -> Agent:
    """Build an agent from ``"random"``, ``"greedy"`` or an agent instance."""
    if isinstance(spec, str):
        if spec == "random":
            return RandomAgent(seed)
        if spec == "greedy":
            return GreedyAgent(seed)
        raise ValueError(f"unknown agent spec {spec!r}")
    if hasattr(spec, "act"):
        return spec
    if callable(spec):
        class _Fn:
            def act(self, game, player):
                return spec(game, player)

        return _Fn()
    raise TypeError(f"cannot build an agent from {spec!r}")
