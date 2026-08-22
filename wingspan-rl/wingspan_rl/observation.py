"""Fixed-size observation encoding.

The encoder is deliberately explicit: every section has a declared width, so
the layout is stable across games, deck sizes and player counts, and
:meth:`ObservationEncoder.describe` can tell you what each slice means.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bonus import N_BONUS_CARDS
from .cards import BirdCard, enumerate_payments
from .constants import (
    MAX_BIRDS_PER_HABITAT,
    MAX_HAND,
    MAX_OPTIONS,
    N_BOARD_SLOTS,
    N_FOOD,
    N_HABITAT,
    N_NEST,
    N_ROUNDS,
    TRAY_SIZE,
    Habitat,
    Timing,
)
from .decisions import DECISION_KIND_INDEX, DECISION_KINDS, OPTION_KINDS
from .goals import N_GOALS
from .powers import POWER_KINDS
from .state import N_FACES, GameState, PlayerBoard

N_TIMING = len(Timing)
N_POWER_KINDS = len(POWER_KINDS)
N_OPTION_KINDS = len(OPTION_KINDS)
N_DECISION_KINDS = len(DECISION_KINDS)

#: Width of the per-card feature block.
CARD_FEATURES = (
    1              # points
    + N_FOOD       # cost by food type
    + 1            # wild cost
    + 1            # total cost
    + N_HABITAT    # habitats
    + N_NEST       # nest type
    + 1            # egg capacity
    + 1            # wingspan
    + 2            # predator, passerine
    + N_TIMING     # power timing
    + N_POWER_KINDS
)
#: Compact card block used for options (where 40 slots would be expensive).
OPTION_CARD_FEATURES = 1 + 1 + N_HABITAT + 1 + N_TIMING

MAX_OPPONENTS = 4
OPPONENT_FEATURES = N_HABITAT + 1 + N_FOOD + 1 + 1 + 1 + 1

SECTIONS: Tuple[Tuple[str, int], ...] = (
    ("global", N_ROUNDS + 4 + 3 + N_FACES + N_FOOD + 1 + 5),
    ("goals", N_ROUNDS * (N_GOALS + 2)),
    ("self", N_FOOD + 2 + 2 + 2 * N_HABITAT + 1),
    ("board", N_BOARD_SLOTS * (CARD_FEATURES + 5)),
    ("hand", MAX_HAND * (CARD_FEATURES + 2)),
    ("tray", TRAY_SIZE * (CARD_FEATURES + 1)),
    ("opponents", MAX_OPPONENTS * OPPONENT_FEATURES),
    ("bonus", N_BONUS_CARDS),
    ("decision", N_DECISION_KINDS + 2),
    ("options", MAX_OPTIONS * (N_OPTION_KINDS + 2 + OPTION_CARD_FEATURES)),
)
OBS_SIZE = sum(width for _, width in SECTIONS)


def section_slices() -> Dict[str, slice]:
    out, start = {}, 0
    for name, width in SECTIONS:
        out[name] = slice(start, start + width)
        start += width
    return out


class _Writer:
    """Append-only cursor over a preallocated observation buffer."""

    def __init__(self, size: int):
        self.buf = np.zeros(size, dtype=np.float32)
        self.pos = 0

    def put(self, *values: float) -> None:
        for value in values:
            self.buf[self.pos] = value
            self.pos += 1

    def put_block(self, values: Sequence[float]) -> None:
        n = len(values)
        self.buf[self.pos:self.pos + n] = values
        self.pos += n

    def onehot(self, index: Optional[int], size: int) -> None:
        if index is not None and 0 <= index < size:
            self.buf[self.pos + index] = 1.0
        self.pos += size

    def skip_to(self, target: int) -> None:
        if self.pos > target:
            raise AssertionError(f"observation section overflow: {self.pos} > {target}")
        self.pos = target


class ObservationEncoder:
    """Encodes a :class:`~wingspan_rl.engine.WingspanGame` for one player."""

    size = OBS_SIZE

    def __init__(self, cards: Sequence[BirdCard]):
        self.cards = list(cards)
        self.card_matrix = np.zeros((len(self.cards), CARD_FEATURES), dtype=np.float32)
        self.option_matrix = np.zeros((len(self.cards), OPTION_CARD_FEATURES), dtype=np.float32)
        for card in self.cards:
            self.card_matrix[card.id] = self._card_features(card)
            self.option_matrix[card.id] = self._option_card_features(card)

    # -- static card features ---------------------------------------------
    @staticmethod
    def _card_features(card: BirdCard) -> np.ndarray:
        cost = card.costs[0]
        out: List[float] = [card.points / 9.0]
        out.extend(n / 3.0 for n in cost.food)
        out.append(cost.wild / 3.0)
        out.append(cost.total / 5.0)
        out.extend(1.0 if Habitat(h) in card.habitats else 0.0 for h in range(N_HABITAT))
        out.extend(1.0 if int(card.nest) == i else 0.0 for i in range(N_NEST))
        out.append(card.egg_capacity / 6.0)
        out.append(card.wingspan / 250.0)
        out.append(1.0 if card.predator else 0.0)
        out.append(1.0 if card.passerine else 0.0)
        out.extend(1.0 if int(card.power.timing) == i else 0.0 for i in range(N_TIMING))
        out.extend(1.0 if card.power.kind_index == i else 0.0 for i in range(N_POWER_KINDS))
        return np.asarray(out, dtype=np.float32)

    @staticmethod
    def _option_card_features(card: BirdCard) -> np.ndarray:
        out: List[float] = [card.points / 9.0, card.costs[0].total / 5.0]
        out.extend(1.0 if Habitat(h) in card.habitats else 0.0 for h in range(N_HABITAT))
        out.append(card.egg_capacity / 6.0)
        out.extend(1.0 if int(card.power.timing) == i else 0.0 for i in range(N_TIMING))
        return np.asarray(out, dtype=np.float32)

    # -- encoding -----------------------------------------------------------
    def encode(self, game, player: int) -> np.ndarray:  # noqa: C901 - flat by design
        state: GameState = game.state
        board = state.players[player]
        w = _Writer(self.size)
        marks = section_slices()

        # --- global
        w.onehot(state.round_index, N_ROUNDS)
        turns = max(state.turns_left[player], 0) if state.turns_left else 0
        w.put(
            turns / 8.0,
            1.0 if state.current_player == player else 0.0,
            1.0 if state.start_player == player else 0.0,
            1.0 if state.done else 0.0,
        )
        w.put(len(state.deck) / len(self.cards),
              len(state.discard) / len(self.cards),
              len(state.tray) / TRAY_SIZE)
        w.put_block([c / 5.0 for c in state.feeder.counts()])
        w.put_block([c / 5.0 for c in state.feeder.food_counts()])
        w.put(state.num_players / 5.0)
        w.onehot(player, 5)
        w.skip_to(marks["goals"].start)

        # --- round goals
        for r in range(N_ROUNDS):
            w.onehot(state.goals[r] if r < len(state.goals) else None, N_GOALS)
            w.put(1.0 if r == state.round_index else 0.0,
                  board.round_goal_points[r] / 7.0)
        w.skip_to(marks["self"].start)

        # --- own resources
        w.put_block([f / 8.0 for f in board.food])
        w.put(len(board.hand) / 20.0, len(board.bonus_cards) / 5.0)
        w.put(board.bird_count() / float(N_BOARD_SLOTS), board.total_eggs() / 25.0)
        for habitat in Habitat:
            w.put(board.action_value(habitat) / 5.0)
        for habitat in Habitat:
            w.put(board.exchanges_available(habitat) / 2.0)
        w.put(board.egg_space(self.cards) / 25.0)
        w.skip_to(marks["board"].start)

        # --- own mat
        for habitat in Habitat:
            row = board.habitats[habitat]
            for column in range(MAX_BIRDS_PER_HABITAT):
                if column < len(row):
                    placed = row[column]
                    card = self.cards[placed.card_id]
                    w.put_block(self.card_matrix[placed.card_id])
                    w.put(
                        1.0,
                        placed.eggs / 6.0,
                        1.0 if placed.eggs >= card.egg_capacity else 0.0,
                        sum(placed.cached) / 5.0,
                        placed.tucked / 5.0,
                    )
                else:
                    w.pos += CARD_FEATURES + 5
        w.skip_to(marks["hand"].start)

        # --- hand
        for i in range(MAX_HAND):
            if i < len(board.hand):
                card_id = board.hand[i]
                card = self.cards[card_id]
                w.put_block(self.card_matrix[card_id])
                affordable = bool(enumerate_payments(board.food, card.costs, limit=1))
                playable = bool(game.playable_habitats(board, card))
                w.put(1.0 if affordable else 0.0, 1.0 if playable else 0.0)
            else:
                w.pos += CARD_FEATURES + 2
        w.skip_to(marks["tray"].start)

        # --- face-up tray
        for i in range(TRAY_SIZE):
            if i < len(state.tray):
                w.put_block(self.card_matrix[state.tray[i]])
                w.put(1.0)
            else:
                w.pos += CARD_FEATURES + 1
        w.skip_to(marks["opponents"].start)

        # --- opponents, in turn order after the observer
        for i, other_index in enumerate(state.opponents(player)[:MAX_OPPONENTS]):
            other: PlayerBoard = state.players[other_index]
            for habitat in Habitat:
                w.put(len(other.habitats[habitat]) / float(MAX_BIRDS_PER_HABITAT))
            w.put(other.total_eggs() / 25.0)
            w.put_block([f / 8.0 for f in other.food])
            w.put(len(other.hand) / 20.0)
            w.put(len(other.bonus_cards) / 5.0)
            w.put(sum(self.cards[p.card_id].points for p in other.all_birds()) / 50.0)
            w.put(sum(other.round_goal_points) / 20.0)
        w.skip_to(marks["bonus"].start)

        # --- bonus cards held (public to the agent that owns them)
        for bonus_id in board.bonus_cards:
            if bonus_id < N_BONUS_CARDS:
                w.buf[marks["bonus"].start + bonus_id] = 1.0
        w.skip_to(marks["decision"].start)

        # --- pending decision
        pending = game.pending
        if pending is not None:
            w.onehot(DECISION_KIND_INDEX.get(pending.kind), N_DECISION_KINDS)
            w.put(1.0 if pending.player == player else 0.0,
                  len(pending.options) / float(MAX_OPTIONS))
        else:
            w.pos += N_DECISION_KINDS + 2
        w.skip_to(marks["options"].start)

        # --- the concrete options on the table
        if pending is not None:
            per = N_OPTION_KINDS + 2 + OPTION_CARD_FEATURES
            base = marks["options"].start
            for i, option in enumerate(pending.options[:MAX_OPTIONS]):
                at = base + i * per
                w.buf[at + option.kind_index] = 1.0
                w.buf[at + N_OPTION_KINDS] = 1.0
                w.buf[at + N_OPTION_KINDS + 1] = option.value / 5.0
                start = at + N_OPTION_KINDS + 2
                if option.card_id is not None:
                    w.buf[start:start + OPTION_CARD_FEATURES] = self.option_matrix[
                        option.card_id
                    ]
                elif option.vector is not None:
                    values = list(option.vector)[:OPTION_CARD_FEATURES]
                    w.buf[start:start + len(values)] = values
        w.pos = self.size
        return w.buf

    def describe(self) -> Dict[str, slice]:
        return section_slices()
