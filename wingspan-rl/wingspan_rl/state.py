"""Mutable game state: birdfeeder, player mats and the shared table."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .cards import BirdCard, FoodCounts, ZERO_FOOD, food_str
from .constants import (
    EXCHANGE_COLUMNS,
    FEEDER_DICE,
    HABITAT_ACTION_VALUE,
    MAX_BIRDS_PER_HABITAT,
    N_FOOD,
    Food,
    Habitat,
    NestType,
)

#: The six faces of a birdfeeder die.  The last face yields both foods.
DIE_FACES: Tuple[Tuple[Food, ...], ...] = (
    (Food.INVERTEBRATE,),
    (Food.SEED,),
    (Food.FISH,),
    (Food.FRUIT,),
    (Food.RODENT,),
    (Food.INVERTEBRATE, Food.SEED),
)
N_FACES = len(DIE_FACES)


class Birdfeeder:
    """Five dice in a dice tower; ``dice`` holds the face index of each die."""

    def __init__(self, rng: random.Random, n_dice: int = FEEDER_DICE):
        self.rng = rng
        self.n_dice = n_dice
        self.dice: List[int] = []
        self.roll_all()

    def roll_all(self) -> None:
        self.dice = [self.rng.randrange(N_FACES) for _ in range(self.n_dice)]
        self._reroll_if_needed()

    def _reroll_if_needed(self) -> None:
        # Rule: if the feeder is empty, or every die shows the same face, reroll.
        guard = 0
        while self.dice and len(set(self.dice)) == 1 and guard < 50:
            self.dice = [self.rng.randrange(N_FACES) for _ in range(len(self.dice))]
            guard += 1

    def refill(self) -> None:
        """Roll every die that is not currently in the feeder (end of round)."""
        self.dice = [self.rng.randrange(N_FACES) for _ in range(self.n_dice)]
        self._reroll_if_needed()

    @property
    def is_empty(self) -> bool:
        return not self.dice

    def available_faces(self) -> List[int]:
        return sorted(set(self.dice))

    def faces_yielding(self, food: Food) -> List[int]:
        return sorted({f for f in set(self.dice) if food in DIE_FACES[f]})

    def take(self, face: int) -> Tuple[Food, ...]:
        """Remove one die showing ``face`` and return the food it grants."""
        self.dice.remove(face)
        if not self.dice:
            self.roll_all()
        else:
            self._reroll_if_needed()
        return DIE_FACES[face]

    def counts(self) -> List[int]:
        out = [0] * N_FACES
        for die in self.dice:
            out[die] += 1
        return out

    def food_counts(self) -> List[int]:
        out = [0] * N_FOOD
        for die in self.dice:
            for food in DIE_FACES[die]:
                out[int(food)] += 1
        return out

    def __str__(self) -> str:
        from .constants import FOOD_SYMBOLS

        return " ".join(
            "".join(FOOD_SYMBOLS[f] for f in DIE_FACES[d]) for d in sorted(self.dice)
        )


@dataclass
class PlacedBird:
    """A bird card on a player's mat, with the tokens sitting on it."""

    card_id: int
    habitat: Habitat
    column: int
    eggs: int = 0
    cached: FoodCounts = ZERO_FOOD
    tucked: int = 0
    pink_used: bool = False

    def copy(self) -> "PlacedBird":
        return PlacedBird(
            self.card_id, self.habitat, self.column, self.eggs, self.cached,
            self.tucked, self.pink_used,
        )


@dataclass
class PlayerBoard:
    index: int
    food: List[int] = field(default_factory=lambda: [0] * N_FOOD)
    hand: List[int] = field(default_factory=list)
    bonus_cards: List[int] = field(default_factory=list)
    habitats: Dict[Habitat, List[PlacedBird]] = field(
        default_factory=lambda: {h: [] for h in Habitat}
    )
    #: Populated at game end by :func:`wingspan_rl.scoring.score_player`.
    round_goal_points: List[int] = field(default_factory=lambda: [0, 0, 0, 0])

    # -- queries -----------------------------------------------------------
    def all_birds(self) -> Iterator[PlacedBird]:
        for habitat in Habitat:
            yield from self.habitats[habitat]

    def bird_count(self) -> int:
        return sum(len(v) for v in self.habitats.values())

    def total_eggs(self) -> int:
        return sum(p.eggs for p in self.all_birds())

    def total_food(self) -> int:
        return sum(self.food)

    def row_size(self, habitat: Habitat) -> int:
        return len(self.habitats[habitat])

    def has_space(self, habitat: Habitat) -> bool:
        return len(self.habitats[habitat]) < MAX_BIRDS_PER_HABITAT

    def next_column(self, habitat: Habitat) -> int:
        return len(self.habitats[habitat])

    def action_value(self, habitat: Habitat) -> int:
        """Strength of ``habitat``'s action, given how many birds sit there."""
        column = min(len(self.habitats[habitat]), MAX_BIRDS_PER_HABITAT - 1)
        return HABITAT_ACTION_VALUE[habitat][column]

    def exchanges_available(self, habitat: Habitat) -> int:
        n = len(self.habitats[habitat])
        return sum(1 for col in EXCHANGE_COLUMNS if col <= n)

    def egg_space(self, deck: Sequence[BirdCard]) -> int:
        return sum(deck[p.card_id].egg_capacity - p.eggs for p in self.all_birds())

    def birds_with_egg_space(self, deck: Sequence[BirdCard]) -> List[PlacedBird]:
        return [p for p in self.all_birds() if p.eggs < deck[p.card_id].egg_capacity]

    def find(self, habitat: Habitat, column: int) -> Optional[PlacedBird]:
        row = self.habitats[habitat]
        return row[column] if column < len(row) else None

    # -- mutation ----------------------------------------------------------
    def gain_food(self, food: Food, count: int = 1) -> None:
        self.food[int(food)] += count

    def pay_food(self, payment: Sequence[int]) -> None:
        for i, n in enumerate(payment):
            if n > self.food[i]:
                raise ValueError("cannot pay food the player does not have")
            self.food[i] -= n

    def place(self, card: BirdCard, habitat: Habitat) -> PlacedBird:
        if not self.has_space(habitat):
            raise ValueError(f"{habitat.name} row is full")
        placed = PlacedBird(card.id, habitat, len(self.habitats[habitat]))
        self.habitats[habitat].append(placed)
        return placed

    def __str__(self) -> str:
        return (
            f"P{self.index} food={food_str(self.food)} hand={len(self.hand)} "
            f"birds={self.bird_count()} eggs={self.total_eggs()}"
        )


@dataclass
class GameState:
    num_players: int
    players: List[PlayerBoard]
    deck: List[int]
    discard: List[int]
    tray: List[int]
    feeder: Birdfeeder
    bonus_deck: List[int]
    bonus_discard: List[int]
    goals: List[int]
    rng: random.Random = field(default_factory=random.Random)
    round_index: int = 0
    turns_left: List[int] = field(default_factory=list)
    current_player: int = 0
    start_player: int = 0
    #: Filled in as each round is scored: ``round_scores[round][player]``.
    round_scores: List[List[int]] = field(default_factory=list)
    done: bool = False

    def board(self, player: int) -> PlayerBoard:
        return self.players[player]

    def opponents(self, player: int) -> List[int]:
        """Other players, in turn order starting after ``player``."""
        return [(player + i) % self.num_players for i in range(1, self.num_players)]

    def draw_card(self) -> Optional[int]:
        if not self.deck:
            if not self.discard:
                return None
            self.deck = self.discard
            self.discard = []
            self.rng.shuffle(self.deck)
        return self.deck.pop()

    def draw_tray(self, slot: int) -> Optional[int]:
        if slot >= len(self.tray):
            return None
        card = self.tray.pop(slot)
        self.refill_tray()
        return card

    def refill_tray(self) -> None:
        from .constants import TRAY_SIZE

        while len(self.tray) < TRAY_SIZE:
            card = self.draw_card()
            if card is None:
                break
            self.tray.append(card)

    def draw_bonus(self) -> Optional[int]:
        if not self.bonus_deck:
            if not self.bonus_discard:
                return None
            self.bonus_deck = self.bonus_discard
            self.bonus_discard = []
        return self.bonus_deck.pop()
