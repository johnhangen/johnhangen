"""Bonus cards: end-of-game objectives kept secret in hand."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Sequence, Tuple

from .constants import Food, Habitat, NestType

if TYPE_CHECKING:  # pragma: no cover
    from .cards import BirdCard
    from .state import PlayerBoard

#: ``(threshold, points)`` pairs, ascending; the highest threshold reached wins.
Tiers = Tuple[Tuple[int, int], ...]


@dataclass
class BonusCard:
    id: int
    name: str
    text: str
    tiers: Tiers
    counter: Callable[["PlayerBoard", Sequence["BirdCard"]], int]

    def count(self, board: "PlayerBoard", deck: Sequence["BirdCard"]) -> int:
        return self.counter(board, deck)

    def score(self, board: "PlayerBoard", deck: Sequence["BirdCard"]) -> int:
        n = self.count(board, deck)
        points = 0
        for threshold, value in self.tiers:
            if n >= threshold:
                points = value
        return points


def _birds(board: "PlayerBoard", deck):
    for placed in board.all_birds():
        yield placed, deck[placed.card_id]


def _count_birds(pred) -> Callable[["PlayerBoard", Sequence["BirdCard"]], int]:
    def counter(board, deck):
        return sum(1 for placed, card in _birds(board, deck) if pred(placed, card))

    return counter


def _habitat_birds(habitat: Habitat):
    def counter(board, deck):
        return len(board.habitats[habitat])

    return counter


def _habitat_eggs(habitat: Habitat):
    def counter(board, deck):
        return sum(p.eggs for p in board.habitats[habitat])

    return counter


def _nest_eggs(nest: NestType):
    def counter(board, deck):
        return sum(p.eggs for p, card in _birds(board, deck) if card.nest_matches(nest))

    return counter


TIER_3_5 = ((3, 3), (5, 6))
TIER_2_4 = ((2, 3), (4, 6))
TIER_4_6 = ((4, 3), (6, 6))
TIER_EGGS = ((4, 3), (7, 6))


def build_bonus_cards() -> List[BonusCard]:
    cards: List[BonusCard] = []

    def add(name, text, tiers, counter):
        cards.append(BonusCard(len(cards), name, text, tiers, counter))

    for habitat, label in (
        (Habitat.FOREST, "Forest"),
        (Habitat.GRASSLAND, "Prairie"),
        (Habitat.WETLAND, "Wetland"),
    ):
        add(
            f"{label} Ecologist",
            f"Birds in your {habitat.name.lower()}",
            TIER_3_5,
            _habitat_birds(habitat),
        )
        add(
            f"{label} Manager",
            f"Eggs in your {habitat.name.lower()}",
            TIER_EGGS,
            _habitat_eggs(habitat),
        )

    for nest in (NestType.BOWL, NestType.CAVITY, NestType.GROUND, NestType.PLATFORM):
        label = nest.name.capitalize()
        add(
            f"{label} Nest Specialist",
            f"Birds with a {nest.name.lower()} nest (star nests count)",
            TIER_3_5,
            _count_birds(lambda p, c, n=nest: c.nest_matches(n)),
        )
        add(
            f"{label} Nest Ranger",
            f"Eggs in {nest.name.lower()} nests (star nests count)",
            TIER_EGGS,
            _nest_eggs(nest),
        )

    for food in Food:
        add(
            f"{food.name.capitalize()} Enthusiast",
            f"Birds that eat {food.name.lower()}",
            TIER_3_5,
            _count_birds(lambda p, c, f=food: f in c.food_types),
        )

    add("Large Bird Specialist", "Birds with wingspan over 65cm", TIER_3_5,
        _count_birds(lambda p, c: c.wingspan > 65))
    add("Small Bird Specialist", "Birds with wingspan under 30cm", TIER_3_5,
        _count_birds(lambda p, c: c.wingspan < 30))
    add("Passerine Specialist", "Passerine birds", TIER_4_6,
        _count_birds(lambda p, c: c.passerine))
    add("Raptor Specialist", "Birds with a predator power", TIER_2_4,
        _count_birds(lambda p, c: c.predator))
    add("Bird Bander", "Birds with at least 1 egg", TIER_4_6,
        _count_birds(lambda p, c: p.eggs > 0))
    add("Food Hoarder", "Food cached on your birds", TIER_EGGS,
        lambda board, deck: sum(sum(p.cached) for p in board.all_birds()))
    add("Card Collector", "Cards tucked behind your birds", TIER_EGGS,
        lambda board, deck: sum(p.tucked for p in board.all_birds()))
    add("Frugal Ornithologist", "Birds costing 1 food or less", TIER_3_5,
        _count_birds(lambda p, c: c.min_cost <= 1))
    add("Big Spender", "Birds costing 4 food or more", ((2, 4), (3, 7)),
        _count_birds(lambda p, c: c.min_cost >= 4))
    add("Behaviorist", "Birds with a 'when activated' power", TIER_4_6,
        _count_birds(lambda p, c: c.power.timing.name == "BROWN"))
    add("Anatomist", "Birds with no power", TIER_2_4,
        _count_birds(lambda p, c: c.power.kind == "none"))
    add("Prolific Layer", "Birds with an egg capacity of 5 or more", TIER_3_5,
        _count_birds(lambda p, c: c.egg_capacity >= 5))
    add("Omnivore Expert", "Birds that eat 2 or more food types", TIER_3_5,
        _count_birds(lambda p, c: len(c.food_types) >= 2))
    return cards


BONUS_CARDS: List[BonusCard] = build_bonus_cards()
N_BONUS_CARDS = len(BONUS_CARDS)
