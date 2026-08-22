"""End-of-round goals and their competitive scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Sequence

from .constants import GOAL_PLACE_POINTS, Habitat, NestType

if TYPE_CHECKING:  # pragma: no cover
    from .cards import BirdCard
    from .state import PlayerBoard


@dataclass
class Goal:
    id: int
    name: str
    counter: Callable[["PlayerBoard", Sequence["BirdCard"]], int]

    def count(self, board: "PlayerBoard", deck: Sequence["BirdCard"]) -> int:
        return self.counter(board, deck)


def build_goals() -> List[Goal]:
    goals: List[Goal] = []

    def add(name, counter):
        goals.append(Goal(len(goals), name, counter))

    for habitat in Habitat:
        add(f"birds in {habitat.name.lower()}",
            lambda b, d, h=habitat: len(b.habitats[h]))
        add(f"eggs in {habitat.name.lower()}",
            lambda b, d, h=habitat: sum(p.eggs for p in b.habitats[h]))
    for nest in (NestType.BOWL, NestType.CAVITY, NestType.GROUND, NestType.PLATFORM):
        add(f"eggs in {nest.name.lower()} nests",
            lambda b, d, n=nest: sum(p.eggs for p in b.all_birds()
                                     if d[p.card_id].nest_matches(n)))
    add("total birds", lambda b, d: sum(len(v) for v in b.habitats.values()))
    add("total eggs", lambda b, d: sum(p.eggs for p in b.all_birds()))
    add("food cost of played birds",
        lambda b, d: sum(d[p.card_id].min_cost for p in b.all_birds()))
    add("birds with no power",
        lambda b, d: sum(1 for p in b.all_birds() if d[p.card_id].power.kind == "none"))
    add("brown powers on birds",
        lambda b, d: sum(1 for p in b.all_birds()
                         if d[p.card_id].power.timing.name == "BROWN"))
    add("bonus cards in hand", lambda b, d: len(b.bonus_cards))
    add("cards in hand", lambda b, d: len(b.hand))
    add("food cached on birds", lambda b, d: sum(sum(p.cached) for p in b.all_birds()))
    return goals


GOALS: List[Goal] = build_goals()
N_GOALS = len(GOALS)


def score_goal(counts: Sequence[int], round_index: int) -> List[int]:
    """Competitive scoring for one round goal.

    Players tied for a place all receive that place's points and the places
    they consume are skipped.  A count of zero never scores.
    """
    points = GOAL_PLACE_POINTS[round_index]
    result = [0] * len(counts)
    ordered = sorted({c for c in counts if c > 0}, reverse=True)
    place = 0
    for value in ordered:
        if place >= len(points):
            break
        winners = [i for i, c in enumerate(counts) if c == value]
        for i in winners:
            result[i] = points[place]
        place += len(winners)
    return result
