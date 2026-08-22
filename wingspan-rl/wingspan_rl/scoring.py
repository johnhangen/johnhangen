"""Final scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .bonus import BonusCard
from .cards import BirdCard
from .constants import HABITAT_BY_NAME, NEST_BY_NAME, Timing
from .state import GameState, PlayerBoard


@dataclass
class ScoreBreakdown:
    birds: int = 0
    end_powers: int = 0
    bonus: int = 0
    goals: int = 0
    eggs: int = 0
    cached_food: int = 0
    tucked_cards: int = 0

    @property
    def total(self) -> int:
        return (
            self.birds
            + self.end_powers
            + self.bonus
            + self.goals
            + self.eggs
            + self.cached_food
            + self.tucked_cards
        )

    def as_dict(self) -> dict:
        out = {
            "birds": self.birds,
            "end_powers": self.end_powers,
            "bonus": self.bonus,
            "goals": self.goals,
            "eggs": self.eggs,
            "cached_food": self.cached_food,
            "tucked_cards": self.tucked_cards,
        }
        out["total"] = self.total
        return out


def _end_power_points(board: PlayerBoard, deck: Sequence[BirdCard]) -> int:
    total = 0
    for placed in board.all_birds():
        power = deck[placed.card_id].power
        if power.timing is not Timing.GAME_END:
            continue
        per = power.get("per", "eggs_on_this")
        amount = int(power.get("amount", 1))
        if per == "eggs_on_this":
            total += amount * placed.eggs
        elif per == "cached_food":
            total += amount * sum(placed.cached)
        elif per == "tucked_cards":
            total += amount * placed.tucked
        elif per == "birds_in_habitat":
            habitat = HABITAT_BY_NAME[power.get("habitat", "forest")]
            total += amount * len(board.habitats[habitat])
        elif per == "birds_with_nest":
            nest = NEST_BY_NAME[power.get("nest", "bowl")]
            total += amount * sum(
                1 for p in board.all_birds() if deck[p.card_id].nest_matches(nest)
            )
    return total


def score_player(
    board: PlayerBoard,
    deck: Sequence[BirdCard],
    bonus_cards: Sequence[BonusCard],
) -> ScoreBreakdown:
    out = ScoreBreakdown()
    for placed in board.all_birds():
        card = deck[placed.card_id]
        out.birds += card.points
        out.eggs += placed.eggs
        out.cached_food += sum(placed.cached)
        out.tucked_cards += placed.tucked
    out.end_powers = _end_power_points(board, deck)
    out.bonus = sum(bonus_cards[b].score(board, deck) for b in board.bonus_cards)
    out.goals = sum(board.round_goal_points)
    return out


def final_scores(
    state: GameState,
    deck: Sequence[BirdCard],
    bonus_cards: Sequence[BonusCard],
) -> List[ScoreBreakdown]:
    return [score_player(board, deck, bonus_cards) for board in state.players]


def winners(
    state: GameState,
    deck: Sequence[BirdCard],
    bonus_cards: Sequence[BonusCard],
) -> List[int]:
    """Indices of the winning players (ties broken by leftover food + cards)."""
    scores = final_scores(state, deck, bonus_cards)
    best = max(s.total for s in scores)
    tied = [i for i, s in enumerate(scores) if s.total == best]
    if len(tied) == 1:
        return tied
    def tiebreak(i: int) -> int:
        board = state.players[i]
        return board.total_food() + len(board.hand)
    best_tb = max(tiebreak(i) for i in tied)
    return [i for i in tied if tiebreak(i) == best_tb]
