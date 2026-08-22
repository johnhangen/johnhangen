"""The decision protocol between the rules engine and an agent.

The engine is written as a single generator that yields a :class:`Decision`
every time it needs input.  Every decision carries the concrete list of legal
:class:`Option` objects, each already bound to the action id an agent must
emit to pick it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

#: Semantic tag on every option, one-hot encoded in the observation.
OPTION_KINDS = (
    "pass",
    "play_bird",
    "gain_food",
    "lay_eggs",
    "draw_cards",
    "habitat",
    "payment",
    "die",
    "tray_card",
    "deck_card",
    "bird",
    "hand_card",
    "food",
    "bonus_card",
    "keep_card",
    "exchange",
)
OPTION_KIND_INDEX = {k: i for i, k in enumerate(OPTION_KINDS)}


@dataclass
class Option:
    """One legal choice.

    ``card_id`` and ``vector`` are what make an option *readable*: an agent
    seeing slot 27 needs to know whether it means "pay 2 seeds" or "take the
    fish die".  Options that refer to a card expose it through ``card_id``;
    the rest describe themselves with a short numeric ``vector`` (food spent,
    food gained, the habitat, ...) that the observation encoder copies in.
    """

    action_id: int
    kind: str
    label: str
    payload: Any = None
    card_id: Optional[int] = None
    value: float = 0.0
    vector: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        if self.kind not in OPTION_KIND_INDEX:
            raise ValueError(f"unknown option kind {self.kind!r}")

    @property
    def kind_index(self) -> int:
        return OPTION_KIND_INDEX[self.kind]


@dataclass
class Decision:
    player: int
    kind: str
    prompt: str
    options: List[Option] = field(default_factory=list)

    def action_ids(self) -> List[int]:
        return [o.action_id for o in self.options]

    def by_action(self, action_id: int) -> Option:
        for option in self.options:
            if option.action_id == action_id:
                return option
        raise KeyError(f"action {action_id} is not legal for: {self.prompt}")

    def __str__(self) -> str:
        opts = ", ".join(f"[{o.action_id}] {o.label}" for o in self.options)
        return f"P{self.player} {self.prompt}: {opts}"


#: Every decision kind the engine can raise, one-hot encoded in observations.
DECISION_KINDS = (
    "main",
    "setup_discard",
    "setup_bonus",
    "choose_food",
    "play_which",
    "play_where",
    "play_pay",
    "spend_egg",
    "take_die",
    "lay_egg",
    "draw_card",
    "tuck",
    "discard_egg",
    "repeat_power",
    "keep_bonus",
    "exchange_card_food",
    "exchange_food_egg",
    "exchange_egg_card",
)
DECISION_KIND_INDEX = {k: i for i, k in enumerate(DECISION_KINDS)}
