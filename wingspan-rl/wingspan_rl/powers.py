"""Bird power descriptions.

A power is pure data: a ``kind`` string plus parameters.  The rules engine
(:mod:`wingspan_rl.engine`) owns the implementation of every kind, which keeps
card data serialisable and lets users swap in their own card sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .constants import FOOD_BY_NAME, FOOD_NAMES, HABITAT_BY_NAME, TIMING_BY_NAME, Timing

#: Every implemented power kind.  Index in this tuple is the one-hot position
#: used by the observation encoder, so only ever append to it.
POWER_KINDS = (
    "none",
    # --- brown / when-activated -------------------------------------------
    "gain_food_supply",
    "gain_food_feeder",
    "lay_eggs",
    "draw_cards",
    "draw_from_tray",
    "tuck_from_hand",
    "cache_food",
    "predator_hunt",
    "all_players_gain_food",
    "repeat_brown",
    "discard_egg_for_food",
    # --- white / when-played ----------------------------------------------
    "play_extra_bird",
    "gain_bonus_card",
    "all_players_draw",
    "lay_eggs_each_bird",
    # --- pink / once between turns ----------------------------------------
    "on_opponent_action",
    # --- yellow / game end -------------------------------------------------
    "end_points_per",
)
POWER_KIND_INDEX = {k: i for i, k in enumerate(POWER_KINDS)}

#: Kinds that only ever appear as a 'when activated' (brown) power.
BROWN_ONLY_KINDS = frozenset(
    {
        "gain_food_feeder",
        "draw_from_tray",
        "tuck_from_hand",
        "cache_food",
        "predator_hunt",
        "all_players_gain_food",
        "repeat_brown",
        "discard_egg_for_food",
    }
)
#: Kinds that only ever appear as a 'when played' (white) power.
WHITE_ONLY_KINDS = frozenset(
    {
        "play_extra_bird",
        "gain_bonus_card",
        "all_players_draw",
        "lay_eggs_each_bird",
    }
)
#: Kinds usable as either brown or white.
SHARED_KINDS = frozenset({"gain_food_supply", "lay_eggs", "draw_cards"})


@dataclass
class Power:
    kind: str
    timing: Timing = Timing.NONE
    params: Dict[str, Any] = field(default_factory=dict)
    text: str = ""

    def __post_init__(self) -> None:
        if self.kind not in POWER_KIND_INDEX:
            raise ValueError(f"unknown power kind: {self.kind!r}")
        if not self.text:
            self.text = describe(self)

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    @property
    def kind_index(self) -> int:
        return POWER_KIND_INDEX[self.kind]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "timing": self.timing.name.lower(),
            "params": dict(self.params),
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Power":
        params = dict(data.get("params", {}))
        return cls(
            kind=data["kind"],
            timing=TIMING_BY_NAME[data.get("timing", "none")],
            params=params,
            text=data.get("text", ""),
        )


def _food_label(value: Any) -> str:
    if value in (None, "any", "wild"):
        return "food"
    if isinstance(value, str):
        return FOOD_NAMES[FOOD_BY_NAME[value]]
    return FOOD_NAMES[value]


def describe(power: Power) -> str:  # noqa: C901 - a flat lookup table reads best
    """Human readable rules text, generated from the power's parameters."""
    p, k = power.params, power.kind
    n = p.get("count", 1)
    if k == "none":
        return ""
    if k == "gain_food_supply":
        return f"Gain {n} {_food_label(p.get('food'))} from the supply."
    if k == "gain_food_feeder":
        return f"Gain {n} {_food_label(p.get('food'))} from the birdfeeder."
    if k == "lay_eggs":
        where = {"this": "on this bird", "any": "on any bird"}.get(p.get("target", "this"))
        if p.get("nest"):
            where = f"on any bird with a {p['nest']} nest"
        return f"Lay {n} egg(s) {where}."
    if k == "draw_cards":
        return f"Draw {n} card(s)."
    if k == "draw_from_tray":
        return f"Draw {n} face-up card(s) from the tray."
    if k == "tuck_from_hand":
        bonus = p.get("then")
        extra = {
            None: "",
            "egg": " If you do, lay 1 egg on this bird.",
            "food": " If you do, gain 1 food from the supply.",
            "draw": " If you do, draw 1 card.",
        }[bonus]
        return f"Tuck {n} card(s) from your hand behind this bird.{extra}"
    if k == "cache_food":
        return f"Cache {n} {_food_label(p.get('food'))} from the supply on this bird."
    if k == "predator_hunt":
        reward = {
            "cache": "cache it on this bird",
            "tuck": "tuck it behind this bird",
        }[p.get("reward", "tuck")]
        return (
            f"Look at a card from the deck. If its wingspan is less than "
            f"{p.get('threshold', 50)}cm, {reward}. Otherwise discard it."
        )
    if k == "all_players_gain_food":
        return f"Each player gains 1 {_food_label(p.get('food'))} from the supply."
    if k == "repeat_brown":
        return "Activate the 'when activated' power of another bird in this habitat."
    if k == "discard_egg_for_food":
        return f"You may discard 1 egg to gain {n} {_food_label(p.get('food'))}."
    if k == "play_extra_bird":
        hab = p.get("habitat")
        where = "in any habitat" if hab is None else f"in the {hab}"
        return f"Play an additional bird {where}. Pay its normal cost."
    if k == "gain_bonus_card":
        return "Draw 2 bonus cards and keep 1."
    if k == "all_players_draw":
        return "Each player draws 1 card from the deck."
    if k == "lay_eggs_each_bird":
        return f"Lay {n} egg(s) on each bird with a {p.get('nest', 'bowl')} nest."
    if k == "on_opponent_action":
        trigger = {
            "gain_food": "takes the 'gain food' action",
            "lay_eggs": "takes the 'lay eggs' action",
            "draw_cards": "takes the 'draw cards' action",
            "play_bird": "plays a bird",
        }[p.get("trigger", "gain_food")]
        effect = {
            "gain_food_feeder": "gain 1 food from the birdfeeder",
            "lay_egg": "lay 1 egg on this bird",
            "draw_card": "draw 1 card from the deck",
        }[p.get("effect", "gain_food_feeder")]
        return f"When another player {trigger}, {effect}."
    if k == "end_points_per":
        per = {
            "eggs_on_this": "egg on this bird",
            "cached_food": "food cached on this bird",
            "tucked_cards": "card tucked behind this bird",
            "birds_in_habitat": f"bird in your {p.get('habitat', 'forest')}",
            "birds_with_nest": f"bird with a {p.get('nest', 'bowl')} nest",
        }[p.get("per", "eggs_on_this")]
        return f"Game end: {p.get('amount', 1)} point(s) per {per}."
    raise ValueError(f"no description for power kind {k!r}")  # pragma: no cover


def validate(power: Power) -> None:
    """Raise if a power's parameters are inconsistent with its kind."""
    p = power.params
    if power.kind in BROWN_ONLY_KINDS and power.timing is not Timing.BROWN:
        raise ValueError(f"{power.kind} must be a brown power")
    if power.kind in WHITE_ONLY_KINDS and power.timing is not Timing.WHITE:
        raise ValueError(f"{power.kind} must be a white power")
    if power.kind in SHARED_KINDS and power.timing not in (Timing.BROWN, Timing.WHITE):
        raise ValueError(f"{power.kind} must be a brown or white power")
    if power.kind == "on_opponent_action" and power.timing is not Timing.PINK:
        raise ValueError("on_opponent_action must be a pink power")
    if power.kind == "end_points_per" and power.timing is not Timing.GAME_END:
        raise ValueError("end_points_per must be a game-end power")
    if "food" in p and p["food"] not in (None, "any") and p["food"] not in FOOD_BY_NAME:
        raise ValueError(f"unknown food {p['food']!r}")
    if p.get("habitat") is not None and p["habitat"] not in HABITAT_BY_NAME:
        raise ValueError(f"unknown habitat {p['habitat']!r}")
    describe(power)


NO_POWER = Power(kind="none", timing=Timing.NONE, text="")
