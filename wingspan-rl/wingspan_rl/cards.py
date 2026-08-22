"""Bird cards, food costs and deck loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .constants import (
    FOOD_BY_NAME,
    FOOD_NAMES,
    FOOD_SYMBOLS,
    HABITAT_BY_NAME,
    HABITAT_NAMES,
    N_FOOD,
    NEST_BY_NAME,
    NEST_NAMES,
    Food,
    Habitat,
    NestType,
)
from .powers import NO_POWER, Power, validate as validate_power

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DECK_PATH = DATA_DIR / "birds.json"

FoodCounts = Tuple[int, int, int, int, int]
ZERO_FOOD: FoodCounts = (0, 0, 0, 0, 0)


def food_counts(mapping: Dict[Any, int]) -> FoodCounts:
    """Build a 5-tuple of food counts from a ``{food: n}`` mapping."""
    counts = [0] * N_FOOD
    for key, value in mapping.items():
        food = FOOD_BY_NAME[key] if isinstance(key, str) else Food(key)
        counts[int(food)] += int(value)
    return tuple(counts)  # type: ignore[return-value]


def add_food(a: Sequence[int], b: Sequence[int]) -> FoodCounts:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]


def sub_food(a: Sequence[int], b: Sequence[int]) -> FoodCounts:
    out = tuple(x - y for x, y in zip(a, b))
    if any(v < 0 for v in out):
        raise ValueError(f"cannot subtract {b} from {a}")
    return out  # type: ignore[return-value]


def food_str(counts: Sequence[int]) -> str:
    parts = [f"{n}{FOOD_SYMBOLS[Food(i)]}" for i, n in enumerate(counts) if n]
    return " ".join(parts) if parts else "-"


@dataclass(frozen=True)
class Cost:
    """One way of paying for a bird: exact foods plus ``wild`` any-food slots."""

    food: FoodCounts = ZERO_FOOD
    wild: int = 0

    @property
    def total(self) -> int:
        return sum(self.food) + self.wild

    def to_dict(self) -> Dict[str, int]:
        out = {FOOD_NAMES[Food(i)]: n for i, n in enumerate(self.food) if n}
        if self.wild:
            out["wild"] = self.wild
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "Cost":
        wild = int(data.get("wild", 0))
        return cls(food=food_counts({k: v for k, v in data.items() if k != "wild"}), wild=wild)

    def __str__(self) -> str:
        base = food_str(self.food)
        if self.wild:
            base = (base + " " if base != "-" else "") + f"{self.wild}*"
        return base


@dataclass
class BirdCard:
    id: int
    name: str
    points: int
    habitats: Tuple[Habitat, ...]
    costs: Tuple[Cost, ...]
    nest: NestType
    egg_capacity: int
    wingspan: int
    predator: bool = False
    passerine: bool = False
    power: Power = field(default_factory=lambda: NO_POWER)

    # -- derived -----------------------------------------------------------
    @property
    def food_types(self) -> frozenset:
        out = set()
        for cost in self.costs:
            out.update(Food(i) for i, n in enumerate(cost.food) if n)
        return frozenset(out)

    @property
    def min_cost(self) -> int:
        return min(cost.total for cost in self.costs)

    def lives_in(self, habitat: Habitat) -> bool:
        return habitat in self.habitats

    def nest_matches(self, nest: NestType) -> bool:
        return self.nest is NestType.STAR or self.nest is nest

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "points": self.points,
            "habitats": [HABITAT_NAMES[h] for h in self.habitats],
            "costs": [c.to_dict() for c in self.costs],
            "nest": NEST_NAMES[self.nest],
            "egg_capacity": self.egg_capacity,
            "wingspan": self.wingspan,
            "predator": self.predator,
            "passerine": self.passerine,
            "power": self.power.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BirdCard":
        power_data = data.get("power")
        power = Power.from_dict(power_data) if power_data else NO_POWER
        return cls(
            id=int(data["id"]),
            name=data["name"],
            points=int(data["points"]),
            habitats=tuple(HABITAT_BY_NAME[h] for h in data["habitats"]),
            costs=tuple(Cost.from_dict(c) for c in data["costs"]),
            nest=NEST_BY_NAME[data["nest"]],
            egg_capacity=int(data["egg_capacity"]),
            wingspan=int(data["wingspan"]),
            predator=bool(data.get("predator", False)),
            passerine=bool(data.get("passerine", False)),
            power=power,
        )

    def __str__(self) -> str:
        cost = " / ".join(str(c) for c in self.costs)
        habs = "".join(HABITAT_NAMES[h][0].upper() for h in self.habitats)
        nest = NEST_NAMES[self.nest]
        return f"{self.name} [{self.points}pt {habs} {cost} {nest}:{self.egg_capacity}]"


def validate_card(card: BirdCard) -> None:
    if not card.habitats:
        raise ValueError(f"{card.name}: needs at least one habitat")
    if not card.costs:
        raise ValueError(f"{card.name}: needs at least one cost option")
    if not 1 <= card.egg_capacity <= 6:
        raise ValueError(f"{card.name}: egg capacity out of range")
    if card.points < 0:
        raise ValueError(f"{card.name}: negative points")
    validate_power(card.power)


def load_deck(path: Optional[Path | str] = None) -> List[BirdCard]:
    """Load a bird deck from JSON (defaults to the bundled deck)."""
    path = Path(path) if path is not None else DEFAULT_DECK_PATH
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    cards = [BirdCard.from_dict(entry) for entry in data["birds"]]
    ids = {c.id for c in cards}
    if len(ids) != len(cards):
        raise ValueError("duplicate card ids in deck")
    for card in cards:
        validate_card(card)
    return cards


def save_deck(cards: Iterable[BirdCard], path: Path | str) -> None:
    payload = {"birds": [c.to_dict() for c in cards]}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")


# --- payments -------------------------------------------------------------


def enumerate_payments(
    supply: Sequence[int], costs: Sequence[Cost], limit: int = 64
) -> List[FoodCounts]:
    """All distinct ways to pay one of ``costs`` out of ``supply``.

    Results are sorted (cheapest food indices first) and de-duplicated across
    the alternative cost options, so the option list an agent sees is stable.
    """
    seen: set = set()
    out: List[FoodCounts] = []
    for cost in costs:
        if any(cost.food[i] > supply[i] for i in range(N_FOOD)):
            continue
        remaining = [supply[i] - cost.food[i] for i in range(N_FOOD)]
        if cost.wild == 0:
            payment = cost.food
            if payment not in seen:
                seen.add(payment)
                out.append(payment)
            continue
        if sum(remaining) < cost.wild:
            continue
        for combo in combinations_with_replacement(range(N_FOOD), cost.wild):
            extra = [0] * N_FOOD
            ok = True
            for idx in combo:
                extra[idx] += 1
                if extra[idx] > remaining[idx]:
                    ok = False
                    break
            if not ok:
                continue
            payment = tuple(cost.food[i] + extra[i] for i in range(N_FOOD))
            if payment not in seen:
                seen.add(payment)
                out.append(payment)
    out.sort()
    return out[:limit]


def can_afford(supply: Sequence[int], costs: Sequence[Cost]) -> bool:
    return bool(enumerate_payments(supply, costs, limit=1))
