"""Generate the bundled bird deck (``wingspan_rl/data/birds.json``).

Species names and wingspans are real; every game statistic (habitat, food
cost, nest, egg capacity, points, power) is generated here by a seeded,
rule-based procedure.  This keeps the environment self-contained and
reproducible without transcribing a published card list.  Swap in your own
JSON deck with ``load_deck(path)`` if you have licensed card data.

Run:  python tools/generate_birds.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wingspan_rl.cards import BirdCard, Cost, food_counts, save_deck  # noqa: E402
from wingspan_rl.constants import Food, Habitat, NestType, Timing  # noqa: E402
from wingspan_rl.powers import Power  # noqa: E402

SEED = 20240817
DECK_SIZE = 170

# (common name, wingspan in cm, passerine?)
SPECIES = [
    ("American Robin", 41, 1), ("Northern Cardinal", 31, 1), ("Blue Jay", 41, 1),
    ("American Crow", 100, 1), ("Common Raven", 117, 1), ("Black-capped Chickadee", 21, 1),
    ("Tufted Titmouse", 26, 1), ("White-breasted Nuthatch", 26, 1),
    ("Red-breasted Nuthatch", 22, 1), ("Brown Creeper", 20, 1), ("House Wren", 15, 1),
    ("Carolina Wren", 29, 1), ("Marsh Wren", 15, 1), ("Winter Wren", 14, 1),
    ("Eastern Bluebird", 33, 1), ("Mountain Bluebird", 36, 1), ("Wood Thrush", 33, 1),
    ("Hermit Thrush", 29, 1), ("Veery", 30, 1), ("Gray Catbird", 30, 1),
    ("Northern Mockingbird", 36, 1), ("Brown Thrasher", 32, 1), ("European Starling", 39, 1),
    ("Cedar Waxwing", 30, 1), ("Bohemian Waxwing", 34, 1), ("House Sparrow", 24, 1),
    ("American Goldfinch", 22, 1), ("House Finch", 25, 1), ("Purple Finch", 25, 1),
    ("Pine Siskin", 22, 1), ("Red Crossbill", 27, 1), ("Evening Grosbeak", 36, 1),
    ("Rose-breasted Grosbeak", 32, 1), ("Indigo Bunting", 21, 1), ("Painted Bunting", 21, 1),
    ("Lazuli Bunting", 22, 1), ("Eastern Towhee", 26, 1), ("Spotted Towhee", 27, 1),
    ("Song Sparrow", 21, 1), ("White-throated Sparrow", 23, 1), ("White-crowned Sparrow", 24, 1),
    ("Chipping Sparrow", 21, 1), ("Field Sparrow", 20, 1), ("Savannah Sparrow", 20, 1),
    ("Vesper Sparrow", 24, 1), ("Grasshopper Sparrow", 18, 1), ("Dark-eyed Junco", 23, 1),
    ("Lark Bunting", 27, 1), ("Bobolink", 29, 1), ("Eastern Meadowlark", 36, 1),
    ("Western Meadowlark", 37, 1), ("Red-winged Blackbird", 33, 1),
    ("Yellow-headed Blackbird", 42, 1), ("Brown-headed Cowbird", 36, 1),
    ("Common Grackle", 43, 1), ("Baltimore Oriole", 29, 1), ("Bullock's Oriole", 31, 1),
    ("Orchard Oriole", 25, 1), ("Yellow Warbler", 20, 1), ("Common Yellowthroat", 17, 1),
    ("American Redstart", 20, 1), ("Black-and-white Warbler", 21, 1),
    ("Yellow-rumped Warbler", 23, 1), ("Prothonotary Warbler", 22, 1), ("Ovenbird", 24, 1),
    ("Northern Waterthrush", 23, 1), ("Blackburnian Warbler", 22, 1),
    ("Magnolia Warbler", 19, 1), ("Chestnut-sided Warbler", 20, 1), ("Eastern Phoebe", 27, 1),
    ("Say's Phoebe", 33, 1), ("Eastern Kingbird", 38, 1), ("Western Kingbird", 39, 1),
    ("Great Crested Flycatcher", 34, 1), ("Willow Flycatcher", 22, 1),
    ("Eastern Wood-Pewee", 26, 1), ("Vermilion Flycatcher", 25, 1),
    ("Scissor-tailed Flycatcher", 38, 1), ("Horned Lark", 32, 1), ("Barn Swallow", 34, 1),
    ("Tree Swallow", 35, 1), ("Cliff Swallow", 30, 1), ("Purple Martin", 45, 1),
    ("Bank Swallow", 27, 1), ("Red-eyed Vireo", 25, 1), ("Warbling Vireo", 22, 1),
    ("Blue-headed Vireo", 23, 1), ("Loggerhead Shrike", 32, 1), ("American Pipit", 27, 1),
    ("American Dipper", 23, 1), ("Golden-crowned Kinglet", 18, 1),
    ("Ruby-crowned Kinglet", 18, 1), ("Blue-gray Gnatcatcher", 16, 1), ("Bushtit", 15, 1),
    ("Verdin", 17, 1), ("Canyon Wren", 19, 1), ("Rock Wren", 22, 1), ("Cactus Wren", 28, 1),
    ("Steller's Jay", 44, 1), ("California Scrub-Jay", 39, 1), ("Clark's Nutcracker", 61, 1),
    ("Black-billed Magpie", 61, 1), ("Canada Jay", 45, 1), ("Boreal Chickadee", 22, 1),
    ("Pine Grosbeak", 36, 1), ("Common Redpoll", 22, 1), ("Snow Bunting", 32, 1),
    ("Lapland Longspur", 28, 1),
    ("Mallard", 88, 0), ("Wood Duck", 73, 0), ("Northern Pintail", 88, 0),
    ("American Wigeon", 84, 0), ("Blue-winged Teal", 58, 0), ("Green-winged Teal", 58, 0),
    ("Canvasback", 84, 0), ("Redhead", 84, 0), ("Bufflehead", 55, 0),
    ("Common Goldeneye", 79, 0), ("Hooded Merganser", 61, 0), ("Common Merganser", 86, 0),
    ("Ruddy Duck", 47, 0), ("Canada Goose", 152, 0), ("Snow Goose", 137, 0),
    ("Trumpeter Swan", 203, 0), ("Tundra Swan", 168, 0), ("American Coot", 61, 0),
    ("Common Loon", 132, 0), ("Pied-billed Grebe", 45, 0), ("Horned Grebe", 47, 0),
    ("Western Grebe", 61, 0), ("Double-crested Cormorant", 132, 0),
    ("American White Pelican", 274, 0), ("Brown Pelican", 203, 0), ("Great Blue Heron", 183, 0),
    ("Great Egret", 134, 0), ("Snowy Egret", 100, 0), ("Green Heron", 66, 0),
    ("Black-crowned Night-Heron", 112, 0), ("American Bittern", 107, 0), ("White Ibis", 97, 0),
    ("Roseate Spoonbill", 127, 0), ("Sandhill Crane", 200, 0), ("Whooping Crane", 230, 0),
    ("Killdeer", 46, 0), ("American Avocet", 72, 0), ("Black-necked Stilt", 71, 0),
    ("Spotted Sandpiper", 37, 0), ("Greater Yellowlegs", 70, 0), ("Willet", 66, 0),
    ("Long-billed Curlew", 90, 0), ("Marbled Godwit", 78, 0), ("Dunlin", 37, 0),
    ("Least Sandpiper", 33, 0), ("Wilson's Snipe", 44, 0), ("American Woodcock", 45, 0),
    ("Ring-billed Gull", 122, 0), ("Herring Gull", 144, 0), ("Laughing Gull", 104, 0),
    ("Common Tern", 78, 0), ("Black Tern", 61, 0), ("Caspian Tern", 127, 0),
    ("Belted Kingfisher", 51, 0), ("Osprey", 163, 0), ("Bald Eagle", 203, 0),
    ("Golden Eagle", 200, 0), ("Red-tailed Hawk", 122, 0), ("Red-shouldered Hawk", 100, 0),
    ("Cooper's Hawk", 79, 0), ("Sharp-shinned Hawk", 58, 0), ("Northern Harrier", 109, 0),
    ("Swainson's Hawk", 127, 0), ("Broad-winged Hawk", 86, 0), ("American Kestrel", 56, 0),
    ("Merlin", 61, 0), ("Peregrine Falcon", 104, 0), ("Prairie Falcon", 102, 0),
    ("Turkey Vulture", 173, 0), ("Black Vulture", 150, 0), ("Great Horned Owl", 122, 0),
    ("Barred Owl", 107, 0), ("Barn Owl", 107, 0), ("Eastern Screech-Owl", 51, 0),
    ("Burrowing Owl", 53, 0), ("Snowy Owl", 142, 0), ("Northern Saw-whet Owl", 43, 0),
    ("Great Gray Owl", 137, 0), ("Wild Turkey", 125, 0), ("Ring-necked Pheasant", 79, 0),
    ("Greater Prairie-Chicken", 70, 0), ("Northern Bobwhite", 33, 0), ("California Quail", 33, 0),
    ("Gambel's Quail", 36, 0), ("Ruffed Grouse", 56, 0), ("Mourning Dove", 45, 0),
    ("Rock Pigeon", 64, 0), ("Band-tailed Pigeon", 66, 0), ("Inca Dove", 29, 0),
    ("Greater Roadrunner", 48, 0), ("Yellow-billed Cuckoo", 46, 0),
    ("Common Nighthawk", 61, 0), ("Eastern Whip-poor-will", 45, 0), ("Chimney Swift", 30, 0),
    ("Ruby-throated Hummingbird", 11, 0), ("Anna's Hummingbird", 12, 0),
    ("Rufous Hummingbird", 11, 0), ("Broad-tailed Hummingbird", 13, 0),
    ("Downy Woodpecker", 30, 0), ("Hairy Woodpecker", 38, 0), ("Northern Flicker", 51, 0),
    ("Pileated Woodpecker", 74, 0), ("Red-bellied Woodpecker", 42, 0),
    ("Red-headed Woodpecker", 42, 0), ("Acorn Woodpecker", 43, 0),
    ("Yellow-bellied Sapsucker", 40, 0), ("Lewis's Woodpecker", 52, 0),
    ("American Three-toed Woodpecker", 38, 0), ("Sora", 36, 0), ("Virginia Rail", 33, 0),
    ("Common Gallinule", 56, 0), ("Least Tern", 51, 0), ("Black Skimmer", 112, 0),
    ("Atlantic Puffin", 53, 0), ("Common Murre", 66, 0), ("Black Guillemot", 58, 0),
    ("Northern Gannet", 180, 0), ("Magnificent Frigatebird", 217, 0),
]

WATER_WORDS = (
    "duck", "goose", "swan", "teal", "wigeon", "pintail", "merganser", "goldeneye",
    "bufflehead", "canvasback", "redhead", "mallard", "coot", "loon", "grebe",
    "cormorant", "pelican", "heron", "egret", "bittern", "ibis", "spoonbill", "crane",
    "gull", "tern", "skimmer", "puffin", "murre", "guillemot", "gannet", "frigatebird",
    "sandpiper", "yellowlegs", "willet", "curlew", "godwit", "dunlin", "snipe",
    "avocet", "stilt", "killdeer", "rail", "sora", "gallinule", "kingfisher", "dipper",
    "woodcock", "waterthrush", "marsh wren",
)
RAPTOR_WORDS = (
    "hawk", "eagle", "falcon", "kestrel", "merlin", "harrier", "osprey", "owl",
    "vulture", "shrike", "roadrunner",
)
GRASS_WORDS = (
    "sparrow", "meadowlark", "longspur", "bobolink", "lark", "quail", "bobwhite",
    "prairie-chicken", "pheasant", "turkey", "grouse", "dove", "pigeon", "bunting",
    "blackbird", "cowbird", "pipit", "burrowing", "kingbird", "swallow", "martin",
)
FOREST_WORDS = (
    "woodpecker", "sapsucker", "flicker", "chickadee", "titmouse", "nuthatch",
    "creeper", "warbler", "vireo", "thrush", "veery", "jay", "crow", "raven",
    "magpie", "nutcracker", "kinglet", "grosbeak", "finch", "siskin", "crossbill",
    "waxwing", "tanager", "cuckoo", "hummingbird", "swift", "phoebe", "flycatcher",
    "pewee", "oriole", "redstart", "ovenbird", "gnatcatcher", "wren", "towhee",
)


def _has(name: str, words) -> bool:
    low = name.lower()
    return any(w in low for w in words)


def pick_habitats(rng: random.Random, name: str) -> tuple:
    if _has(name, WATER_WORDS):
        base = [Habitat.WETLAND]
        if rng.random() < 0.35:
            base.append(rng.choices([Habitat.FOREST, Habitat.GRASSLAND],
                                    weights=[0.35, 0.65])[0])
    elif _has(name, RAPTOR_WORDS):
        base = [rng.choice([Habitat.FOREST, Habitat.GRASSLAND])]
        if rng.random() < 0.55:
            other = Habitat.GRASSLAND if base[0] is Habitat.FOREST else Habitat.FOREST
            base.append(other)
    elif _has(name, GRASS_WORDS):
        base = [Habitat.GRASSLAND]
        if rng.random() < 0.3:
            base.append(rng.choice([Habitat.FOREST, Habitat.WETLAND]))
    elif _has(name, FOREST_WORDS):
        base = [Habitat.FOREST]
        if rng.random() < 0.42:
            base.append(rng.choices([Habitat.GRASSLAND, Habitat.WETLAND],
                                    weights=[0.7, 0.3])[0])
    else:
        base = [rng.choice(list(Habitat))]
        if rng.random() < 0.25:
            base.append(rng.choice(list(Habitat)))
    return tuple(sorted(set(base)))


def is_predator(name: str) -> bool:
    return _has(name, ("hawk", "eagle", "falcon", "kestrel", "merlin", "harrier",
                       "osprey", "owl", "shrike", "roadrunner", "heron", "loon"))


def pick_foods(rng: random.Random, name: str, wingspan: int) -> list:
    if is_predator(name):
        pool = [Food.RODENT, Food.RODENT, Food.FISH, Food.INVERTEBRATE]
    elif _has(name, ("gull", "tern", "pelican", "cormorant", "merganser", "puffin",
                     "murre", "guillemot", "gannet", "kingfisher", "skimmer", "grebe")):
        pool = [Food.FISH, Food.FISH, Food.INVERTEBRATE]
    elif _has(name, WATER_WORDS):
        pool = [Food.INVERTEBRATE, Food.FISH, Food.SEED]
    elif _has(name, ("hummingbird", "waxwing", "thrasher", "catbird", "tanager",
                     "oriole", "robin")):
        pool = [Food.FRUIT, Food.FRUIT, Food.INVERTEBRATE]
    elif _has(name, ("sparrow", "finch", "grosbeak", "siskin", "crossbill", "junco",
                     "quail", "bobwhite", "dove", "pigeon", "turkey", "pheasant",
                     "redpoll", "bunting", "longspur", "jay", "nutcracker")):
        pool = [Food.SEED, Food.SEED, Food.FRUIT, Food.INVERTEBRATE]
    else:
        pool = [Food.INVERTEBRATE, Food.INVERTEBRATE, Food.SEED, Food.FRUIT]
    return pool


def pick_nest(rng: random.Random, name: str) -> NestType:
    if rng.random() < 0.09:
        return NestType.STAR
    if _has(name, ("woodpecker", "sapsucker", "flicker", "chickadee", "titmouse",
                   "nuthatch", "bluebird", "wood duck", "screech-owl", "saw-whet",
                   "merganser", "swift", "martin", "starling", "wren", "verdin",
                   "goldeneye", "bufflehead", "kestrel", "puffin", "guillemot")):
        return NestType.CAVITY
    if _has(name, ("hawk", "eagle", "osprey", "heron", "egret", "ibis", "spoonbill",
                   "cormorant", "crane", "stork", "vulture", "dove", "pigeon")):
        return NestType.PLATFORM
    if _has(name, ("sparrow", "plover", "killdeer", "sandpiper", "tern", "gull",
                   "duck", "goose", "swan", "teal", "quail", "bobwhite", "grouse",
                   "pheasant", "turkey", "meadowlark", "bobolink", "longspur",
                   "nighthawk", "whip-poor-will", "woodcock", "snipe", "rail", "sora",
                   "avocet", "stilt", "curlew", "godwit", "dunlin", "murre", "gannet",
                   "burrowing", "harrier", "prairie-chicken", "lark", "gallinule",
                   "coot", "loon", "grebe", "pipit", "junco", "towhee")):
        return NestType.GROUND
    return NestType.BOWL


def build_cost(rng: random.Random, name: str, wingspan: int) -> tuple:
    size = 1 + wingspan // 45
    size = max(1, min(5, size + rng.choice([-1, 0, 0, 1])))
    pool = pick_foods(rng, name, wingspan)
    counts: dict = {}
    wild = 0
    for _ in range(size):
        if rng.random() < 0.18:
            wild += 1
        else:
            food = rng.choice(pool)
            counts[food] = counts.get(food, 0) + 1
    if not counts and wild == 0:
        counts[rng.choice(pool)] = 1
    primary = Cost(food=food_counts(counts), wild=wild)
    costs = [primary]
    # Cheap birds sometimes offer an either/or cost.
    if primary.total == 1 and primary.wild == 0 and rng.random() < 0.3:
        alt = rng.choice([f for f in pool if food_counts(counts)[int(f)] == 0] or pool)
        costs.append(Cost(food=food_counts({alt: 1})))
    return tuple(costs)


def build_power(rng: random.Random, name: str, habitats: tuple, wingspan: int,
                foods: list, predator: bool) -> Power:
    roll = rng.random()
    if roll < 0.10:
        return Power("none", Timing.NONE)
    if roll < 0.17:
        per = rng.choice(["eggs_on_this", "cached_food", "tucked_cards",
                          "birds_in_habitat", "birds_with_nest"])
        params = {"per": per, "amount": 1}
        if per == "birds_in_habitat":
            params["habitat"] = rng.choice(list(Habitat)).name.lower()
        if per == "birds_with_nest":
            params["nest"] = rng.choice([NestType.BOWL, NestType.CAVITY,
                                         NestType.GROUND, NestType.PLATFORM]).name.lower()
        return Power("end_points_per", Timing.GAME_END, params)
    if roll < 0.25:
        return Power(
            "on_opponent_action",
            Timing.PINK,
            {
                "trigger": rng.choice(["gain_food", "lay_eggs", "draw_cards", "play_bird"]),
                "effect": rng.choice(["gain_food_feeder", "lay_egg", "draw_card"]),
            },
        )
    if roll < 0.45:  # white / when played
        choice = rng.random()
        if choice < 0.2:
            return Power("play_extra_bird", Timing.WHITE,
                         {"habitat": rng.choice(list(Habitat)).name.lower()})
        if choice < 0.35:
            return Power("gain_bonus_card", Timing.WHITE, {})
        if choice < 0.5:
            return Power("draw_cards", Timing.WHITE, {"count": rng.choice([1, 2, 2, 3])})
        if choice < 0.65:
            return Power("gain_food_supply", Timing.WHITE,
                         {"food": rng.choice(foods).name.lower(), "count": rng.choice([1, 2])})
        if choice < 0.8:
            return Power("lay_eggs", Timing.WHITE,
                         {"count": rng.choice([1, 2]), "target": "any"})
        if choice < 0.9:
            return Power("lay_eggs_each_bird", Timing.WHITE,
                         {"count": 1, "nest": rng.choice(["bowl", "cavity", "ground",
                                                          "platform"])})
        return Power("all_players_draw", Timing.WHITE, {})
    # brown / when activated
    if predator and rng.random() < 0.6:
        return Power("predator_hunt", Timing.BROWN,
                     {"threshold": rng.choice([40, 50, 65, 75, 100]),
                      "reward": rng.choice(["cache", "tuck"]),
                      "food": rng.choice([Food.RODENT, Food.FISH]).name.lower()})
    choice = rng.random()
    if choice < 0.18:
        return Power("gain_food_feeder", Timing.BROWN,
                     {"food": rng.choice(foods).name.lower() if rng.random() < 0.6 else "any",
                      "count": 1})
    if choice < 0.33:
        return Power("gain_food_supply", Timing.BROWN,
                     {"food": rng.choice(foods).name.lower(), "count": 1})
    if choice < 0.47:
        return Power("lay_eggs", Timing.BROWN,
                     {"count": 1, "target": rng.choice(["this", "any", "any"])})
    if choice < 0.60:
        return Power("draw_cards", Timing.BROWN, {"count": 1})
    if choice < 0.67:
        return Power("draw_from_tray", Timing.BROWN, {"count": 1})
    if choice < 0.78:
        return Power("tuck_from_hand", Timing.BROWN,
                     {"count": 1, "then": rng.choice([None, "egg", "food", "draw"])})
    if choice < 0.88:
        return Power("cache_food", Timing.BROWN,
                     {"food": rng.choice(foods).name.lower(), "count": 1})
    if choice < 0.93:
        return Power("all_players_gain_food", Timing.BROWN,
                     {"food": rng.choice(foods).name.lower()})
    if choice < 0.97:
        return Power("discard_egg_for_food", Timing.BROWN,
                     {"food": rng.choice(foods).name.lower(), "count": 2})
    return Power("repeat_brown", Timing.BROWN, {})


STRONG_KINDS = {"play_extra_bird", "gain_bonus_card", "repeat_brown", "lay_eggs_each_bird"}


def build_card(rng: random.Random, index: int, name: str, wingspan: int,
               passerine: int) -> BirdCard:
    habitats = pick_habitats(rng, name)
    predator = is_predator(name)
    foods = pick_foods(rng, name, wingspan)
    costs = build_cost(rng, name, wingspan)
    power = build_power(rng, name, habitats, wingspan, foods, predator)
    nest = pick_nest(rng, name)
    egg_capacity = max(1, min(6, 6 - wingspan // 38 + rng.choice([0, 0, 1])))

    points = costs[0].total * 1.6 + rng.choice([0, 0, 1])
    if power.kind == "none":
        points += 2
    elif power.kind in STRONG_KINDS:
        points -= 1.5
    if len(habitats) > 1:
        points -= 0.5
    if power.timing is Timing.GAME_END:
        points -= 1
    points = int(max(1, min(9, round(points))))

    return BirdCard(
        id=index,
        name=name,
        points=points,
        habitats=habitats,
        costs=costs,
        nest=nest,
        egg_capacity=egg_capacity,
        wingspan=wingspan,
        predator=predator,
        passerine=bool(passerine),
        power=power,
    )


def main() -> None:
    rng = random.Random(SEED)
    species = sorted(SPECIES)
    rng.shuffle(species)
    species = species[:DECK_SIZE]
    species.sort()
    cards = [build_card(rng, i, name, span, pas)
             for i, (name, span, pas) in enumerate(species)]
    out = ROOT / "wingspan_rl" / "data" / "birds.json"
    save_deck(cards, out)

    from collections import Counter
    print(f"wrote {len(cards)} cards -> {out}")
    print("habitats:", Counter(h.name for c in cards for h in c.habitats))
    print("timings :", Counter(c.power.timing.name for c in cards))
    print("nests   :", Counter(c.nest.name for c in cards))
    print("points  :", Counter(c.points for c in cards))
    print("cost    :", Counter(c.costs[0].total for c in cards))


if __name__ == "__main__":
    main()
