"""Core enums and rule constants for the Wingspan RL environment."""

from __future__ import annotations

from enum import IntEnum


class Food(IntEnum):
    INVERTEBRATE = 0
    SEED = 1
    FISH = 2
    FRUIT = 3
    RODENT = 4


N_FOOD = len(Food)

FOOD_NAMES = {
    Food.INVERTEBRATE: "invertebrate",
    Food.SEED: "seed",
    Food.FISH: "fish",
    Food.FRUIT: "fruit",
    Food.RODENT: "rodent",
}
FOOD_BY_NAME = {v: k for k, v in FOOD_NAMES.items()}
FOOD_SYMBOLS = {
    Food.INVERTEBRATE: "\N{BUG}",
    Food.SEED: "\N{EAR OF MAIZE}",
    Food.FISH: "\N{FISH}",
    Food.FRUIT: "\N{RED APPLE}",
    Food.RODENT: "\N{RAT}",
}


class Habitat(IntEnum):
    FOREST = 0
    GRASSLAND = 1
    WETLAND = 2


N_HABITAT = len(Habitat)

HABITAT_NAMES = {
    Habitat.FOREST: "forest",
    Habitat.GRASSLAND: "grassland",
    Habitat.WETLAND: "wetland",
}
HABITAT_BY_NAME = {v: k for k, v in HABITAT_NAMES.items()}


class NestType(IntEnum):
    BOWL = 0
    CAVITY = 1
    GROUND = 2
    PLATFORM = 3
    STAR = 4  # wild nest, counts as every type


N_NEST = len(NestType)

NEST_NAMES = {
    NestType.BOWL: "bowl",
    NestType.CAVITY: "cavity",
    NestType.GROUND: "ground",
    NestType.PLATFORM: "platform",
    NestType.STAR: "star",
}
NEST_BY_NAME = {v: k for k, v in NEST_NAMES.items()}


class Timing(IntEnum):
    """When a bird power triggers."""

    NONE = 0
    BROWN = 1  # when activated (habitat action)
    WHITE = 2  # when played
    PINK = 3  # once between turns, on an opponent's action
    GAME_END = 4  # scored at the end of the game


TIMING_NAMES = {
    Timing.NONE: "none",
    Timing.BROWN: "brown",
    Timing.WHITE: "white",
    Timing.PINK: "pink",
    Timing.GAME_END: "game_end",
}
TIMING_BY_NAME = {v: k for k, v in TIMING_NAMES.items()}

# --- Player mat -----------------------------------------------------------

MAX_BIRDS_PER_HABITAT = 5
N_BOARD_SLOTS = N_HABITAT * MAX_BIRDS_PER_HABITAT

#: Eggs you must pay to play a bird into a given column (0-indexed).
EGG_COST_BY_COLUMN = (0, 1, 1, 2, 2)

#: Strength of a habitat action, indexed by how many birds are already there.
HABITAT_ACTION_VALUE = {
    Habitat.FOREST: (1, 1, 2, 2, 3),      # food taken from the birdfeeder
    Habitat.GRASSLAND: (2, 2, 3, 3, 4),   # eggs laid
    Habitat.WETLAND: (1, 1, 2, 2, 3),     # cards drawn
}

#: Columns that unlock an optional exchange (card->food, food->egg, egg->card).
#: A row with ``n`` birds unlocks one exchange per entry <= n.
EXCHANGE_COLUMNS = (1, 3)

# --- Game flow ------------------------------------------------------------

N_ROUNDS = 4
ROUND_TURNS = (8, 7, 6, 5)
TRAY_SIZE = 3
FEEDER_DICE = 5
STARTING_HAND = 5
STARTING_BONUS = 2
#: Total (cards + food) a player keeps during setup; the rest is discarded.
SETUP_KEEP = 5

#: Round-goal points for 1st/2nd/3rd place, by round index.
GOAL_PLACE_POINTS = (
    (4, 1, 0),
    (5, 2, 1),
    (6, 3, 2),
    (7, 4, 3),
)

# --- Action space ---------------------------------------------------------

MAX_HAND = 24      # playable hand slots exposed to the agent
MAX_OPTIONS = 40   # slots for sub-decisions (payments, targets, dice, ...)

ACTION_PLAY_BIRD_BASE = 0
ACTION_GAIN_FOOD = MAX_HAND
ACTION_LAY_EGGS = MAX_HAND + 1
ACTION_DRAW_CARDS = MAX_HAND + 2
ACTION_OPTION_BASE = MAX_HAND + 3
N_ACTIONS = ACTION_OPTION_BASE + MAX_OPTIONS
