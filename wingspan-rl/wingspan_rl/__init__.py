"""A reinforcement-learning environment for the board game Wingspan."""

from .agents import Agent, GreedyAgent, RandomAgent, make_agent
from .bonus import BONUS_CARDS, BonusCard
from .cards import BirdCard, Cost, load_deck, save_deck
from .constants import (
    ACTION_DRAW_CARDS,
    ACTION_GAIN_FOOD,
    ACTION_LAY_EGGS,
    ACTION_OPTION_BASE,
    ACTION_PLAY_BIRD_BASE,
    MAX_HAND,
    MAX_OPTIONS,
    N_ACTIONS,
    Food,
    Habitat,
    NestType,
    Timing,
)
from .decisions import Decision, Option
from .engine import GameConfig, WingspanGame
from .goals import GOALS, Goal
from .observation import OBS_SIZE, ObservationEncoder
from .powers import Power
from .render import render_text
from .scoring import ScoreBreakdown, final_scores, score_player, winners

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "BirdCard",
    "BONUS_CARDS",
    "BonusCard",
    "Cost",
    "Decision",
    "Food",
    "GOALS",
    "GameConfig",
    "Goal",
    "GreedyAgent",
    "Habitat",
    "MAX_HAND",
    "MAX_OPTIONS",
    "N_ACTIONS",
    "NestType",
    "OBS_SIZE",
    "ObservationEncoder",
    "Option",
    "Power",
    "RandomAgent",
    "ScoreBreakdown",
    "Timing",
    "WingspanGame",
    "ACTION_DRAW_CARDS",
    "ACTION_GAIN_FOOD",
    "ACTION_LAY_EGGS",
    "ACTION_OPTION_BASE",
    "ACTION_PLAY_BIRD_BASE",
    "final_scores",
    "load_deck",
    "make_agent",
    "render_text",
    "save_deck",
    "score_player",
    "winners",
]


def make_env(**kwargs):
    """Create the single-agent Gymnasium environment (requires gymnasium)."""
    from .env import WingspanEnv

    return WingspanEnv(**kwargs)


def make_aec_env(**kwargs):
    """Create the multi-agent AEC environment."""
    from .aec_env import WingspanAECEnv

    return WingspanAECEnv(**kwargs)
