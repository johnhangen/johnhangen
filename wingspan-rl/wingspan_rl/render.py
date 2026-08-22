"""Human-readable rendering of a game state."""

from __future__ import annotations

from typing import Optional

from .cards import food_str
from .constants import HABITAT_ACTION_VALUE, HABITAT_NAMES, MAX_BIRDS_PER_HABITAT, Habitat
from .goals import GOALS


def render_bird(game, placed) -> str:
    card = game.cards[placed.card_id]
    bits = [card.name]
    if placed.eggs:
        bits.append("o" * placed.eggs)
    if sum(placed.cached):
        bits.append(f"cache:{food_str(placed.cached)}")
    if placed.tucked:
        bits.append(f"tuck:{placed.tucked}")
    return " ".join(bits)


def render_board(game, player: int) -> str:
    board = game.state.board(player)
    lines = [f"Player {player}: food {food_str(board.food)} | hand {len(board.hand)} "
             f"| bonus {len(board.bonus_cards)} | eggs {board.total_eggs()}"]
    for habitat in Habitat:
        row = board.habitats[habitat]
        value = HABITAT_ACTION_VALUE[habitat][min(len(row), MAX_BIRDS_PER_HABITAT - 1)]
        cells = " | ".join(render_bird(game, p) for p in row) or "-"
        lines.append(f"  {HABITAT_NAMES[habitat]:<9} ({value}): {cells}")
    return "\n".join(lines)


def render_text(game, player: Optional[int] = None) -> str:
    state = game.state
    lines = []
    if state.done:
        lines.append("=== game over ===")
    else:
        lines.append(
            f"=== round {state.round_index + 1}/4 | turns left "
            f"{state.turns_left} | first player P{state.start_player} ==="
        )
    goal_names = [GOALS[g].name for g in state.goals]
    lines.append("goals: " + ", ".join(
        f"{'>' if i == state.round_index else ''}R{i + 1} {name}"
        for i, name in enumerate(goal_names)
    ))
    lines.append(f"feeder: {state.feeder} | deck {len(state.deck)} | discard {len(state.discard)}")
    lines.append("tray: " + (", ".join(str(game.cards[c]) for c in state.tray) or "-"))
    for index in range(state.num_players):
        lines.append(render_board(game, index))
    if state.done:
        for index, score in enumerate(game.scores()):
            lines.append(f"P{index} score {score.total}: {score.as_dict()}")
        lines.append(f"winner(s): {game.winners()}")
    elif game.pending is not None:
        lines.append(str(game.pending))
    if player is not None and not state.done:
        board = state.board(player)
        lines.append(f"P{player} hand: " + ", ".join(
            str(game.cards[c]) for c in board.hand) or "-")
    return "\n".join(lines)
