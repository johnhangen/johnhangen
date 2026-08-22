import random

import pytest

from conftest import assert_invariants, make_game, play_random

from wingspan_rl.constants import (
    ACTION_DRAW_CARDS,
    ACTION_GAIN_FOOD,
    ACTION_LAY_EGGS,
    N_ACTIONS,
    ROUND_TURNS,
)
from wingspan_rl.agents import GreedyAgent, RandomAgent


@pytest.mark.parametrize("num_players", [1, 2, 3, 4, 5])
def test_random_games_finish_with_invariants(num_players):
    rng = random.Random(num_players)
    game = make_game(num_players=num_players, seed=num_players)
    play_random(game, rng, check=assert_invariants)
    assert game.done
    assert game.pending is None
    assert len(game.scores()) == num_players
    assert all(s.total > 0 for s in game.scores())
    assert 1 <= len(game.winners()) <= num_players


def test_each_player_gets_the_scheduled_number_of_turns():
    rng = random.Random(0)
    game = make_game(num_players=3, seed=4)
    main_decisions = [0, 0, 0]
    while not game.done:
        if game.pending.kind == "main":
            main_decisions[game.pending.player] += 1
        game.step(rng.choice(game.legal_action_ids()))
    assert main_decisions == [sum(ROUND_TURNS)] * 3


def test_action_mask_matches_pending_options():
    game = make_game(seed=11)
    mask = game.action_mask()
    assert mask.shape == (N_ACTIONS,)
    assert sorted(int(i) for i in mask.nonzero()[0]) == sorted(game.legal_action_ids())


def test_illegal_action_is_rejected():
    game = make_game(seed=2)
    illegal = next(a for a in range(N_ACTIONS) if a not in game.legal_action_ids())
    with pytest.raises(KeyError):
        game.step(illegal)


def test_step_after_game_over_raises():
    game = make_game(seed=3)
    play_random(game, random.Random(3))
    with pytest.raises(RuntimeError):
        game.step(0)


def test_same_seed_and_policy_replays_identically():
    def run(seed):
        game = make_game(num_players=2, seed=seed)
        agents = [GreedyAgent(1), RandomAgent(2)]
        actions = []
        while not game.done:
            player = game.current_player
            action = agents[player].act(game, player)
            actions.append(action)
            game.step(action)
        return actions, [s.total for s in game.scores()]

    assert run(21) == run(21)
    assert run(21) != run(22)


def test_main_actions_are_always_available():
    game = make_game(seed=8)
    rng = random.Random(0)
    seen_main = 0
    while not game.done and seen_main < 30:
        if game.pending.kind == "main":
            seen_main += 1
            legal = game.legal_action_ids()
            for action in (ACTION_GAIN_FOOD, ACTION_LAY_EGGS, ACTION_DRAW_CARDS):
                assert action in legal
        game.step(rng.choice(game.legal_action_ids()))
    assert seen_main == 30


def test_auto_resolve_single_option_can_be_disabled():
    """Forced choices are hidden by default, and only add steps when shown."""
    results = []
    for auto in (True, False):
        game = make_game(seed=5, auto_resolve_single=auto)
        agents = [GreedyAgent(1, noise=0.0), GreedyAgent(2, noise=0.0)]
        steps = 0
        while not game.done:
            player = game.current_player
            game.step(agents[player].act(game, player))
            steps += 1
        results.append((steps, [s.total for s in game.scores()]))
    (auto_steps, auto_scores), (forced_steps, forced_scores) = results
    assert forced_steps > auto_steps
    assert forced_scores == auto_scores


def test_game_log_records_events():
    game = make_game(seed=6, keep_log=True)
    play_random(game, random.Random(6))
    assert any("round 1" in line for line in game.log)
    assert any("final scores" in line for line in game.log)


def test_small_custom_deck_forces_reshuffles():
    """The discard pile is reshuffled when the deck runs out."""
    from wingspan_rl.cards import load_deck
    from wingspan_rl.engine import GameConfig, WingspanGame

    cards = load_deck()[:40]
    game = WingspanGame(GameConfig(num_players=4, seed=1, keep_log=False), cards=cards)
    play_random(game, random.Random(1), check=assert_invariants)
    assert game.done
    assert all(s.total > 0 for s in game.scores())


def test_custom_deck_must_have_contiguous_ids():
    from wingspan_rl.cards import load_deck
    from wingspan_rl.engine import GameConfig, WingspanGame

    cards = load_deck()[10:20]
    with pytest.raises(ValueError):
        WingspanGame(GameConfig(num_players=2, seed=0), cards=cards)
