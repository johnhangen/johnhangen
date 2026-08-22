import random

from conftest import make_game

from wingspan_rl.agents import GreedyAgent, RandomAgent, make_agent


def _play(agents, seed):
    game = make_game(num_players=len(agents), seed=seed)
    while not game.done:
        player = game.current_player
        action = agents[player].act(game, player)
        assert action in game.legal_action_ids(), "agent produced an illegal action"
        game.step(action)
    return game


def test_agents_only_produce_legal_actions():
    for seed in range(3):
        _play([GreedyAgent(seed), RandomAgent(seed), GreedyAgent(seed + 1)], seed)


def test_greedy_beats_random_over_many_games():
    greedy_wins = 0
    games = 20
    for seed in range(games):
        agents = [GreedyAgent(seed), RandomAgent(seed + 500)]
        game = _play(agents, seed)
        if 0 in game.winners():
            greedy_wins += 1
    assert greedy_wins >= int(0.75 * games)


def test_greedy_outscores_random_on_average():
    greedy_total = random_total = 0
    for seed in range(10):
        game = _play([GreedyAgent(seed), RandomAgent(seed + 900)], seed)
        scores = [s.total for s in game.scores()]
        greedy_total += scores[0]
        random_total += scores[1]
    assert greedy_total > random_total * 1.2


def test_make_agent_accepts_specs_instances_and_callables():
    assert isinstance(make_agent("random", 0), RandomAgent)
    assert isinstance(make_agent("greedy", 0), GreedyAgent)
    agent = GreedyAgent(1)
    assert make_agent(agent) is agent
    functional = make_agent(lambda game, player: game.legal_action_ids()[0])
    game = make_game(seed=1)
    assert functional.act(game, 0) == game.legal_action_ids()[0]


def test_greedy_weights_are_tunable():
    agent = GreedyAgent(0, w_egg_late=9.9)
    assert agent.W_EGG_LATE == 9.9
    assert GreedyAgent(0).W_EGG_LATE != 9.9
