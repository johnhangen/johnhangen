import random

import pytest

pytest.importorskip("gymnasium")

from wingspan_rl.aec_env import WingspanAECEnv  # noqa: E402


def play(env, seed=0):
    env.reset(seed=seed)
    rng = random.Random(seed)
    steps = 0
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        if termination or truncation:
            env.step(None)
            if not env.agents:
                break
            continue
        legal = [i for i, m in enumerate(obs["action_mask"]) if m]
        assert legal, "an acting agent must have at least one legal action"
        env.step(rng.choice(legal))
        steps += 1
    return steps, info


def test_aec_episode_runs_to_completion():
    env = WingspanAECEnv(num_players=2, reward_mode="sparse")
    steps, info = play(env, seed=1)
    assert steps > 20
    assert env.agents == []
    assert sorted(info["scores"]) == sorted(info["scores"])


def test_rewards_are_zero_sum_in_sparse_mode():
    env = WingspanAECEnv(num_players=2, reward_mode="sparse")
    env.reset(seed=2)
    rng = random.Random(2)
    totals = {a: 0.0 for a in env.possible_agents}
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        totals[agent] += reward
        if termination or truncation:
            env.step(None)
            if not env.agents:
                break
            continue
        env.step(rng.choice([i for i, m in enumerate(obs["action_mask"]) if m]))
    assert sum(totals.values()) == 0.0


def test_only_the_selected_agent_may_act():
    env = WingspanAECEnv(num_players=2)
    env.reset(seed=3)
    other = next(a for a in env.agents if a != env.agent_selection)
    env.agent_selection, saved = other, env.agent_selection
    with pytest.raises(RuntimeError):
        env.step(0)
    env.agent_selection = saved


def test_observation_masks_are_agent_specific():
    env = WingspanAECEnv(num_players=2)
    env.reset(seed=4)
    acting = env.agent_selection
    idle = next(a for a in env.agents if a != acting)
    assert env.observe(acting)["action_mask"].sum() > 0
    assert env.observe(idle)["action_mask"].sum() == 0


def test_pettingzoo_api_conformance():
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import api_test

    api_test(WingspanAECEnv(num_players=2), num_cycles=200, verbose_progress=False)
