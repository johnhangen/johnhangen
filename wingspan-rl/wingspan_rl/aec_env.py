"""PettingZoo-style AEC environment for self-play and multi-agent training.

``pettingzoo`` is optional: if it is installed we subclass :class:`AECEnv` so
the env drops into the PettingZoo tooling, and otherwise we fall back to a
small local base class that implements the same ``last``/``agent_iter``
interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - depends on the user's environment
    from pettingzoo import AECEnv as _AECBase

    HAS_PETTINGZOO = True
except ImportError:  # pragma: no cover
    HAS_PETTINGZOO = False

    class _AECBase:  # minimal stand-in with the parts we rely on
        agents: List[str] = []
        rewards: Dict[str, float] = {}
        _cumulative_rewards: Dict[str, float] = {}
        terminations: Dict[str, bool] = {}
        truncations: Dict[str, bool] = {}
        infos: Dict[str, dict] = {}
        agent_selection: str = ""

        _skip_agent_selection: Optional[str] = None

        def observe(self, agent: str):  # pragma: no cover - overridden
            raise NotImplementedError

        def _clear_rewards(self):
            for agent in self.rewards:
                self.rewards[agent] = 0.0

        def _accumulate_rewards(self):
            for agent, reward in self.rewards.items():
                self._cumulative_rewards[agent] += reward

        def _was_dead_step(self, action):
            if action is not None:
                raise ValueError("a dead agent may only be stepped with None")
            agent = self.agent_selection
            for mapping in (self.terminations, self.truncations, self.rewards,
                            self._cumulative_rewards, self.infos):
                mapping.pop(agent, None)
            self.agents.remove(agent)
            dead = [a for a in self.agents
                    if self.terminations[a] or self.truncations[a]]
            if dead:
                if self._skip_agent_selection is None:
                    self._skip_agent_selection = self.agent_selection
                self.agent_selection = dead[0]
            elif self._skip_agent_selection is not None:
                self.agent_selection = self._skip_agent_selection
                self._skip_agent_selection = None
            self._clear_rewards()

        def last(self, observe: bool = True):
            agent = self.agent_selection
            obs = self.observe(agent) if observe else None
            return (
                obs,
                self._cumulative_rewards.get(agent, 0.0),
                self.terminations.get(agent, False),
                self.truncations.get(agent, False),
                self.infos.get(agent, {}),
            )

        def agent_iter(self, max_iter: int = 2**31):
            count = 0
            while self.agents and count < max_iter:
                yield self.agent_selection
                count += 1

        def close(self):
            pass


from gymnasium import spaces  # noqa: E402

from .constants import N_ACTIONS  # noqa: E402
from .engine import GameConfig, WingspanGame  # noqa: E402
from .observation import OBS_SIZE, ObservationEncoder  # noqa: E402
from .render import render_text  # noqa: E402
from .scoring import score_player  # noqa: E402


class WingspanAECEnv(_AECBase):
    """Turn-based multi-agent Wingspan.

    Agents are named ``player_0 ... player_{n-1}``.  Exactly one agent acts at
    a time - whichever one the engine is waiting on, which includes an
    opponent's 'once between turns' powers.
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "wingspan_aec_v0",
                "is_parallelizable": False}

    def __init__(
        self,
        num_players: int = 2,
        reward_mode: str = "dense",
        render_mode: Optional[str] = None,
        deck_path: Optional[str] = None,
        **game_options: Any,
    ):
        super().__init__()
        self.num_players = num_players
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.deck_path = deck_path
        self.game_options = game_options

        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agents: List[str] = []
        self.game: Optional[WingspanGame] = None
        self.encoder: Optional[ObservationEncoder] = None
        self._last_scores: Dict[str, float] = {}

        obs_space = spaces.Dict(
            {
                "observation": spaces.Box(0.0, np.inf, (OBS_SIZE,), dtype=np.float32),
                "action_mask": spaces.Box(0, 1, (N_ACTIONS,), dtype=np.int8),
            }
        )
        self._observation_spaces = {a: obs_space for a in self.possible_agents}
        self._action_spaces = {a: spaces.Discrete(N_ACTIONS) for a in self.possible_agents}

    # ------------------------------------------------------------------ API
    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def seat(self, agent: str) -> int:
        return self.possible_agents.index(agent)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        config = GameConfig(
            num_players=self.num_players, seed=seed, deck_path=self.deck_path,
            **self.game_options,
        )
        self.game = WingspanGame(config)
        if self.encoder is None:
            self.encoder = ObservationEncoder(self.game.cards)
        self.agents = list(self.possible_agents)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._last_scores = {a: self._score(a) for a in self.agents}
        self._skip_agent_selection = None
        self.agent_selection = self.possible_agents[self.game.current_player]
        self._update_infos()

    def observe(self, agent: str) -> Dict[str, np.ndarray]:
        assert self.game is not None and self.encoder is not None
        mask = np.zeros(N_ACTIONS, dtype=np.int8)
        if self.game.pending is not None and self.game.pending.player == self.seat(agent):
            mask[self.game.pending.action_ids()] = 1
        return {
            "observation": self.encoder.encode(self.game, self.seat(agent)),
            "action_mask": mask,
        }

    def step(self, action: Optional[int]) -> None:
        assert self.game is not None
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            self._update_infos()
            return
        if self.game.pending is None or self.game.pending.player != self.seat(agent):
            raise RuntimeError(f"{agent} is not the acting agent")

        self._cumulative_rewards[agent] = 0.0
        self.game.step(int(action))
        self._clear_rewards()

        if self.reward_mode == "dense":
            for other in self.agents:
                score = self._score(other)
                self.rewards[other] = (score - self._last_scores[other]) / 10.0
                self._last_scores[other] = score

        if self.game.done:
            finals = self._final_rewards()
            for other in self.agents:
                self.rewards[other] += finals[other]
                self.terminations[other] = True
        else:
            self.agent_selection = self.possible_agents[self.game.pending.player]

        self._accumulate_rewards()
        self._update_infos()

    def render(self):
        assert self.game is not None
        text = render_text(self.game)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self):  # pragma: no cover - nothing to release
        pass

    # -------------------------------------------------------------- internals
    def _score(self, agent: str) -> float:
        assert self.game is not None
        board = self.game.state.board(self.seat(agent))
        return float(score_player(board, self.game.cards, self.game.bonus_cards).total)

    def _final_rewards(self) -> Dict[str, float]:
        assert self.game is not None
        totals = [s.total for s in self.game.scores()]
        winners = self.game.winners()
        out = {}
        for agent in self.possible_agents:
            seat = self.seat(agent)
            others = [t for i, t in enumerate(totals) if i != seat]
            margin = totals[seat] - (max(others) if others else 0)
            if self.reward_mode == "sparse":
                out[agent] = (1.0 if len(winners) == 1 else 0.0) if seat in winners else -1.0
            elif self.reward_mode == "score":
                out[agent] = totals[seat] / 100.0
            elif self.reward_mode == "dense":
                out[agent] = margin / 20.0
            else:
                out[agent] = margin / 50.0
        return out

    def _update_infos(self) -> None:
        assert self.game is not None
        pending = self.game.pending
        for agent in list(self.infos):
            info: Dict[str, Any] = {"round": self.game.state.round_index}
            if pending is not None and pending.player == self.seat(agent):
                info["decision"] = pending.kind
                info["prompt"] = pending.prompt
                info["legal_actions"] = pending.action_ids()
            if self.game.done:
                info["scores"] = [s.total for s in self.game.scores()]
                info["winners"] = self.game.winners()
            self.infos[agent] = info
