# wingspan-rl

A reinforcement-learning environment for the board game **Wingspan** — the full
engine (four rounds, three habitats, bird powers, bonus cards, end-of-round
goals), a Gymnasium single-agent env, a PettingZoo AEC multi-agent env, and
scripted baselines to train against.

```bash
pip install -e ".[dev]"          # numpy + gymnasium + pettingzoo + pytest
python -m wingspan_rl.cli demo   # watch two bots play
```

```python
import wingspan_rl

env = wingspan_rl.make_env(num_players=2, opponents="greedy", reward_mode="dense")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(info["legal_actions"][0])
```

## Why it is built this way

Wingspan is not a "pick one of four actions" game. Playing a bird means
choosing the card, the habitat, which eggs to spend, and which food to pay
with; a single *gain food* action can cascade through five brown powers, each
with its own choices, and an opponent's pink power can interrupt in the middle.

So the engine is written as **one Python generator that yields a decision
whenever it needs input**:

```python
game = wingspan_rl.WingspanGame(num_players=2, seed=0)
while not game.done:
    decision = game.pending          # who must choose, and the legal options
    game.step(decision.options[0].action_id)
```

Every point where a human would be asked something is a `Decision` carrying the
concrete, legal `Option` list — nothing else is reachable. Agents never need to
know *why* they are being asked; they read the mask and pick.

## The RL interfaces

### Single agent (Gymnasium)

`WingspanEnv` seats one learner against scripted bots.

| | |
|---|---|
| observation | `Box(3320,)` float32, all features in `[0, 1]` |
| action | `Discrete(67)`, mostly illegal at any moment |
| mask | `info["action_mask"]`, plus `env.action_masks()` for sb3-contrib |
| reward | `dense` (default), `sparse`, `score`, `score_diff` |

```python
env = wingspan_rl.make_env(
    num_players=3,
    opponents=["greedy", "greedy", "random"],  # per seat; the learner's is ignored
    agent_seat=0,
    reward_mode="dense",
    illegal_action="raise",                    # or "penalize" for -1 and continue
)
```

It is also registered with gymnasium:

```python
import gymnasium as gym
import wingspan_rl  # registers Wingspan-v0

env = gym.make("Wingspan-v0", num_players=2, opponents="greedy")
mask = env.unwrapped.action_masks()   # wrappers hide the helper
```

**Use the mask.** Only 3–10 of the 67 actions are legal at a typical decision,
so unmasked PPO spends nearly all of its samples on rejected actions.

### Multi-agent (PettingZoo AEC)

`WingspanAECEnv` exposes every seat, which is what you want for self-play. It
passes `pettingzoo.test.api_test`, and falls back to a small built-in base
class if PettingZoo is not installed.

```python
env = wingspan_rl.make_aec_env(num_players=2, reward_mode="sparse")
env.reset(seed=0)
for agent in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()
    if termination or truncation:
        env.step(None); continue
    legal = obs["action_mask"].nonzero()[0]
    env.step(policy(obs["observation"], legal))
```

### Action space (67 discrete)

| ids | meaning |
|---|---|
| `0 – 23` | play the bird in hand slot *i* (hand is kept sorted) |
| `24` | gain food (forest) |
| `25` | lay eggs (grassland) |
| `26` | draw cards (wetland) |
| `27 – 66` | option *i* of the current sub-decision |

Sub-decision slots are contextual — slot 27 might be "pay 1 seed + 1 fruit",
"take the fish die" or "lay the egg on the Wood Duck" — which is why
`info["option_labels"]` and the option block of the observation exist. The
observation encodes each live option's kind, its numeric value and, when it
refers to a card, that card's features, so the meaning of a slot is always
visible in the input rather than something the network has to memorise.

Decisions with a single legal option are resolved automatically
(`auto_resolve_single=True`); a deterministic policy plays an identical game
either way, it just sees fewer no-choice steps.

### Observation layout

`ObservationEncoder.describe()` returns the exact slices:

| section | width | contents |
|---|---|---|
| `global` | 28 | round, turns left, deck/discard/tray sizes, birdfeeder dice, seat |
| `goals` | 80 | the four round goals, which is active, points already banked |
| `self` | 16 | food, hand size, bird/egg counts, action strengths, exchanges |
| `board` | 720 | 15 mat slots × (card features + eggs, cached food, tucked) |
| `hand` | 1080 | 24 hand slots × (card features + affordable/playable flags) |
| `tray` | 132 | the three face-up cards |
| `opponents` | 52 | per opponent: birds by habitat, eggs, food, hand, points |
| `bonus` | 32 | which bonus cards you hold |
| `decision` | 20 | what you are being asked, and whether it is your turn |
| `options` | 1160 | 40 option slots × (kind, value, referenced card) |

Card features (43 per card) are static: points, cost by food type, wild slots,
habitats, nest, egg capacity, wingspan, predator/passerine flags, power timing
and power kind.

## Rules implemented

* 4 rounds of 8/7/6/5 turns, first-player marker passing each round.
* Setup: keep any of 5 cards, 1 food per card discarded, 1 of 2 bonus cards.
* Three habitat actions that strengthen as the row fills — food `1,1,2,2,3`,
  eggs `2,2,3,3,4`, cards `1,1,2,2,3` — plus the mat's exchange arrows
  (card→food, food→egg, egg→card).
* Column egg costs `0,1,1,2,2`, five birds per habitat.
* Birdfeeder: five dice, six faces, the invertebrate+seed face grants both,
  automatic reroll when the feeder empties or all dice match.
* Bird powers: brown (activated right-to-left when you take the habitat
  action), white (when played), pink (once between your turns, on an
  opponent's action) and yellow (game end) — 17 power kinds covering food
  gain/caching, egg laying, drawing, tucking, predator hunts, repeating
  another bird's power, playing extra birds and drawing bonus cards.
* 18 end-of-round goals scored competitively (4/1/0 … 7/4/3, ties consume
  places, zero never scores) and 32 bonus cards with tiered scoring.
* Final score: birds + eggs + cached food + tucked cards + bonus cards +
  round goals + game-end powers; ties broken by leftover food and cards.

### Deliberate simplifications

* **Card data is generated, not transcribed.** `tools/generate_birds.py`
  builds the 170-card deck from a table of real species (names and wingspans
  are factual) with seeded, rule-based mechanics. It is balanced and
  reproducible, but it is *not* the published card list. Point
  `load_deck(path)` / `WingspanEnv(deck_path=...)` at your own JSON to swap in
  a different set — see `wingspan_rl/data/birds.json` for the schema.
* The exchange arrows are modelled as unlocking at columns 2 and 4 of each row.
* Nectar and the Oceania/European/Asia expansions are not implemented.
* A power that asks for a specific food from the feeder grants only that food,
  even when taken from the combined die.
* Brown powers resolve automatically rather than offering a "decline" choice,
  except where declining is meaningful (tucking, discarding eggs, repeats).

## Baselines

`GreedyAgent` is a tuned heuristic — it plays birds early, lays eggs late, and
spends its most abundant food first. It is the default opponent.

```
$ python -m wingspan_rl.cli benchmark --agents greedy,random --games 100
100 games, 2 players
  seat 0 ( greedy): wins    93  mean  62.79  sd 11.34  max 89
  seat 1 ( random): wins     7  mean  36.65  sd 10.86  max 62
```

Two greedy agents against each other land around 60 points a game, which is in
the range of a casual human two-player game.

## Training

```bash
pip install "stable-baselines3>=2.0" sb3-contrib torch
python examples/train_maskable_ppo.py --timesteps 400000 --opponents greedy
```

The example wraps the env in sb3-contrib's `ActionMasker` and trains
`MaskablePPO` (~900 env-steps/s on CPU with 8 parallel envs). Other examples:

* `examples/random_rollout.py` — drive the raw engine and render a game.
* `examples/selfplay_aec.py` — self-play loop over the AEC env.

## Command line

```bash
python -m wingspan_rl.cli demo --agents greedy,greedy --verbose
python -m wingspan_rl.cli benchmark --agents greedy,random --games 200
python -m wingspan_rl.cli --players 3 play --seat 0    # play against the bots
```

## Project layout

```
wingspan_rl/
  constants.py     enums and rule constants (mat values, action ids)
  cards.py         BirdCard, costs, payment enumeration, deck IO
  powers.py        power data model, descriptions, validation
  bonus.py         32 bonus cards         goals.py    18 round goals
  state.py         birdfeeder, player mats, table state
  decisions.py     Decision/Option protocol
  engine.py        the rules engine (one generator, decision by decision)
  scoring.py       final scoring and tie-breaks
  observation.py   fixed-size encoding    agents.py   random + greedy baselines
  env.py           Gymnasium env          aec_env.py  PettingZoo AEC env
  render.py        text rendering         cli.py      demo/benchmark/play
  data/birds.json  the generated 170-card deck
tools/generate_birds.py   regenerates the deck deterministically
tests/                    75 tests: rules, powers, scoring, envs, agents
examples/                 rollout, PPO training, self-play
```

```bash
python -m pytest            # 75 tests, ~1s
```

*Wingspan is designed by Elizabeth Hargrave and published by Stonemaier Games.
This is an unofficial, fan-made research environment with no affiliation to
either, and it ships no published card data.*
