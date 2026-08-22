"""Command line entry point: watch bots play, benchmark agents, or play yourself."""

from __future__ import annotations

import argparse
import statistics
import sys
from typing import List, Optional, Sequence

from .agents import make_agent
from .engine import GameConfig, WingspanGame
from .render import render_text


def _build_game(args, seed: Optional[int]) -> WingspanGame:
    return WingspanGame(
        GameConfig(num_players=args.players, seed=seed, deck_path=args.deck,
                   keep_log=getattr(args, "log", False))
    )


def cmd_demo(args) -> int:
    game = _build_game(args, args.seed)
    agents = [make_agent(spec, args.seed) for spec in _agent_specs(args, game)]
    while not game.done:
        player = game.current_player
        action = agents[player].act(game, player)
        if args.verbose:
            option = game.pending.by_action(action)
            print(f"P{player}: {option.label}   ({game.pending.prompt})")
        game.step(action)
    print(render_text(game))
    if args.log:
        print("\n".join(game.log))
    return 0


def cmd_benchmark(args) -> int:
    totals: List[List[int]] = [[] for _ in range(args.players)]
    wins = [0] * args.players
    for i in range(args.games):
        seed = (args.seed or 0) + i
        game = _build_game(args, seed)
        agents = [make_agent(spec, seed + 17 * s)
                  for s, spec in enumerate(_agent_specs(args, game))]
        while not game.done:
            player = game.current_player
            game.step(agents[player].act(game, player))
        for index, score in enumerate(game.scores()):
            totals[index].append(score.total)
        for winner in game.winners():
            wins[winner] += 1
    specs = _agent_specs(args, None)
    print(f"{args.games} games, {args.players} players")
    for index in range(args.players):
        mean = statistics.mean(totals[index])
        stdev = statistics.pstdev(totals[index])
        print(f"  seat {index} ({specs[index]:>7}): "
              f"wins {wins[index]:>4}  mean {mean:6.2f}  sd {stdev:5.2f}  "
              f"max {max(totals[index])}")
    return 0


def cmd_play(args) -> int:
    game = _build_game(args, args.seed)
    bots = {seat: make_agent(args.bot, args.seed) for seat in range(args.players)
            if seat != args.seat}
    while not game.done:
        player = game.current_player
        if player != args.seat:
            game.step(bots[player].act(game, player))
            continue
        print(render_text(game, args.seat))
        decision = game.pending
        print(f"\n{decision.prompt}")
        for index, option in enumerate(decision.options):
            print(f"  [{index}] {option.label}")
        raw = input("choice> ").strip()
        if raw in ("q", "quit", "exit"):
            return 0
        try:
            choice = int(raw)
            action = decision.options[choice].action_id
        except (ValueError, IndexError):
            print("not a valid choice")
            continue
        game.step(action)
    print(render_text(game, args.seat))
    return 0


def _agent_specs(args, game) -> List[str]:
    specs = [s.strip() for s in args.agents.split(",") if s.strip()]
    if not specs:
        specs = ["greedy"]
    while len(specs) < args.players:
        specs.append(specs[-1])
    return specs[: args.players]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="wingspan", description=__doc__)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deck", default=None, help="path to a custom bird deck JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="play one bot game and print the result")
    demo.add_argument("--agents", default="greedy,greedy")
    demo.add_argument("--verbose", action="store_true")
    demo.add_argument("--log", action="store_true", help="print the rules log")
    demo.set_defaults(func=cmd_demo)

    bench = sub.add_parser("benchmark", help="compare agents over many games")
    bench.add_argument("--agents", default="greedy,random")
    bench.add_argument("--games", type=int, default=100)
    bench.set_defaults(func=cmd_benchmark)

    play = sub.add_parser("play", help="play against the bots in the terminal")
    play.add_argument("--seat", type=int, default=0)
    play.add_argument("--bot", default="greedy")
    play.add_argument("--agents", default="greedy")
    play.set_defaults(func=cmd_play)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
