"""The Wingspan rules engine.

The whole game is expressed as one Python generator that yields a
:class:`~wingspan_rl.decisions.Decision` whenever it needs a player to choose
something, and resumes when the choice is sent back in.  That keeps multi-step
effects (pay eggs, then food, then resolve a 'when played' power that plays
another bird) readable, and gives every agent exactly one uniform interface:
look at ``game.pending``, pick a legal action id, call ``game.step``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Sequence

from .bonus import BONUS_CARDS, BonusCard
from .cards import BirdCard, enumerate_payments, load_deck
from .constants import (
    ACTION_DRAW_CARDS,
    ACTION_GAIN_FOOD,
    ACTION_LAY_EGGS,
    ACTION_OPTION_BASE,
    ACTION_PLAY_BIRD_BASE,
    EGG_COST_BY_COLUMN,
    FOOD_BY_NAME,
    FOOD_NAMES,
    HABITAT_BY_NAME,
    HABITAT_NAMES,
    MAX_HAND,
    MAX_OPTIONS,
    NEST_BY_NAME,
    N_ACTIONS,
    N_FOOD,
    N_ROUNDS,
    ROUND_TURNS,
    SETUP_KEEP,
    STARTING_BONUS,
    STARTING_HAND,
    TRAY_SIZE,
    Food,
    Habitat,
    NestType,
    Timing,
)
from .decisions import Decision, Option
from .goals import GOALS, score_goal
from .powers import Power
from .scoring import ScoreBreakdown, final_scores, winners
from .state import DIE_FACES, Birdfeeder, GameState, PlacedBird, PlayerBoard

Flow = Generator[Decision, Any, None]

MAIN_ACTION_KINDS = {
    ACTION_GAIN_FOOD: "gain_food",
    ACTION_LAY_EGGS: "lay_eggs",
    ACTION_DRAW_CARDS: "draw_cards",
}


@dataclass
class GameConfig:
    num_players: int = 2
    seed: Optional[int] = None
    deck_path: Optional[str] = None
    use_bonus_cards: bool = True
    #: Resolve decisions that have a single legal option without asking.
    auto_resolve_single: bool = True
    #: Guard against pathological 'repeat another power' chains.
    max_power_depth: int = 3
    keep_log: bool = True


class WingspanGame:
    """A full game of Wingspan, driven one decision at a time."""

    def __init__(
        self,
        config: Optional[GameConfig] = None,
        cards: Optional[Sequence[BirdCard]] = None,
        bonus_cards: Optional[Sequence[BonusCard]] = None,
        **overrides: Any,
    ):
        self.config = config or GameConfig()
        for key, value in overrides.items():
            if not hasattr(self.config, key):
                raise TypeError(f"unknown config option {key!r}")
            setattr(self.config, key, value)
        if not 1 <= self.config.num_players <= 5:
            raise ValueError("Wingspan supports 1-5 players")

        self.cards: List[BirdCard] = list(cards) if cards else load_deck(self.config.deck_path)
        self.cards.sort(key=lambda c: c.id)
        if [c.id for c in self.cards] != list(range(len(self.cards))):
            raise ValueError("card ids must be a contiguous range starting at 0")
        self.bonus_cards: List[BonusCard] = list(bonus_cards) if bonus_cards else list(BONUS_CARDS)

        self.rng = random.Random(self.config.seed)
        self.log: List[str] = []
        self.state = self._new_state()
        self.pending: Optional[Decision] = None
        self._flow: Flow = self._run_game()
        self._pump(first=True)

    # ------------------------------------------------------------------ setup
    def _new_state(self) -> GameState:
        n = self.config.num_players
        deck = [c.id for c in self.cards]
        self.rng.shuffle(deck)
        bonus_deck = [b.id for b in self.bonus_cards]
        self.rng.shuffle(bonus_deck)
        goals = self.rng.sample(range(len(GOALS)), N_ROUNDS)
        state = GameState(
            num_players=n,
            players=[PlayerBoard(index=i) for i in range(n)],
            deck=deck,
            discard=[],
            tray=[],
            feeder=Birdfeeder(self.rng),
            bonus_deck=bonus_deck,
            bonus_discard=[],
            goals=goals,
            rng=self.rng,
            turns_left=[ROUND_TURNS[0]] * n,
        )
        state.refill_tray()
        return state

    # ------------------------------------------------------------- public API
    @property
    def done(self) -> bool:
        return self.state.done

    @property
    def current_player(self) -> int:
        return self.pending.player if self.pending else self.state.current_player

    def legal_action_ids(self) -> List[int]:
        return self.pending.action_ids() if self.pending else []

    def action_mask(self):
        import numpy as np

        mask = np.zeros(N_ACTIONS, dtype=bool)
        if self.pending is not None:
            mask[self.pending.action_ids()] = True
        return mask

    def step(self, action_id: int) -> None:
        """Apply ``action_id`` to the pending decision and run to the next one."""
        if self.pending is None:
            raise RuntimeError("game is over")
        option = self.pending.by_action(int(action_id))
        self._pump(payload=option.payload)

    def scores(self) -> List[ScoreBreakdown]:
        return final_scores(self.state, self.cards, self.bonus_cards)

    def winners(self) -> List[int]:
        return winners(self.state, self.cards, self.bonus_cards)

    def card(self, card_id: int) -> BirdCard:
        return self.cards[card_id]

    # --------------------------------------------------------------- plumbing
    def _pump(self, first: bool = False, payload: Any = None) -> None:
        try:
            decision = next(self._flow) if first else self._flow.send(payload)
            while self.config.auto_resolve_single and len(decision.options) == 1:
                decision = self._flow.send(decision.options[0].payload)
        except StopIteration:
            self.pending = None
            self.state.done = True
            return
        self.pending = decision

    def _note(self, message: str) -> None:
        if self.config.keep_log:
            self.log.append(message)

    def _ask(
        self,
        player: int,
        kind: str,
        prompt: str,
        options: List[Option],
        default: Any = None,
    ) -> Generator[Decision, Any, Any]:
        """Yield a decision built from choice-space options; return the payload."""
        if not options:
            return default
        options = options[:MAX_OPTIONS]
        for i, option in enumerate(options):
            option.action_id = ACTION_OPTION_BASE + i
        payload = yield Decision(player=player, kind=kind, prompt=prompt, options=options)
        return payload

    @staticmethod
    def _pass_option(label: str = "pass") -> Option:
        return Option(action_id=-1, kind="pass", label=label, payload=None)

    # ------------------------------------------------------------------ flow
    def _run_game(self) -> Flow:
        yield from self._setup()
        for round_index in range(N_ROUNDS):
            self.state.round_index = round_index
            self._note(f"--- round {round_index + 1} "
                       f"(goal: {GOALS[self.state.goals[round_index]].name}) ---")
            yield from self._run_round(round_index)
            self._end_round(round_index)
        self.state.done = True
        totals = [s.total for s in self.scores()]
        self._note(f"final scores: {totals}")

    def _setup(self) -> Flow:
        state = self.state
        for board in state.players:
            for _ in range(STARTING_HAND):
                card_id = state.draw_card()
                if card_id is not None:
                    board.hand.append(card_id)
            board.hand.sort()
            if self.config.use_bonus_cards:
                for _ in range(STARTING_BONUS):
                    bonus_id = state.draw_bonus()
                    if bonus_id is not None:
                        board.bonus_cards.append(bonus_id)

        for board in state.players:
            state.current_player = board.index
            yield from self._setup_keep(board)
            if self.config.use_bonus_cards:
                yield from self._setup_bonus(board)

    def _setup_keep(self, board: PlayerBoard) -> Flow:
        """Keep any number of the starting cards; gain 1 food per card discarded."""
        for _ in range(SETUP_KEEP):
            if not board.hand:
                break
            options = [
                Option(0, "keep_card", f"discard {self.cards[c].name}", payload=c, card_id=c)
                for c in board.hand
            ]
            options.append(self._pass_option("keep the rest"))
            card_id = yield from self._ask(
                board.index,
                "setup_discard",
                f"discard a starting card for 1 food (hand {len(board.hand)})",
                options,
            )
            if card_id is None:
                break
            board.hand.remove(card_id)
            self.state.discard.append(card_id)
            food = yield from self._choose_food(board.index, "gain 1 food from the supply")
            board.gain_food(food)
        self._note(f"P{board.index} keeps {len(board.hand)} cards, "
                   f"{board.total_food()} food")

    def _setup_bonus(self, board: PlayerBoard) -> Flow:
        if len(board.bonus_cards) <= 1:
            return
        options = [
            Option(0, "bonus_card", self.bonus_cards[b].name, payload=b)
            for b in board.bonus_cards
        ]
        keep = yield from self._ask(board.index, "setup_bonus", "keep 1 bonus card", options)
        for bonus_id in list(board.bonus_cards):
            if bonus_id != keep:
                self.state.bonus_discard.append(bonus_id)
        board.bonus_cards = [keep]

    def _run_round(self, round_index: int) -> Flow:
        state = self.state
        turns = ROUND_TURNS[round_index]
        state.turns_left = [turns] * state.num_players
        order = [(state.start_player + k) % state.num_players
                 for k in range(state.num_players)]
        for _ in range(turns):
            for player in order:
                state.current_player = player
                state.turns_left[player] -= 1
                yield from self._take_turn(player)

    def _end_round(self, round_index: int) -> None:
        state = self.state
        goal = GOALS[state.goals[round_index]]
        counts = [goal.count(b, self.cards) for b in state.players]
        points = score_goal(counts, round_index)
        state.round_scores.append(points)
        for board, pts in zip(state.players, points):
            board.round_goal_points[round_index] = pts
        self._note(f"goal '{goal.name}' counts={counts} points={points}")
        state.discard.extend(state.tray)
        state.tray = []
        state.refill_tray()
        state.feeder.refill()
        state.start_player = (state.start_player + 1) % state.num_players
        for board in state.players:
            for placed in board.all_birds():
                placed.pink_used = False

    # ------------------------------------------------------------------ turns
    def _take_turn(self, player: int) -> Flow:
        board = self.state.board(player)
        for placed in board.all_birds():
            placed.pink_used = False
        option = yield from self._ask_main(player)
        if option is None:
            return
        if option.kind == "play_bird":
            yield from self._play_bird_flow(player, card_id=option.payload)
            action_kind = "play_bird"
        elif option.kind == "gain_food":
            yield from self._gain_food_action(player)
            action_kind = "gain_food"
        elif option.kind == "lay_eggs":
            yield from self._lay_eggs_action(player)
            action_kind = "lay_eggs"
        else:
            yield from self._draw_cards_action(player)
            action_kind = "draw_cards"
        yield from self._trigger_pink(player, action_kind)

    def _ask_main(self, player: int) -> Generator[Decision, Any, Optional[Option]]:
        board = self.state.board(player)
        options: List[Option] = []
        for index, card_id in enumerate(board.hand[:MAX_HAND]):
            if self._playable_habitats(board, self.cards[card_id]):
                options.append(
                    Option(
                        action_id=ACTION_PLAY_BIRD_BASE + index,
                        kind="play_bird",
                        label=f"play {self.cards[card_id].name}",
                        payload=card_id,
                        card_id=card_id,
                    )
                )
        options.append(Option(ACTION_GAIN_FOOD, "gain_food",
                              f"gain food ({board.action_value(Habitat.FOREST)})",
                              payload="gain_food",
                              value=board.action_value(Habitat.FOREST)))
        options.append(Option(ACTION_LAY_EGGS, "lay_eggs",
                              f"lay eggs ({board.action_value(Habitat.GRASSLAND)})",
                              payload="lay_eggs",
                              value=board.action_value(Habitat.GRASSLAND)))
        options.append(Option(ACTION_DRAW_CARDS, "draw_cards",
                              f"draw cards ({board.action_value(Habitat.WETLAND)})",
                              payload="draw_cards",
                              value=board.action_value(Habitat.WETLAND)))
        by_payload = {o.payload: o for o in options}
        prompt = (f"round {self.state.round_index + 1}, "
                  f"{self.state.turns_left[player] + 1} turns left - choose an action")
        payload = yield Decision(player=player, kind="main", prompt=prompt, options=options)
        return by_payload[payload]

    def _playable_habitats(
        self,
        board: PlayerBoard,
        card: BirdCard,
        habitat_filter: Optional[Habitat] = None,
    ) -> List[Habitat]:
        if not enumerate_payments(board.food, card.costs, limit=1):
            return []
        eggs = board.total_eggs()
        out = []
        for habitat in card.habitats:
            if habitat_filter is not None and habitat is not habitat_filter:
                continue
            if not board.has_space(habitat):
                continue
            if eggs < EGG_COST_BY_COLUMN[board.next_column(habitat)]:
                continue
            out.append(habitat)
        return out

    # ------------------------------------------------------------ play a bird
    def _play_bird_flow(
        self,
        player: int,
        card_id: Optional[int] = None,
        habitat_filter: Optional[Habitat] = None,
        optional: bool = False,
    ) -> Generator[Decision, Any, bool]:
        board = self.state.board(player)
        if card_id is None:
            options = [
                Option(0, "play_bird", f"play {self.cards[c].name}", payload=c, card_id=c)
                for c in board.hand[:MAX_HAND]
                if self._playable_habitats(board, self.cards[c], habitat_filter)
            ]
            if not options:
                return False
            if optional:
                options.append(self._pass_option("play no extra bird"))
            card_id = yield from self._ask(player, "play_which", "play which bird?", options)
            if card_id is None:
                return False
        card = self.cards[card_id]
        habitats = self._playable_habitats(board, card, habitat_filter)
        if not habitats:
            return False
        options = [
            Option(0, "habitat", HABITAT_NAMES[h], payload=h,
                   value=EGG_COST_BY_COLUMN[board.next_column(h)])
            for h in habitats
        ]
        habitat = yield from self._ask(
            player, "play_where", f"place {card.name} in which habitat?", options
        )
        egg_cost = EGG_COST_BY_COLUMN[board.next_column(habitat)]
        for _ in range(egg_cost):
            yield from self._spend_egg(player, f"pay 1 egg to play {card.name}")
        payments = enumerate_payments(board.food, card.costs, limit=MAX_OPTIONS)
        options = [
            Option(0, "payment", self._payment_label(p), payload=p) for p in payments
        ]
        payment = yield from self._ask(
            player, "play_pay", f"pay for {card.name}", options, default=None
        )
        if payment is None:
            return False
        board.pay_food(payment)
        board.hand.remove(card_id)
        placed = board.place(card, habitat)
        self._note(f"P{player} plays {card.name} in {HABITAT_NAMES[habitat]}")
        if card.power.timing is Timing.WHITE:
            yield from self._apply_power(player, placed, card.power)
        return True

    @staticmethod
    def _payment_label(payment: Sequence[int]) -> str:
        parts = [f"{n} {FOOD_NAMES[Food(i)]}" for i, n in enumerate(payment) if n]
        return " + ".join(parts) if parts else "free"

    def _spend_egg(self, player: int, prompt: str) -> Flow:
        board = self.state.board(player)
        options = [
            Option(0, "bird", f"{self.cards[p.card_id].name} ({p.eggs} eggs)", payload=p,
                   card_id=p.card_id, value=p.eggs)
            for p in board.all_birds()
            if p.eggs > 0
        ]
        placed = yield from self._ask(player, "spend_egg", prompt, options)
        if placed is not None:
            placed.eggs -= 1

    def _choose_food(
        self, player: int, prompt: str, allowed: Optional[Sequence[Food]] = None
    ) -> Generator[Decision, Any, Food]:
        foods = list(allowed) if allowed is not None else list(Food)
        options = [
            Option(0, "food", FOOD_NAMES[f], payload=f, value=int(f)) for f in foods
        ]
        food = yield from self._ask(player, "choose_food", prompt, options, default=foods[0])
        return food

    # -------------------------------------------------------- habitat actions
    def _gain_food_action(self, player: int) -> Flow:
        board = self.state.board(player)
        count = board.action_value(Habitat.FOREST)
        for _ in range(count):
            yield from self._take_die(player, f"gain food ({count} total)")
        for _ in range(board.exchanges_available(Habitat.FOREST)):
            if not board.hand or self.state.feeder.is_empty:
                break
            options = [
                Option(0, "hand_card", f"discard {self.cards[c].name}", payload=c, card_id=c)
                for c in board.hand[:MAX_OPTIONS - 1]
            ]
            options.append(self._pass_option("no exchange"))
            card_id = yield from self._ask(
                player, "exchange_card_food", "discard 1 card to gain 1 food?", options
            )
            if card_id is None:
                break
            board.hand.remove(card_id)
            self.state.discard.append(card_id)
            yield from self._take_die(player, "gain 1 food (exchange)")
        yield from self._activate_brown(player, Habitat.FOREST)

    def _take_die(
        self,
        player: int,
        prompt: str,
        food_filter: Optional[Food] = None,
    ) -> Generator[Decision, Any, bool]:
        feeder = self.state.feeder
        board = self.state.board(player)
        faces = (feeder.faces_yielding(food_filter) if food_filter is not None
                 else feeder.available_faces())
        if not faces:
            return False
        options = [
            Option(0, "die", "+".join(FOOD_NAMES[f] for f in DIE_FACES[face]), payload=face)
            for face in faces
        ]
        face = yield from self._ask(player, "take_die", prompt, options)
        if face is None:
            return False
        foods = feeder.take(face)
        if len(foods) > 1 and food_filter is None:
            # A combined face grants both foods.
            for food in foods:
                board.gain_food(food)
        elif food_filter is not None:
            board.gain_food(food_filter)
        else:
            board.gain_food(foods[0])
        return True

    def _lay_eggs_action(self, player: int) -> Flow:
        board = self.state.board(player)
        count = board.action_value(Habitat.GRASSLAND)
        for _ in range(count):
            laid = yield from self._lay_one_egg(player, f"lay an egg ({count} total)")
            if not laid:
                break
        for _ in range(board.exchanges_available(Habitat.GRASSLAND)):
            if board.total_food() == 0 or not board.birds_with_egg_space(self.cards):
                break
            options = [
                Option(0, "food", f"spend 1 {FOOD_NAMES[Food(i)]}", payload=Food(i))
                for i in range(N_FOOD)
                if board.food[i] > 0
            ]
            options.append(self._pass_option("no exchange"))
            food = yield from self._ask(
                player, "exchange_food_egg", "spend 1 food to lay 1 extra egg?", options
            )
            if food is None:
                break
            board.food[int(food)] -= 1
            yield from self._lay_one_egg(player, "lay 1 egg (exchange)")
        yield from self._activate_brown(player, Habitat.GRASSLAND)

    def _lay_one_egg(
        self,
        player: int,
        prompt: str,
        nest: Optional[NestType] = None,
        target: Optional[PlacedBird] = None,
    ) -> Generator[Decision, Any, bool]:
        board = self.state.board(player)
        if target is not None:
            if target.eggs >= self.cards[target.card_id].egg_capacity:
                return False
            target.eggs += 1
            return True
        candidates = [
            p for p in board.birds_with_egg_space(self.cards)
            if nest is None or self.cards[p.card_id].nest_matches(nest)
        ]
        if not candidates:
            return False
        options = [
            Option(0, "bird",
                   f"{self.cards[p.card_id].name} "
                   f"({p.eggs}/{self.cards[p.card_id].egg_capacity})",
                   payload=p, card_id=p.card_id, value=p.eggs)
            for p in candidates
        ]
        placed = yield from self._ask(player, "lay_egg", prompt, options)
        if placed is None:
            return False
        placed.eggs += 1
        return True

    def _draw_cards_action(self, player: int) -> Flow:
        board = self.state.board(player)
        count = board.action_value(Habitat.WETLAND)
        for _ in range(count):
            drawn = yield from self._draw_one_card(player, f"draw a card ({count} total)")
            if not drawn:
                break
        for _ in range(board.exchanges_available(Habitat.WETLAND)):
            if board.total_eggs() == 0:
                break
            options = [
                Option(0, "bird", f"discard 1 egg from {self.cards[p.card_id].name}",
                       payload=p, card_id=p.card_id, value=p.eggs)
                for p in board.all_birds() if p.eggs > 0
            ]
            options.append(self._pass_option("no exchange"))
            placed = yield from self._ask(
                player, "exchange_egg_card", "discard 1 egg to draw 1 card?", options
            )
            if placed is None:
                break
            placed.eggs -= 1
            yield from self._draw_one_card(player, "draw 1 card (exchange)")
        yield from self._activate_brown(player, Habitat.WETLAND)

    def _draw_one_card(
        self, player: int, prompt: str, tray_only: bool = False
    ) -> Generator[Decision, Any, bool]:
        state = self.state
        board = state.board(player)
        options = [
            Option(0, "tray_card", f"tray: {self.cards[c].name}", payload=("tray", i),
                   card_id=c)
            for i, c in enumerate(state.tray)
        ]
        if not tray_only and state.deck:
            options.append(Option(0, "deck_card", "deck (face down)", payload=("deck", 0)))
        if not options:
            return False
        source = yield from self._ask(player, "draw_card", prompt, options)
        if source is None:
            return False
        where, index = source
        card_id = state.draw_tray(index) if where == "tray" else state.draw_card()
        if card_id is None:
            return False
        board.hand.append(card_id)
        board.hand.sort()
        return True

    # ----------------------------------------------------------------- powers
    def _activate_brown(self, player: int, habitat: Habitat) -> Flow:
        board = self.state.board(player)
        for placed in list(reversed(board.habitats[habitat])):
            power = self.cards[placed.card_id].power
            if power.timing is Timing.BROWN:
                yield from self._apply_power(player, placed, power)

    def _trigger_pink(self, actor: int, action_kind: str) -> Flow:
        for player in self.state.opponents(actor):
            board = self.state.board(player)
            for placed in list(board.all_birds()):
                power = self.cards[placed.card_id].power
                if power.timing is not Timing.PINK or placed.pink_used:
                    continue
                if power.get("trigger") != action_kind:
                    continue
                placed.pink_used = True
                yield from self._apply_power(player, placed, power)

    def _apply_power(  # noqa: C901 - one dispatch table, kept flat on purpose
        self,
        player: int,
        placed: PlacedBird,
        power: Power,
        depth: int = 0,
    ) -> Flow:
        if depth > self.config.max_power_depth:
            return
        board = self.state.board(player)
        card = self.cards[placed.card_id]
        kind = power.kind
        count = int(power.get("count", 1))
        self._note(f"P{player} activates {card.name}: {power.text}")

        if kind in ("none", "end_points_per"):
            return

        if kind == "gain_food_supply":
            food = power.get("food")
            for _ in range(count):
                if food in (None, "any"):
                    chosen = yield from self._choose_food(player, "gain 1 food (power)")
                    board.gain_food(chosen)
                else:
                    board.gain_food(FOOD_BY_NAME[food])
            return

        if kind == "gain_food_feeder":
            food = power.get("food")
            wanted = None if food in (None, "any") else FOOD_BY_NAME[food]
            for _ in range(count):
                got = yield from self._take_die(player, "gain food from the feeder (power)",
                                                food_filter=wanted)
                if not got:
                    break
            return

        if kind == "lay_eggs":
            nest = NEST_BY_NAME[power.get("nest")] if power.get("nest") else None
            target = placed if power.get("target", "this") == "this" else None
            for _ in range(count):
                laid = yield from self._lay_one_egg(
                    player, "lay an egg (power)", nest=nest, target=target
                )
                if not laid:
                    break
            return

        if kind == "lay_eggs_each_bird":
            nest = NEST_BY_NAME[power.get("nest", "bowl")]
            for other in list(board.all_birds()):
                if self.cards[other.card_id].nest_matches(nest):
                    for _ in range(count):
                        yield from self._lay_one_egg(player, "", target=other)
            return

        if kind == "draw_cards":
            for _ in range(count):
                card_id = self.state.draw_card()
                if card_id is None:
                    break
                board.hand.append(card_id)
            board.hand.sort()
            return

        if kind == "draw_from_tray":
            for _ in range(count):
                got = yield from self._draw_one_card(player, "draw a face-up card (power)",
                                                     tray_only=True)
                if not got:
                    break
            return

        if kind == "tuck_from_hand":
            for _ in range(count):
                if not board.hand:
                    break
                options = [
                    Option(0, "hand_card", f"tuck {self.cards[c].name}", payload=c, card_id=c)
                    for c in board.hand[:MAX_OPTIONS - 1]
                ]
                options.append(self._pass_option("tuck nothing"))
                card_id = yield from self._ask(
                    player, "tuck", f"tuck a card behind {card.name}?", options
                )
                if card_id is None:
                    break
                board.hand.remove(card_id)
                placed.tucked += 1
                then = power.get("then")
                if then == "egg":
                    yield from self._lay_one_egg(player, "", target=placed)
                elif then == "food":
                    chosen = yield from self._choose_food(player, "gain 1 food (power)")
                    board.gain_food(chosen)
                elif then == "draw":
                    drawn = self.state.draw_card()
                    if drawn is not None:
                        board.hand.append(drawn)
                        board.hand.sort()
            return

        if kind == "cache_food":
            food = power.get("food")
            cached = list(placed.cached)
            for _ in range(count):
                if food in (None, "any"):
                    chosen = yield from self._choose_food(player, "cache 1 food (power)")
                else:
                    chosen = FOOD_BY_NAME[food]
                cached[int(chosen)] += 1
            placed.cached = tuple(cached)
            return

        if kind == "predator_hunt":
            card_id = self.state.draw_card()
            if card_id is None:
                return
            hunted = self.cards[card_id]
            if hunted.wingspan < int(power.get("threshold", 50)):
                if power.get("reward", "tuck") == "tuck":
                    # A tucked card leaves the game rather than going to the discard.
                    placed.tucked += 1
                else:
                    food = FOOD_BY_NAME[power.get("food", "rodent")]
                    cached = list(placed.cached)
                    cached[int(food)] += 1
                    placed.cached = tuple(cached)
                    self.state.discard.append(card_id)
                self._note(f"  hunt succeeds on {hunted.name} ({hunted.wingspan}cm)")
            else:
                self.state.discard.append(card_id)
                self._note(f"  hunt fails on {hunted.name} ({hunted.wingspan}cm)")
            return

        if kind == "all_players_gain_food":
            food = FOOD_BY_NAME[power.get("food", "seed")]
            for other in self.state.players:
                other.gain_food(food)
            return

        if kind == "all_players_draw":
            for other in self.state.players:
                card_id = self.state.draw_card()
                if card_id is None:
                    break
                other.hand.append(card_id)
                other.hand.sort()
            return

        if kind == "discard_egg_for_food":
            if board.total_eggs() == 0:
                return
            options = [
                Option(0, "bird", f"discard 1 egg from {self.cards[p.card_id].name}",
                       payload=p, card_id=p.card_id, value=p.eggs)
                for p in board.all_birds() if p.eggs > 0
            ]
            options.append(self._pass_option("keep your eggs"))
            target = yield from self._ask(
                player, "discard_egg", f"discard 1 egg for {count} food?", options
            )
            if target is None:
                return
            target.eggs -= 1
            food = power.get("food")
            for _ in range(count):
                if food in (None, "any"):
                    chosen = yield from self._choose_food(player, "gain 1 food (power)")
                    board.gain_food(chosen)
                else:
                    board.gain_food(FOOD_BY_NAME[food])
            return

        if kind == "repeat_brown":
            options = []
            for other in board.habitats[placed.habitat]:
                if other is placed:
                    continue
                other_power = self.cards[other.card_id].power
                if other_power.timing is Timing.BROWN:
                    options.append(
                        Option(0, "bird", f"repeat {self.cards[other.card_id].name}",
                               payload=other, card_id=other.card_id)
                    )
            if not options:
                return
            options.append(self._pass_option("repeat nothing"))
            target = yield from self._ask(
                player, "repeat_power", "repeat which bird's power?", options
            )
            if target is None:
                return
            yield from self._apply_power(
                player, target, self.cards[target.card_id].power, depth=depth + 1
            )
            return

        if kind == "play_extra_bird":
            habitat = HABITAT_BY_NAME[power.get("habitat")] if power.get("habitat") else None
            yield from self._play_bird_flow(
                player, habitat_filter=habitat, optional=True
            )
            return

        if kind == "gain_bonus_card":
            drawn = [b for b in (self.state.draw_bonus(), self.state.draw_bonus())
                     if b is not None]
            if not drawn:
                return
            options = [
                Option(0, "bonus_card", self.bonus_cards[b].name, payload=b) for b in drawn
            ]
            keep = yield from self._ask(player, "keep_bonus", "keep 1 bonus card", options)
            for bonus_id in drawn:
                if bonus_id == keep:
                    board.bonus_cards.append(bonus_id)
                else:
                    self.state.bonus_discard.append(bonus_id)
            return

        if kind == "on_opponent_action":
            effect = power.get("effect", "gain_food_feeder")
            if effect == "gain_food_feeder":
                yield from self._take_die(player, "gain 1 food (pink power)")
            elif effect == "lay_egg":
                yield from self._lay_one_egg(player, "", target=placed)
            elif effect == "draw_card":
                card_id = self.state.draw_card()
                if card_id is not None:
                    board.hand.append(card_id)
                    board.hand.sort()
            return

        raise ValueError(f"unhandled power kind {kind!r}")  # pragma: no cover
