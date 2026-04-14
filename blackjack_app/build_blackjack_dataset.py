from collections import Counter
from itertools import combinations_with_replacement
from pathlib import Path
import sys
from typing import Iterable, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blackjack_engine import format_for_gpt, recommend_action


RANKS: List[str] = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
NON_TEN_RANKS = {"A", "2", "3", "4", "5", "6", "7", "8", "9"}
TWO_DECK_LIMITS = {rank: 8 for rank in NON_TEN_RANKS}
TWO_DECK_LIMITS.update({"10": 8, "J": 8, "Q": 8, "K": 8})

MAX_PLAYER_CARDS = 5
OUTPUT_PATH = Path("data/train.txt")


def cards_within_two_decks(cards: Sequence[str]) -> bool:
    counts = Counter(cards)
    return all(counts[rank] <= TWO_DECK_LIMITS[rank] for rank in counts)


def canonical_hand(cards: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(cards, key=lambda rank: (RANKS.index(rank), rank)))


def valid_state(player_cards: Sequence[str], dealer_card: str) -> bool:
    all_cards = list(player_cards) + [dealer_card]
    return cards_within_two_decks(all_cards)


def serialize_state(player_cards: Sequence[str], dealer_card: str) -> str:
    result = recommend_action(
        player_cards=list(player_cards),
        dealer_card=dealer_card,
        can_double=True,
        can_split=True,
        dealer_hits_soft_17=False,
        deck_count=2,
    )
    return format_for_gpt(result)


def opening_states() -> Iterable[Tuple[Tuple[str, ...], str]]:
    for player_cards in combinations_with_replacement(RANKS, 2):
        hand = canonical_hand(player_cards)
        for dealer_card in RANKS:
            if valid_state(hand, dealer_card):
                yield hand, dealer_card


def later_hit_states() -> Iterable[Tuple[Tuple[str, ...], str]]:
    for card_count in range(3, MAX_PLAYER_CARDS + 1):
        for player_cards in combinations_with_replacement(RANKS, card_count):
            hand = canonical_hand(player_cards)
            for dealer_card in RANKS:
                if not valid_state(hand, dealer_card):
                    continue

                result = recommend_action(
                    player_cards=list(hand),
                    dealer_card=dealer_card,
                    can_double=True,
                    can_split=True,
                    dealer_hits_soft_17=False,
                    deck_count=2,
                )
                hand_info = result["hand_info"]

                if hand_info["is_bust"]:
                    continue

                yield hand, dealer_card


def build_dataset_lines() -> List[str]:
    serialized_lines = set()

    for player_cards, dealer_card in opening_states():
        serialized_lines.add(serialize_state(player_cards, dealer_card))

    for player_cards, dealer_card in later_hit_states():
        serialized_lines.add(serialize_state(player_cards, dealer_card))

    return sorted(serialized_lines)


def main() -> None:
    lines = build_dataset_lines()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} training examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
