# blackjack_engine.py
# Basic two-deck blackjack decision engine for an MVP.
#
# What this file does:
# - evaluates blackjack hands
# - detects soft totals and pairs
# - recommends an action using standard two-deck basic-strategy-style rules
# - returns a structured explanation payload for your GPT layer
#
# Supported actions:
# - "Hit"
# - "Stand"
# - "Double"
# - "Split"
#
# Notes:
# - This is a rules-based MVP engine, not a full casino EV simulator.
# - It assumes dealer stands on soft 17 by default unless changed in rules.
# - Surrender and insurance are omitted for simplicity.
# - After splitting aces / resplitting rules are not fully modeled here.
#
# Example:
#   state = GameState(
#       player_cards=["8", "8"],
#       dealer_card="6",
#       can_double=True,
#       can_split=True,
#   )
#   result = best_action(state)
#   print(result["recommended_action"])
#   print(result["explanation"])

from dataclasses import dataclass, asdict
from typing import List, Dict, Any


# --------------------------------------------------
# Card utilities
# --------------------------------------------------
RANK_VALUES = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 10,
    "Q": 10,
    "K": 10,
    "A": 11,
}


def normalize_card(card: str) -> str:
    """Normalize card input into supported ranks."""
    card = str(card).strip().upper()
    if card in {"T"}:
        return "10"
    if card in {"J", "Q", "K", "A", "2", "3", "4", "5", "6", "7", "8", "9", "10"}:
        return card
    raise ValueError(f"Unsupported card rank: {card}")


def card_value(card: str) -> int:
    """Return blackjack value for a single card rank."""
    return RANK_VALUES[normalize_card(card)]


def dealer_upcard_value(card: str) -> int:
    """Dealer upcard numeric value used by strategy tables."""
    return card_value(card)


# --------------------------------------------------
# Data model
# --------------------------------------------------
@dataclass
class GameState:
    player_cards: List[str]
    dealer_card: str
    can_double: bool = True
    can_split: bool = True
    dealer_hits_soft_17: bool = False
    deck_count: int = 2

    def normalized(self) -> "GameState":
        return GameState(
            player_cards=[normalize_card(c) for c in self.player_cards],
            dealer_card=normalize_card(self.dealer_card),
            can_double=self.can_double,
            can_split=self.can_split,
            dealer_hits_soft_17=self.dealer_hits_soft_17,
            deck_count=self.deck_count,
        )


# --------------------------------------------------
# Hand evaluation
# --------------------------------------------------
def hand_value(cards: List[str]) -> int:
    """
    Returns the best blackjack hand total <= 21 if possible.
    Handles aces as 11 or 1.
    """
    cards = [normalize_card(c) for c in cards]
    total = sum(card_value(c) for c in cards)
    ace_count = sum(1 for c in cards if c == "A")

    while total > 21 and ace_count > 0:
        total -= 10
        ace_count -= 1

    return total


def is_soft_hand(cards: List[str]) -> bool:
    """
    True if the hand contains an ace counted as 11 in the final best total.
    """
    cards = [normalize_card(c) for c in cards]
    total = sum(card_value(c) for c in cards)
    ace_count = sum(1 for c in cards if c == "A")

    while total > 21 and ace_count > 0:
        total -= 10
        ace_count -= 1

       # Any ace not converted from 11 down to 1 means the hand remains soft.
    return ace_count > 0 and total <= 21


def soft_total(cards: List[str]) -> int:
    """
    Returns the 'soft total' label for a soft hand.
    Example: A,6 -> 17
    """
    return hand_value(cards)


def is_pair(cards: List[str]) -> bool:
    """True if exactly two cards of equal split value are present."""
    if len(cards) != 2:
        return False
    c1 = normalize_card(cards[0])
    c2 = normalize_card(cards[1])

    # Treat all 10-value cards as equivalent for splitting logic only if same value.
    # Standard splitting usually requires actual same rank in real casinos,
    # but for an MVP we use equal blackjack value.
    return card_value(c1) == card_value(c2)


def is_blackjack(cards: List[str]) -> bool:
    """Natural blackjack: exactly two cards totaling 21."""
    return len(cards) == 2 and hand_value(cards) == 21


def is_bust(cards: List[str]) -> bool:
    return hand_value(cards) > 21


# --------------------------------------------------
# Strategy tables
# --------------------------------------------------
def pair_action(pair_rank: str, dealer: int, can_double: bool = True) -> str:
    """
    Two-deck-inspired pair strategy for MVP.
    Assumes split is available.
    """
    pair_rank = normalize_card(pair_rank)

    # Convert all 10-value ranks into "10"
    if card_value(pair_rank) == 10 and pair_rank != "A":
        pair_rank = "10"

    if pair_rank == "A":
        return "Split"
    if pair_rank == "10":
        return "Stand"
    if pair_rank == "9":
        if dealer in [2, 3, 4, 5, 6, 8, 9]:
            return "Split"
        return "Stand"
    if pair_rank == "8":
        return "Split"
    if pair_rank == "7":
        if dealer in [2, 3, 4, 5, 6, 7]:
            return "Split"
        return "Hit"
    if pair_rank == "6":
        if dealer in [2, 3, 4, 5, 6]:
            return "Split"
        return "Hit"
    if pair_rank == "5":
        return hard_total_action(10, dealer, can_double=can_double)
    if pair_rank == "4":
        if dealer in [5, 6]:
            return "Split"
        return "Hit"
    if pair_rank in ["3", "2"]:
        if dealer in [2, 3, 4, 5, 6, 7]:
            return "Split"
        return "Hit"

    return "Hit"


def soft_hand_action(total: int, dealer: int, can_double: bool = True) -> str:
    """
    Soft total basic strategy for two-card soft hands.
    total should be 13..20 typically (A,2 through A,9).
    """
    if total in [20, 19]:
        # Some charts double soft 19 vs 6 in some games; keep MVP simple.
        return "Stand"

    if total == 18:
        if dealer in [3, 4, 5, 6] and can_double:
            return "Double"
        if dealer in [2, 7, 8]:
            return "Stand"
        return "Hit"

    if total == 17:
        if dealer in [3, 4, 5, 6] and can_double:
            return "Double"
        return "Hit"

    if total in [15, 16]:
        if dealer in [4, 5, 6] and can_double:
            return "Double"
        return "Hit"

    if total in [13, 14]:
        if dealer in [5, 6] and can_double:
            return "Double"
        return "Hit"

    return "Hit"


def hard_total_action(total: int, dealer: int, can_double: bool = True) -> str:
    """
    Hard total basic strategy.
    """
    if total >= 17:
        return "Stand"

    if total in [13, 14, 15, 16]:
        if dealer in [2, 3, 4, 5, 6]:
            return "Stand"
        return "Hit"

    if total == 12:
        if dealer in [4, 5, 6]:
            return "Stand"
        return "Hit"

    if total == 11:
        return "Double" if can_double else "Hit"

    if total == 10:
        if dealer in [2, 3, 4, 5, 6, 7, 8, 9]:
            return "Double" if can_double else "Hit"
        return "Hit"

    if total == 9:
        if dealer in [3, 4, 5, 6]:
            return "Double" if can_double else "Hit"
        return "Hit"

    return "Hit"


# --------------------------------------------------
# Explanation helpers
# --------------------------------------------------
def classify_hand(cards: List[str]) -> Dict[str, Any]:
    total = hand_value(cards)
    soft = is_soft_hand(cards)
    pair = is_pair(cards)
    blackjack = is_blackjack(cards)

    return {
        "cards": [normalize_card(c) for c in cards],
        "total": total,
        "is_soft": soft,
        "is_pair": pair,
        "is_blackjack": blackjack,
        "is_bust": is_bust(cards),
    }


def action_reason(state: GameState, action: str, hand_info: Dict[str, Any]) -> str:
    dealer = dealer_upcard_value(state.dealer_card)
    total = hand_info["total"]

    if hand_info["is_blackjack"]:
        return "This is a natural blackjack, which is already the strongest starting hand."

    if hand_info["is_pair"]:
        pair_rank = normalize_card(state.player_cards[0])
        if pair_rank == "A":
            return "Splitting aces gives you a chance to build two stronger hands instead of keeping a single soft 12."
        if card_value(pair_rank) == 8:
            return "Splitting 8s breaks up a weak hard 16, which is usually one of the worst totals to keep together."
        if action == "Split":
            return f"Pair strategy favors splitting here against a dealer {state.dealer_card} to improve long-run outcomes."
        if action == "Stand":
            return f"Keeping this pair together is stronger than splitting it against a dealer {state.dealer_card}."
        if action == "Hit":
            return f"This pair is usually played as a regular hand against a dealer {state.dealer_card}, and hitting gives the better long-run result."

    if hand_info["is_soft"]:
        if action == "Double":
            return f"A soft {total} against a dealer {state.dealer_card} is a strong doubling spot because you can improve without busting on one extra card."
        if action == "Stand":
            return f"A soft {total} is already strong enough to stand here against a dealer {state.dealer_card}."
        return f"A soft {total} against a dealer {state.dealer_card} often benefits from taking another card because the ace gives extra flexibility."

    if action == "Stand":
        return f"A hard {total} against a dealer {state.dealer_card} is usually strong enough, or the dealer is weak enough, that standing is preferred."
    if action == "Double":
        return f"A hard {total} against a dealer {state.dealer_card} is a favorable doubling opportunity because the starting total is strong."
    if action == "Hit":
        return f"A hard {total} against a dealer {state.dealer_card} is usually too weak to stand on, so hitting is preferred."

    return "This action is recommended by the blackjack strategy engine."


def simple_confidence(state: GameState, action: str, hand_info: Dict[str, Any]) -> float:
    """
    Lightweight confidence score for MVP UI.
    Not a true probability.
    """
    dealer = dealer_upcard_value(state.dealer_card)
    total = hand_info["total"]

    if hand_info["is_blackjack"]:
        return 0.99

    if action == "Split" and hand_info["is_pair"]:
        rank = normalize_card(state.player_cards[0])
        if rank == "A" or card_value(rank) == 8:
            return 0.96
        return 0.85

    if action == "Double":
        if total in [10, 11]:
            return 0.9
        if hand_info["is_soft"] and total in [17, 18]:
            return 0.84
        return 0.8

    if action == "Stand":
        if total >= 17:
            return 0.94
        if total in [12, 13, 14, 15, 16] and dealer in [4, 5, 6]:
            return 0.86
        return 0.78

    if action == "Hit":
        if total <= 11:
            return 0.95
        if total in [12, 13, 14, 15, 16] and dealer in [7, 8, 9, 10, 11]:
            return 0.88
        return 0.8

    return 0.75


# --------------------------------------------------
# Main engine
# --------------------------------------------------
def best_action(state: GameState) -> Dict[str, Any]:
    """
    Main recommendation function.
    Returns a structured dict that can be fed into your app or GPT layer.
    """
    state = state.normalized()
    hand_info = classify_hand(state.player_cards)
    dealer = dealer_upcard_value(state.dealer_card)

    # Immediate terminal cases
    if hand_info["is_blackjack"]:
        action = "Stand"
        explanation = action_reason(state, action, hand_info)
        return {
            "state": asdict(state),
            "hand_info": hand_info,
            "recommended_action": action,
            "confidence": simple_confidence(state, action, hand_info),
            "explanation": explanation,
        }

    if hand_info["is_bust"]:
        return {
            "state": asdict(state),
            "hand_info": hand_info,
            "recommended_action": "Bust",
            "confidence": 1.0,
            "explanation": "The hand total is over 21, so the hand is already bust.",
        }

    # Pair strategy first if exactly two cards and splitting allowed
    if len(state.player_cards) == 2 and hand_info["is_pair"] and state.can_split:
        pair_rank = normalize_card(state.player_cards[0])
        action = pair_action(pair_rank, dealer, can_double=state.can_double)
        explanation = action_reason(state, action, hand_info)
        return {
            "state": asdict(state),
            "hand_info": hand_info,
            "recommended_action": action,
            "confidence": simple_confidence(state, action, hand_info),
            "explanation": explanation,
        }

    # Soft hand strategy next for exactly two cards containing an ace
    if len(state.player_cards) == 2 and hand_info["is_soft"]:
        action = soft_hand_action(hand_info["total"], dealer, can_double=state.can_double)
        explanation = action_reason(state, action, hand_info)
        return {
            "state": asdict(state),
            "hand_info": hand_info,
            "recommended_action": action,
            "confidence": simple_confidence(state, action, hand_info),
            "explanation": explanation,
        }

    # Otherwise use hard total strategy
    action = hard_total_action(hand_info["total"], dealer, can_double=state.can_double)
    explanation = action_reason(state, action, hand_info)
    return {
        "state": asdict(state),
        "hand_info": hand_info,
        "recommended_action": action,
        "confidence": simple_confidence(state, action, hand_info),
        "explanation": explanation,
    }


# --------------------------------------------------
# Convenience API for UI / app
# --------------------------------------------------
def recommend_action(
    player_cards: List[str],
    dealer_card: str,
    can_double: bool = True,
    can_split: bool = True,
    dealer_hits_soft_17: bool = False,
    deck_count: int = 2,
) -> Dict[str, Any]:
    """
    Simple wrapper for app code.
    """
    state = GameState(
        player_cards=player_cards,
        dealer_card=dealer_card,
        can_double=can_double,
        can_split=can_split,
        dealer_hits_soft_17=dealer_hits_soft_17,
        deck_count=deck_count,
    )
    return best_action(state)


def format_for_gpt(result: Dict[str, Any]) -> str:
    """
    Converts engine output into a compact text prompt for your nanoGPT model.
    """
    state = result["state"]
    hand = result["hand_info"]
    return (
        f"Player: {','.join(state['player_cards'])} | "
        f"Dealer: {state['dealer_card']} | "
        f"Total: {hand['total']} | "
        f"Soft: {hand['is_soft']} | "
        f"Pair: {hand['is_pair']} | "
        f"Action: {result['recommended_action']} | "
        f"Reason: {result['explanation']}"
    )


# --------------------------------------------------
# Quick test
# --------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        (["8", "8"], "6"),
        (["10", "6"], "7"),
        (["A", "7"], "9"),
        (["5", "5"], "6"),
        (["9", "9"], "7"),
        (["10", "2"], "4"),
        (["A", "6"], "3"),
    ]

    for player_cards, dealer_card in test_cases:
        result = recommend_action(player_cards, dealer_card)
        print("=" * 70)
        print(f"Player: {player_cards} | Dealer: {dealer_card}")
        print(f"Recommended Action: {result['recommended_action']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Explanation: {result['explanation']}")
        print("GPT Prompt Format:")
        print(format_for_gpt(result))
