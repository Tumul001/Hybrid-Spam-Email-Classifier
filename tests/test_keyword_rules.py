"""
test_keyword_rules.py — Tests for the keyword rules engine.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from spam_detector.keyword_rules import match_rules, combined_keyword_boost, SPAM_RULES, RULE_COUNT


def test_rule_count_above_minimum():
    """We must have at least 50 rules — guards against accidental deletions."""
    assert RULE_COUNT >= 50, f"Expected 50+ rules, got {RULE_COUNT}"


def test_unsubscribe_is_marketing():
    matched = match_rules("Click here to unsubscribe from our mailing list.")
    names = [r.name for r in matched]
    assert "unsubscribe" in names


def test_phishing_account_suspended():
    matched = match_rules("Your account has been suspended. Verify your identity now.")
    names = [r.name for r in matched]
    assert "account_suspended" in names
    assert "verify_account" in names


def test_scam_lottery():
    matched = match_rules("Congratulations! You have won the lottery sweepstakes prize!")
    names = [r.name for r in matched]
    assert "you_have_won" in names
    assert "lottery_winner" in names


def test_ham_no_signals():
    matched = match_rules("Hi Sarah, can we reschedule the meeting to 3pm?")
    assert len(matched) == 0, f"Expected 0 signals for normal email, got {[r.name for r in matched]}"


def test_combined_boost_empty():
    assert combined_keyword_boost([]) == 0.0


def test_combined_boost_increases_with_more_rules():
    one   = [r for r in SPAM_RULES if r.name == "unsubscribe"]
    two   = [r for r in SPAM_RULES if r.name in ("unsubscribe", "newsletter")]
    assert combined_keyword_boost(two) > combined_keyword_boost(one)


def test_combined_boost_capped_below_one():
    boost = combined_keyword_boost(SPAM_RULES)  # all rules
    assert boost <= 1.0
    assert boost >= 0.99  # with 100+ rules it should be near maximum


def test_categories_present():
    from spam_detector.keyword_rules import CATEGORIES
    expected = {"marketing", "phishing", "scam", "urgency", "financial"}
    assert expected.issubset(set(CATEGORIES))


def test_new_categories_present():
    from spam_detector.keyword_rules import CATEGORIES
    expected = {"health", "tech", "gambling", "crypto", "adult"}
    assert expected.issubset(set(CATEGORIES))
