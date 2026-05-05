"""
test_preprocess.py — Tests for text cleaning utility.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spam_detector.preprocess import clean_text


def test_clean_text_strips_html():
    result = clean_text("<b>Hello</b> <i>World</i>")
    assert "<" not in result and ">" not in result


def test_clean_text_strips_urls():
    result = clean_text("Visit https://example.com or www.test.org for details")
    assert "http" not in result
    assert "example" not in result


def test_clean_text_strips_emails():
    result = clean_text("Contact us at support@company.com for help")
    assert "@" not in result


def test_clean_text_lowercases():
    result = clean_text("HELLO WORLD")
    assert result == "hello world"


def test_clean_text_handles_none():
    result = clean_text(None)
    assert isinstance(result, str)
    assert result == ""


def test_clean_text_collapses_whitespace():
    result = clean_text("hello    world   test")
    assert "  " not in result
