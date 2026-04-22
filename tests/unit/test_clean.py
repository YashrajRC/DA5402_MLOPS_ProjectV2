"""Unit tests for text cleaning."""
from src.data.clean import clean_text


def test_clean_text_lowercases():
    assert clean_text("HELLO World") == "hello world"


def test_clean_text_strips_urls():
    assert "http" not in clean_text("check http://example.com/page")


def test_clean_text_strips_mentions():
    assert "@user" not in clean_text("hi @user how are you")


def test_clean_text_collapses_whitespace():
    assert clean_text("hello    world\n\nhi") == "hello world hi"


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_clean_text_preserves_punctuation():
    assert "!" in clean_text("I feel great!")
    assert "?" in clean_text("Are you okay?")
