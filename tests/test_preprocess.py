import pandas as pd

from spam_detector.preprocess import clean_text, normalize_label, prepare_dataframe


def test_normalize_label_maps_expected_values() -> None:
    assert normalize_label("spam") == "spam"
    assert normalize_label("1") == "spam"
    assert normalize_label("ham") == "not spam"
    assert normalize_label("0") == "not spam"


def test_clean_text_removes_urls_and_html() -> None:
    cleaned = clean_text("<b>Hello</b> visit https://example.com now!")
    assert "http" not in cleaned
    assert "<b>" not in cleaned
    assert cleaned == cleaned.lower()


def test_prepare_dataframe_accepts_category_message_columns() -> None:
    df = pd.DataFrame(
        {
            "Category": ["spam", "ham"],
            "Message": ["Win a prize now", "See you at 5"],
        }
    )

    prepared = prepare_dataframe(df)

    assert list(prepared.columns) == ["text", "label"]
    assert set(prepared["label"].unique()) == {"spam", "not spam"}
