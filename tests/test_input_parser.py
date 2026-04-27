import pytest
from src.core.input_parser import parse_channels_or_labels, validate_and_parse_drug_event


def test_parse_channels_or_labels_valid_input():
    """Test parsing a mix of numbers and labels."""
    result = parse_channels_or_labels("1, 2, 3, A, B, C")
    assert "error" not in result
    assert result["channels"] == [1, 2, 3]
    assert result["labels"] == ["A", "B", "C"]


def test_parse_channels_only():
    """Test parsing only numerical channels."""
    result = parse_channels_or_labels("5, 10, 15")
    assert "error" not in result
    assert result["channels"] == [5, 10, 15]
    assert result["labels"] == []


def test_parse_labels_only():
    """Test parsing only labels."""
    result = parse_channels_or_labels("X, Y, Z")
    assert "error" not in result
    assert result["channels"] == []
    assert result["labels"] == ["X", "Y", "Z"]


def test_parse_empty_input():
    result = parse_channels_or_labels("")
    assert result == {"error": "No input provided."}


def test_parse_whitespace_input():
    result = parse_channels_or_labels("   ")
    assert result == {"error": "No input provided."}


def test_parse_trailing_commas():
    """Ensure trailing commas do not create empty elements."""
    result = parse_channels_or_labels("1, 2, 3,")
    assert "error" not in result
    assert result["channels"] == [1, 2, 3]
    assert result["labels"] == []


def test_parse_extra_spaces():
    """Test handling of extra spaces around inputs."""
    result = parse_channels_or_labels("  4 ,   8 , X , Y  ")
    assert "error" not in result
    assert result["channels"] == [4, 8]
    assert result["labels"] == ["X", "Y"]


def test_parse_duplicate_values():
    """Test handling of duplicate numbers and labels."""
    result = parse_channels_or_labels("1, 1, 2, 2, A, A, B, B")
    assert "error" not in result
    assert result["channels"] == [1, 2]
    assert result["labels"] == ["A", "B"]


def test_parse_large_numbers():
    """Test handling of large numerical inputs."""
    result = parse_channels_or_labels("1000000, 500000, label")
    assert "error" not in result
    assert result["channels"] == [500000, 1000000]
    assert result["labels"] == ["label"]


def test_parse_unicode_input():
    """Test handling of Unicode input."""
    result = parse_channels_or_labels("α, β, γ, 1, 2, 3")
    assert "error" not in result
    assert result["channels"] == [1, 2, 3]
    assert result["labels"] == ["α", "β", "γ"]


def test_parse_mixed_case_labels():
    """Test case sensitivity of labels (order follows case-insensitive sort)."""
    result = parse_channels_or_labels("A, a, B, b, 1, 2")
    assert "error" not in result
    assert result["channels"] == [1, 2]
    assert set(result["labels"]) == {"A", "a", "B", "b"}
    assert len(result["labels"]) == 4


def test_parse_leading_commas():
    """Ensure leading commas do not cause issues."""
    result = parse_channels_or_labels(", 1, 2, 3, A, B")
    assert "error" not in result
    assert result["channels"] == [1, 2, 3]
    assert result["labels"] == ["A", "B"]


def test_parse_only_commas():
    """Test handling of input with only commas."""
    result = parse_channels_or_labels(",,,,,")
    assert result == {"error": "No valid input provided."}


def test_parse_whitespace_between_commas():
    """Test handling of extra spaces between commas."""
    result = parse_channels_or_labels(" 1 ,   2 , , , 3 , A ,    B ")
    assert "error" not in result
    assert result["channels"] == [1, 2, 3]
    assert result["labels"] == ["A", "B"]


def test_parse_float_like_strings():
    """Test handling of float-like numbers that should not be converted."""
    result = parse_channels_or_labels("1.0, 2.5, 3.1, label")
    assert "error" not in result
    assert result["channels"] == []
    assert result["labels"] == ["1.0", "2.5", "3.1", "label"]


# --- validate_and_parse_drug_event ---

def test_drug_event_missing_name():
    with pytest.raises(ValueError, match="name"):
        validate_and_parse_drug_event(name="", peri_drug="", start_text="100", end_text="")


def test_drug_event_invalid_start():
    with pytest.raises(ValueError, match="start"):
        validate_and_parse_drug_event(name="Drug", peri_drug="", start_text="abc", end_text="")


def test_drug_event_acute_no_end():
    result = validate_and_parse_drug_event(name="Drug", peri_drug="", start_text="100", end_text="")
    assert result["start"] == 100.0
    assert result["end"] is None
    assert result["type"] == "Acute"


def test_drug_event_end_max():
    result = validate_and_parse_drug_event(name="Drug", peri_drug="", start_text="100", end_text="max")
    assert result["end"] == float("inf")
    assert result["type"] == "Continuous"


def test_drug_event_continuous_with_end():
    result = validate_and_parse_drug_event(name="Drug", peri_drug="", start_text="100", end_text="200")
    assert result["end"] == 200.0
    assert result["type"] == "Continuous"


def test_drug_event_acute_same_start_end():
    result = validate_and_parse_drug_event(name="Drug", peri_drug="", start_text="100", end_text="100")
    assert result["type"] == "Acute"


def test_drug_event_end_before_start():
    with pytest.raises(ValueError, match="End time"):
        validate_and_parse_drug_event(name="Drug", peri_drug="", start_text="200", end_text="100")


def test_drug_event_invalid_end():
    with pytest.raises(ValueError, match="End time"):
        validate_and_parse_drug_event(name="Drug", peri_drug="", start_text="100", end_text="abc")


def test_drug_event_peri_single_value():
    result = validate_and_parse_drug_event(name="Drug", peri_drug="300", start_text="600", end_text="1200")
    assert result["start_offset"] == 300.0
    assert result["end_offset"] == 1500.0


def test_drug_event_peri_pre_post_pair():
    result = validate_and_parse_drug_event(name="Drug", peri_drug="300/600", start_text="600", end_text="1200")
    assert result["start_offset"] == 300.0
    assert result["end_offset"] == 1800.0


def test_drug_event_peri_clamps_to_zero():
    result = validate_and_parse_drug_event(name="Drug", peri_drug="9999", start_text="100", end_text="200")
    assert result["start_offset"] == 0.0


def test_drug_event_peri_no_end_uses_start():
    result = validate_and_parse_drug_event(name="Drug", peri_drug="60", start_text="300", end_text="")
    assert result["end_offset"] == 360.0


def test_drug_event_invalid_peri():
    with pytest.raises(ValueError, match="Peri-drug"):
        validate_and_parse_drug_event(name="Drug", peri_drug="abc", start_text="100", end_text="")
