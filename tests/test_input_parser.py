from src.core.input_parser import parse_channels_or_labels


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
