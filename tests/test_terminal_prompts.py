from core.terminal_prompts import (
    channels_or_labels_to_export,
)


def test_channels_or_labels_to_export(monkeypatch):
    # Monkeypatch input so that channels_or_labels_to_export returns "1, A"
    monkeypatch.setattr("builtins.input", lambda prompt="": "1, A")
    user_filters = channels_or_labels_to_export()
    # Expect channels: [1] and labels: ["A"]
    assert user_filters["channels_to_include"] == [1]
    assert user_filters["labels_to_include"] == ["A"]
