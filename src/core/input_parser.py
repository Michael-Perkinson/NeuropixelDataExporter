from __future__ import annotations

from typing import TypedDict, NotRequired


class ParseSuccess(TypedDict):
    channels: list[int]
    labels: list[str]


class ParseError(TypedDict):
    error: str


ParseResult = ParseSuccess | ParseError


def parse_channels_or_labels(user_input: str) -> ParseResult:
    """
    Parse comma-separated user input into cluster/channel IDs (ints) and label strings.

    Examples:
      "1,2,3" -> channels=[1,2,3], labels=[]
      "good,3,unknown" -> channels=[3], labels=["good","unknown"]
    """
    raw = user_input.strip()
    if not raw:
        return {"error": "No input provided."}

    inputs = [item.strip() for item in raw.split(",") if item.strip()]
    if not inputs:
        return {"error": "No valid input provided."}

    channels_set: set[int] = set()
    labels_set: set[str] = set()

    for item in inputs:
        # treat pure integers as channels; everything else as a label
        if item.isdigit():
            channels_set.add(int(item))
        else:
            labels_set.add(item)

    return {
        "channels": sorted(channels_set),
        "labels": sorted(labels_set, key=str.lower),
    }


class DrugEvent(TypedDict):
    name: str
    type: str  # "Acute" | "Continuous"
    start: float
    end: float | None
    start_offset: NotRequired[float]
    end_offset: NotRequired[float]


def validate_and_parse_drug_event(
    name: str,
    peri_drug: str,
    start_text: str,
    end_text: str,
) -> DrugEvent:
    if not name:
        raise ValueError("Drug name is required.")

    try:
        start = float(start_text)
    except Exception:
        raise ValueError("Invalid or missing start time.")

    end: float | None
    if end_text.strip():
        if end_text.strip().lower() == "max":
            end = float("inf")
            acute_or_continuous = "Continuous"
        else:
            try:
                end = float(end_text)
            except Exception:
                raise ValueError("End time must be a number or 'max'.")
            if end < start:
                raise ValueError(
                    "End time must be greater than or equal to start time.")
            acute_or_continuous = "Acute" if end == start else "Continuous"
    else:
        end = None
        acute_or_continuous = "Acute"

    result: DrugEvent = {
        "name": name,
        "type": acute_or_continuous,
        "start": start,
        "end": end,
    }

    if peri_drug.strip():
        try:
            if "/" in peri_drug:
                pre_str, post_str = peri_drug.split("/", 1)
                pre_val = float(pre_str)
                post_val = float(post_str)
            else:
                pre_val = post_val = float(peri_drug)
        except Exception:
            raise ValueError(
                "Peri-drug time must be a number or pre/post pair (e.g. 600 or 300/900).")

        result["start_offset"] = max(0.0, start - pre_val)
        result["end_offset"] = (
            end + post_val) if end is not None else (start + post_val)

    return result
