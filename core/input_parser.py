def parse_channels_or_labels(user_input: str) -> dict[str, list[int] | list[str] | str]:
    """
    Parses user input into numerical channels and text labels.

    Args
        user_input: Comma-separated list of numbers and/or labels.

    Returns
        Dictionary with `channels` (list of ints) and `labels` (list of strings).
        If input is invalid, returns a dictionary with an `error` key.
    """
    if not user_input.strip():
        return {"error": "No input provided."}

    inputs = [item.strip() for item in user_input.split(",") if item.strip()]
    if not inputs:
        return {"error": "No valid input provided."}

    channels = sorted(
        {int(item) for item in inputs if item.isdigit() and "." not in item}
    )
    labels = sorted(
        {item for item in inputs if not item.isdigit() or "." in item}
    )

    # Wrap channels in a list if empty or contains only one value
    if len(channels) == 1:
        channels = [channels[0]]  # Ensure it’s a list even if there’s one item

    return {"channels": channels, "labels": labels}


def validate_and_parse_drug_event(name: str, peri_drug: str, start_text: str, end_text: str) -> dict:
    if not name:
        raise ValueError("Drug name is required.")
    
    if not start_text.replace(".", "", 1).isdigit():
        raise ValueError("Invalid or missing start time.")
    start = float(start_text)

    if end_text:
        if not end_text.replace(".", "", 1).isdigit():
            raise ValueError("End time must be a number.")
        end = float(end_text)
        if end < start:
            raise ValueError(
                "End time must be greater than or equal to start time.")
        acute_or_continuous = "Acute" if end == start else "Continuous"
    else:
        end = None
        acute_or_continuous = "Acute"

    result = {
        "name": name,
        "type": acute_or_continuous,
        "start": start,
        "end": end
    }

    if peri_drug:
        try:
            peri_val = float(peri_drug)
            result["start-offset"] = max(0, start - peri_val)
            result["end-offset"] = (end +
                                   peri_val) if end is not None else (start + peri_val)
        except ValueError:
            raise ValueError("Peri-drug time must be a number.")

    return result
