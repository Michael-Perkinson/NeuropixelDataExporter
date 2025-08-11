def channels_or_labels_to_export() -> dict[str, list]:
    """
    Prompt the user to enter channels and/or labels to export.

    The user enters a comma-separated list of items. Numbers are interpreted as channel IDs (int),
    while non-numeric entries are treated as group labels (str). Both lists are sorted before returning.

    Returns
        Dictionary with:
        - "channels_to_include": Sorted list of channel IDs.
        - "labels_to_include": Sorted list of group labels.
    """
    user_input = input(
        "Enter the channels or labels to export (separated by commas): "
    ).strip()
    inputs = [item.strip() for item in user_input.split(",") if item.strip()]

    channels = sorted({int(item) for item in inputs if item.isdigit()})
    labels = sorted({item for item in inputs if not item.isdigit()})

    if channels:
        print(f"Selected channels: {channels}")
    else:
        print("No numeric channels selected.")

    if labels:
        print(f"Selected labels: {labels}")
    else:
        print("No group labels selected.")

    return {"channels_to_include": channels, "labels_to_include": labels}


def drug_application_time() -> float | None:
    """
    Prompt the user to enter the drug application time in seconds.

    Returns
        Drug application time as a float, or None if skipped.
    """
    try:
        drug_time = input("Enter drug application time (s), or press Enter to skip: ")
        return float(drug_time) if drug_time else None
    except ValueError:
        print("Invalid input. Defaulting to no drug application time.")
        return None


def start_and_end_time(max_time: float) -> tuple[float, float]:
    """
    Prompt the user for start and end times for analysis.

    Pressing "Enter" selects defaults (0 and max_time).

    Args
        max_time: Maximum available recording time.

    Returns
        Tuple (start_time, end_time).
    """

    def parse_input(prompt: str, default: float) -> float:
        value = input(prompt).strip()
        return float(value) if value else default

    start_time = parse_input("Enter start time (s), or press Enter to use 0: ", 0.0)
    end_time = parse_input(
        f"Enter end time (s), or press Enter to use {max_time:.2f}: ", max_time
    )

    return start_time, end_time


def prompt_for_baseline(
    max_time: float, min_time: float = 0
) -> tuple[float | None, float | None]:
    """
    Prompt the user to specify a baseline period for analysis.

    Args
        max_time: Maximum allowed baseline time.
        min_time: Minimum allowed baseline time (default: 0).

    Returns
        Tuple (baseline_start, baseline_end) if valid, otherwise (None, None).
    """
    if input("Specify a baseline period? (y/n): ").strip().lower() != "y":
        return None, None

    try:
        baseline_start = float(
            input(f"Enter baseline start time (≥ {min_time:.2f}s): ")
        )
        baseline_end = float(input(f"Enter baseline end time (≤ {max_time:.2f}s): "))

        if baseline_start < min_time:
            print(f"Baseline start must be ≥ {min_time:.2f}s. Using {min_time:.2f}s.")
            baseline_start = min_time
        if baseline_end > max_time:
            print(f"Baseline end must be ≤ {max_time:.2f}s. Using {max_time:.2f}s.")
            baseline_end = max_time

        return baseline_start, baseline_end
    except ValueError:
        print("Invalid baseline input. Defaulting to no baseline.")
        return None, None
