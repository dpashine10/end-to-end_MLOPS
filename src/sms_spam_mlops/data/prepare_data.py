"""Load, normalize, and validate the UCI SMS Spam Collection."""

from pathlib import Path

import pandas as pd

LABEL_MAP = {"ham": "not_spam", "spam": "spam"}


def normalize_label(value: object) -> str:
    """Normalize a raw UCI label and reject unknown values."""

    label = str(value).strip().lower()
    if label not in LABEL_MAP:
        raise ValueError(f"Unsupported label {value!r}; expected ham or spam")
    return LABEL_MAP[label]


def load_and_validate(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read the tab-separated UCI file and return clean records and quality stats."""

    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Dataset file is missing or empty: {path}")
    frame = pd.read_csv(path, sep="\t", header=None, names=["label", "message"], dtype=str)
    if frame.empty or frame[["label", "message"]].isna().any().any():
        frame = frame.dropna(subset=["label", "message"])
    frame["message"] = frame["message"].astype(str).str.strip()
    frame = frame[frame["message"] != ""].copy()
    frame["label"] = frame["label"].map(normalize_label)

    duplicate_count = int(frame.duplicated(subset=["message"], keep=False).sum())
    conflicts = frame.groupby("message")["label"].nunique()
    conflict_count = int((conflicts > 1).sum())
    if conflict_count:
        raise ValueError(f"Found {conflict_count} messages with conflicting labels")
    frame = frame.drop_duplicates(subset=["message"]).reset_index(drop=True)
    stats: dict[str, object] = {
        "records": len(frame),
        "duplicate_records_removed": duplicate_count,
        "label_distribution": frame["label"].value_counts().to_dict(),
    }
    if set(frame["label"]) != {"spam", "not_spam"}:
        raise ValueError("Dataset must contain both spam and ham/not_spam classes")
    return frame, stats