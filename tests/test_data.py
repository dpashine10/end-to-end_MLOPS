from pathlib import Path

import pytest

from sms_spam_mlops.data.prepare_data import load_and_validate, normalize_label


def test_normalize_label() -> None:
    assert normalize_label(" HAM ") == "not_spam"
    assert normalize_label("spam") == "spam"
    with pytest.raises(ValueError):
        normalize_label("unknown")


def test_load_removes_duplicates_and_reports_distribution(tmp_path: Path) -> None:
    path = tmp_path / "SMSSpamCollection"
    path.write_text("ham\thello\nspam\tfree prize\nham\thello\n", encoding="utf-8")
    frame, stats = load_and_validate(path)
    assert len(frame) == 2
    assert stats["duplicate_records_removed"] == 2


def test_conflicting_duplicate_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "SMSSpamCollection"
    path.write_text("ham\tduplicate\nspam\tduplicate\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting"):
        load_and_validate(path)