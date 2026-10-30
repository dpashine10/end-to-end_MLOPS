from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sms_spam_mlops.config import Settings


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    model_path = tmp_path / "model.joblib"
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=500)),
    ])
    pipeline.fit(["free prize claim now", "meeting at noon", "win cash today", "see you later"], ["spam", "not_spam", "spam", "not_spam"])
    joblib.dump(pipeline, model_path)
    return Settings(model_path=model_path, report_dir=tmp_path, cors_origins="http://test")