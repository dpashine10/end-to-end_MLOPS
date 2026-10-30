"""Load the accepted model artifact and provide inference helpers."""

from pathlib import Path
from typing import Any

import joblib


class ModelService:
    """Single loaded model used by the API process."""

    def __init__(self, model_path: Path, model_version: str) -> None:
        self.model_path = model_path
        self.model_version = model_version
        self.pipeline: Any | None = None
        self.load_error: str | None = None
        try:
            if not model_path.is_file():
                raise FileNotFoundError(f"Model artifact not found: {model_path}")
            self.pipeline = joblib.load(model_path)
        except Exception as exc:  # noqa: BLE001 - expose a stable health state to the API
            self.load_error = str(exc)

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    def predict(self, text: str) -> tuple[str, float]:
        if self.pipeline is None:
            raise RuntimeError(self.load_error or "Model is unavailable")
        label = str(self.pipeline.predict([text])[0])
        probabilities = self.pipeline.predict_proba([text])[0]
        classes = list(self.pipeline.classes_)
        spam_probability = float(probabilities[classes.index("spam")])
        return label, spam_probability