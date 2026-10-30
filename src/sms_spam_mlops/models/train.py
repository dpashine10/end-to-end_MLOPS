"""Train, evaluate, and atomically accept the SMS spam pipeline."""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from sms_spam_mlops.config import settings
from sms_spam_mlops.data.prepare_data import load_and_validate
from sms_spam_mlops.models.evaluate import candidate_is_accepted, evaluate_predictions


def train_model(dataset_path: Path, artifact_dir: Path, report_dir: Path) -> dict[str, object]:
    """Train a candidate, evaluate it once on held-out data, and accept on threshold."""

    started = time.perf_counter()
    frame, data_stats = load_and_validate(dataset_path)
    texts_train, texts_test, labels_train, labels_test = train_test_split(
        frame["message"], frame["label"], test_size=0.2, random_state=settings.random_seed, stratify=frame["label"]
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1000, random_state=settings.random_seed)),
    ])
    search = GridSearchCV(
        pipeline,
        {"tfidf__ngram_range": [(1, 1), (1, 2)], "tfidf__min_df": [1, 2], "classifier__C": [0.5, 1.0, 2.0]},
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
    )
    search.fit(texts_train, labels_train)
    predictions = search.predict(texts_test)
    metrics = evaluate_predictions(labels_test.tolist(), predictions.tolist())
    metadata: dict[str, object] = {
        "model_version": settings.model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": settings.random_seed,
        "dataset": data_stats | {"train_records": len(texts_train), "test_records": len(texts_test)},
        "best_params": search.best_params_,
        "metrics": metrics,
        "minimum_accepted_f1": settings.min_accepted_f1,
        "accepted": candidate_is_accepted(metrics, settings.min_accepted_f1),
        "training_seconds": round(time.perf_counter() - started, 3),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "latest_evaluation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not metadata["accepted"]:
        raise RuntimeError(f"Candidate rejected: spam F1 {metrics['f1_spam']:.3f} is below threshold")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    temporary_model = artifact_dir / f"{settings.model_filename}.tmp"
    joblib.dump(search.best_estimator_, temporary_model)
    temporary_model.replace(settings.resolved_model_path())
    (artifact_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=settings.data_dir / settings.dataset_filename)
    parser.add_argument("--artifact-dir", type=Path, default=settings.artifact_dir)
    parser.add_argument("--report-dir", type=Path, default=settings.report_dir)
    args = parser.parse_args()
    print(json.dumps(train_model(args.dataset, args.artifact_dir, args.report_dir), indent=2))


if __name__ == "__main__":
    main()