from sms_spam_mlops.models.evaluate import candidate_is_accepted
from sms_spam_mlops.models.predict import ModelService


def test_acceptance_policy() -> None:
    assert candidate_is_accepted({"f1_spam": 0.81}, 0.80)
    assert not candidate_is_accepted({"f1_spam": 0.79}, 0.80)


def test_serialized_model_predicts(test_settings) -> None:
    service = ModelService(test_settings.model_path, "test")
    predicted, probability = service.predict("free prize claim now")
    assert predicted in {"spam", "not_spam"}
    assert 0 <= probability <= 1


def test_missing_model_is_reported(tmp_path) -> None:
    service = ModelService(tmp_path / "missing.joblib", "test")
    assert not service.loaded