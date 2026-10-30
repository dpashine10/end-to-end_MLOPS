from fastapi.testclient import TestClient

from sms_spam_mlops.api.app import create_app


def test_health_predict_and_metrics(test_settings) -> None:
    client = TestClient(create_app(test_settings))
    page = client.get("/")
    assert page.status_code == 200
    assert "SMS Spam Detector" in page.text
    assert client.get("/script.js").status_code == 200
    assert client.get("/health").json()["model_loaded"] is True
    response = client.post("/predict", json={"text": "free prize claim now"})
    assert response.status_code == 200
    assert response.json()["predicted_class"] in {"spam", "not_spam"}
    assert "sms_http_requests_total" in client.get("/metrics").text


def test_invalid_and_oversized_messages(test_settings) -> None:
    client = TestClient(create_app(test_settings))
    assert client.post("/predict", json={"text": " "}).status_code == 422
    test_settings.max_message_length = 3
    assert client.post("/predict", json={"text": "long message"}).status_code == 413


def test_missing_model_returns_degraded_health(tmp_path) -> None:
    from sms_spam_mlops.config import Settings

    client = TestClient(create_app(Settings(model_path=tmp_path / "none.joblib")))
    assert client.get("/health").json()["status"] == "degraded"
    assert client.post("/predict", json={"text": "hello"}).status_code == 503
