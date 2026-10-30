"""FastAPI application for SMS spam inference."""

import time
import json
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator

from sms_spam_mlops import __version__
from sms_spam_mlops.config import Settings, settings
from sms_spam_mlops.models.predict import ModelService


class PredictionRequest(BaseModel):
    """Validated user message."""

    text: str = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


def create_app(runtime_settings: Settings = settings) -> FastAPI:
    """Create an isolated app and metric registry for production or tests."""

    registry = CollectorRegistry(auto_describe=True)
    requests = Counter("sms_http_requests_total", "HTTP requests", ["endpoint", "status"], registry=registry)
    prediction_counts = Counter("sms_predictions_total", "Predictions by class", ["predicted_class"], registry=registry)
    prediction_failures = Counter("sms_prediction_failures_total", "Prediction failures", registry=registry)
    latency = Histogram("sms_http_request_duration_seconds", "HTTP request latency", ["endpoint"], registry=registry)
    model_status = Gauge("sms_model_loaded", "Whether the model is loaded", registry=registry)
    model_info = Gauge("sms_model_info", "Model metadata", ["version"], registry=registry)
    evaluation_metrics = {
        name: Gauge(f"sms_model_eval_{name}", f"Latest offline evaluation {name}", registry=registry)
        for name in ("accuracy", "precision_spam", "recall_spam", "f1_spam", "macro_f1")
    }
    evaluation_path = runtime_settings.report_dir / "latest_evaluation.json"
    if evaluation_path.is_file():
        try:
            latest_metrics = json.loads(evaluation_path.read_text(encoding="utf-8")).get("metrics", {})
            for name, gauge in evaluation_metrics.items():
                if name in latest_metrics:
                    gauge.set(float(latest_metrics[name]))
        except (OSError, ValueError, TypeError):
            pass
    service = ModelService(runtime_settings.resolved_model_path(), runtime_settings.model_version)
    model_status.set(1 if service.loaded else 0)
    model_info.labels(version=runtime_settings.model_version).set(1)

    app = FastAPI(title="SMS Spam Detector API", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.cors_origin_list(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.state.model_service = service

    @app.middleware("http")
    async def record_http_metrics(request: Request, call_next: Any) -> Any:
        started = time.perf_counter()
        response = await call_next(request)
        endpoint = request.url.path if request.url.path in {"/", "/health", "/predict", "/metrics"} else "other"
        requests.labels(endpoint=endpoint, status=str(response.status_code)).inc()
        latency.labels(endpoint=endpoint).observe(time.perf_counter() - started)
        return response

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "sms-spam-detector", "status": "ok", "docs": "/docs"}

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok" if service.loaded else "degraded", "model_loaded": service.loaded, "model_version": service.model_version}

    @app.post("/predict")
    def predict(payload: PredictionRequest) -> dict[str, object]:
        if len(payload.text) > runtime_settings.max_message_length:
            raise HTTPException(status_code=413, detail=f"text exceeds {runtime_settings.max_message_length} characters")
        try:
            predicted_class, probability = service.predict(payload.text)
        except RuntimeError as exc:
            prediction_failures.inc()
            raise HTTPException(status_code=503, detail="The model is unavailable") from exc
        except Exception as exc:  # noqa: BLE001 - prevent inference details leaking to clients
            prediction_failures.inc()
            raise HTTPException(status_code=500, detail="Prediction failed") from exc
        prediction_counts.labels(predicted_class=predicted_class).inc()
        return {
            "predicted_class": predicted_class,
            "label": "Spam" if predicted_class == "spam" else "Not Spam",
            "spam_probability": probability,
            "model_version": service.model_version,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()