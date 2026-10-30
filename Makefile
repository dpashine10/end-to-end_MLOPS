.PHONY: setup data train test api docker-up docker-down lint
setup:
	uv sync --extra dev
data:
	uv run download-data
train:
	uv run train-model
test:
	uv run pytest
lint:
	uv run ruff check .
api:
	uv run uvicorn sms_spam_mlops.api.app:app --app-dir src --host 0.0.0.0 --port 8000
docker-up:
	docker compose up --build -d
docker-down:
	docker compose down