# SMS Spam Detection - End-to-End MLOps

This repository contains a reproducible academic MLOps project for classifying SMS messages as `spam` or `not_spam`. It trains a scikit-learn Pipeline containing TF-IDF and Logistic Regression, serves it through FastAPI, and includes a plain HTML/CSS/JavaScript client, tests, CI, Docker Compose, Prometheus, Grafana, and an optional Kubernetes manifest.

## Architecture

```text
UCI archive -> download/validate -> stratified split -> GridSearchCV -> accepted artifact
                                                                      |
static frontend -> FastAPI /predict -> loaded scikit-learn Pipeline --+
                                      /metrics -> Prometheus -> Grafana
```

The test set is held out before hyperparameter search. Offline quality metrics are calculated only from labeled evaluation data. Production request metrics are operational metrics and do not measure production accuracy.

## Dataset

The default source is the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), downloaded from its documented archive URL. The archive contains `SMSSpamCollection`, a tab-separated file with `label` (`ham` or `spam`) and `message` columns. Consult the UCI record for current license and attribution terms; this project fetches the data during setup instead of committing it. `src/sms_spam_mlops/data/prepare_data.py` removes empty rows, normalizes labels, removes exact duplicate messages, and rejects conflicting duplicate labels.

## Repository map

```text
src/sms_spam_mlops/   configuration, data, model, and API package
frontend/              static user interface
tests/                 fixture-based unit and API tests
monitoring/            Prometheus and Grafana provisioning
deployment/kubernetes/ optional k3s/Kubernetes manifest
artifacts/             accepted model artifact, ignored by Git
reports/               evaluation JSON, ignored by Git
.github/workflows/     CI and manual/scheduled training workflow
```

## Local setup

Prerequisites: Python 3.11, `uv` (recommended), and Git. Docker is needed only for the Compose deployment.

From the repository root:

```bash
uv sync --extra dev
cp .env.example .env
uv run download-data
uv run train-model
uv run pytest
uv run ruff check .
```

`uv run download-data` fetches the dataset only when a valid local copy is absent. Use `--force` to refresh it. The training command writes `artifacts/sms_spam_pipeline.joblib`, `artifacts/model_metadata.json`, and `reports/latest_evaluation.json`. Candidate models with spam F1 below `SMS_MIN_ACCEPTED_F1` are rejected and do not replace the current model. The default threshold is a project policy, not a universal benchmark.

Run the API and the web page from the repository root:

```bash
uv run uvicorn sms_spam_mlops.api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` to use the web page. The same server provides the API at `/predict`, `/health`, `/ready`, and `/metrics`, plus OpenAPI at `/docs`. The `/api` route returns a small API status document. `/health` is the process liveness check; `/ready` returns 503 until a model is available. Without a trained model the page remains viewable, but inference responds that the model is unavailable.

The default `frontend/config.js` uses a same-origin API, which works with the API server above and in the Docker deployment. When hosting the static files separately (for example on Cloudflare Pages), edit `frontend/config.js` so `apiBaseUrl` is the full public API origin such as `https://api.example.com`; this is a public URL, not a secret. Do not open the HTML with a `file://` URL when testing API calls.

## Metrics and monitoring

From the repository root, after a model has been trained:

```bash
docker compose up --build -d
```

Compose serves the website and API on `http://127.0.0.1:8000`, Prometheus on `http://127.0.0.1:9090`, and Grafana on `http://127.0.0.1:3000`. All ports bind to loopback by default. On a VM, reach monitoring through an authenticated reverse proxy or SSH tunnel. Set `GRAFANA_ADMIN_PASSWORD` in `.env` before deployment; never use the example password publicly. Stop the stack with:

```bash
docker compose down
```

The API exposes bounded-label request count, status, latency, prediction class, model loaded status, model version, and latest accepted offline evaluation metrics. The evaluation gauges are read from `artifacts/model_metadata.json`; rejected runs cannot overwrite metrics for the current accepted model. They are not inferred from unlabeled user messages.

## CI and training workflow

`.github/workflows/ci.yml` runs on pushes and pull requests to `main`, installs pinned dependencies, runs Ruff and pytest, and builds the Docker image. It uses only temporary test fixtures. `.github/workflows/train-evaluate.yml` is manual by default and also has a monthly schedule; it downloads the public data, trains, applies the acceptance threshold, and uploads evaluation output. It does not deploy or push a candidate model. No cloud secrets are required by these workflows.

## Docker VM deployment

On an Ubuntu or Oracle Cloud VM, first create a DNS record pointing your domain to the VM and allow only SSH, HTTP, and HTTPS in the cloud firewall. Then, from the VM:

```bash
sudo apt update && sudo apt install -y git docker.io docker-compose-plugin
sudo usermod -aG docker "$USER"
git clone <YOUR_REPOSITORY_URL> sms-spam-mlops
cd sms-spam-mlops
cp .env.example .env
```

Install `uv` on the VM, then edit `.env`: set a strong `GRAFANA_ADMIN_PASSWORD`, set `SMS_CORS_ORIGINS` to the final frontend origin, and acquire/train the model from the repository root before building. Model artifacts are gitignored and are not included by cloning the repository; training on the VM creates the files that the image needs:

```bash
uv sync --extra dev
uv run download-data
uv run train-model
ls -lh artifacts/sms_spam_pipeline.joblib artifacts/model_metadata.json
docker compose build
docker compose up -d
docker compose ps
```

Compose binds the API, Prometheus, and Grafana to loopback. Put Nginx or a managed load balancer in front of the API for TLS, forward only the API host, and set the frontend API URL to the resulting HTTPS origin. HTTPS, DNS, registry authentication, backups, and firewall rules remain operator tasks; this repository does not claim them as configured.

## Cloudflare Pages frontend

1. Create a Pages project from the repository, using `frontend` as the build output directory and no build command.
2. Before deploying, change `frontend/config.js` to the public API origin, for example `https://api.example.com`.
3. Configure the API VM's `SMS_CORS_ORIGINS` to the exact Pages origin, such as `https://sms-spam.pages.dev`.
4. Deploy the static directory and verify the browser can reach `GET /health` and `POST /predict`.

The frontend contains no credentials. A custom Pages domain and HTTPS certificate are configured in Cloudflare, not by this repository.

## Kubernetes extension

`deployment/kubernetes/deployment.yaml` is optional. Replace `YOUR_REGISTRY/sms-spam-mlops:latest` with a real image, ensure the accepted model artifact is included in that image or mounted through a controlled artifact process, then apply it to a k3s cluster:

```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl get pods,service
```

Docker Compose is the supported baseline and Kubernetes is not required.

## Configuration

Settings use the `SMS_` environment prefix. Important values are `SMS_DATA_DIR`, `SMS_ARTIFACT_DIR`, `SMS_REPORT_DIR`, `SMS_MODEL_VERSION`, `SMS_RANDOM_SEED`, `SMS_MIN_ACCEPTED_F1`, `SMS_MAX_MESSAGE_LENGTH`, and `SMS_CORS_ORIGINS`. Never put credentials in `.env.example`, frontend files, images, logs, or metric labels. The API intentionally does not log complete submitted messages.

## Troubleshooting and limitations

- A degraded health response means the accepted model artifact is missing or invalid; run data acquisition and training, then restart the API.
- If Compose says the API is unhealthy, inspect it from the server with `docker compose logs --tail=100 api` and query `curl -i http://127.0.0.1:8000/ready`. The Docker health check tests that the API process responds; `/ready` reports whether its model loaded. Since model artifacts are excluded from Git, create them with the documented download and training commands on the server (then rebuild the API image) or provide them through a controlled artifact transfer.
- A download failure is surfaced as an error; no fabricated fallback data is created.
- Predictions are statistical estimates and may be wrong for new languages, adversarial text, or changing spam campaigns.
- The project has no online drift detection or automatic deployment. A future extension could add a review queue, signed model artifacts, richer data versioning, and scheduled drift reports.
- `make setup`, `make data`, `make train`, `make test`, `make lint`, `make api`, `make docker-up`, and `make docker-down` are shortcuts for the commands above.

Implemented: data acquisition and validation, reproducible training/evaluation, threshold-based artifact acceptance, FastAPI inference, static frontend, tests, CI, training workflow, Docker Compose, Prometheus/Grafana provisioning, and optional Kubernetes deployment. Optional/manual: cloud VM provisioning, DNS, HTTPS reverse proxy, Cloudflare project creation, registry publishing, and Kubernetes operations.
