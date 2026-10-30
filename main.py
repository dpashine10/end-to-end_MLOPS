"""Development entry point for the SMS Spam MLOps project."""

from sms_spam_mlops.api.app import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
