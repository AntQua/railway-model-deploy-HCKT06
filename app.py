import json
import logging
import os
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model,
    CharField,
    IntegerField,
    TextField,
)
from playhouse.db_url import connect
from playhouse.shortcuts import model_to_dict


# =========================================================
# Logging
# =========================================================
class CustomRailwayLogFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)


def get_logger() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(CustomRailwayLogFormatter())
    logger.addHandler(handler)

    return logger


logger = get_logger()


# =========================================================
# App + DB
# =========================================================
app = Flask(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///predictions.db")
DB = connect(DATABASE_URL)


class BaseModel(Model):
    class Meta:
        database = DB


class PredictionRequest(BaseModel):
    port_code = IntegerField()
    traffic = CharField(max_length=50)
    request_json = TextField()
    prediction_json = TextField()


class UpdateRecord(BaseModel):
    date = CharField(max_length=20)
    port_code = IntegerField()
    traffic = CharField(max_length=50)
    true_value = IntegerField()
    request_json = TextField()


DB.connect(reuse_if_open=True)
DB.create_tables([PredictionRequest, UpdateRecord], safe=True)


# =========================================================
# Load artifacts
# =========================================================
ARTIFACTS_READY = True
columns: List[str] = []
pipeline = None

try:
    with open("columns.json", "r", encoding="utf-8") as fh:
        columns = json.load(fh)

    pipeline = joblib.load("pipeline.pickle")
    logger.info("Model artifacts loaded successfully.")
except FileNotFoundError as e:
    ARTIFACTS_READY = False
    logger.warning("Model artifacts not found yet: %s", str(e))
except Exception as e:
    ARTIFACTS_READY = False
    logger.exception("Failed to load model artifacts: %s", str(e))


# =========================================================
# Constants / validation
# =========================================================
VALID_TRAFFIC = {"people", "vehicles", "containers"}
FORECAST_MONTHS = ["Sep 2025", "Oct 2025", "Nov 2025", "Dec 2025", "Jan 2026", "Feb 2026"]


def json_error(message: str, status_code: int = 400):
    return jsonify({"error": message}), status_code


def get_request_json() -> Optional[Dict[str, Any]]:
    if not request.is_json:
        return None
    return request.get_json(silent=True)


def normalize_traffic(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    if value in VALID_TRAFFIC:
        return value
    return None


def validate_predict_payload(payload: Dict[str, Any]) -> Optional[str]:
    required_keys = {"port_code", "traffic"}

    missing = required_keys - payload.keys()
    if missing:
        return f"Missing required fields: {sorted(missing)}"

    if not isinstance(payload["port_code"], int):
        return "Field 'port_code' must be an integer."

    traffic = normalize_traffic(payload["traffic"])
    if traffic is None:
        return "Field 'traffic' must be one of: people, vehicles, containers."

    return None


def validate_update_payload(payload: Dict[str, Any]) -> Optional[str]:
    required_keys = {"date", "port_code", "traffic", "true_value"}

    missing = required_keys - payload.keys()
    if missing:
        return f"Missing required fields: {sorted(missing)}"

    if not isinstance(payload["date"], str) or not payload["date"].strip():
        return "Field 'date' must be a non-empty string."

    if not isinstance(payload["port_code"], int):
        return "Field 'port_code' must be an integer."

    traffic = normalize_traffic(payload["traffic"])
    if traffic is None:
        return "Field 'traffic' must be one of: people, vehicles, containers."

    if not isinstance(payload["true_value"], int):
        return "Field 'true_value' must be an integer."

    return None


# =========================================================
# Feature builder
# =========================================================
def build_forecast_features(port_code: int, traffic: str) -> pd.DataFrame:
    """
    Build the 6 rows the model needs for Sep 2025 to Feb 2026.

    IMPORTANT:
    This implementation assumes your trained pipeline expects at least:
      - port_code
      - traffic
      - date

    If your real columns.json expects different feature names, adjust this
    function so the returned DataFrame matches training exactly.
    """
    rows = []
    for forecast_date in FORECAST_MONTHS:
        rows.append(
            {
                "port_code": port_code,
                "traffic": traffic,
                "date": forecast_date,
            }
        )

    df = pd.DataFrame(rows)

    if columns:
        for col in columns:
            if col not in df.columns:
                df[col] = None
        df = df[columns]

    return df


def fallback_predictions(port_code: int, traffic: str) -> List[int]:
    """
    Safe fallback so the server does not crash if artifacts are unavailable.
    This keeps the API responsive during setup/testing.
    """
    logger.warning(
        "Using fallback predictions for port_code=%s, traffic=%s",
        port_code,
        traffic,
    )
    return [0, 0, 0, 0, 0, 0]


def make_predictions(port_code: int, traffic: str) -> List[int]:
    if not ARTIFACTS_READY or pipeline is None:
        return fallback_predictions(port_code, traffic)

    features = build_forecast_features(port_code, traffic)
    preds = pipeline.predict(features)

    cleaned = []
    for pred in preds:
        try:
            value = int(round(float(pred)))
            cleaned.append(max(value, 0))
        except Exception:
            cleaned.append(0)

    if len(cleaned) != 6:
        logger.warning("Model returned %s predictions instead of 6.", len(cleaned))
        if len(cleaned) < 6:
            cleaned.extend([0] * (6 - len(cleaned)))
        else:
            cleaned = cleaned[:6]

    return cleaned


# =========================================================
# Routes
# =========================================================
@app.route("/", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "artifacts_ready": ARTIFACTS_READY,
            "endpoints": ["/predict", "/update", "/list-db-contents"],
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    payload = get_request_json()

    if payload is None:
        logger.warning("Invalid JSON payload on /predict")
        return json_error("Request must contain valid JSON.", 400)

    logger.info("Predict request: %s", payload)

    validation_error = validate_predict_payload(payload)
    if validation_error:
        logger.warning("Predict validation error: %s", validation_error)
        return json_error(validation_error, 400)

    port_code = payload["port_code"]
    traffic = normalize_traffic(payload["traffic"])

    try:
        predictions = make_predictions(port_code, traffic)

        response = {
            "port_code": port_code,
            "traffic": traffic,
            "prediction": predictions,
        }

        PredictionRequest.create(
            port_code=port_code,
            traffic=traffic,
            request_json=json.dumps(payload),
            prediction_json=json.dumps(response),
        )

        logger.info("Predict response: %s", response)
        return jsonify(response), 200

    except Exception as e:
        logger.exception("Unhandled error in /predict: %s", str(e))
        return json_error("Internal server error.", 500)


@app.route("/update", methods=["POST"])
def update():
    payload = get_request_json()

    if payload is None:
        logger.warning("Invalid JSON payload on /update")
        return json_error("Request must contain valid JSON.", 400)

    logger.info("Update request: %s", payload)

    validation_error = validate_update_payload(payload)
    if validation_error:
        logger.warning("Update validation error: %s", validation_error)
        return json_error(validation_error, 400)

    date = payload["date"].strip()
    port_code = payload["port_code"]
    traffic = normalize_traffic(payload["traffic"])
    true_value = payload["true_value"]

    try:
        UpdateRecord.create(
            date=date,
            port_code=port_code,
            traffic=traffic,
            true_value=true_value,
            request_json=json.dumps(payload),
        )

        response = {
            "date": date,
            "port_code": port_code,
            "traffic": traffic,
            "true_value": true_value,
        }

        logger.info("Update saved: %s", response)
        return jsonify(response), 200

    except Exception as e:
        logger.exception("Unhandled error in /update: %s", str(e))
        return json_error("Internal server error.", 500)


@app.route("/list-db-contents", methods=["GET"])
def list_db_contents():
    try:
        return jsonify(
            {
                "predictions": [model_to_dict(row) for row in PredictionRequest.select()],
                "updates": [model_to_dict(row) for row in UpdateRecord.select()],
            }
        )
    except Exception as e:
        logger.exception("Unhandled error in /list-db-contents: %s", str(e))
        return json_error("Internal server error.", 500)


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)