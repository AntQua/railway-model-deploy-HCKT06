import json
import logging
import os
from typing import Any, Dict, List, Optional

import joblib
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
predictions_store = None

try:
    with open("columns.json", "r", encoding="utf-8") as fh:
        columns = json.load(fh)

    predictions_store = joblib.load("predictions_store.pickle")
    logger.info("Prediction store loaded successfully.")

except FileNotFoundError as e:
    ARTIFACTS_READY = False
    logger.warning("Artifacts not found yet: %s", str(e))

except Exception as e:
    ARTIFACTS_READY = False
    logger.exception("Failed to load artifacts: %s", str(e))


# =========================================================
# Constants / validation
# =========================================================
VALID_TRAFFIC = {"people", "vehicles", "containers"}


def json_error(message: str, status_code: int):
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
# Prediction helpers
# =========================================================
def fallback_predictions(port_code: int, traffic: str) -> str:
    logger.warning(
        "Using fallback predictions for port_code=%s, traffic=%s",
        port_code,
        traffic,
    )
    clean_list = [0, 0, 0, 0, 0, 0]
    return " ".join(str(p) for p in clean_list)


def clean_prediction_list(preds: Any) -> str:
    if not isinstance(preds, (list, tuple)):
        clean_list = [0, 0, 0, 0, 0, 0]
        return " ".join(str(p) for p in clean_list)

    cleaned = []
    for pred in preds:
        try:
            value = int(round(float(pred)))
            cleaned.append(max(value, 0))
        except Exception:
            cleaned.append(0)

    if len(cleaned) != 6:
        logger.warning("Prediction list length is %s instead of 6.", len(cleaned))
        if len(cleaned) < 6:
            cleaned.extend([0] * (6 - len(cleaned)))
        else:
            cleaned = cleaned[:6]

    return " ".join(str(p) for p in cleaned)


def get_predictions_from_store(port_code: int, traffic: str) -> Optional[str]:
    """
    Tries several common key formats in case the pickle was saved
    with tuple keys, string keys, or nested dictionaries.
    """
    if predictions_store is None:
        return None

    if isinstance(predictions_store, dict):
        tuple_key = (port_code, traffic)
        if tuple_key in predictions_store:
            logger.info("Found predictions using tuple key: %s", tuple_key)
            return clean_prediction_list(predictions_store[tuple_key])

        str_key_underscore = f"{port_code}_{traffic}"
        if str_key_underscore in predictions_store:
            logger.info("Found predictions using string key: %s", str_key_underscore)
            return clean_prediction_list(predictions_store[str_key_underscore])

        str_key_dash = f"{port_code}-{traffic}"
        if str_key_dash in predictions_store:
            logger.info("Found predictions using string key: %s", str_key_dash)
            return clean_prediction_list(predictions_store[str_key_dash])

        str_key_pipe = f"{port_code}|{traffic}"
        if str_key_pipe in predictions_store:
            logger.info("Found predictions using string key: %s", str_key_pipe)
            return clean_prediction_list(predictions_store[str_key_pipe])

        nested_level_1 = predictions_store.get(port_code)
        if isinstance(nested_level_1, dict) and traffic in nested_level_1:
            logger.info(
                "Found predictions using nested dict key: [%s][%s]",
                port_code,
                traffic,
            )
            return clean_prediction_list(nested_level_1[traffic])

        nested_level_1 = predictions_store.get(str(port_code))
        if isinstance(nested_level_1, dict) and traffic in nested_level_1:
            logger.info(
                "Found predictions using nested string dict key: [%s][%s]",
                str(port_code),
                traffic,
            )
            return clean_prediction_list(nested_level_1[traffic])

    logger.warning(
        "No predictions found in store for port_code=%s, traffic=%s",
        port_code,
        traffic,
    )
    return None


def make_predictions(port_code: int, traffic: str) -> str:
    if not ARTIFACTS_READY or predictions_store is None:
        return fallback_predictions(port_code, traffic)

    try:
        preds = get_predictions_from_store(port_code, traffic)
        if preds is None:
            return fallback_predictions(port_code, traffic)
        return preds

    except Exception as e:
        logger.exception("Error accessing predictions_store: %s", str(e))
        return fallback_predictions(port_code, traffic)


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
        return json_error(validation_error, 422)

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
        return json_error(validation_error, 422)

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
    app.run(host="0.0.0.0", port=port, debug=False)