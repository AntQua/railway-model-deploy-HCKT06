import json
import logging
import os
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import CharField, IntegerField, Model, TextField
from playhouse.db_url import connect
from playhouse.shortcuts import model_to_dict


# =========================================================
# Logging
# =========================================================
class CustomRailwayLogFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "time": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
        })


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

FUTURE_MONTHS = pd.date_range(start="2025-09-01", periods=6, freq="MS")


class BaseModel(Model):
    class Meta:
        database = DB


class Prediction(BaseModel):
    date = CharField(max_length=20)
    port_code = IntegerField()
    traffic = CharField(max_length=50)
    prediction = IntegerField()
    true_value = IntegerField(null=True)


DB.connect(reuse_if_open=True)
DB.create_tables([Prediction], safe=True)


# =========================================================
# Load artifacts & seed DB
# =========================================================
ARTIFACTS_READY = False

try:
    predictions_store = joblib.load("predictions_store.pickle")

    if Prediction.select().count() == 0:
        rows = []
        for (code, traffic), preds in predictions_store.items():
            for date, pred in zip(FUTURE_MONTHS, preds):
                rows.append({
                    "date": date.strftime("%b %Y"),
                    "port_code": int(code),
                    "traffic": traffic,
                    "prediction": int(pred),
                    "true_value": None,
                })
        Prediction.insert_many(rows).execute()
        logger.info("Inserted %s prediction rows into DB.", len(rows))
    else:
        logger.info("Predictions already in DB, skipping insert.")

    ARTIFACTS_READY = True

except Exception as e:
    logger.warning("Failed to load artifacts: %s", str(e))


# =========================================================
# Validation helpers
# =========================================================
VALID_TRAFFIC = {"people", "vehicles", "containers"}


def json_error(message: str, status: int = 422):
    return jsonify({"error": message}), status


def get_request_json() -> Optional[Dict[str, Any]]:
    if not request.is_json:
        return None
    return request.get_json(silent=True)


def normalize_traffic(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if value in VALID_TRAFFIC else None


def validate_predict_payload(payload: Dict) -> Optional[str]:
    missing = {"port_code", "traffic"} - payload.keys()
    if missing:
        return f"Missing required fields: {sorted(missing)}"
    if not isinstance(payload["port_code"], int):
        return "Field 'port_code' must be an integer."
    if normalize_traffic(payload["traffic"]) is None:
        return "Field 'traffic' must be one of: people, vehicles, containers."
    return None


def validate_update_payload(payload: Dict) -> Optional[str]:
    missing = {"date", "port_code", "traffic", "true_value"} - payload.keys()
    if missing:
        return f"Missing required fields: {sorted(missing)}"
    if not isinstance(payload["date"], str) or not payload["date"].strip():
        return "Field 'date' must be a non-empty string."
    if not isinstance(payload["port_code"], int):
        return "Field 'port_code' must be an integer."
    if normalize_traffic(payload["traffic"]) is None:
        return "Field 'traffic' must be one of: people, vehicles, containers."
    if not isinstance(payload["true_value"], int):
        return "Field 'true_value' must be an integer."
    return None


# =========================================================
# Routes
# =========================================================
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "artifacts_ready": ARTIFACTS_READY,
        "endpoints": ["/predict", "/update", "/list-db-contents"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    payload = get_request_json()
    if payload is None:
        return json_error("Request must contain valid JSON.")

    logger.info("Predict request: %s", payload)

    error = validate_predict_payload(payload)
    if error:
        return json_error(error)

    port_code = payload["port_code"]
    traffic = normalize_traffic(payload["traffic"])

    try:
        rows = (
            Prediction
            .select()
            .where(Prediction.port_code == port_code, Prediction.traffic == traffic)
            .order_by(Prediction.date)
        )

        if not rows.exists():
            logger.warning("No predictions in DB for port_code=%s, traffic=%s", port_code, traffic)
            return json_error(f"No predictions found for port_code={port_code}, traffic={traffic}.", 404)

        prediction = " ".join(str(r.prediction) for r in rows)

        response = {"port_code": port_code, "traffic": traffic, "prediction": prediction}
        logger.info("Predict response: %s", response)
        return jsonify(response), 200

    except Exception as e:
        logger.exception("Unhandled error in /predict: %s", str(e))
        return json_error("Internal server error.", 500)


@app.route("/update", methods=["POST"])
def update():
    payload = get_request_json()
    if payload is None:
        return json_error("Request must contain valid JSON.")

    logger.info("Update request: %s", payload)

    error = validate_update_payload(payload)
    if error:
        return json_error(error)

    date = payload["date"].strip()
    port_code = payload["port_code"]
    traffic = normalize_traffic(payload["traffic"])
    true_value = payload["true_value"]

    try:
        updated = (
            Prediction
            .update(true_value=true_value)
            .where(
                Prediction.port_code == port_code,
                Prediction.traffic == traffic,
                Prediction.date == date,
            )
            .execute()
        )

        if updated == 0:
            logger.warning("No row found to update for port_code=%s, traffic=%s, date=%s", port_code, traffic, date)
            return json_error(f"No prediction found for port_code={port_code}, traffic={traffic}, date={date}.", 404)

        response = {"date": date, "port_code": port_code, "traffic": traffic, "true_value": true_value}
        logger.info("Update saved: %s", response)
        return jsonify(response), 200

    except Exception as e:
        logger.exception("Unhandled error in /update: %s", str(e))
        return json_error("Internal server error.", 500)


@app.route("/list-db-contents", methods=["GET"])
def list_db_contents():
    try:
        return jsonify({
            "predictions": [model_to_dict(r) for r in Prediction.select()],
        })
    except Exception as e:
        logger.exception("Unhandled error in /list-db-contents: %s", str(e))
        return json_error("Internal server error.", 500)


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)