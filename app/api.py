from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict
import uuid
from fastapi import FastAPI, Request
from starlette.responses import Response
import json
from app.schemas import PredictPayload, valid_features
from config import config
from config.config import logger
from guess_price import main, predict
from redis_om import get_redis_connection

from .custom.monitoring import instrumentator
from .schemas import valid_features
from fastapi.middleware.cors import CORSMiddleware

# from prometheus_fastapi_instrumentator import Instrumentator

redis_db = get_redis_connection(host="192.168.1.136", port="6379")
# Define application
app = FastAPI(
    title="Guess_price - predict house values via ml",
    description="predict house values via ml.",
    version="0.1",
)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

instrumentator.instrument(app)
instrumentator.expose(app, include_in_schema=False, should_gzip=True)
# Instrumentator().instrument(app).expose(app)


@app.on_event("startup")
def load_artifacts():
    global artifacts
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id=run_id)
    logger.info("Ready for inference!")


def construct_response(f):
    """Construct a JSON response for an endpoint."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs) -> Dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap


@app.get("/", tags=["General"])
@construct_response
async def root():
    def _index(request: Request) -> Dict:
        """Health check."""
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {},
        }
        return response


@app.get("/performance", tags=["Performance"])
@construct_response
def _performance(request: Request, filter: str = None) -> Dict:
    """Get the performance metrics."""
    performance = artifacts["performance"]
    data = {"performance": performance.get(filter, performance)}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response


@app.get("/args", tags=["Arguments"])
@construct_response
def _args(request: Request) -> Dict:
    """Get all arguments used for the run."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "args": vars(artifacts["args"]),
        },
    }
    return response


@app.get("/args/{arg}", tags=["Arguments"])
@construct_response
def _arg(request: Request, arg: str) -> Dict:
    """Get a specific parameter's value used for the run."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            arg: vars(artifacts["args"]).get(arg, ""),
        },
    }
    return response


@app.post("/predict", tags=["Prediction"])
def _predict(request: Request, response: Response, payload: valid_features) -> Dict:
    """Predict tags for a list of texts."""
    print(payload)
    # print(payload.dict())
    # texts = [item.val for item in payload.records]
    texts = [item for k, item in payload.dict().items()]
    print(texts)
    predictions = predict.predict(texts=texts, artifacts=artifacts)
    # print(predictions)
    response_ = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predictions": predictions},
    }
    k = str(uuid.uuid4())
    time = datetime.now().isoformat(sep=" ", timespec="milliseconds")
    d = {"id": k, "time": time, "response": response_}
    redis_db.rpush("pred_data", json.dumps(d))
    # print(response_)
    response.headers["X-model-predict"] = str(
        [i["predicted_price"] for i in response_["data"]["predictions"]]
    )
    print(response.headers["X-model-predict"])
    return response_
