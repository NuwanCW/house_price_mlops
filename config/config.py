import logging
import sys
from ast import Try
from pathlib import Path

import mlflow

# import pretty_errors
from rich.logging import RichHandler

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
STORE_DIR = Path(BASE_DIR, "stores")
LOGS_DIR = Path(BASE_DIR, "logs")

# Stores
MODEL_REGISTRY = Path(STORE_DIR, "model")
BLOB_STORE = Path(STORE_DIR, "blob")

# create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
BLOB_STORE.mkdir(parents=True, exist_ok=True)

# MLFlow model registry
mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))

# Logger
logging_config = {
    "version": 1,
    "disabel_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)

# Assets
train_path = Path(DATA_DIR, "train.csv")
test_path = Path(DATA_DIR, "test.csv")
valid_columns = [
    "LotArea",
    "OverallQual",
    "YearRemodAdd",
    "BsmtQual",
    "BsmtFinSF1",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "GrLivArea",
    "GarageCars",
    "SalePrice",
    "YrSold",
]
