"""
Run load tests:
locust -f load_test/locustfile.py --host http://127.0.0.1:8000
"""
import json
from locust import HttpUser, task
import pandas as pd
import random

feature_columns = [
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
]

new_names = {
    "LotArea": "LotArea",
    "OverallQual": "OverallQual",
    "YearRemodAdd": "YearRemodAdd",
    "BsmtQual": "BsmtQual",
    "BsmtFinSF1": "BsmtFinSF1",
    "TotalBsmtSF": "TotalBsmtSF",
    "1stFlrSF": "FirststFlrSF",
    "2ndFlrSF": "SecondndFlrSF",
    "GrLivArea": "GrLivArea",
    "GarageCars": "GarageCars",
}
# dataset = (
#     pd.read_csv(
#         "../data/train.csv",
#         delimiter=";",
#     )
#     .rename(columns=feature_columns)
#     .drop("quality", axis=1)
#     .to_dict(orient="records")
# )
dataset = pd.read_csv("../data/train.csv")
dataset = dataset[feature_columns].rename(columns=new_names).to_dict(orient="records")


class HousePredictionUser(HttpUser):
    # @task(1)
    # def healthcheck(self):
    #     self.client.get("/healthcheck")

    @task(10)
    def prediction(self):
        record = random.choice(dataset).copy()
        print(record)
        self.client.post("/predict", json=record)

    @task(2)
    def prediction_bad_value(self):
        record = random.choice(dataset).copy()
        corrupt_key = random.choice(list(record.keys()))
        record[corrupt_key] = "bad data"
        record = json.dumps(record)
        self.client.post("/predict", json=record)