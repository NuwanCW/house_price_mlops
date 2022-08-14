import argparse
import os
import time, json
import typing as t
from random import randint, choice
from collections import defaultdict
import pandas as pd
import requests

# from gradient_boosting_model.config.core import config
# from gradient_boosting_model.processing.data_management import load_dataset

# LOCAL_URL = f'http://{os.getenv("DB_HOST", "localhost")}:5000'
data_dict = defaultdict(list)
url = "http://0.0.0.0:8000/predict"
headers = {"content-type": "application/json", "accept": "application/json"}
# HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}
data = {
    "LotArea_map": {"min": 1470, "max": 56600},
    "OverallQual_map": {"min": 1, "max": 10},
    "YearRemodAdd": {"min": 1950, "max": 2010},
    "BsmtQual": ("Gd", "TA", "Ex", "Fa"),
    "BsmtFinSF1": {"min": 0, "max": 5644},
    "TotalBsmtSF": {"min": 0, "max": 6110},
    "1stFlrSF_map": {"min": 407, "max": 5095},
    "2ndFlrSF_map": {"min": 0, "max": 1862},
    "GrLivArea": {"min": 334, "max": 5642},
    "GarageCars": {"min": 0, "max": 4},
}
# "1stFlrSF_map": {"min": 407, "max": 5095},
# "OverallQual_map": {"min": 1, "max": 10},
# "LotArea_map": {"min": 1470, "max": 56600},
# "BsmtFinSF1": {"min": 0, "max": 5644},
# "TotalBsmtSF": {"min": 0, "max": 6110},
# "GrLivArea": {"min": 334, "max": 5642},


def _generate_random_int(value_ranges) -> int:
    """Generate random integer within a min and max range."""
    random_value = randint(int(value_ranges["min"]), int(value_ranges["max"]))
    return int(random_value)


def _select_random_category(value_options: t.Sequence) -> str:
    """Select random category given a sequence of categories."""
    random_category = choice(value_options)
    # d = json.loads(random_category)
    return str(random_category)


def prepare(data, n=2):
    n = randint(1, 7)
    data_records = []
    print(n)
    for i in range(n):
        val = []
        for k, v in data.items():
            if k != "BsmtQual":
                # print(v)
                val.append(_generate_random_int(value_ranges=v))
            else:
                val.append(_select_random_category(value_options=v))
            # print({"val": val})
        record = {"val": val}
        data_records.append(record)
    data_vals = {
        "records": data_records,
    }
    return str(data_vals).replace("'", '"')


for j in range(500):
    test_data = prepare(data)
    r = requests.post(url, data=test_data, headers=headers)

    # # post data to url
    data_ = r.json()
    print(data_)
    # print(data)
    d = randint(1, 20)
    time.sleep(d)
# print(test_data)
# response = requests.post(
#             f"{LOCAL_URL}",
#             headers=HEADERS,
#             json=[data.to_dict()],
#         )


# for i in range(len(test_data)):
#     print(test_data[i])


# def _prepare_inputs(dataframe: pd.DataFrame) -> pd.DataFrame:
#     """Prepare input data by removing key rows with NA values."""
#     clean_inputs_df = dataframe.dropna(
#         subset=config.model_config.features + ["KitchenQual", "LotFrontage"]
#     ).copy()

#     clean_inputs_df.loc[:, "FirstFlrSF"] = clean_inputs_df["FirstFlrSF"].apply(
#         _generate_random_int, value_ranges=1stFlrSF_map
#     )
#     clean_inputs_df.loc[:, "SecondFlrSF"] = clean_inputs_df["SecondFlrSF"].apply(
#         _generate_random_int, value_ranges=2ndFlrSF_map
#     )
#     clean_inputs_df.loc[:, "LotArea"] = clean_inputs_df["LotArea"].apply(
#         _generate_random_int, value_ranges=LotArea_map
#     )

#     clean_inputs_df.loc[:, "BsmtQual"] = clean_inputs_df["BsmtQual"].apply(
#         _select_random_category, value_options=BsmtQual
#     )

#     return clean_inputs_df


# def populate_database(n_predictions: int = 500, anomaly: bool = False) -> None:
#     """
#     Manipulate the test data to generate random
#     predictions and save them to the database.
#     Before running this script, ensure that the
#     API and Database docker containers are running.
#     """

#     print(f"Preparing to generate: {n_predictions} predictions.")

#     # Load the gradient boosting test dataset which
#     # is included in the model package
#     test_inputs_df = load_dataset(file_name="test.csv")
#     clean_inputs_df = _prepare_inputs(dataframe=test_inputs_df)
#     if len(clean_inputs_df) < n_predictions:
#         print(
#             f"If you want {n_predictions} predictions, you need to"
#             "extend the script to handle more predictions."
#         )

#     if anomaly:
#         # set extremely low values to generate an outlier
#         n_predictions = 1
#         clean_inputs_df.loc[:, "FirstFlrSF"] = 1
#         clean_inputs_df.loc[:, "LotArea"] = 1
#         clean_inputs_df.loc[:, "OverallQual"] = 1
#         clean_inputs_df.loc[:, "GrLivArea"] = 1

#     clean_inputs_df = clean_inputs_df.where(pd.notnull(clean_inputs_df), None)
#     for index, data in clean_inputs_df.iterrows():
#         if index > n_predictions:
#             if anomaly:
#                 print('Created 1 anomaly')
#             break

#         response = requests.post(
#             f"{LOCAL_URL}/v1/predictions/regression",
#             headers=HEADERS,
#             json=[data.to_dict()],
#         )
#         response.raise_for_status()

#         if index % 50 == 0:
#             print(f"{index} predictions complete")

#             # prevent overloading the server
#             time.sleep(0.5)

#     print("Prediction generation complete.")


# if __name__ == "__main__":
#     anomaly = False
#     parser = argparse.ArgumentParser(
#         description='Send random requests to House Price API.')
#     parser.add_argument('--anomaly', help="generate unusual inputs")
#     args = parser.parse_args()
#     if args.anomaly:
#         print("Generating unusual inputs")
#         anomaly = True

#     populate_database(n_predictions=500, anomaly=anomaly)
