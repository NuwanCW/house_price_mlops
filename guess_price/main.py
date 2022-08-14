import json
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from sqlite3 import DateFromTicks
from typing import Dict, List

import joblib
import mlflow
import optuna
import pandas as pd
import typer
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import pickle
from config import config
from config.config import logger
from guess_price import data, predict, train, util

warnings.filterwarnings("ignore")

# Initialize Typer CLI app
app = typer.Typer()


@app.command()
def etl_data():
    """Extract, load and transform our data assets."""
    # Extract
    # projects = utils.load_json_from_url(url=config.PROJECTS_URL)
    # tags = utils.load_json_from_url(url=config.TAGS_URL)
    # train_path = utils.load_json_from_url(url=config.PROJECTS_URL)
    # test_path = utils.load_json_from_url(url=config.TAGS_URL)

    # Transform
    df = pd.read_csv(config.train_path)
    df = df[config.valid_columns]
    df = df[df.SalePrice.notnull()]  # drop rows w/ no tag

    # print(df.head())
    # get columns as date, categorical and numerical types
    vars_cat = [
        var for var in df.columns if df[var].dtypes == "O"
    ]  # we know this is just a one column
    vars_num = [var for var in df.columns if df[var].dtypes != "O" and var not in ["Id"]]

    # missing imputation
    imputer = SimpleImputer(strategy="most_frequent")
    df[vars_num] = imputer.fit_transform(df[vars_num])

    imputer = SimpleImputer(strategy="constant", fill_value="missing")
    df[vars_cat] = imputer.fit_transform(df[vars_cat])

    def elapsed_years(df, var):
        # capture difference between year variable and year the house was sold
        df[var] = df["YrSold"] - df[var]
        return df

    for var in ["YearRemodAdd"]:
        df = elapsed_years(df, var)
    df = df[config.valid_columns]
    # encoding = {}
    # for i,v in enumerate(df[vars_cat[0]].unique()):
    #     encoding[i]=v
    # ordinal_enc = OrdinalEncoder()
    # df[vars_cat] = ordinal_enc.fit_transform(df[vars_cat])
    # ordinal_encode = Path(config.DATA_DIR, "encode_cat.pkl")
    # with open(ordinal_encode, "wb") as f:
    #     pickle.dump(ordinal_enc, f)

    df.drop(columns=["YrSold"], inplace=True)

    # Load
    train_cleaned_fp = Path(config.DATA_DIR, "train_cleaned.json")

    util.save_dict(d=df.to_dict(orient="records"), filepath=train_cleaned_fp)
    # util.save_dict(d=ordinal_enc.to_dict(orient="records"), filepath=ordinal_encode)
    logger.info("✅ ETL on data is complete, data saved!")


# @app.command()
# def label_data(args_fp: str = "config/args.json") -> None:
#     """Label data with constraints.
#     Args:
#         args_fp (str): location of args.
#     """
#     # Load projects
#     projects_fp = Path(config.DATA_DIR, "projects.json")
#     projects = utils.load_dict(filepath=projects_fp)
#     df = pd.DataFrame(projects)

#     # Load tags
#     tags_dict = {}
#     tags_fp = Path(config.DATA_DIR, "tags.json")
#     for item in utils.load_dict(filepath=tags_fp):
#         key = item.pop("tag")
#         tags_dict[key] = item

#     # Label with constrains
#     args = Namespace(**utils.load_dict(filepath=args_fp))
#     df = df[df.tag.notnull()]  # remove projects with no label
#     df = data.replace_oos_labels(
#         df=df, labels=tags_dict.keys(), label_col="tag", oos_label="other"
#     )
#     df = data.replace_minority_labels(
#         df=df, label_col="tag", min_freq=args.min_freq, new_label="other"
#     )

#     # Save clean labeled data
#     labeled_projects_fp = Path(config.DATA_DIR, "labeled_projects.json")
#     utils.save_dict(d=df.to_dict(orient="records"), filepath=labeled_projects_fp)
#     logger.info("✅ Saved labeled data!")


@app.command()
def train_model(
    args_fp: str = "config/args.json",
    experiment_name: str = "baselines_1",
    run_name: str = "gbr",
    test_run: bool = False,
) -> None:
    """Train a model given arguments.
    Args:
        args_fp (str): location of args.
        experiment_name (str): name of experiment.
        run_name (str): name of specific run in experiment.
        test_run (bool, optional): If True, artifacts will not be saved. Defaults to False.
    """
    # Load labeled data
    projects_fp = Path(config.DATA_DIR, "train_cleaned.json")
    projects = util.load_dict(filepath=projects_fp)
    df = pd.DataFrame(projects)

    # Train
    args = Namespace(**util.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")
        artifacts = train.train(df=df, args=args)
        performance = artifacts["performance"]
        logger.info(json.dumps(performance, indent=2))

        # Log metrics and parameters
        performance = artifacts["performance"]
        mlflow.log_metrics({"MSE": performance["overall"]["MSE"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            util.save_dict(vars(artifacts["args"]), Path(dp, "args.json"), cls=NumpyEncoder)
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            util.save_dict(performance, Path(dp, "performance.json"))
            joblib.dump(artifacts["ordinal_enc"], Path(dp, "ordinal_enc.pkl"))
            mlflow.log_artifacts(dp)

    # Save to config
    if not test_run:  # pragma: no cover, actual run
        open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
        util.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))


@app.command()
def optimize(
    args_fp: str = "config/args.json",
    study_name: str = "optimization",
    num_trials: int = 100,
) -> None:
    """Optimize hyperparameters.
    Args:
        args_fp (str): location of args.
        study_name (str): name of optimization study.
        num_trials (int): number of trials to run in study.
    """
    # Load labeled data
    projects_fp = Path(config.DATA_DIR, "train_cleaned.json")
    projects = util.load_dict(filepath=projects_fp)
    df = pd.DataFrame(projects)

    # Optimize
    args = Namespace(**util.load_dict(filepath=args_fp))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction="minimize", pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="MSE")
    study.optimize(
        lambda trial: train.objective(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # Best trial
    trials_df = study.trials_dataframe()
    # print(trials_df.head())
    trials_df = trials_df.sort_values(["user_attrs_MSE"], ascending=False)
    args = {**args.__dict__, **study.best_trial.params}
    util.save_dict(d=args, filepath=args_fp, cls=NumpyEncoder)
    logger.info(f"\nBest value (MSE): {study.best_trial.value}")
    logger.info(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")


def load_artifacts(run_id: str = None) -> Dict:
    """Load artifacts for a given run_id.
    Args:
        run_id (str): id of run to load artifacts from.
    Returns:
        Dict: run's artifacts.
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()

    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")

    # Load objects from run
    args = Namespace(**util.load_dict(filepath=Path(artifacts_dir, "args.json")))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    ordinal_enc = joblib.load(Path(artifacts_dir, "ordinal_enc.pkl"))
    performance = util.load_dict(filepath=Path(artifacts_dir, "performance.json"))

    return {
        "args": args,
        "model": model,
        "performance": performance,
        "ordinal_enc": ordinal_enc,
    }


@app.command()
def predict_tag(text: List = [], run_id: str = None) -> None:
    """Predict tag for text.
    Args:
        text (str): input text to predict label for.
        run_id (str, optional): run id to load artifacts for prediction. Defaults to None.
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=text, artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2))
    return prediction


if __name__ == "__main__":
    app()  # pragma: no cover, live app
