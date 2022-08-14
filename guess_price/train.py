import json
from argparse import Namespace
from typing import Dict
import argparse
import mlflow
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, SGDRegressor
import lightgbm as lgb
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score
from config.config import logger
from guess_price import data, evaluate, predict, util


def train(args: Namespace, df: pd.DataFrame, trial: optuna.trial._trial.Trial = None) -> Dict:
    """Train model on data.
    Args:
        args (Namespace): arguments to use for training.
        df (pd.DataFrame): data for training.
        trial (optuna.trial._trial.Trial, optional): optimization trial. Defaults to None.
    Raises:
        optuna.TrialPruned: early stopping of trial if it's performing poorly.
    Returns:
        Dict: artifacts from the run.
    """

    # Setup
    util.set_seeds()
    if args.shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    df = df[: args.subset]  # None = all samples
    ordinal_enc = OrdinalEncoder()
    df["BsmtQual"] = ordinal_enc.fit_transform(df[["BsmtQual"]])
    # df = data.preprocess(df, lower=args.lower, stem=args.stem)
    # label_encoder = data.LabelEncoder().fit(df.tag)
    X_train, X_val, X_test, y_train, y_val, y_test = data.get_data_splits(
        X=df[df.columns[~df.columns.isin(["SalePrice"])]].to_numpy(),
        y=df.SalePrice.to_numpy(),
    )
    # print(y_test)
    # test_df = pd.DataFrame({"text": X_test, "tag": label_encoder.decode(y_test)})

    # Oversample
    # oversample = RandomOverSampler(sampling_strategy="all")
    # X_over, y_over = oversample.fit_resample(X_train, y_train)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

    def namespace_to_dict(namespace):
        return {
            k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
            for k, v in vars(namespace).items()
        }

    model = lgb.train(
        namespace_to_dict(args),
        lgb_train,
        num_boost_round=200,
        valid_sets=lgb_eval,
        callbacks=[lgb.early_stopping(stopping_rounds=10)],
    )

    # Training
    # for epoch in range(args.num_epochs):
    # model = SGDRegressor(early_stopping=True)
    # model.fit(X_train, y_train)
    train_mse = mean_squared_error(
        y_train, model.predict(X_train, num_iteration=model.best_iteration)
    )
    train_r2 = r2_score(y_train, model.predict(X_train))
    # print(model.predict(X_train), y_train)
    val_mse = mean_squared_error(y_val, model.predict(X_val, num_iteration=model.best_iteration))
    val_r2 = r2_score(y_train, model.predict(X_train))
    # if not epoch % 10:
    #     logger.info(
    #         f"Epoch: {epoch:02d} | " f"train_mse: {train_mse:.5f}, " f"val_mse: {val_mse:.5f}"
    #     )
    # print(f"train_mse: {train_mse:.5f}, " f"val_mse: {val_mse:.5f}")

    logger.info(f"train_mse: {train_mse:.5f}, " f"val_mse: {val_mse:.5f}")
    # Log
    if not trial:  # mlflow.log_metrics({"train_loss": train_mse, "val_loss": val_mse}, step=epoch)
        mlflow.log_metrics({"train_loss": train_mse, "val_loss": val_mse})

    # Pruning (for optimization in next section)
    if trial:  # pragma: no cover, optuna pruning
        # trial.report(val_mse, epoch)
        trial.report(val_mse, 1)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Threshold
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    performance = evaluate.get_metrics(y_true=y_test, y_pred=y_pred)

    return {
        "args": args,
        "model": model,
        "performance": performance,
        "ordinal_enc": ordinal_enc,
    }


def objective(args: Namespace, df: pd.DataFrame, trial: optuna.trial._trial.Trial) -> float:
    """Objective function for optimization trials.
    Args:
        args (Namespace): arguments to use for training.
        df (pd.DataFrame): data for training.
        trial (optuna.trial._trial.Trial, optional): optimization trial.
    Returns:
        float: metric value to be used for optimization.
    """
    # Parameters to tune
    args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-2, 1e0)
    args.lambda_l1 = (trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),)
    args.lambda_l2 = (trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),)
    args.num_leaves = (trial.suggest_int("num_leaves", 2, 256),)
    args.feature_fraction = (trial.suggest_float("feature_fraction", 0.4, 1.0),)
    args.bagging_fraction = (trial.suggest_float("bagging_fraction", 0.4, 1.0),)
    args.bagging_freq = (trial.suggest_int("bagging_freq", 1, 7),)
    args.min_child_samples = (trial.suggest_int("min_child_samples", 5, 100),)
    # args.optimizer = trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])
    args.boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])

    # Train & evaluate
    artifacts = train(args=args, df=df, trial=trial)

    # Set additional attributes
    overall_performance = artifacts["performance"]["overall"]
    logger.info(json.dumps(overall_performance, indent=2))
    trial.set_user_attr("MSE", overall_performance["MSE"])

    return overall_performance["MSE"]
