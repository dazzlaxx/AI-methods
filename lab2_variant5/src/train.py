"""Обучение модели LightGBM с препроцессингом."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score,
)

from src.features import full_transform, RAW_FEATURES, TARGET, RANDOM_STATE


def train_model(X: pd.DataFrame, y: pd.Series, params: dict) -> Pipeline:
    """Обучает LightGBM за препроцессингом full_transform.

    Parameters
    ----------
    X : DataFrame с RAW_FEATURES (6 колонок)
    y : бинарная целевая (0/1)
    params : параметры LGBMClassifier

    Returns
    -------
    sklearn Pipeline (fe → model)
    """
    params = {**params, 'random_state': RANDOM_STATE, 'verbose': -1, 'n_jobs': -1}
    model = lgb.LGBMClassifier(**params)
    pipe = Pipeline([
        ('fe', FunctionTransformer(full_transform, validate=False)),
        ('model', model),
    ])
    pipe.fit(X, y)
    return pipe


def evaluate(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Считает метрики на выборке (PR-AUC, ROC-AUC, F1, P, R при пороге 0.5)."""
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        'pr_auc':  float(average_precision_score(y, proba)),
        'roc_auc': float(roc_auc_score(y, proba)),
        'f1':      float(f1_score(y, pred, zero_division=0)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall':  float(recall_score(y, pred, zero_division=0)),
    }
