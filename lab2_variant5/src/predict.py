"""Инференс: загрузка сохранённой модели и предсказание по списку словарей."""
import json
import os
from typing import List

import joblib
import numpy as np
import pandas as pd

from src.features import RAW_FEATURES

DEFAULT_MODEL_PATH = os.environ.get('MODEL_PATH', 'models/model.joblib')
DEFAULT_SCHEMA_PATH = os.environ.get('SCHEMA_PATH', 'models/input_schema.json')
DEFAULT_THRESHOLD = 0.022

_MODEL = None
_SCHEMA = None
_THRESHOLD = None


def load_model(model_path: str = DEFAULT_MODEL_PATH,
               schema_path: str = DEFAULT_SCHEMA_PATH,
               threshold: float = DEFAULT_THRESHOLD) -> None:
    """Загружает модель один раз (вызывается при старте сервиса)."""
    global _MODEL, _SCHEMA, _THRESHOLD
    _MODEL = joblib.load(model_path)
    with open(schema_path, encoding='utf-8') as f:
        _SCHEMA = json.load(f)
    _THRESHOLD = float(threshold)


def get_threshold() -> float:
    return _THRESHOLD


def get_schema() -> dict:
    return _SCHEMA


def predict_proba(records: List[dict]) -> np.ndarray:
    """Возвращает массив вероятностей для списка объектов."""
    if _MODEL is None:
        raise RuntimeError('Модель не загружена: вызовите load_model() при старте')
    df = pd.DataFrame(records)
    #Гарантируем порядок колонок
    df = df[RAW_FEATURES]
    return _MODEL.predict_proba(df)[:, 1]


def predict_decision(records: List[dict]) -> List[int]:
    """Возвращает решения 0/1 по порогу."""
    proba = predict_proba(records)
    return (proba >= _THRESHOLD).astype(int).tolist()
