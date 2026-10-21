"""Признаки варианта 5 (отказ оборудования).

Импортируется и ноутбуком (Шаг 3), и сервисом (Шаг 4),
и модулем предсказания (predict.py).
"""
import numpy as np
import pandas as pd

RANDOM_STATE = 42

#Категориальная колонка (единственная в наборе AI4I)
CAT_COLS = ['type']

#Сырые признаки (6 штук) — вход модели
RAW_FEATURES = [
    'type',
    'air_temperature',
    'process_temperature',
    'rotational_speed',
    'torque',
    'tool_wear',
]

#елевая переменная
TARGET = 'machine_failure'


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Добавляет физические признаки.

    Из 5 кандидатов из ЛР1 оставлены 4 (power_per_wear удалён 
    по результатам абляции). Именно на этих 4 обучена финальная 
    модель LightGBM из ЛР1 — сохранение их гарантирует 
    совместимость с model.joblib.
    """
    X = X.copy()

    #1. Мощность, Вт = момент (Н·м) × угловая скорость (рад/с)
    X['power_w'] = X['torque'] * X['rotational_speed'] * 2 * np.pi / 60

    #2. Разность температур: процесса и воздуха, К
    X['temp_diff_k'] = X['process_temperature'] - X['air_temperature']

    #3. Произведение износа на момент
    X['wear_x_torque'] = X['tool_wear'] * X['torque']

    #4. Отношение скорости к моменту (устойчиво к нулю)
    X['speed_per_torque'] = X['rotational_speed'] / (X['torque'] + 1.0)

    return X


def to_category(X: pd.DataFrame) -> pd.DataFrame:
    """Приводит type к category для LightGBM."""
    X = X.copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype('category')
    return X


def full_transform(X: pd.DataFrame) -> pd.DataFrame:
    """Полный конвейер: add_features + to_category."""
    return to_category(add_features(X))
