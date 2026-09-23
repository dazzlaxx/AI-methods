"""Признаковые преобразования для варианта 5.Модуль нужен, чтобы сохранённый model.joblib корректно загружался
в другом процессе. Все функции принимают и возвращают DataFrame.
"""
import numpy as np
import pandas as pd

CAT_COLS = ['type']


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Новые физические признаки для задачи отказа оборудования."""
    X = X.copy()

    #1. Мощность, Вт = момент (Н·м) × угловая скорость (рад/с)
    X['power_w'] = (X['torque'] *
                    X['rotational_speed'] *
                    2 * np.pi / 60)

    #2. Разность температур: процесса и воздуха, К
    X['temp_diff_k'] = X['process_temperature'] - X['air_temperature']

    #3. Произведение износа на момент
    X['wear_x_torque'] = X['tool_wear'] * X['torque']

    #4. Мощность на единицу износа (износ + 1, чтобы избежать деления на 0)
    X['power_per_wear'] = X['power_w'] / (X['tool_wear'] + 1.0)

    #5. Отношение скорости к моменту (устойчиво к нулю)
    X['speed_per_torque'] = X['rotational_speed'] / (X['torque'] + 1.0)

    return X


def to_category(X: pd.DataFrame) -> pd.DataFrame:
    """Приводит известные категориальные столбцы к типу category."""
    X = X.copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype('category')
    return X
