"""FastAPI-сервис для прогноза отказа оборудования (ЛР2, вариант 5)."""
import os
import json
from contextlib import asynccontextmanager
from typing import List, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_model, predict_proba, get_threshold, get_schema
from src.features import RAW_FEATURES


#Pydantic-схема входа

class MachineInput(BaseModel):
    type: Literal['L', 'M', 'H'] = Field(..., description='Тип станка')
    air_temperature: float = Field(..., ge=295, le=305,
                                   description='Темп. воздуха, К')
    process_temperature: float = Field(..., ge=305, le=315,
                                       description='Темп. процесса, К')
    rotational_speed: float = Field(..., ge=1000, le=2900,
                                    description='Скорость, об/мин')
    torque: float = Field(..., ge=0, le=80,
                          description='Момент, Н·м')
    tool_wear: float = Field(..., ge=0, le=260,
                             description='Износ, мин')

    model_config = {'extra': 'forbid'}   #запрещаем лишние поля


class PredictionResponse(BaseModel):
    proba: float
    decision: int
    threshold: float


class BatchResponse(BaseModel):
    proba: List[float]
    decision: List[int]
    threshold: float

#Lifespan: загрузка модели при старте
@asynccontextmanager
async def lifespan(app: FastAPI):
    #Модель загружается ОДИН раз при старте сервиса
    model_path = os.environ.get('MODEL_PATH', 'models/model.joblib')
    schema_path = os.environ.get('SCHEMA_PATH', 'models/input_schema.json')
    threshold = float(os.environ.get('THRESHOLD', '0.022'))
    load_model(model_path, schema_path, threshold)
    print(f'[startup] model loaded from {model_path}, threshold={threshold}')
    yield


app = FastAPI(
    title='Equipment Failure Prediction (LR2, variant 5)',
    version='1.0.0',
    lifespan=lifespan,
)

#Эндпоинты
@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/model-info')
def model_info():
    return {
        'model_name': 'lab2_equipment_failure',
        'version': '1',
        'threshold': get_threshold(),
        'features': RAW_FEATURES,
        'positive_class': 'machine_failure',
    }


@app.post('/predict', response_model=PredictionResponse)
def predict_one(item: MachineInput):
    record = item.model_dump()
    proba = float(predict_proba([record])[0])
    return PredictionResponse(
        proba=proba,
        decision=int(proba >= get_threshold()),
        threshold=get_threshold(),
    )


@app.post('/predict-batch', response_model=BatchResponse)
def predict_batch(items: List[MachineInput]):
    if not items:
        raise HTTPException(status_code=400, detail='empty batch')
    records = [it.model_dump() for it in items]
    proba = predict_proba(records)
    decisions = (proba >= get_threshold()).astype(int).tolist()
    return BatchResponse(
        proba=proba.tolist(),
        decision=decisions,
        threshold=get_threshold(),
    )
