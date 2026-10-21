"""Тесты FastAPI-сервиса (ЛР2, вариант 5)."""
import pytest
from fastapi.testclient import TestClient

from src.service import app
from src.predict import load_model


@pytest.fixture(scope='module')
def client():
    load_model('models/model.joblib', 'models/input_schema.json', 0.022)
    return TestClient(app)


VALID = {
    'type': 'L',
    'air_temperature': 300.0,
    'process_temperature': 310.0,
    'rotational_speed': 1500.0,
    'torque': 40.0,
    'tool_wear': 100.0,
}


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


def test_predict_ok(client):
    r = client.post('/predict', json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body['proba'] <= 1.0
    assert body['decision'] in (0, 1)
    assert body['threshold'] == 0.022


def test_predict_wrong_type(client):
    bad = {**VALID, 'type': 'X'}
    r = client.post('/predict', json=bad)
    assert r.status_code == 422


def test_predict_missing_field(client):
    bad = {k: v for k, v in VALID.items() if k != 'tool_wear'}
    r = client.post('/predict', json=bad)
    assert r.status_code == 422


def test_predict_out_of_range(client):
    bad = {**VALID, 'tool_wear': -5}
    r = client.post('/predict', json=bad)
    assert r.status_code == 422


def test_predict_batch_1000(client):
    r = client.post('/predict-batch', json=[VALID] * 1000)
    assert r.status_code == 200
    body = r.json()
    assert len(body['proba']) == 1000
    assert len(body['decision']) == 1000
    assert all(0.0 <= p <= 1.0 for p in body['proba'])


def test_predict_batch_empty(client):
    r = client.post('/predict-batch', json=[])
    assert r.status_code == 400


def test_model_info(client):
    r = client.get('/model-info')
    assert r.status_code == 200
    body = r.json()
    assert body['model_name'] == 'lab2_equipment_failure'
    assert body['threshold'] == 0.022
    assert len(body['features']) == 6
