"""Нагрузочный сценарий для ЛР2, вариант 5.

Смешанный профиль:
  - 80% — одиночные /predict (потоковый режим, каждую минуту)
  - 20% — батч /predict-batch (на 50 объектов)

Требование варианта 5: p95 ≤ 200 мс для одиночного запроса.
"""
from locust import HttpUser, task, between


SINGLE = {
    'type': 'L',
    'air_temperature': 300.0,
    'process_temperature': 310.0,
    'rotational_speed': 1500.0,
    'torque': 40.0,
    'tool_wear': 100.0,
}

BATCH_50 = [SINGLE] * 50


class ScoringUser(HttpUser):
    wait_time = between(0.01, 0.05)   #имитация потока

    @task(8)
    def predict_one(self):
        self.client.post('/predict', json=SINGLE)

    @task(2)
    def predict_batch_50(self):
        self.client.post('/predict-batch', json=BATCH_50)
