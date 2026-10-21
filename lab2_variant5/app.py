"""Gradio-интерфейс для ЛР2, вариант 5."""
import gradio as gr
import requests

API_URL = 'http://127.0.0.1:8000'


def score(machine_type, air_t, proc_t, speed, torque, wear):
    payload = {
        'type': machine_type,
        'air_temperature': float(air_t),
        'process_temperature': float(proc_t),
        'rotational_speed': float(speed),
        'torque': float(torque),
        'tool_wear': float(wear),
    }
    try:
        r = requests.post(f'{API_URL}/predict', json=payload, timeout=5)
    except Exception as e:
        return f'Ошибка соединения: {e}'
    if r.status_code != 200:
        return f'Ошибка валидации ({r.status_code}):\n{r.json()}'
    out = r.json()
    verdict = 'ОСТАНОВИТЬ' if out['decision'] else 'РАбОТАТЬ'
    return (
        f"Вероятность отказа: {out['proba']:.4f}\n"
        f"Порог: {out['threshold']}\n"
        f"Решение: {verdict}"
    )


demo = gr.Interface(
    fn=score,
    inputs=[
        gr.Dropdown(['L', 'M', 'H'], label='Тип станка', value='L'),
        gr.Number(label='Темп. воздуха, К', value=300.0),
        gr.Number(label='Темп. процесса, К', value=310.0),
        gr.Number(label='Скорость, об/мин', value=1500.0),
        gr.Number(label='Момент, Н·м', value=40.0),
        gr.Number(label='Износ, мин', value=100.0),
    ],
    outputs=gr.Textbox(label='Результат', lines=5),
    title='Прогноз отказа оборудования (ЛР2, вариант 5)',
    description='Модель LightGBM (PR-AUC 0.910 на тесте). Сервис FastAPI.',
)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860,
                share=True, show_error=True)
