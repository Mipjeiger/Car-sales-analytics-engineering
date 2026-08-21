from datetime import datetime
from kafka import KafkaProducer
import json
import uuid

class KafkaEventProducer:
    def __init__(self, bootstrap_servers='localhost:9092'):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def send_prediction_event(self, prediction_data):
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "prediction",
            "data": prediction_data,
            'timestamp': datetime.now().isoformat()
        }
        self.producer.send('predictions', event)

    def send_training_event(self, model_data):
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "model_trained",
            "data": model_data,
            'timestamp': datetime.now().isoformat()
        }
        self.producer.send('training', event)

    def close(self):
        self.producer.close()