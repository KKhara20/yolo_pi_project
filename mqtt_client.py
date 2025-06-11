import paho.mqtt.client as mqtt
import json
from config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, CLIENT_ID

class MqttPublisher:
    def __init__(self):
        self.client = mqtt.Client(CLIENT_ID)
        self.client.connect(MQTT_BROKER, MQTT_PORT)

    def publish_detection(self, track_id, x, y):
        msg = {"id": track_id, "x": x, "y": y}
        self.client.publish(MQTT_TOPIC, json.dumps(msg))
